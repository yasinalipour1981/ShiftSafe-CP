"""Extension E experiment runner with progress tracking."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.adaptive_cp import AdaptiveJackknifeCP
from src.baselines_e import E_METHOD_NAMES, fit_e_baseline
from src.base_models import fit_base_models
from src.conformal import (
    physical_feature_indices,
    sample_k_indices,
    stratified_sample_k_indices,
)
from src.datasets import (
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)
from src.metrics import evaluate_all_methods
from src.pubstyle import method_uses_source_lgbm
from src.validity import annotate_guaranteed, get_guarantee_scope, is_guaranteed

logger = logging.getLogger(__name__)

FOLDS_E = ("D1", "D2", "D4")
N_JK_REPS = 10
PRIMARY_K = 20
PRIMARY_ALPHA = 0.10


class ExtensionEProgress:
    """User-visible progress tracker (JSON + markdown)."""

    def __init__(self, out_dir: Path, total_tasks: int):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total = total_tasks
        self.completed = 0
        self.errors: list[str] = []
        self.started = datetime.now(timezone.utc)
        self._task_times: list[float] = []
        self._write(seed=-1, fold="-", method="(starting)", status="ok")

    def tick(self, seed: int, fold: str, method: str, elapsed: float) -> None:
        self.completed += 1
        self._task_times.append(elapsed)
        self._write(seed=seed, fold=fold, method=method, status="ok")

    def error(self, seed: int, fold: str, method: str, msg: str) -> None:
        self.completed += 1
        self.errors.append(f"{seed}/{fold}/{method}: {msg}")
        self._write(seed=seed, fold=fold, method=method, status="error", error=msg)

    def _eta_sec(self) -> float | None:
        if not self._task_times:
            return None
        avg = sum(self._task_times) / len(self._task_times)
        remaining = max(0, self.total - self.completed)
        return avg * remaining

    def _write(
        self,
        seed: int,
        fold: str,
        method: str,
        status: str,
        error: str | None = None,
    ) -> None:
        pct = 100.0 * self.completed / max(self.total, 1)
        eta = self._eta_sec()
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed": self.completed,
            "total": self.total,
            "percent": round(pct, 2),
            "last_task": {"seed": seed, "fold": fold, "method": method, "status": status},
            "errors": self.errors,
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "elapsed_seconds": (datetime.now(timezone.utc) - self.started).total_seconds(),
        }
        json_path = self.out_dir / "progress.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        md_lines = [
            "# Extension E Progress",
            "",
            f"**Updated:** {payload['updated_at']}",
            f"**Progress:** {self.completed}/{self.total} ({pct:.1f}%)",
        ]
        if eta is not None:
            md_lines.append(f"**ETA:** ~{eta / 60:.1f} min")
        md_lines.extend([
            "",
            f"**Last:** seed={seed} fold={fold} method={method} → {status}",
            "",
            "## Errors",
        ])
        if self.errors:
            md_lines.extend(f"- {e}" for e in self.errors[-20:])
        else:
            md_lines.append("- (none)")
        (self.out_dir / "PROGRESS.md").write_text("\n".join(md_lines), encoding="utf-8")


#: Calibration draw used by this module. The submitted version used the
#: strength-quartile draw; the 2026-09 revision re-runs the external benchmark on
#: the outcome-independent random draw so that every method, including the
#: comparators, sees identical draws (Reviewer 1 c.1, Reviewer 2 c.4).
DRAW_SCHEME = "strength_quartile"


def derive_k_indices(y_pool: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Re-derive K indices; must match the confirmatory draws of this run."""
    if DRAW_SCHEME == "strength_quartile":
        return stratified_sample_k_indices(y_pool, K, seed=seed)
    return sample_k_indices(y_pool, K, seed=seed, scheme=DRAW_SCHEME)


def assert_k_index_match(
    y_pool: np.ndarray,
    K: int,
    seed: int,
    reference: np.ndarray,
    *,
    fold: str,
) -> np.ndarray:
    """Assert K-index equality with reference (AdaptiveJackknifeCP indices)."""
    derived = derive_k_indices(y_pool, K, seed)
    ref = np.asarray(reference, dtype=int)
    if not np.array_equal(np.sort(derived), np.sort(ref)):
        raise AssertionError(
            f"K-index mismatch fold={fold} seed={seed}: "
            f"derived={sorted(derived.tolist())} ref={sorted(ref.tolist())}"
        )
    return ref


def save_k_indices(
    out_dir: Path,
    fold: str,
    seed: int,
    indices: np.ndarray,
) -> None:
    kdir = out_dir / "k_indices"
    kdir.mkdir(parents=True, exist_ok=True)
    path = kdir / f"fold_{fold}_seed{seed}_K{PRIMARY_K}.json"
    with open(path, "w") as f:
        json.dump({"fold": fold, "seed": seed, "K": PRIMARY_K, "indices": indices.tolist()}, f)


def _load_reference_k_indices(
    out_dir: Path,
    fold: str,
    seed: int,
) -> np.ndarray | None:
    path = out_dir / "k_indices" / f"fold_{fold}_seed{seed}_K{PRIMARY_K}.json"
    if path.exists():
        with open(path) as f:
            return np.array(json.load(f)["indices"], dtype=int)
    return None


def _jkplus_bootstrap_intervals(
    base_model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
    seed: int,
    n_reps: int,
    use_gpu: bool,
    held_out_domain: str | None = None,
    calib_domains: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Average JKplus-source intervals over bootstrap subsamples.

    Subsample is drawn from source calib only (already excluding held-out fold).
    LOO LightGBM refits run on CPU by default to avoid GPU OOM from O(n) models.
    """
    import gc

    from src.baselines_e import JKPlusSourceCP

    if held_out_domain is not None and calib_domains is not None:
        n_leak = int(np.sum(np.asarray(calib_domains) == held_out_domain))
        if n_leak > 0:
            raise AssertionError(
                f"JKplus calib pool LEAKAGE before subsample: {n_leak} rows from '{held_out_domain}'"
            )

    # Force CPU for LOO swarm: GPU LOO×300 models OOM'd the full E-run on D4
    jk_use_gpu = False
    _ = use_gpu  # reserved if a future low-memory GPU path is added

    n = len(X_calib)
    max_n = 300
    lowers, uppers = [], []
    n_models_list: list[int] = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + 1000 * rep)
        if n > max_n:
            idx = rng.choice(n, max_n, replace=False)
            X_sub, y_sub = X_calib[idx], y_calib[idx]
            dom_sub = None if calib_domains is None else np.asarray(calib_domains)[idx]
        else:
            X_sub, y_sub = X_calib, y_calib
            dom_sub = calib_domains
        jk = JKPlusSourceCP(
            base_model, X_sub, y_sub, alpha=alpha, seed=seed + rep,
            max_calib_samples=max_n, use_gpu=jk_use_gpu,
            held_out_domain=held_out_domain,
            calib_domains=dom_sub,
        )
        jk.fit()
        n_models_list.append(int(jk.diagnostics.get("n_models_trained", 0)))
        lo, hi = jk.predict_intervals(X_eval, alpha=alpha)
        lowers.append(lo)
        uppers.append(hi)
        if jk._jk is not None:
            jk._jk.release_loo_models()
        del jk
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info(
            "JKplus bootstrap rep %d/%d done (held_out=%s, n_models=%d)",
            rep + 1, n_reps, held_out_domain, n_models_list[-1],
        )
    diag = {
        "n_reps": n_reps,
        "n_models_trained_per_rep": n_models_list,
        "n_models_trained_mean": float(np.mean(n_models_list)) if n_models_list else 0.0,
        "max_calib_samples": max_n,
        "held_out_domain": held_out_domain,
        "jk_use_gpu": jk_use_gpu,
    }
    return np.mean(lowers, axis=0), np.mean(uppers, axis=0), diag


def run_extension_e(
    config: dict[str, Any],
    *,
    seeds: list[int],
    folds: tuple[str, ...] = FOLDS_E,
    methods: tuple[str, ...] = E_METHOD_NAMES,
    K: int = PRIMARY_K,
    alpha: float = PRIMARY_ALPHA,
    alpha_sweep: list[float] | None = None,
    smoke: bool = False,
    skip_jk: bool = False,
    output_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Run Extension E baselines on confirmatory protocol."""
    global DRAW_SCHEME
    ext_cfg = config.get("extension_e", {}) or {}
    DRAW_SCHEME = ext_cfg.get("calibration_draw", "strength_quartile")
    exact_jkp = bool(ext_cfg.get("exact_jackknife_plus", False))
    out_dir = Path(config.get("results_dir", "results")) / ext_cfg.get(
        "output_subdir", "extension_e")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_csv) if output_csv else out_dir / "EXTENSION_E_results.csv"

    per_seed_methods = [m for m in methods if m != "JKplus-source"]
    n_tasks = len(seeds) * len(folds) * len(per_seed_methods)
    if "JKplus-source" in methods and not skip_jk:
        n_tasks += len(folds)
    progress = ExtensionEProgress(out_dir, n_tasks)

    datasets = load_all_datasets(config, seed=seeds[0])
    train_calib_ratio = config.get("evaluation", {}).get("train_calib_split", 0.8)
    if isinstance(config.get("experiment"), dict):
        train_calib_ratio = config["experiment"].get("train_calib_ratio", train_calib_ratio)

    lodo_splits = create_lodo_splits(datasets, train_calib_ratio, seed=seeds[0])
    lodo_splits = {k: v for k, v in lodo_splits.items() if k in folds}

    use_gpu = config.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    n_trials = config.get("n_optuna_trials", 0)
    alpha_levels = alpha_sweep or [alpha]

    all_rows: list[pd.DataFrame] = []
    jk_cache: dict[str, dict[str, Any]] = {}

    # JKplus-source: once per fold, amortized (10 bootstrap reps)
    if "JKplus-source" in methods and not skip_jk:
        for fold, (df_train, df_calib, df_test) in lodo_splits.items():
            feature_cols = get_feature_columns(df_train)
            df_train = align_dataframe_features(df_train)
            df_calib = align_dataframe_features(df_calib)
            df_test = align_dataframe_features(df_test)
            X_train = df_train[feature_cols].values.astype(np.float64)
            y_train = df_train["strength_mpa"].values.astype(np.float64)
            X_calib = df_calib[feature_cols].values.astype(np.float64)
            y_calib = df_calib["strength_mpa"].values.astype(np.float64)
            X_test_full = df_test[feature_cols].values.astype(np.float64)
            y_test_full = df_test["strength_mpa"].values.astype(np.float64)

            base_models = fit_base_models(
                X_train, y_train, X_calib, y_calib,
                config=config, device=device, n_trials=n_trials,
            )
            primary = base_models["LightGBM"]

            # Source-only hard assert: calib domains must exclude held-out fold
            if "domain_id" in df_calib.columns and "domain_id" in df_test.columns:
                calib_domains = df_calib["domain_id"].astype(str).values
                test_domains = set(df_test["domain_id"].astype(str).unique())
            elif "dataset" in df_calib.columns:
                calib_domains = df_calib["dataset"].astype(str).values
                test_domains = {str(fold)}
            else:
                calib_domains = np.array([f"source_not_{fold}"] * len(df_calib))
                test_domains = {str(fold)}
            test_dom = sorted(test_domains)[0]
            n_leak = int(np.sum(np.isin(calib_domains, list(test_domains))))
            if n_leak > 0:
                raise AssertionError(
                    f"LODO fold '{fold}': {n_leak} calib rows share held-out domain_id {test_domains}"
                )
            # Prefix check (domain_id like D4_mondal vs fold key D4)
            n_prefix = int(np.sum([str(d).startswith(str(fold)) for d in calib_domains]))
            if n_prefix > 0:
                raise AssertionError(
                    f"LODO fold '{fold}': {n_prefix} calib rows have domain_id prefix '{fold}'"
                )

            n_jk = 1 if smoke else N_JK_REPS
            cache_dir = out_dir / "jk_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"jk_{fold}_alpha{alpha:.2f}.npz"
            t0 = time.time()
            try:
                if cache_path.exists() and not smoke:
                    data = np.load(cache_path)
                    lo, hi = data["lo"], data["hi"]
                    y_cached = data["y"]
                    if len(y_cached) != len(y_test_full):
                        raise RuntimeError(f"JK cache length mismatch for {fold}")
                    jk_diag = {"n_models_trained_mean": float(data.get("n_models_mean", -1)), "from_cache": True}
                    logger.info("JKplus-source fold=%s loaded from cache %s", fold, cache_path)
                else:
                    lo, hi, jk_diag = _jkplus_bootstrap_intervals(
                        primary, X_calib, y_calib, X_test_full, alpha,
                        seed=142, n_reps=n_jk, use_gpu=False,
                        held_out_domain=test_dom,
                        calib_domains=calib_domains,
                    )
                    np.savez_compressed(
                        cache_path, lo=lo, hi=hi, y=y_test_full,
                        n_models_mean=jk_diag.get("n_models_trained_mean", 0.0),
                    )
                logger.info(
                    "JKplus-source fold=%s n_models_trained_mean=%.1f n_reps=%d diag=%s",
                    fold, jk_diag.get("n_models_trained_mean"), n_jk, jk_diag,
                )
                print(
                    f"[JKplus VERIFY] fold={fold} held_out={test_dom} "
                    f"n_models_trained={jk_diag.get('n_models_trained_per_rep', 'cache')} "
                    f"n_calib={len(X_calib)} max_subsample=300 leak_assert=PASS",
                    flush=True,
                )
                jk_cache[fold] = {"lo": lo, "hi": hi, "y": y_test_full, "diag": jk_diag}
                progress.tick(0, fold, "JKplus-source", time.time() - t0)
            except Exception as exc:
                import traceback

                logger.error("JKplus-source fold=%s FAILED: %s\n%s", fold, exc, traceback.format_exc())
                progress.error(0, fold, "JKplus-source", str(exc))
                raise

    for seed in seeds:
        set_seed = seed
        lodo_splits = create_lodo_splits(datasets, train_calib_ratio, seed=set_seed)
        lodo_splits = {k: v for k, v in lodo_splits.items() if k in folds}

        for fold_idx, (fold, (df_train, df_calib, df_test)) in enumerate(lodo_splits.items()):
            feature_cols = get_feature_columns(df_train)
            df_train = align_dataframe_features(df_train)
            df_calib = align_dataframe_features(df_calib)
            df_test = align_dataframe_features(df_test)
            X_train = df_train[feature_cols].values.astype(np.float64)
            y_train = df_train["strength_mpa"].values.astype(np.float64)
            X_calib = df_calib[feature_cols].values.astype(np.float64)
            y_calib = df_calib["strength_mpa"].values.astype(np.float64)
            X_test_full = df_test[feature_cols].values.astype(np.float64)
            y_test_full = df_test["strength_mpa"].values.astype(np.float64)
            f_cm_idx = feature_cols.index("f_cm") if "f_cm" in feature_cols else None
            phys_idx = physical_feature_indices(feature_cols)

            t0_fit = time.time()
            base_models = fit_base_models(
                X_train, y_train, X_calib, y_calib,
                config=config, device=device, n_trials=n_trials,
            )
            fit_time = time.time() - t0_fit
            primary = base_models["LightGBM"]

            # Reference K indices via AdaptiveJackknifeCP (confirmatory protocol)
            ajk_ref = AdaptiveJackknifeCP(
                base_model=primary, alpha=alpha, K=K, seed=set_seed,
                exact_jkp=exact_jkp,
                sampler=(None if DRAW_SCHEME == "strength_quartile" else
                         (lambda Xp, yp, k, sd: sample_k_indices(
                             yp, k, seed=sd, scheme=DRAW_SCHEME))),
            )
            ajk_ref.fit(X_test_full, y_test_full, physical_indices=phys_idx)
            k_idx = assert_k_index_match(
                y_test_full, K, set_seed, ajk_ref.calib_target_indices, fold=fold,
            )
            save_k_indices(out_dir, fold, set_seed, k_idx)

            eval_mask = np.ones(len(y_test_full), dtype=bool)
            eval_mask[k_idx] = False
            X_eval = X_test_full[eval_mask]
            y_eval = y_test_full[eval_mask]

            conformal_methods: dict[str, Any] = {}
            method_calib_times: dict[str, float] = {}

            for method in per_seed_methods:
                t0 = time.time()
                try:
                    cp = fit_e_baseline(
                        method,
                        base_model=primary,
                        X_train=X_train,
                        y_train=y_train,
                        X_calib=X_calib,
                        y_calib=y_calib,
                        X_target_pool=X_test_full,
                        y_target_pool=y_test_full,
                        X_eval=X_eval,
                        k_indices=k_idx,
                        phys_indices=phys_idx,
                        dr_feature_indices=phys_idx,
                        f_cm_col_index=f_cm_idx,
                        alpha=alpha,
                        K=K,
                        seed=set_seed,
                        use_gpu=use_gpu,
                    )
                    conformal_methods[method] = cp
                    method_calib_times[method] = time.time() - t0
                    progress.tick(set_seed, fold, method, method_calib_times[method])
                except Exception as exc:
                    logger.warning("E baseline %s failed seed=%d fold=%s: %s",
                                   method, set_seed, fold, exc)
                    progress.error(set_seed, fold, method, str(exc))

            # JKplus-source from cache (same for all seeds; eval mask applied)
            if "JKplus-source" in methods and fold in jk_cache:
                lo_full, hi_full = jk_cache[fold]["lo"], jk_cache[fold]["hi"]

                class _JKCached:
                    method_name = "JKplus-source"
                    guarantee_scope = "source"

                    def predict_intervals(self, X, alpha=None):
                        return lo_full[eval_mask], hi_full[eval_mask]

                    def predict(self, X):
                        lo, hi = self.predict_intervals(X)
                        return (lo + hi) / 2

                conformal_methods["JKplus-source"] = _JKCached()

            if not conformal_methods:
                continue

            # Alpha sweep only for cheap split arms
            cheap = {"CQR-target", "TabPFN-target", "WeightedCP-v2"}
            method_alphas = {}
            for m in conformal_methods:
                if m in cheap and alpha_sweep:
                    method_alphas[m] = alpha_sweep
                else:
                    method_alphas[m] = [alpha]

            rows = []
            for m, pred in conformal_methods.items():
                for a in method_alphas[m]:
                    from src.metrics import evaluate_method
                    try:
                        row = evaluate_method(pred, X_eval, y_eval, a, m)
                        row["guarantee_scope"] = get_guarantee_scope(m)
                        row["guaranteed"] = is_guaranteed(m, K, a) if m != "JKplus-source" else True
                        row["fold"] = fold
                        row["fold_idx"] = fold_idx
                        row["seed"] = set_seed
                        row["K"] = K
                        row["fit_time_sec"] = fit_time
                        row["source_fit_time_sec"] = (
                            fit_time if method_uses_source_lgbm(m) else 0.0
                        )
                        row["calib_time_sec"] = method_calib_times.get(m, 0.0)
                        rows.append(row)
                    except Exception as exc:
                        logger.warning("Eval skip %s alpha=%.2f: %s", m, a, exc)

            if rows:
                df_fold = pd.DataFrame(rows)
                df_fold = annotate_guaranteed(df_fold)
                all_rows.append(df_fold)

            if smoke:
                break
        if smoke:
            break

        # Checkpoint after each seed (survives process kill)
        if all_rows:
            df_seed = pd.concat(all_rows, ignore_index=True)
            df_seed.to_csv(csv_path, index=False)

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(csv_path, index=False)
    logger.info("Extension E results saved: %s (%d rows)", csv_path, len(df_all))
    return df_all

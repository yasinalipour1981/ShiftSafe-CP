"""Revision experiments for Computers and Concrete 26M-08-239 (major revision).

Addresses, in one grid:

  * Reviewer 1 comment 1 / Reviewer 2 comment 4 -- the calibration draw must not
    use the response. Every method is re-run under three draws:
    ``strength_quartile`` (as submitted, outcome-dependent), ``random`` (simple
    random, deployment-realistic) and ``covariate`` (stratified on w/cm and age,
    both fixed at mix-design time).
  * Reviewer 1 comment 2 / Reviewer 2 comment 3 -- exact Barber et al. (2021)
    Jackknife+ is run alongside the practical construction at every draw.
  * Reviewer 1 comment 3 -- ``TargetOnlyJKplus`` uses the identical leave-one-out
    calibration but a candidate pool restricted to target-only predictors, so the
    contribution of source transfer is separated from that of spending the whole
    budget K through LOO rather than a fit/calibration split.
  * Reviewer 1 comment 6 / Reviewer 2 comment 6 -- ``StackedOnlyCP`` isolates the
    adaptive candidate-selection step.

Primary operating point: K = 20, alpha = 0.10, folds D1/D2/D4, seeds 142-191.
Output: results/revision/REVISION_draws_K20.csv (one row per
seed x fold x draw x method).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_cp import (  # noqa: E402
    AdaptiveJackknifeCP,
    AdaptiveSplitCP,
    LocalOnlyJackknifeCP,
    StackedOnlyCP,
)
from src.base_models import fit_base_models  # noqa: E402
from src.conformal import (  # noqa: E402
    SplitConformal,
    TargetOnlyCP,
    TransferCalibratedCP,
    fit_source_sigma_predictor,
    physical_feature_indices,
    sample_k_indices,
)
from src.datasets import (  # noqa: E402
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)
from src.metrics import (  # noqa: E402
    empirical_coverage,
    mean_interval_width,
    winkler_score,
)

FOLDS = ("D1", "D2", "D4")
K, ALPHA = 20, 0.10
DRAWS = ("strength_quartile", "random", "covariate")


def make_sampler(scheme: str, strata_indices: list[int]):
    """Return a ``(X_pool, y_pool, K, seed) -> indices`` closure for `scheme`."""

    def _sampler(X_pool, y_pool, k, seed, _s=scheme, _si=strata_indices):
        return sample_k_indices(
            y_pool, k, seed=seed, scheme=_s, X=X_pool, strata_indices=_si,
        )

    return _sampler


def _score(y, lo, hi, alpha=ALPHA) -> dict:
    return {
        "coverage": float(empirical_coverage(y, lo, hi)),
        "width": float(mean_interval_width(lo, hi)),
        "winkler": float(winkler_score(y, lo, hi, alpha=alpha)),
        "n_eval": int(len(y)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=142)
    ap.add_argument("--seed-end", type=int, default=191)
    ap.add_argument("--out", type=str, default="results/revision/REVISION_draws_K20.csv")
    ap.add_argument("--draws", type=str, default=",".join(DRAWS))
    args = ap.parse_args()
    draws = tuple(d.strip() for d in args.draws.split(",") if d.strip())

    with open(ROOT / "configs" / "final_confirmatory.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config["device"])
    ratio = config.get("experiment", {}).get("train_calib_ratio", 0.8)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.time()

    for seed in range(args.seed_start, args.seed_end + 1):
        datasets = load_all_datasets(config, seed=seed)
        splits = create_lodo_splits(datasets, ratio, seed=seed)

        for fold in FOLDS:
            df_train, df_calib, df_test = (
                align_dataframe_features(d) for d in splits[fold]
            )
            feature_cols = get_feature_columns(df_train)
            X_tr = df_train[feature_cols].values.astype(np.float64)
            y_tr = df_train["strength_mpa"].values.astype(np.float64)
            X_ca = df_calib[feature_cols].values.astype(np.float64)
            y_ca = df_calib["strength_mpa"].values.astype(np.float64)
            X_te = df_test[feature_cols].values.astype(np.float64)
            y_te = df_test["strength_mpa"].values.astype(np.float64)

            phys_idx = physical_feature_indices(feature_cols)
            # Pre-test strata: water/cement ratio and age, fixed at mix design.
            strata_idx = [
                feature_cols.index(c) for c in ("w_cm", "age") if c in feature_cols
            ]

            base = fit_base_models(
                X_tr, y_tr, X_ca, y_ca, config=config, device=device, n_trials=0,
            )["LightGBM"]
            sigma_model, sigma_floor = fit_source_sigma_predictor(
                base, X_ca, y_ca, seed=seed, use_gpu=(config["device"] == "cuda"),
            )

            for draw in draws:
                sampler = make_sampler(draw, strata_idx)
                fitted: dict[str, tuple] = {}

                def _add(name, obj, fit_call):
                    t0 = time.time()
                    fit_call()
                    fitted[name] = (obj, time.time() - t0)

                ajk = AdaptiveJackknifeCP(
                    base, alpha=ALPHA, K=K, seed=seed, sampler=sampler)
                _add("AdaptiveJackknifeCP", ajk,
                     lambda: ajk.fit(X_te, y_te, physical_indices=phys_idx))

                ajk_x = AdaptiveJackknifeCP(
                    base, alpha=ALPHA, K=K, seed=seed, sampler=sampler, exact_jkp=True)
                _add("AdaptiveJackknifeCP-exactJK+", ajk_x,
                     lambda: ajk_x.fit(X_te, y_te, physical_indices=phys_idx))

                toj = LocalOnlyJackknifeCP(
                    base, alpha=ALPHA, K=K, seed=seed, sampler=sampler)
                _add("TargetOnlyJKplus", toj,
                     lambda: toj.fit(X_te, y_te, physical_indices=phys_idx))

                toj_x = LocalOnlyJackknifeCP(
                    base, alpha=ALPHA, K=K, seed=seed, sampler=sampler, exact_jkp=True)
                _add("TargetOnlyJKplus-exactJK+", toj_x,
                     lambda: toj_x.fit(X_te, y_te, physical_indices=phys_idx))

                soc = StackedOnlyCP(
                    base, alpha=ALPHA, K=K, seed=seed, sampler=sampler)
                _add("StackedOnlyCP", soc,
                     lambda: soc.fit(X_te, y_te, physical_indices=phys_idx))

                toc = TargetOnlyCP(alpha=ALPHA, K=K, seed=seed, sampler=sampler)
                _add("TargetOnlyCP", toc,
                     lambda: toc.fit(X_te, y_te, physical_indices=phys_idx))

                asc = AdaptiveSplitCP(
                    base, alpha=ALPHA, K=K, seed=seed, sampler=sampler)
                _add("AdaptiveSplitCP", asc,
                     lambda: asc.fit(X_te, y_te, physical_indices=phys_idx))

                tcal = TransferCalibratedCP(
                    base_model=base, sigma_model=sigma_model, sigma_floor=sigma_floor,
                    alpha=ALPHA, K=K, fine_tune_mode="affine_localsigma", seed=seed,
                    sampler=sampler,
                )
                _add("TransferCal-CP-affine-localsigma", tcal,
                     lambda: tcal.fit(X_te, y_te))

                # Every target-anchored method must share one calibration draw.
                draw_idx = np.asarray(ajk.calib_target_indices, dtype=int)
                for nm, (obj, _t) in fitted.items():
                    got = set(np.asarray(obj.calib_target_indices).tolist())
                    assert got == set(draw_idx.tolist()), (
                        f"calibration draw mismatch for {nm} "
                        f"({draw}, {fold}, seed {seed})"
                    )

                mask = np.ones(len(y_te), dtype=bool)
                mask[draw_idx] = False
                X_ev, y_ev = X_te[mask], y_te[mask]

                # Source-calibrated split CP: no target labels, same evaluation set.
                split_cp = SplitConformal(base, X_ca, y_ca, ALPHA)
                t0 = time.time()
                split_cp.fit()
                t_split = time.time() - t0
                lo, hi = split_cp.predict_intervals(X_ev)
                rows.append(dict(
                    seed=seed, fold=fold, draw=draw, method="SplitCP-source",
                    K=K, alpha=ALPHA, calib_time_sec=t_split,
                    selected_candidate="", **_score(y_ev, lo, hi),
                ))

                for nm, (obj, t_fit) in fitted.items():
                    lo, hi = obj.predict_intervals(X_ev)
                    diag = getattr(obj, "diagnostics", {}) or {}
                    rows.append(dict(
                        seed=seed, fold=fold, draw=draw, method=nm,
                        K=K, alpha=ALPHA, calib_time_sec=t_fit,
                        selected_candidate=str(
                            diag.get("winner_full", diag.get("winner", ""))
                        ),
                        **_score(y_ev, lo, hi),
                    ))

        pd.DataFrame(rows).to_csv(out_path, index=False)
        el = time.time() - t_start
        done = seed - args.seed_start + 1
        tot = args.seed_end - args.seed_start + 1
        print(
            f"[{done}/{tot}] seed {seed} done | {len(rows)} rows | "
            f"{el / 60:.1f} min elapsed | eta {(el / done) * (tot - done) / 60:.1f} min",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(df)} rows)\n")
    for draw in draws:
        sub = df[df.draw == draw]
        if sub.empty:
            continue
        print(f"--- draw = {draw} : pooled over D1/D2/D4 ---")
        print(sub.groupby("method")[["coverage", "width", "winkler"]].mean().round(3))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

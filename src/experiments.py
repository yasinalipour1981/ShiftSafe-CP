"""LODO experiment orchestration."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.base_models import fit_base_models, predict_base_models
from src.baselines_uq import fit_uq_baselines
from src.compliance import compute_compliance_probabilities
from src.conformal import (
    TransferCalibratedCP,
    TargetOnlyCP,
    build_shiftsafe_ablations,
    fit_conformal_method,
    fit_source_sigma_predictor,
    min_k_for_alpha,
    physical_feature_indices,
    sample_k_indices,
)
from src.adaptive_cp import AdaptiveJackknifeCP, AdaptiveSplitCP
from src.datasets import (
    align_dataframe_features,
    create_lodo_splits,
    generate_derived_features,
    get_feature_columns,
    load_all_datasets,
    save_lodo_splits,
    save_processed_datasets,
)
from src.metrics import evaluate_all_methods, statistical_significance_test
from src.pubstyle import method_uses_source_lgbm
from src.validity import annotate_guaranteed
from src.shift_analysis import run_shift_analysis
from src.utils import append_execution_log, create_results_manifest, log_gpu_memory

logger = logging.getLogger(__name__)


def _timed_fit(method_calib_times: dict[str, float], name: str, fn):
    """Run fn() and accumulate wall-clock calibration time for one method."""
    t0 = time.time()
    result = fn()
    method_calib_times[name] = method_calib_times.get(name, 0.0) + (time.time() - t0)
    return result


class LODOExperiment:
    """Leave-one-dataset-out cross-laboratory evaluation."""

    def __init__(self, config: dict[str, Any], seed: int = 0):
        self.config = config
        self.seed = seed
        self.results_dir = Path(config.get("results_dir", "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: list[pd.DataFrame] = []
        self.timing_results: list[dict[str, Any]] = []
        self.compliance_results: list[pd.DataFrame] = []
        self.point_accuracy: list[dict[str, Any]] = []

    def run_full_pipeline(self) -> pd.DataFrame:
        logger.info("=" * 80)
        logger.info("SHIFTSAFE-CP: FULL PIPELINE (seed=%d)", self.seed)
        logger.info("=" * 80)

        # Stage 1: Load datasets
        logger.info("\n[STAGE 1] Loading and harmonizing datasets...")
        datasets = load_all_datasets(self.config, seed=self.seed)
        if len(datasets) < 2:
            raise RuntimeError("Need at least 2 datasets for LODO evaluation")

        save_processed_datasets(datasets, self.config.get("processed_data_dir", "data/processed"))
        append_execution_log(f"Stage 1 complete: {len(datasets)} datasets loaded (seed={self.seed})")

        # Stage 2: Shift analysis
        logger.info("\n[STAGE 2] Shift quantification...")
        shift_dir = self.results_dir / "shift_analysis"
        run_shift_analysis(
            datasets, shift_dir, device=self.config.get("device", "cuda"), seed=self.seed
        )
        append_execution_log("Stage 2 complete: shift analysis")

        # Stage 3: LODO evaluation
        logger.info("\n[STAGE 3] LODO evaluation...")
        train_calib_ratio = self.config.get("evaluation", {}).get("train_calib_split", 0.8)
        if isinstance(self.config.get("experiment"), dict):
            train_calib_ratio = self.config["experiment"].get(
                "train_calib_ratio", train_calib_ratio
            )

        lodo_splits = create_lodo_splits(datasets, train_calib_ratio, seed=self.seed)
        save_lodo_splits(lodo_splits, self.config.get("processed_data_dir", "data/processed"))

        held_out_filter = self.config.get("evaluation", {}).get("held_out_folds")
        if held_out_filter:
            lodo_splits = {
                k: v for k, v in lodo_splits.items() if k in held_out_filter
            }
            logger.info("LODO filter: running folds %s only", list(lodo_splits.keys()))

        alpha_levels = self.config.get("evaluation", {}).get("alpha_levels", [0.05, 0.10, 0.20])
        n_trials = self.config.get("n_optuna_trials", 20)
        use_gpu = self.config.get("device", "cuda") == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_gpu else "cpu")

        methods_to_eval = self.config.get("methods_to_evaluate", [
            "SplitCP", "Jackknife+", "CQR", "ShiftSafe-CP",
            "ShiftSafe-CP-L1", "ShiftSafe-CP-L1-L2", "ShiftSafe-CP-L1-L3",
            "GPR", "NGBoost", "DeepEnsemble",
        ])
        protocol = self.config.get("evaluation", {}).get("protocol", "v2")
        if protocol in ("v3", "v_final"):
            return self._run_lodo_v3(
                lodo_splits, alpha_levels, n_trials, use_gpu, device, methods_to_eval
            )

        for fold_idx, (held_out, (df_train, df_calib, df_test)) in enumerate(
            lodo_splits.items()
        ):
            logger.info(
                "\n--- FOLD %d/%d: Held-out = %s ---",
                fold_idx + 1,
                len(lodo_splits),
                held_out,
            )

            feature_cols = get_feature_columns(df_train)
            df_train = align_dataframe_features(df_train)
            df_calib = align_dataframe_features(df_calib)
            df_test = align_dataframe_features(df_test)
            X_train = df_train[feature_cols].values.astype(np.float64)
            y_train = df_train["strength_mpa"].values.astype(np.float64)
            X_calib = df_calib[feature_cols].values.astype(np.float64)
            y_calib = df_calib["strength_mpa"].values.astype(np.float64)
            X_test = df_test[feature_cols].values.astype(np.float64)
            y_test = df_test["strength_mpa"].values.astype(np.float64)

            # Fit base learners
            t0 = time.time()
            base_models = fit_base_models(
                X_train, y_train, X_calib, y_calib,
                config=self.config,
                device=device,
                n_trials=n_trials,
            )
            fit_time = time.time() - t0
            log_gpu_memory(logger, "after base models")

            primary = base_models.get("LightGBM")
            if primary is None:
                raise RuntimeError("LightGBM base model required")

            # Point accuracy for all base learners
            for name, model in base_models.items():
                preds = predict_base_models({name: model}, X_test)[name]
                from src.metrics import rmse, mae, r_squared
                self.point_accuracy.append({
                    "fold": held_out,
                    "seed": self.seed,
                    "model": name,
                    "rmse": rmse(y_test, preds),
                    "mae": mae(y_test, preds),
                    "r_squared": r_squared(y_test, preds),
                })

            # Conformal methods
            conformal_methods: dict[str, Any] = {}
            method_calib_times: dict[str, float] = {}

            conformal_names = [
                m for m in methods_to_eval
                if m in ("SplitCP", "Jackknife+", "CQR")
                or m.startswith("ShiftSafe")
            ]
            # ShiftSafe-CP v2 needs unlabeled target covariates (X only, never
            # y_test) and the explicit f_cm feature index.
            f_cm_idx = feature_cols.index("f_cm") if "f_cm" in feature_cols else None
            dr_idx = physical_feature_indices(feature_cols)
            for name in conformal_names:
                try:
                    if name.startswith("ShiftSafe"):
                        ablations = build_shiftsafe_ablations(
                            primary, X_train, y_train, X_calib, y_calib,
                            X_target_unlabeled=X_test,
                            f_cm_col_index=f_cm_idx,
                            dr_feature_indices=dr_idx,
                            config=self.config,
                            device=self.config.get("device", "cuda"),
                            seed=self.seed,
                        )
                        cp = ablations.get(name)
                        if cp:
                            def _fit_shift(_cp=cp, _n=name):
                                _cp.fit()
                                return _cp
                            conformal_methods[name] = _timed_fit(
                                method_calib_times, name, _fit_shift,
                            )
                    else:
                        def _fit_conf(_n=name):
                            return fit_conformal_method(
                                _n, primary, X_train, y_train, X_calib, y_calib,
                                config=self.config, use_gpu=use_gpu,
                                seed=self.seed,
                            )
                        conformal_methods[name] = _timed_fit(
                            method_calib_times, name, _fit_conf,
                        )
                except Exception as e:
                    logger.warning("Failed to fit %s: %s", name, e)

            # UQ baselines
            uq_names = [m for m in methods_to_eval if m in ("GPR", "NGBoost", "DeepEnsemble")]
            if uq_names:
                ensemble = base_models.get("DeepMLP")
                uq_models = fit_uq_baselines(
                    np.vstack([X_train, X_calib]),
                    np.concatenate([y_train, y_calib]),
                    ensemble=ensemble,
                    seed=self.seed,
                )
                for name in uq_names:
                    if name in uq_models:
                        conformal_methods[name] = uq_models[name]
                        method_calib_times[name] = 0.0  # fit bundled above

            calib_time = sum(method_calib_times.values())

            # Evaluate
            df_fold = evaluate_all_methods(
                conformal_methods, X_test, y_test, alpha_levels=alpha_levels
            )
            df_fold["fold"] = held_out
            df_fold["fold_idx"] = fold_idx
            df_fold["seed"] = self.seed
            self.all_results.append(df_fold)

            for method_name in conformal_methods:
                self.timing_results.append({
                    "method": method_name,
                    "fold": held_out,
                    "seed": self.seed,
                    "fit_time_sec": fit_time,
                    "source_fit_time_sec": fit_time if method_uses_source_lgbm(method_name) else 0.0,
                    "calib_time_sec": method_calib_times.get(method_name, 0.0),
                    "infer_time_per_1000_sec": df_fold[
                        df_fold["method"] == method_name
                    ]["inference_time_sec"].mean()
                    * (1000 / max(len(X_test), 1)),
                })

            # Compliance at alpha=0.10
            alpha_comp = 0.10
            intervals = {}
            for name, pred in conformal_methods.items():
                try:
                    lo, hi = pred.predict_intervals(X_test, alpha=alpha_comp)
                    intervals[name] = (lo, hi)
                except Exception:
                    pass

            # Mandatory ShiftSafe-CP diagnostics per fold (Stage-4 patch)
            self._save_fold_diagnostics(
                held_out, conformal_methods, intervals, X_test, y_test, alpha_comp
            )
            if intervals:
                comp_df = compute_compliance_probabilities(intervals, grade="M25")
                comp_df["fold"] = held_out
                comp_df["seed"] = self.seed
                self.compliance_results.append(comp_df)

            logger.info("  Fold complete: %d methods evaluated", len(conformal_methods))

        # Aggregate
        eval_subdir = self.results_dir / "lodo_evaluation"
        eval_subdir.mkdir(parents=True, exist_ok=True)

        df_all = pd.concat(self.all_results, ignore_index=True)
        df_all.to_csv(eval_subdir / "LODO_all_results.csv", index=False)
        df_all.to_csv(self.results_dir / "LODO_all_results.csv", index=False)

        if self.timing_results:
            pd.DataFrame(self.timing_results).to_csv(
                self.results_dir / "lodo_evaluation" / "timing_results.csv", index=False
            )

        if self.point_accuracy:
            pd.DataFrame(self.point_accuracy).to_csv(
                self.results_dir / "lodo_evaluation" / "point_accuracy.csv", index=False
            )

        if self.compliance_results:
            pd.concat(self.compliance_results).to_csv(
                self.results_dir / "lodo_evaluation" / "TABLE_7_compliance_decisions.csv",
                index=False,
            )

        # Statistical tests
        tests = []
        compare = self.config.get("statistical_tests", {}).get("compare_methods", [
            ["ShiftSafe-CP", "SplitCP"],
            ["ShiftSafe-CP", "Jackknife+"],
            ["ShiftSafe-CP", "CQR"],
        ])
        for pair in compare:
            if len(pair) == 2:
                tests.append(
                    statistical_significance_test(df_all, pair[0], pair[1])
                )
        def _json_safe(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(v) for v in obj]
            if isinstance(obj, (np.bool_, np.generic)):
                return obj.item()
            return obj

        with open(self.results_dir / "lodo_evaluation" / "wilcoxon_tests.json", "w") as f:
            json.dump(_json_safe(tests), f, indent=2)

        create_results_manifest(
            self.results_dir,
            self.config,
            stage="lodo_evaluation",
            output_files=["LODO_all_results.csv"],
        )

        append_execution_log(f"Stage 3 complete: LODO evaluation (seed={self.seed})")
        logger.info("Pipeline complete for seed=%d", self.seed)
        return df_all

    def _run_lodo_v3(
        self,
        lodo_splits: dict,
        alpha_levels: list[float],
        n_trials: int,
        use_gpu: bool,
        device: torch.device,
        methods_to_eval: list[str],
    ) -> pd.DataFrame:
        """Stage-4 v3: target-anchored transfer calibration with K-sweep."""
        eval_cfg = self.config.get("evaluation", {})
        K_sweep = eval_cfg.get("K_sweep", [20])
        max_frac = eval_cfg.get("max_target_calib_fraction", 0.30)
        primary_alpha = 0.10
        # 2026-09 revision: the calibration draw must not use the response
        # (Reviewer 1 c.1 / Reviewer 2 c.4), and the primary construction is the
        # exact Barber jackknife-plus (Reviewer 1 c.2 / Reviewer 2 c.3). Both
        # default to the originally submitted behaviour so that the earlier
        # results remain reproducible from the archived configs.
        draw_scheme = eval_cfg.get("calibration_draw", "strength_quartile")
        use_exact_jkp = bool(eval_cfg.get("exact_jackknife_plus", False))
        logger.info("calibration draw=%s, exact_jkp=%s", draw_scheme, use_exact_jkp)

        for fold_idx, (held_out, (df_train, df_calib, df_test)) in enumerate(
            lodo_splits.items()
        ):
            logger.info("\n--- FOLD %d/%d: Held-out = %s (v3) ---",
                        fold_idx + 1, len(lodo_splits), held_out)

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

            t0 = time.time()
            base_models = fit_base_models(
                X_train, y_train, X_calib, y_calib,
                config=self.config, device=device, n_trials=n_trials,
            )
            fit_time = time.time() - t0
            primary = base_models.get("LightGBM")
            if primary is None:
                raise RuntimeError("LightGBM base model required")

            sigma_model, sigma_floor = fit_source_sigma_predictor(
                primary, X_calib, y_calib, seed=self.seed, use_gpu=use_gpu,
            )
            f_cm_idx = feature_cols.index("f_cm") if "f_cm" in feature_cols else None
            dr_idx = physical_feature_indices(feature_cols)
            phys_idx = dr_idx

            if draw_scheme == "strength_quartile":
                fold_sampler = None
            else:
                strata_idx = [
                    feature_cols.index(c) for c in ("w_cm", "age") if c in feature_cols
                ]

                def fold_sampler(X_pool, y_pool, k, sd, _s=draw_scheme, _si=strata_idx):
                    return sample_k_indices(
                        y_pool, k, seed=sd, scheme=_s, X=X_pool, strata_indices=_si,
                    )

            n_target = len(y_test_full)
            max_K = int(max_frac * n_target)
            K_values = [
                k for k in K_sweep
                if k <= max_K and k >= min_k_for_alpha(primary_alpha)
            ]
            if not K_values:
                logger.warning("No valid K for fold %s (n=%d)", held_out, n_target)
                continue

            for K in K_values:
                conformal_methods: dict[str, Any] = {}
                method_calib_times: dict[str, float] = {}

                # Source-calibrated baselines
                for name in ("SplitCP", "Jackknife+", "CQR"):
                    if name not in methods_to_eval:
                        continue
                    try:
                        def _fit_src(_n=name):
                            return fit_conformal_method(
                                _n, primary, X_train, y_train, X_calib, y_calib,
                                config=self.config, use_gpu=use_gpu, seed=self.seed,
                            )
                        conformal_methods[name] = _timed_fit(
                            method_calib_times, name, _fit_src,
                        )
                    except Exception as e:
                        logger.warning("Failed %s: %s", name, e)

                # v3 target-anchored methods (define eval mask via calib indices)
                v3_calib_idx: np.ndarray | None = None

                if "TargetOnlyCP" in methods_to_eval:
                    def _fit_toc():
                        toc = TargetOnlyCP(alpha=primary_alpha, K=K, seed=self.seed,
                                           sampler=fold_sampler)
                        toc.fit(X_test_full, y_test_full, physical_indices=phys_idx)
                        return toc
                    toc = _timed_fit(method_calib_times, "TargetOnlyCP", _fit_toc)
                    conformal_methods["TargetOnlyCP"] = toc
                    v3_calib_idx = toc.calib_target_indices

                for v3_name, mode in (
                    ("TransferCal-CP", "none"),
                    ("TransferCal-CP-affine", "affine"),
                    ("TransferCal-CP-affine-localsigma", "affine_localsigma"),
                    ("TransferCal-CP-boost", "residual_boost"),
                ):
                    if v3_name not in methods_to_eval:
                        continue
                    def _fit_tc(_vn=v3_name, _m=mode):
                        tc = TransferCalibratedCP(
                            base_model=primary,
                            sigma_model=sigma_model,
                            sigma_floor=sigma_floor,
                            alpha=primary_alpha,
                            K=K,
                            fine_tune_mode=_m,
                            seed=self.seed,
                            sampler=fold_sampler,
                        )
                        tc.fit(X_test_full, y_test_full)
                        return tc
                    tc = _timed_fit(method_calib_times, v3_name, _fit_tc)
                    conformal_methods[v3_name] = tc
                    if v3_calib_idx is not None:
                        assert set(v3_calib_idx) == set(tc.calib_target_indices), (
                            "v3 calib index mismatch across methods"
                        )
                    v3_calib_idx = tc.calib_target_indices

                if "AdaptiveJackknifeCP" in methods_to_eval:
                    def _fit_ajk():
                        ajk = AdaptiveJackknifeCP(
                            base_model=primary, alpha=primary_alpha, K=K, seed=self.seed,
                            sampler=fold_sampler, exact_jkp=use_exact_jkp,
                        )
                        ajk.fit(X_test_full, y_test_full, physical_indices=phys_idx)
                        return ajk
                    ajk = _timed_fit(method_calib_times, "AdaptiveJackknifeCP", _fit_ajk)
                    conformal_methods["AdaptiveJackknifeCP"] = ajk
                    v3_calib_idx = ajk.calib_target_indices

                if "AdaptiveSplitCP" in methods_to_eval:
                    def _fit_asc():
                        asc = AdaptiveSplitCP(
                            base_model=primary, alpha=primary_alpha, K=K, seed=self.seed,
                            sampler=fold_sampler,
                        )
                        asc.fit(X_test_full, y_test_full, physical_indices=phys_idx)
                        return asc
                    asc = _timed_fit(method_calib_times, "AdaptiveSplitCP", _fit_asc)
                    conformal_methods["AdaptiveSplitCP"] = asc
                    if v3_calib_idx is not None:
                        assert set(v3_calib_idx) == set(asc.calib_target_indices)
                    v3_calib_idx = asc.calib_target_indices

                if v3_calib_idx is not None:
                    v3_eval_mask = np.ones(n_target, dtype=bool)
                    v3_eval_mask[v3_calib_idx] = False
                else:
                    v3_eval_mask = np.ones(n_target, dtype=bool)

                calib_idx = np.where(~v3_eval_mask)[0]
                test_idx = np.where(v3_eval_mask)[0]
                assert set(calib_idx).isdisjoint(set(test_idx))

                X_eval = X_test_full[test_idx]
                y_eval = y_test_full[test_idx]

                if any(m.startswith("ShiftSafe") for m in methods_to_eval):
                    try:
                        ablations = build_shiftsafe_ablations(
                            primary, X_train, y_train, X_calib, y_calib,
                            X_target_unlabeled=X_eval,
                            f_cm_col_index=f_cm_idx,
                            dr_feature_indices=dr_idx,
                            config=self.config,
                            device=self.config.get("device", "cuda"),
                            seed=self.seed,
                        )
                        for name, cp in ablations.items():
                            if name in methods_to_eval:
                                def _fit_ab(_c=cp, _n=name):
                                    _c.fit()
                                    return _c
                                conformal_methods[name] = _timed_fit(
                                    method_calib_times, name, _fit_ab,
                                )
                    except Exception as e:
                        logger.warning("ShiftSafe v2 failed: %s", e)

                calib_time = sum(method_calib_times.values())

                # Skip (K, alpha) combos that violate min-K (logged once per fold)
                valid_alphas = []
                for a in alpha_levels:
                    mk = min_k_for_alpha(a)
                    if K >= mk + 1:
                        valid_alphas.append(a)
                    else:
                        logger.info(
                            "  Skip K=%d at alpha=%.2f (need K>=%d)", K, a, mk + 1,
                        )
                if not valid_alphas:
                    continue

                df_fold = evaluate_all_methods(
                    conformal_methods, X_eval, y_eval, alpha_levels=valid_alphas
                )
                df_fold = annotate_guaranteed(df_fold)
                df_fold["fold"] = held_out
                df_fold["fold_idx"] = fold_idx
                df_fold["seed"] = self.seed
                df_fold["K"] = K
                self.all_results.append(df_fold)

                for method_name in conformal_methods:
                    self.timing_results.append({
                        "method": method_name,
                        "fold": held_out,
                        "seed": self.seed,
                        "K": K,
                        "fit_time_sec": fit_time,
                        "source_fit_time_sec": fit_time if method_uses_source_lgbm(method_name) else 0.0,
                        "calib_time_sec": method_calib_times.get(method_name, 0.0),
                        "infer_time_per_1000_sec": df_fold[
                            df_fold["method"] == method_name
                        ]["inference_time_sec"].mean()
                        * (1000 / max(len(X_eval), 1)),
                    })

                alpha_comp = primary_alpha
                intervals = {}
                for name, pred in conformal_methods.items():
                    try:
                        lo, hi = pred.predict_intervals(X_eval, alpha=alpha_comp)
                        intervals[name] = (lo, hi)
                    except Exception:
                        pass

                self._save_fold_diagnostics_v3(
                    held_out, conformal_methods, intervals, X_eval, y_eval,
                    alpha_comp, K, calib_idx,
                )

        eval_subdir = self.results_dir / "lodo_evaluation"
        eval_subdir.mkdir(parents=True, exist_ok=True)
        df_all = pd.concat(self.all_results, ignore_index=True)
        df_all.to_csv(eval_subdir / "LODO_all_results.csv", index=False)
        df_all.to_csv(self.results_dir / "LODO_all_results.csv", index=False)

        if self.timing_results:
            pd.DataFrame(self.timing_results).to_csv(
                eval_subdir / "timing_results.csv", index=False
            )

        append_execution_log(f"Stage 3 v3 complete: LODO (seed={self.seed})")
        return df_all

    def _save_fold_diagnostics_v3(
        self,
        held_out: str,
        conformal_methods: dict[str, Any],
        intervals: dict[str, tuple[np.ndarray, np.ndarray]],
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        alpha: float,
        K: int,
        calib_idx: np.ndarray,
    ) -> None:
        diag: dict[str, Any] = {
            "fold": held_out, "seed": self.seed, "alpha": alpha, "K": K,
            "n_calib_target": len(calib_idx),
            "n_eval": len(y_eval),
        }

        def _cov(name: str) -> float | None:
            if name not in intervals:
                return None
            lo, hi = intervals[name]
            return float(np.mean((y_eval >= lo) & (y_eval <= hi)))

        diag["splitcp_coverage"] = _cov("SplitCP")
        for v3_name in (
            "TransferCal-CP", "TransferCal-CP-affine",
            "TransferCal-CP-affine-localsigma",
            "TransferCal-CP-boost", "TargetOnlyCP",
        ):
            diag[f"{v3_name}_coverage"] = _cov(v3_name)

        tc_affine = conformal_methods.get("TransferCal-CP-affine")
        if tc_affine is not None and hasattr(tc_affine, "diagnostics"):
            diag["bias_before_affine"] = tc_affine.diagnostics.get("bias_before_affine")
            diag["bias_after_affine"] = tc_affine.diagnostics.get("bias_after_affine")
            diag["affine_params"] = tc_affine.diagnostics.get("affine_params")

        ajk = conformal_methods.get("AdaptiveJackknifeCP")
        if ajk is not None and hasattr(ajk, "diagnostics"):
            diag["adaptive_selection"] = ajk.diagnostics.get("selection_counts")
            diag["adaptive_winner"] = ajk.diagnostics.get("winner_full")

        ss = conformal_methods.get("ShiftSafe-CP")
        if ss is not None and hasattr(ss, "diagnostics"):
            diag["domain_auc"] = ss.diagnostics.get("domain_auc")
            diag["domain_auc_all_features"] = ss.diagnostics.get("domain_auc_all_features")
            diag["ess"] = ss.diagnostics.get("ess")
            diag["ess_fraction"] = ss.diagnostics.get("ess_fraction")
            last = ss.diagnostics.get("last_predict", {})
            diag["pct_infinite_quantile"] = last.get("pct_infinite_quantile")
            diag["shiftsafe_v2_coverage"] = _cov("ShiftSafe-CP")

        diag_dir = self.results_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_path = diag_dir / f"fold_{held_out}_seed{self.seed}_K{K}.json"
        with open(out_path, "w") as f:
            json.dump(diag, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        logger.info("  v3 diagnostics: %s", out_path)

    def _write_v3_tables(self, df_all: pd.DataFrame, eval_subdir: Path) -> None:
        from src.paper_tables import table_3v3_transfer

        table_3v3_transfer(df_all, eval_subdir, K=20, alpha=0.10)
        self._write_k_sweep_summary(df_all, eval_subdir)

    def _write_k_sweep_summary(self, df_all: pd.DataFrame, eval_subdir: Path) -> None:
        if "K" not in df_all.columns:
            return
        agg = df_all.groupby(["method", "fold", "K", "alpha"]).agg({
            "empirical_coverage": ["mean", "std"],
            "mean_interval_width": ["mean", "std"],
        }).reset_index()
        agg.columns = [
            "method", "fold", "K", "alpha",
            "coverage_mean", "coverage_std", "width_mean", "width_std",
        ]
        agg.to_csv(eval_subdir / "K_sweep_summary.csv", index=False)
        # Stability: width SD across seeds
        if "seed" in df_all.columns:
            stab = df_all.groupby(["method", "fold", "K", "alpha"]).agg(
                width_sd=("mean_interval_width", "std"),
                width_worst=("mean_interval_width", "max"),
                cov_mean=("empirical_coverage", "mean"),
            ).reset_index()
            stab.to_csv(eval_subdir / "stability_summary.csv", index=False)

    def _run_v3_stat_tests(self, df_all: pd.DataFrame, eval_subdir: Path) -> None:
        from src.metrics import holm_corrected_wilcoxon_tests

        df_k = df_all[(df_all["K"] == 20) & (df_all["alpha"] == 0.10)] if "K" in df_all.columns else df_all[df_all["alpha"] == 0.10]
        # Primary pooled test excludes D3 (n=25, K=10 only)
        if "fold" in df_k.columns:
            df_k = df_k[df_k["fold"] != "D3"]
        pairs = [
            ("AdaptiveJackknifeCP", "TargetOnlyCP"),
            ("TransferCal-CP-affine-localsigma", "TargetOnlyCP"),
            ("TransferCal-CP-affine", "TargetOnlyCP"),
            ("TransferCal-CP-boost", "TargetOnlyCP"),
            ("TransferCal-CP", "TargetOnlyCP"),
        ]
        pairs = [(a, b) for a, b in pairs if a in df_k["method"].values and b in df_k["method"].values]
        winkler_tests = holm_corrected_wilcoxon_tests(
            df_k, pairs, metric="winkler_score",
        )
        width_tests = holm_corrected_wilcoxon_tests(
            df_k, pairs, metric="mean_interval_width",
        )
        with open(eval_subdir / "wilcoxon_tests_v3_winkler.json", "w") as f:
            json.dump(winkler_tests, f, indent=2, default=str)
        with open(eval_subdir / "wilcoxon_tests_v3_width.json", "w") as f:
            json.dump(width_tests, f, indent=2, default=str)
        with open(eval_subdir / "wilcoxon_tests_v3.json", "w") as f:
            json.dump(winkler_tests, f, indent=2, default=str)

    def _save_fold_diagnostics(
        self,
        held_out: str,
        conformal_methods: dict[str, Any],
        intervals: dict[str, tuple[np.ndarray, np.ndarray]],
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float,
    ) -> None:
        """Save per-fold ShiftSafe-CP diagnostics (Stage-4 mandatory)."""
        diag: dict[str, Any] = {"fold": held_out, "seed": self.seed, "alpha": alpha}

        def _coverage(name: str) -> float | None:
            if name not in intervals:
                return None
            lo, hi = intervals[name]
            return float(np.mean((y_test >= lo) & (y_test <= hi)))

        # 1. SplitCP coverage = shift-severity baseline
        diag["splitcp_coverage"] = _coverage("SplitCP")

        ss = conformal_methods.get("ShiftSafe-CP")
        if ss is not None and hasattr(ss, "diagnostics"):
            # 2-4. domain AUC, ESS, +inf quantile fraction
            diag["domain_auc"] = ss.diagnostics.get("domain_auc")
            diag["ess"] = ss.diagnostics.get("ess")
            diag["ess_fraction"] = ss.diagnostics.get("ess_fraction")
            last = ss.diagnostics.get("last_predict", {})
            diag["pct_infinite_quantile"] = last.get("pct_infinite_quantile")
            diag["mean_w_test"] = last.get("mean_w_test")
            diag["cells_used"] = last.get("cells_used")
            diag["shiftsafe_coverage"] = _coverage("ShiftSafe-CP")

            # 5. Coverage per Mondrian cell (worst-cell is a headline metric)
            if "ShiftSafe-CP" in intervals and hasattr(ss, "_cell_id"):
                lo, hi = intervals["ShiftSafe-CP"]
                y_pred = ss.base_model.predict(X_test)
                cells = ss._cell_id(np.asarray(X_test, dtype=np.float64), y_pred)
                covered = (y_test >= lo) & (y_test <= hi)
                per_cell = {
                    int(c): {
                        "n": int((cells == c).sum()),
                        "coverage": float(covered[cells == c].mean()),
                    }
                    for c in np.unique(cells)
                }
                diag["coverage_per_cell"] = per_cell
                diag["worst_cell_coverage"] = min(
                    v["coverage"] for v in per_cell.values()
                )

        diag_dir = self.results_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_path = diag_dir / f"fold_{held_out}_seed{self.seed}.json"
        with open(out_path, "w") as f:
            json.dump(diag, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        logger.info("  Diagnostics saved: %s", out_path)

    @property
    def datasets_harmonized(self) -> dict[str, pd.DataFrame]:
        return load_all_datasets(self.config, seed=self.seed)

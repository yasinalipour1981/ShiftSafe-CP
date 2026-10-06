"""Practical vs exact Barber Jackknife+ on confirmatory primary cell."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_cp import AdaptiveJackknifeCP
from src.base_models import fit_base_models
from src.datasets import (
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)
from src.metrics import empirical_coverage, mean_interval_width, winkler_score


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=142)
    parser.add_argument("--seed-end", type=int, default=191)
    args = parser.parse_args()

    cfg_path = ROOT / "configs" / "final_confirmatory.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    folds = ["D1", "D2", "D4"]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    K, alpha = 20, 0.10
    rows = []

    for seed in seeds:
        datasets = load_all_datasets(config, seed=seed)
        ratio = config.get("experiment", {}).get("train_calib_ratio", 0.8)
        splits = create_lodo_splits(datasets, ratio, seed=seed)
        for fold in folds:
            df_train, df_calib, df_test = splits[fold]
            feature_cols = get_feature_columns(df_train)
            for df in (df_train, df_calib, df_test):
                align_dataframe_features(df)
            X_tr = df_train[feature_cols].values.astype(np.float64)
            y_tr = df_train["strength_mpa"].values.astype(np.float64)
            X_ca = df_calib[feature_cols].values.astype(np.float64)
            y_ca = df_calib["strength_mpa"].values.astype(np.float64)
            X_te = df_test[feature_cols].values.astype(np.float64)
            y_te = df_test["strength_mpa"].values.astype(np.float64)
            phys = list(range(min(8, X_te.shape[1])))

            device = torch.device(config["device"])
            models = fit_base_models(
                X_tr, y_tr, X_ca, y_ca, config=config, device=device, n_trials=0,
            )
            primary = models["LightGBM"]

            for exact in (False, True):
                ajk = AdaptiveJackknifeCP(
                    base_model=primary, alpha=alpha, K=K, seed=seed, exact_jkp=exact,
                )
                ajk.fit(X_te, y_te, physical_indices=phys)
                mask = np.ones(len(y_te), dtype=bool)
                mask[ajk.calib_target_indices] = False
                lo, hi = ajk.predict_intervals(X_te[mask])
                y = y_te[mask]
                rows.append({
                    "seed": seed,
                    "fold": fold,
                    "exact_jkp": exact,
                    "construction": ajk.diagnostics["interval_construction"],
                    "K": K,
                    "alpha": alpha,
                    "coverage": empirical_coverage(y, lo, hi),
                    "width": mean_interval_width(lo, hi),
                    "winkler": winkler_score(y, lo, hi, alpha=alpha),
                    "n_eval": int(mask.sum()),
                })
        if (seed - 141) % 10 == 0:
            print(f"seed {seed} done ({len(rows)} rows)")

    out = ROOT / "results" / "ablation" / "EXACT_JKP_sensitivity_confirmatory_K20.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    summary = df.groupby(["exact_jkp", "fold"]).agg(
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        winkler=("winkler", "mean"),
    ).round(4)
    pooled = df.groupby("exact_jkp").agg(
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        winkler=("winkler", "mean"),
    ).round(4)
    print("\nPer-fold means:")
    print(summary)
    print("\nPooled means:")
    print(pooled)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

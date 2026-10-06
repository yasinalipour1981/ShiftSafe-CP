"""D4 smoke: AdaptiveJackknifeCP exact_jkp=True vs False (practical)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
from src.metrics import mean_interval_width, winkler_score


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def main() -> None:
    with open(ROOT / "configs" / "extension_e.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 142
    datasets = load_all_datasets(config, seed=seed)
    ratio = config.get("experiment", {}).get("train_calib_ratio", 0.8)
    splits = create_lodo_splits(datasets, ratio, seed=seed)
    df_train, df_calib, df_test = splits["D4"]
    feature_cols = get_feature_columns(df_train)
    df_train = align_dataframe_features(df_train)
    df_calib = align_dataframe_features(df_calib)
    df_test = align_dataframe_features(df_test)
    X_tr = df_train[feature_cols].values.astype(np.float64)
    y_tr = df_train["strength_mpa"].values.astype(np.float64)
    X_cal = df_calib[feature_cols].values.astype(np.float64)
    y_cal = df_calib["strength_mpa"].values.astype(np.float64)
    X_te = df_test[feature_cols].values.astype(np.float64)
    y_te = df_test["strength_mpa"].values.astype(np.float64)

    device = torch.device(config["device"])
    models = fit_base_models(
        X_tr, y_tr, X_cal, y_cal, config=config, device=device, n_trials=0,
    )
    primary = models["LightGBM"]
    phys = list(range(min(8, X_te.shape[1])))
    k_idx = None
    rows = []
    for exact in (False, True):
        ajk = AdaptiveJackknifeCP(
            primary, alpha=0.10, K=20, seed=seed, exact_jkp=exact,
        )
        ajk.fit(X_te, y_te, physical_indices=phys)
        if k_idx is None:
            k_idx = ajk.calib_target_indices.copy()
        else:
            assert np.array_equal(np.sort(k_idx), np.sort(ajk.calib_target_indices))
        mask = np.ones(len(y_te), dtype=bool)
        mask[ajk.calib_target_indices] = False
        lo, hi = ajk.predict_intervals(X_te[mask], alpha=0.10)
        y = y_te[mask]
        asym = float(np.mean(np.abs((hi - (lo + hi) / 2) - ((lo + hi) / 2 - lo))))
        rows.append({
            "exact_jkp": exact,
            "construction": ajk.diagnostics["interval_construction"],
            "coverage": _coverage(y, lo, hi),
            "width": mean_interval_width(lo, hi),
            "winkler": winkler_score(y, lo, hi, alpha=0.10),
            "mean_asymmetry": asym,
            "n_eval": int(mask.sum()),
        })
    out = ROOT / "results" / "extension_e" / "AJ_exact_vs_practical_D4_smoke.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

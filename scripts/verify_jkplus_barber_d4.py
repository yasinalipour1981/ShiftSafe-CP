"""Verify fixed JKplus intervals are Barber (asymmetric), not base±q."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines_e import JKPlusSourceCP
from src.base_models import fit_base_models
from src.conformal import conformal_quantile
from src.datasets import (
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)


def main() -> None:
    with open(ROOT / "configs" / "extension_e.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed = 142
    datasets = load_all_datasets(config, seed=seed)
    ratio = config.get("experiment", {}).get("train_calib_ratio", 0.8)
    df_train, df_calib, df_test = create_lodo_splits(datasets, ratio, seed=seed)["D4"]
    cols = get_feature_columns(df_train)
    df_train, df_calib, df_test = map(align_dataframe_features, (df_train, df_calib, df_test))
    X_tr = df_train[cols].values.astype(np.float64)
    y_tr = df_train["strength_mpa"].values.astype(np.float64)
    X_cal = df_calib[cols].values.astype(np.float64)
    y_cal = df_calib["strength_mpa"].values.astype(np.float64)
    X_te = df_test[cols].values.astype(np.float64)
    y_te = df_test["strength_mpa"].values.astype(np.float64)
    dom = df_calib["domain_id"].astype(str).values
    assert not np.any(np.char.startswith(dom.astype(str), "D4")), "LEAK"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary = fit_base_models(X_tr, y_tr, X_cal, y_cal, config=config, device=device, n_trials=0)["LightGBM"]
    jk = JKPlusSourceCP(
        primary, X_cal, y_cal, alpha=0.10, seed=seed, max_calib_samples=300,
        use_gpu=device.type == "cuda", held_out_domain="D4_mondal", calib_domains=dom,
    )
    jk.fit()
    lo, hi = jk.predict_intervals(X_te, alpha=0.10)
    base = primary.predict(X_te)
    q = conformal_quantile(jk._jk.scores_calib, 0.10)
    split_lo, split_hi = base - q, base + q
    mid = (lo + hi) / 2
    left = mid - lo
    right = hi - mid
    cov_jk = float(np.mean((y_te >= lo) & (y_te <= hi)))
    cov_split = float(np.mean((y_te >= split_lo) & (y_te <= split_hi)))
    print(f"n_models_trained={jk.diagnostics['n_models_trained']}")
    print(f"mean_width_jk={float(np.mean(hi-lo)):.3f} mean_width_splitstyle={float(np.mean(split_hi-split_lo)):.3f}")
    print(f"mean_left={left.mean():.3f} mean_right={right.mean():.3f} asymmetry={float(np.mean(np.abs(left-right))):.3f}")
    print(f"cov_jk={cov_jk:.3f} cov_splitstyle={cov_split:.3f}")
    print(f"allclose_to_splitstyle={np.allclose(lo, split_lo) and np.allclose(hi, split_hi)}")
    print(f"frac_asymmetric={(np.abs(left-right)>1e-6).mean():.3f}")


if __name__ == "__main__":
    main()

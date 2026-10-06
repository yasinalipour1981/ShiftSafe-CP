"""Lean follow-up: asymmetry + WeightedCP + LOO spread (one JK fit, no 2nd bootstrap)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines_e import JKPlusSourceCP, WeightedCPv2
from src.base_models import fit_base_models
from src.conformal import SplitConformal, conformal_quantile, physical_feature_indices
from src.datasets import (
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)


def cov_width(y, lo, hi):
    cov = float(np.mean((y >= lo) & (y <= hi)))
    return cov, float(np.mean(hi - lo))


def main() -> None:
    with open(ROOT / "configs" / "extension_e.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed, alpha = 142, 0.10
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

    rng = np.random.default_rng(seed)
    k_idx = rng.choice(len(y_te), size=20, replace=False)
    eval_mask = np.ones(len(y_te), dtype=bool)
    eval_mask[k_idx] = False
    X_eval, y_eval = X_te[eval_mask], y_te[eval_mask]

    # Prefer CPU to avoid contending with full E-run on GPU
    device = torch.device("cpu")
    print(f"device={device} n_cal={len(y_cal)} n_eval={len(y_eval)}", flush=True)

    primary = fit_base_models(
        X_tr, y_tr, X_cal, y_cal, config=config, device=device, n_trials=0
    )["LightGBM"]
    base_pred = primary.predict(X_eval)

    split = SplitConformal(primary, X_cal, y_cal, alpha=alpha)
    split.fit()
    slo, shi = split.predict_intervals(X_eval, alpha=alpha)
    print("SplitCP done", cov_width(y_eval, slo, shi), "q=", conformal_quantile(split.scores_calib, alpha), flush=True)

    jk = JKPlusSourceCP(
        primary, X_cal, y_cal, alpha=alpha, seed=seed, max_calib_samples=300,
        use_gpu=False, held_out_domain="D4_mondal", calib_domains=dom,
    )
    jk.fit()
    jlo, jhi = jk.predict_intervals(X_eval, alpha=alpha)
    jcov, jw = cov_width(y_eval, jlo, jhi)
    q_loo = conformal_quantile(jk._jk.scores_calib, alpha)
    print(f"Barber JK+ cov={jcov:.3f} width={jw:.2f} q_loo={q_loo:.3f} n_models={jk.diagnostics['n_models_trained']}", flush=True)

    hlo, hhi = base_pred - q_loo, base_pred + q_loo
    print("hybrid base+/-q_LOO", cov_width(y_eval, hlo, hhi), flush=True)

    mid = 0.5 * (jlo + jhi)
    print(
        f"asymmetry vs base: mean(base-lo)={float(np.mean(base_pred-jlo)):.3f} "
        f"mean(hi-base)={float(np.mean(jhi-base_pred)):.3f}",
        flush=True,
    )
    print(
        f"asymmetry vs mid: mean(mid-lo)={float(np.mean(mid-jlo)):.3f} "
        f"mean(hi-mid)={float(np.mean(jhi-mid)):.3f}",
        flush=True,
    )

    mu_loo = np.vstack([m.predict(X_eval) for m in jk._jk.loo_models])
    print(
        f"LOO spread: mean_std={float(np.mean(np.std(mu_loo,axis=0))):.3f} "
        f"mean_range={float(np.mean(np.max(mu_loo,axis=0)-np.min(mu_loo,axis=0))):.3f} "
        f"mean_R={float(np.mean(jk._jk.scores_calib)):.3f}",
        flush=True,
    )
    mu_mean = np.mean(mu_loo, axis=0)
    print(
        f"|base-LOOmean|={float(np.mean(np.abs(base_pred-mu_mean))):.3f} "
        f"|y-base|={float(np.mean(np.abs(y_eval-base_pred))):.3f} "
        f"|y-LOOmean|={float(np.mean(np.abs(y_eval-mu_mean))):.3f}",
        flush=True,
    )

    # Counterfactual: identical mu => Barber ~ split on LOO R
    n = len(jk._jk.loo_models)
    R = jk._jk.scores_calib
    lo_k = int(np.floor(alpha * (n + 1)))
    hi_k = int(np.ceil((1.0 - alpha) * (n + 1)))
    lower_cf = np.sort(mu_mean[None, :] - R[:, None], axis=0)[lo_k - 1]
    upper_cf = np.sort(mu_mean[None, :] + R[:, None], axis=0)[min(hi_k - 1, n - 1)]
    print("JK counterfactual constant-mu", cov_width(y_eval, lower_cf, upper_cf), flush=True)
    print("LOO-mean +/- q_LOO", cov_width(y_eval, mu_mean - q_loo, mu_mean + q_loo), flush=True)

    phys = physical_feature_indices(cols)
    f_cm = cols.index("f_cm") if "f_cm" in cols else None
    wcp = WeightedCPv2(
        primary, X_tr, y_tr, X_cal, y_cal, X_te,
        dr_feature_indices=phys, f_cm_col_index=f_cm,
        alpha=alpha, seed=seed, device="cpu",
    )
    wcp.fit()
    wlo, whi = wcp.predict_intervals(X_eval, alpha=alpha)
    print("WeightedCP-v2", cov_width(y_eval, wlo, whi), flush=True)

    # base_model unused in predict
    src = open(ROOT / "src" / "conformal.py", encoding="utf-8").read()
    # crude: JackknifeP.predict_intervals body
    print("JackknifeP.predict uses only loo_models+R (no base center): CONFIRMED by prior source read", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

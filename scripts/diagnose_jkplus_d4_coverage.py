"""Diagnose why JKplus-source D4 smoke ~72% cov vs expected SplitCP collapse."""
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
    y = np.asarray(y)
    lo, hi = np.asarray(lo), np.asarray(hi)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    w = float(np.mean(hi - lo))
    return cov, w


def main() -> None:
    with open(ROOT / "configs" / "extension_e.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed = 142
    alpha = 0.10
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
    assert not np.any([str(d).startswith("D4") for d in dom]), "LEAK"

    # Match smoke: K=20 holdout from target for eval mask
    rng = np.random.default_rng(seed)
    k_idx = rng.choice(len(y_te), size=20, replace=False)
    eval_mask = np.ones(len(y_te), dtype=bool)
    eval_mask[k_idx] = False
    X_eval, y_eval = X_te[eval_mask], y_te[eval_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    print(f"device={device} n_train={len(y_tr)} n_cal={len(y_cal)} n_test={len(y_te)} n_eval={len(y_eval)}")

    primary = fit_base_models(
        X_tr, y_tr, X_cal, y_cal, config=config, device=device, n_trials=0
    )["LightGBM"]
    base_pred = primary.predict(X_eval)

    # --- SplitCP ---
    split = SplitConformal(primary, X_cal, y_cal, alpha=alpha)
    split.fit()
    slo, shi = split.predict_intervals(X_eval, alpha=alpha)
    scov, sw = cov_width(y_eval, slo, shi)

    # --- broken hybrid: Barber LOO R but base±q ---
    jk = JKPlusSourceCP(
        primary, X_cal, y_cal, alpha=alpha, seed=seed, max_calib_samples=300,
        use_gpu=use_gpu, held_out_domain="D4_mondal", calib_domains=dom,
    )
    jk.fit()
    print(f"n_models_trained={jk.diagnostics['n_models_trained']} n_calib_used={jk.diagnostics['n_calib_used']}")
    q_loo = conformal_quantile(jk._jk.scores_calib, alpha)
    q_split = conformal_quantile(split.scores_calib, alpha)
    hlo, hhi = base_pred - q_loo, base_pred + q_loo
    hcov, hw = cov_width(y_eval, hlo, hhi)

    # --- true Barber JK+ (single fit, smoke path) ---
    jlo, jhi = jk.predict_intervals(X_eval, alpha=alpha)
    jcov, jw = cov_width(y_eval, jlo, jhi)

    # --- WeightedCP-v2 ---
    phys = physical_feature_indices(cols)
    f_cm = cols.index("f_cm") if "f_cm" in cols else None
    wcp = WeightedCPv2(
        primary, X_tr, y_tr, X_cal, y_cal, X_te,
        dr_feature_indices=phys, f_cm_col_index=f_cm,
        alpha=alpha, seed=seed, device=str(device),
    )
    wcp.fit()
    wlo, whi = wcp.predict_intervals(X_eval, alpha=alpha)
    wcov, ww = cov_width(y_eval, wlo, whi)

    # Asymmetry relative to base_model AND relative to JK midpoint
    mid = (jlo + jhi) / 2
    asym_base_lo = float(np.mean(base_pred - jlo))
    asym_base_hi = float(np.mean(jhi - base_pred))
    asym_mid_lo = float(np.mean(mid - jlo))
    asym_mid_hi = float(np.mean(jhi - mid))

    # LOO prediction spread on eval
    mu_loo = np.vstack([m.predict(X_eval) for m in jk._jk.loo_models])  # (n, n_eval)
    loo_std = float(np.mean(np.std(mu_loo, axis=0)))
    loo_range = float(np.mean(np.max(mu_loo, axis=0) - np.min(mu_loo, axis=0)))
    mean_R = float(np.mean(jk._jk.scores_calib))
    # Correlation: does interval width track LOO spread?
    widths = jhi - jlo
    spreads = np.std(mu_loo, axis=0)
    corr_w_spread = float(np.corrcoef(widths, spreads)[0, 1])

    # Does base_model enter predict? Compare swapping base predictions into nothing
    # (JackknifeP.predict_intervals never calls base_model)
    import inspect
    src = inspect.getsource(jk._jk.predict_intervals)
    uses_base = "base_model" in src or "self.base_model" in src

    # Off-by-one probe: alt indices
    n = len(jk._jk.loo_models)
    R = jk._jk.scores_calib
    lo_k = int(np.floor(alpha * (n + 1)))
    hi_k = int(np.ceil((1.0 - alpha) * (n + 1)))
    print(f"Barber ranks: n={n} lo_k={lo_k} hi_k={hi_k} (1-indexed); q_loo={q_loo:.3f} q_split={q_split:.3f}")

    # If μ identical, JK ≈ Split with LOO residuals: build that counterfactual
    mu_mean = np.mean(mu_loo, axis=0)
    lower_cands = np.sort(mu_loo - R[:, None], axis=0)
    upper_cands = np.sort(mu_loo + R[:, None], axis=0)
    # counterfactual: replace mu_loo with constant mean
    lower_cf = np.sort(mu_mean[None, :] - R[:, None], axis=0)
    upper_cf = np.sort(mu_mean[None, :] + R[:, None], axis=0)
    lo_cf = lower_cf[lo_k - 1] if lo_k > 0 else np.full(len(y_eval), -np.inf)
    hi_cf = upper_cf[min(hi_k - 1, n - 1)]
    cfcov, cfw = cov_width(y_eval, lo_cf, hi_cf)

    # Also: LOO residual SplitCP centered at mean LOO pred (not base)
    q = conformal_quantile(R, alpha)
    lo_loo_split, hi_loo_split = mu_mean - q, mu_mean + q
    lscov, lsw = cov_width(y_eval, lo_loo_split, hi_loo_split)

    print("\n=== COVERAGE / WIDTH TABLE (D4 seed=142 eval) ===")
    rows = [
        ("SplitCP (base+/-q_abs)", scov, sw),
        ("broken hybrid (base+/-q_LOO)", hcov, hw),
        ("true Barber JK+", jcov, jw),
        ("JK counterfactual (mean mu +/- R orderstats)", cfcov, cfw),
        ("LOO-mean +/- q_LOO (split-style)", lscov, lsw),
        ("WeightedCP-v2", wcov, ww),
    ]
    print(f"{'method':40s} {'cov':>8s} {'width':>8s}")
    for name, c, w in rows:
        print(f"{name:40s} {c:8.3f} {w:8.2f}")

    print("\n=== ASYMMETRY (true Barber) ===")
    print(f"mean(pred_base - lower)={asym_base_lo:.3f}  mean(upper - pred_base)={asym_base_hi:.3f}")
    print(f"mean(mid - lower)={asym_mid_lo:.3f}  mean(upper - mid)={asym_mid_hi:.3f}")

    print("\n=== LOO / BASE INFLUENCE ===")
    print(f"predict_intervals source mentions base_model: {uses_base}")
    print(f"mean LOO std across models @eval: {loo_std:.3f} MPa")
    print(f"mean LOO range (max-min) @eval: {loo_range:.3f} MPa")
    print(f"mean LOO residual R: {mean_R:.3f} MPa")
    print(f"corr(JK width, LOO pred std): {corr_w_spread:.3f}")
    print(f"mean |base - LOO-mean|: {float(np.mean(np.abs(base_pred - mu_mean))):.3f}")
    print(f"mean |y - base|: {float(np.mean(np.abs(y_eval - base_pred))):.3f}")
    print(f"mean |y - LOO-mean|: {float(np.mean(np.abs(y_eval - mu_mean))):.3f}")

    # Bootstrap averaging artifact check: smoke used n_reps=1, so cannot explain 72%.
    # Train one extra rep and average endpoints with the already-fitted smoke JK.
    print("\n=== BOOTSTRAP AVERAGING (reuse fitted JK as rep0 + 1 extra) ===")
    print("(smoke path uses n_reps=1, averaging cannot explain smoke 72%)")
    print(f"n_reps=1 (fitted): cov={jcov:.3f} width={jw:.2f}")
    rng2 = np.random.default_rng(142 + 1000)
    n_cal = len(X_cal)
    max_n = 300
    if n_cal > max_n:
        idx2 = rng2.choice(n_cal, max_n, replace=False)
        X_sub2, y_sub2 = X_cal[idx2], y_cal[idx2]
        dom2 = dom[idx2]
    else:
        X_sub2, y_sub2, dom2 = X_cal, y_cal, dom
    jk2 = JKPlusSourceCP(
        primary, X_sub2, y_sub2, alpha=alpha, seed=143, max_calib_samples=max_n,
        use_gpu=use_gpu, held_out_domain="D4_mondal", calib_domains=dom2,
    )
    jk2.fit()
    jlo2, jhi2 = jk2.predict_intervals(X_eval, alpha=alpha)
    c2, w2 = cov_width(y_eval, jlo2, jhi2)
    lo_avg = 0.5 * (jlo + jlo2)
    hi_avg = 0.5 * (jhi + jhi2)
    ca, wa = cov_width(y_eval, lo_avg, hi_avg)
    print(f"n_reps=1 (alt seed): cov={c2:.3f} width={w2:.2f}")
    print(f"averaged endpoints (2 reps): cov={ca:.3f} width={wa:.2f}")
    print(f"avg width vs mean of widths: {wa:.2f} vs {0.5*(jw+w2):.2f}")

    # Sanity: fraction of eval points where Barber interval contains y vs SplitCP
    both = ((y_eval >= jlo) & (y_eval <= jhi) & (y_eval >= slo) & (y_eval <= shi)).mean()
    jk_only = ((y_eval >= jlo) & (y_eval <= jhi) & ~((y_eval >= slo) & (y_eval <= shi))).mean()
    print(f"\nfrac covered by both JK+Split={both:.3f}  JK-only={jk_only:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Mondal D4 reconstruction artifact check (v3.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_to_dict, load_config
from src.datasets import (
    generate_derived_features,
    load_all_datasets,
    create_lodo_splits,
    get_feature_columns,
    align_dataframe_features,
)
from src.base_models import fit_base_models
from src.utils import append_execution_log

OFFSET_THRESHOLD_MPA = 5.0


def _mondal_equation_strength(df: pd.DataFrame) -> np.ndarray:
    """Deterministic Mondal equation strengths (no noise) for comparison."""
    strengths = []
    for age in sorted(df["age"].unique()):
        mask = df["age"] == age
        sub = df.loc[mask, ["cement", "fly_ash", "water"]]
        cm = sub["cement"] + sub["fly_ash"]
        w_cm = sub["water"] / cm.clip(lower=1)
        f_cm = sub["fly_ash"] / cm.clip(lower=1)
        age_i = int(age)
        if age_i == 7:
            a0, a1, a2 = 4.65, 2.1, 1.2
        elif age_i == 28:
            a0, a1, a2 = 5.0159, 2.448, 1.533
        elif age_i == 90:
            a0, a1, a2 = 5.46, 2.94, 0.756
        else:
            continue
        s = np.exp(a0 - a1 * w_cm - a2 * f_cm)
        strengths.append(pd.Series(s.values, index=sub.index))
    out = pd.concat(strengths).sort_index()
    return out.reindex(df.index).values


def main() -> int:
    cfg = config_to_dict(load_config("configs/base.yaml"))
    cfg["n_optuna_trials"] = 0
    cfg.setdefault("models", {})
    for m in ("xgboost", "catboost", "deep_mlp"):
        cfg["models"].setdefault(m, {})["enabled"] = False
    cfg["models"].setdefault("lightgbm", {})["enabled"] = True
    cfg["models"]["lightgbm"]["n_trials"] = 0

    datasets = load_all_datasets(cfg, seed=42)
    splits = create_lodo_splits(datasets, 0.8, seed=42)
    df_tr, df_ca, df_te = splits["D4"]
    df_tr = generate_derived_features(df_tr, cfg)
    df_ca = generate_derived_features(df_ca, cfg)
    df_te = generate_derived_features(df_te, cfg)
    fc = get_feature_columns(df_tr)
    df_tr, df_ca, df_te = align_dataframe_features(df_tr), align_dataframe_features(df_ca), align_dataframe_features(df_te)

    X_tr = df_tr[fc].values
    y_tr = df_tr["strength_mpa"].values
    X_ca = df_ca[fc].values
    y_ca = df_ca["strength_mpa"].values
    X_te = df_te[fc].values
    y_te = df_te["strength_mpa"].values

    models = fit_base_models(X_tr, y_tr, X_ca, y_ca, cfg, n_trials=0)
    mu_te = models["LightGBM"].predict(X_te)

    eq_strength = _mondal_equation_strength(df_te)
    offset = float(np.mean(y_te - eq_strength))
    lr = LinearRegression().fit(eq_strength.reshape(-1, 1), mu_te)
    fitted_offset = float(lr.intercept_)

    out_dir = Path(cfg["results_dir"]) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(eq_strength, y_te, alpha=0.5, s=20, label="CSV strength")
    axes[0].plot([5, 80], [5, 80], "k--", alpha=0.4)
    axes[0].set_xlabel("Mondal equation strength (MPa)")
    axes[0].set_ylabel("CSV strength (MPa)")
    axes[0].set_title("Equation vs CSV (D4)")
    axes[0].legend(fontsize=8)

    axes[1].scatter(eq_strength, mu_te, alpha=0.5, s=20, label="Source mu_hat")
    xs = np.linspace(eq_strength.min(), eq_strength.max(), 50)
    axes[1].plot(xs, lr.predict(xs.reshape(-1, 1)), "r-", label=f"fit intercept={fitted_offset:.1f}")
    axes[1].set_xlabel("Mondal equation strength (MPa)")
    axes[1].set_ylabel("Source model mu_hat (MPa)")
    axes[1].set_title("Equation vs source predictor (D4)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "mondal_artifact_scatter.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = {
        "fold": "D4",
        "mean_csv_minus_equation": offset,
        "mean_mu_minus_equation": float(np.mean(mu_te - eq_strength)),
        "fitted_offset_intercept": fitted_offset,
        "fitted_slope": float(lr.coef_[0]),
        "flag_synthetic_artifact": abs(fitted_offset) > OFFSET_THRESHOLD_MPA
            or abs(offset) > OFFSET_THRESHOLD_MPA,
    }

    import json
    with open(out_dir / "mondal_artifact_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Mondal artifact check D4:")
    print(f"  mean(CSV - equation) = {offset:.2f} MPa")
    print(f"  fitted offset (mu_hat vs equation) = {fitted_offset:.2f} MPa")
    if report["flag_synthetic_artifact"]:
        msg = (
            "D4 concept shift partially attributable to synthetic reconstruction "
            f"(offset > {OFFSET_THRESHOLD_MPA} MPa)"
        )
        print(f"  FLAG: {msg}")
        append_execution_log(msg)
    else:
        append_execution_log(
            f"Mondal D4 artifact check: offset={fitted_offset:.2f} MPa (below {OFFSET_THRESHOLD_MPA} threshold)"
        )
    print(f"  Saved: {out_dir / 'mondal_artifact_scatter.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

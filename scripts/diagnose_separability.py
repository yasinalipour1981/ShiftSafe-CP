#!/usr/bin/env python
"""Stage-4 v3: domain separability diagnostics (run before model training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_to_dict, load_config
from src.conformal import PHYSICAL_SUBSPACE_COLS, _domain_classifier_auc
from src.datasets import (
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)
from src.utils import append_execution_log

MONDAL_FOLD = "D4"


def _verdict(auc_physical: float) -> str:
    if auc_physical < 0.75:
        return "artificial separability: use physical-subspace ratio"
    if auc_physical >= 0.90:
        return (
            "genuine disjoint support: weighted CP cannot work; "
            "v3 target-anchored path is primary"
        )
    return "moderate shift: weighted CP may partially help; v3 recommended"


def _top5_coef(clf: LogisticRegression, names: list[str]) -> list[dict]:
    coef = clf.coef_[0]
    order = np.argsort(np.abs(coef))[::-1][:5]
    return [{"feature": names[i], "coef": float(coef[i])} for i in order]


def _plot_overlap(
    src: pd.DataFrame,
    tgt: pd.DataFrame,
    fold: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), PHYSICAL_SUBSPACE_COLS):
        if col not in src.columns:
            continue
        ax.hist(src[col], bins=25, alpha=0.5, density=True, label="source")
        ax.hist(tgt[col], bins=25, alpha=0.5, density=True, label="target")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle(f"Covariate overlap, fold {fold}", fontweight="bold")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"overlap_{fold}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    config = config_to_dict(load_config("configs/base.yaml"))
    results_dir = Path(config.get("results_dir", "results"))
    diag_dir = results_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets(config, seed=42)
    splits = create_lodo_splits(
        datasets, config.get("evaluation", {}).get("train_calib_split", 0.8), seed=42
    )

    rows = []
    for fold, (df_train, df_calib, df_test) in splits.items():
        df_src = align_dataframe_features(pd.concat([df_train, df_calib]))
        df_tgt = align_dataframe_features(df_test)
        feature_cols = get_feature_columns(df_src)
        phys_idx = [feature_cols.index(c) for c in PHYSICAL_SUBSPACE_COLS if c in feature_cols]

        X_src_all = df_src[feature_cols].values.astype(np.float64)
        X_tgt_all = df_tgt[feature_cols].values.astype(np.float64)
        X_src_phys = X_src_all[:, phys_idx]
        X_tgt_phys = X_tgt_all[:, phys_idx]

        auc_all, clf_all, top_all = _domain_classifier_auc(X_src_all, X_tgt_all, seed=42)
        auc_phys, clf_phys, _ = _domain_classifier_auc(X_src_phys, X_tgt_phys, seed=42)
        top5 = _top5_coef(clf_all, feature_cols)

        note = ""
        if fold == MONDAL_FOLD:
            note = (
                "FLAG: D4/Mondal uses synthetic-reconstructed data; "
                "separability may reflect reconstruction noise."
            )

        row = {
            "fold": fold,
            "AUC_all_features": round(auc_all, 4),
            "AUC_physical": round(auc_phys, 4),
            "verdict": _verdict(auc_phys),
            "top5_features": top5,
            "note": note,
        }
        rows.append(row)
        print(f"\n=== Fold {fold} ===")
        print(f"  AUC (all features):  {auc_all:.3f}")
        print(f"  AUC (physical only): {auc_phys:.3f}")
        print(f"  Top-5 |coef|: {top5}")
        print(f"  Verdict: {row['verdict']}")
        if note:
            print(f"  {note}")

        _plot_overlap(df_src, df_tgt, fold, diag_dir)

    df = pd.DataFrame(rows)
    table_path = diag_dir / "separability_summary.csv"
    df[["fold", "AUC_all_features", "AUC_physical", "verdict", "note"]].to_csv(
        table_path, index=False
    )
    with open(diag_dir / "separability_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    append_execution_log(
        "v3 separability diagnostics complete; see results/diagnostics/separability_summary.csv"
    )
    print(f"\nSaved: {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

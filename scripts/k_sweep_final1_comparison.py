#!/usr/bin/env python
"""Side-by-side K-sweep comparison: pre-final.1 vs post-final.1 (AdaptiveJackknifeCP)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pubstyle import METHOD_COLORS, apply_pub_style, save_figure

PRE_PATH = ROOT / "results" / "FINAL_RESULTS_exploratory_pre_final1.csv"
POST_PATH = ROOT / "results" / "FINAL_RESULTS_exploratory_42-51.csv"
OUT_DIR = ROOT / "results" / "appendix"
METHOD = "AdaptiveJackknifeCP"
ALPHA = 0.10
K_VALUES = [10, 15, 20, 30]


def _agg_widths(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["method"] == METHOD) & (df["alpha"] == ALPHA) & (df["K"].isin(K_VALUES))]
    return (
        sub.groupby(["K", "fold"])["mean_interval_width"]
        .mean()
        .reset_index()
        .groupby("K")["mean_interval_width"]
        .agg(mean="mean", std="std")
        .reset_index()
    )


def main() -> None:
    if not PRE_PATH.exists():
        raise FileNotFoundError(f"Pre-final.1 snapshot missing: {PRE_PATH}")
    if not POST_PATH.exists():
        raise FileNotFoundError(f"Post-final.1 results missing: {POST_PATH}")

    pre = pd.read_csv(PRE_PATH)
    post = pd.read_csv(POST_PATH)
    pre_agg = _agg_widths(pre).rename(columns={"mean": "width_pre", "std": "width_pre_sd"})
    post_agg = _agg_widths(post).rename(columns={"mean": "width_post", "std": "width_post_sd"})

    cmp_tbl = pre_agg.merge(post_agg, on="K", how="outer")
    cmp_tbl["delta_width"] = cmp_tbl["width_post"] - cmp_tbl["width_pre"]
    cmp_tbl["pct_change"] = 100 * cmp_tbl["delta_width"] / cmp_tbl["width_pre"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl_path = OUT_DIR / "K_sweep_final1_comparison.csv"
    cmp_tbl.to_csv(tbl_path, index=False)
    print(cmp_tbl.to_string(index=False, float_format="%.2f"))
    print(f"\nTable saved: {tbl_path}")

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(K_VALUES))
    w = 0.35
    pre_vals = [cmp_tbl.loc[cmp_tbl["K"] == k, "width_pre"].iloc[0] for k in K_VALUES]
    post_vals = [cmp_tbl.loc[cmp_tbl["K"] == k, "width_post"].iloc[0] for k in K_VALUES]
    pre_sd = [cmp_tbl.loc[cmp_tbl["K"] == k, "width_pre_sd"].fillna(0).iloc[0] for k in K_VALUES]
    post_sd = [cmp_tbl.loc[cmp_tbl["K"] == k, "width_post_sd"].fillna(0).iloc[0] for k in K_VALUES]

    color = METHOD_COLORS.get(METHOD, "#D55E00")
    ax.bar([i - w / 2 for i in x], pre_vals, w, yerr=pre_sd, capsize=3,
           label="pre-final.1", color=color, alpha=0.45)
    ax.bar([i + w / 2 for i in x], post_vals, w, yerr=post_sd, capsize=3,
           label="post-final.1", color=color, alpha=0.95)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("K (labeled target points)")
    ax.set_ylabel("Mean interval width (MPa)")
    ax.set_title(f"{METHOD} width: pre vs post final.1 (α={ALPHA})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = OUT_DIR / "FIGURE_K_sweep_final1_comparison.pdf"
    save_figure(fig, fig_path)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()

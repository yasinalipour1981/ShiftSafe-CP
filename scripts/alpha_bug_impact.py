"""Regenerate alpha_bug_impact.csv from exploratory snapshots."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pre = ROOT / "results/FINAL_RESULTS_exploratory_pre_final1.csv"
post = ROOT / "results/FINAL_RESULTS_exploratory_42-51.csv"
out = ROOT / "results/diagnostics/alpha_bug_impact.csv"

METHODS = ["TargetOnlyCP", "AdaptiveSplitCP"]
K, ALPHA = 20, 0.10
KEYS = ["method", "fold", "seed", "K", "alpha"]


def main():
    if not pre.exists() or not post.exists():
        print("Missing pre/post exploratory snapshots")
        return
    df_pre = pd.read_csv(pre)
    df_post = pd.read_csv(post)
    b = df_pre[(df_pre.method.isin(METHODS)) & (df_pre.K == K) & (df_pre.alpha == ALPHA)]
    f = df_post[(df_post.method.isin(METHODS)) & (df_post.K == K) & (df_post.alpha == ALPHA)]
    m = b.merge(f, on=KEYS, suffixes=("_baseline", "_fixed"))
    for col in ["empirical_coverage", "mean_interval_width", "winkler_score"]:
        m[f"{col}_delta"] = m[f"{col}_fixed"] - m[f"{col}_baseline"]
    m.to_csv(out, index=False)
    print(f"Wrote {out} ({len(m)} rows)")
    print(m.groupby("method")[["empirical_coverage_delta", "mean_interval_width_delta"]].mean())


if __name__ == "__main__":
    main()

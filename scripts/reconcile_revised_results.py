"""Reconcile the revised confirmatory run with the supplementary revision grid.

Two result files now exist for the primary cell:

  * `results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv` -- the canonical
    pipeline (run_all.py) re-run under the outcome-independent draw and the exact
    Barber construction. This is what the manuscript's tables and figures report.
  * `results/revision/REVISION_draws_K20.csv` -- the supplementary grid, which
    additionally carries the target-only jackknife-plus comparator, the fixed
    stacked ablation and the three calibration draws.

The two use slightly different code paths to reach the same estimand, so this
script prints them side by side. Any headline number quoted in the manuscript or
the response letter must come from the canonical file; the grid is quoted only
for the comparisons that the canonical run does not contain, and the agreement
between them is reported so that a reader can see it is close.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "results" / "FINAL_RESULTS_confirmatory_REVISED_142-191.csv"
GRID = ROOT / "results" / "revision" / "REVISION_draws_K20.csv"
FOLDS = ("D1", "D2", "D4")
K, ALPHA = 20, 0.10


def canon_table(df: pd.DataFrame, folds=FOLDS) -> pd.DataFrame:
    p = df[(df.K == K) & (np.isclose(df.alpha, ALPHA)) & (df.fold.isin(folds))]
    per = p.groupby(["method", "fold"])[
        ["empirical_coverage", "mean_interval_width", "winkler_score"]
    ].mean()
    out = per.groupby("method").mean()
    out.columns = ["coverage", "width", "winkler"]
    return out


def grid_table(df: pd.DataFrame, draw="random", folds=FOLDS) -> pd.DataFrame:
    p = df[(df.draw == draw) & (df.fold.isin(folds))]
    per = p.groupby(["method", "fold"])[["coverage", "width", "winkler"]].mean()
    return per.groupby("method").mean()


def main() -> int:
    if not CANON.exists():
        print(f"missing {CANON}")
        return 1
    c = pd.read_csv(CANON)
    g = pd.read_csv(GRID)
    ct, gt = canon_table(c), grid_table(g)

    print(f"canonical run: {c.seed.nunique()} seeds; grid: {g.seed.nunique()} seeds\n")
    print("=== primary cell, pooled over D1/D2/D4 (each laboratory weighted equally) ===")
    print(f"{'method':36}{'canonical':>28}{'grid (random draw)':>30}")
    print(f"{'':36}{'cov    width  winkler':>28}{'cov    width  winkler':>30}")
    pairs = [
        ("AdaptiveJackknifeCP", "AdaptiveJackknifeCP-exactJK+"),
        ("TargetOnlyCP", "TargetOnlyCP"),
        ("TransferCal-CP-affine-localsigma", "TransferCal-CP-affine-localsigma"),
        ("AdaptiveSplitCP", "AdaptiveSplitCP"),
        ("SplitCP", "SplitCP-source"),
    ]
    for cm, gm in pairs:
        if cm not in ct.index:
            continue
        a = ct.loc[cm]
        b = gt.loc[gm] if gm in gt.index else None
        left = f"{a.coverage:5.3f} {a.width:7.2f} {a.winkler:8.2f}"
        right = (f"{b.coverage:5.3f} {b.width:7.2f} {b.winkler:8.2f}"
                 if b is not None else "-")
        print(f"{cm:36}{left:>28}{right:>30}")

    print("\n=== per laboratory, canonical run (Winkler) ===")
    p = c[(c.K == K) & (np.isclose(c.alpha, ALPHA)) & (c.fold.isin(FOLDS))]
    print(p.pivot_table(index="method", columns="fold",
                        values="winkler_score").round(2).to_string())
    print("\n=== per laboratory, canonical run (coverage) ===")
    print(p.pivot_table(index="method", columns="fold",
                        values="empirical_coverage").round(3).to_string())

    print("\n=== with and without D4 (canonical, Winkler) ===")
    a = canon_table(c, FOLDS)["winkler"]
    b = canon_table(c, ("D1", "D2"))["winkler"]
    print(pd.DataFrame({"all three": a, "without D4": b}).round(2).to_string())

    print("\n=== leave-one-laboratory-out pooled Winkler (canonical) ===")
    lodo = {f"without {d}": canon_table(c, tuple(f for f in FOLDS if f != d))["winkler"]
            for d in FOLDS}
    lodo["all three"] = a
    print(pd.DataFrame(lodo).round(2).to_string())

    # Agreement between the two paths on the shared arms.
    print("\n=== agreement between canonical run and supplementary grid ===")
    for cm, gm in pairs:
        if cm in ct.index and gm in gt.index:
            d_cov = ct.loc[cm, "coverage"] - gt.loc[gm, "coverage"]
            d_w = ct.loc[cm, "winkler"] - gt.loc[gm, "winkler"]
            print(f"  {cm:36} dcoverage {d_cov:+.3f}   dWinkler {d_w:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

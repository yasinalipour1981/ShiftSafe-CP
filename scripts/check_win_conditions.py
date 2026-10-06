#!/usr/bin/env python
"""Evaluate the pre-registered ShiftSafe-CP win conditions from the production results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results/FINAL_RESULTS_ALLSEEDS.csv")
ALPHA = 0.10  # nominal 90% coverage
NOMINAL = 1 - ALPHA
TARGET_METHOD = "ShiftSafe-CP"


def main():
    if not RESULTS.exists():
        print(f"Missing {RESULTS}. Run production pipeline first.")
        return 1

    df = pd.read_csv(RESULTS)
    df_a = df[df["alpha"] == ALPHA].copy()

    print("=" * 70)
    print("WIN CONDITION CHECK (alpha=0.10, nominal 90% coverage)")
    print("=" * 70)

    # Win condition 1: validity under shift
    print("\n[1] Empirical coverage by domain (ShiftSafe-CP vs baselines)")
    pivot_cov = df_a.pivot_table(
        index="fold", columns="method", values="empirical_coverage", aggfunc="mean"
    ).round(3)
    print(pivot_cov.to_string())

    if TARGET_METHOD in pivot_cov.columns:
        ss_cov = pivot_cov[TARGET_METHOD]
        win1 = (ss_cov >= NOMINAL).all()
        print(f"\n  ShiftSafe-CP >= {NOMINAL:.0%} on ALL domains: {'PASS' if win1 else 'FAIL'}")
        for fold, cov in ss_cov.items():
            status = "OK" if cov >= NOMINAL else "BELOW"
            print(f"    {fold}: {cov:.1%} [{status}]")
    else:
        win1 = False
        print(f"  {TARGET_METHOD} not found in results")

    # Win condition 2: interval efficiency vs Jackknife+
    print("\n[2] Mean interval width at alpha=0.10 (lower is better)")
    pivot_w = df_a.pivot_table(
        index="fold", columns="method", values="mean_interval_width", aggfunc="mean"
    ).round(2)
    cols = [c for c in ["ShiftSafe-CP", "Jackknife+", "SplitCP", "CQR"] if c in pivot_w.columns]
    print(pivot_w[cols].to_string())

    win2 = False
    if TARGET_METHOD in pivot_w.columns and "Jackknife+" in pivot_w.columns:
        win2 = (pivot_w[TARGET_METHOD] <= pivot_w["Jackknife+"]).mean() >= 0.5
        print(f"\n  ShiftSafe-CP narrower than Jackknife+ on majority of folds: {'PASS' if win2 else 'FAIL'}")

    # Win condition 3: timing
    print("\n[3] Computational timing (seconds)")
    timing_path = Path("results/lodo_evaluation/timing_results.csv")
    if timing_path.exists():
        t = pd.read_csv(timing_path)
        agg = t.groupby("method")[["fit_time_sec", "calib_time_sec", "infer_time_per_1000_sec"]].mean()
        print(agg.round(3).to_string())
        if TARGET_METHOD in agg.index:
            calib = agg.loc[TARGET_METHOD, "calib_time_sec"]
            infer = agg.loc[TARGET_METHOD, "infer_time_per_1000_sec"]
            win3 = calib < 10 and infer < 10
            print(f"\n  ShiftSafe-CP calib < 10s: {calib:.2f}s {'PASS' if calib < 10 else 'FAIL'}")
            print(f"  ShiftSafe-CP infer/1000 < 10s: {infer:.3f}s {'PASS' if infer < 10 else 'FAIL'}")
        else:
            win3 = False
    else:
        win3 = None
        print("  Timing file not found")

    # Summary table for paper
    print("\n[Summary] ShiftSafe-CP vs SplitCP (Wilcoxon)")
    wilcoxon_path = Path("results/lodo_evaluation/wilcoxon_tests.json")
    if wilcoxon_path.exists():
        tests = json.loads(wilcoxon_path.read_text())
        for t in tests:
            if t.get("method_a") == TARGET_METHOD or t.get("method_b") == TARGET_METHOD:
                print(f"  {t['method_a']} vs {t['method_b']}: p={t['p_value']:.4f}, sig={t['significant_at_0.05']}")

    print("\n" + "=" * 70)
    overall = win1 and win2 and (win3 is not False)
    print(f"OVERALL: {'PASS' if overall else 'NEEDS REVIEW'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

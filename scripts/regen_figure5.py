#!/usr/bin/env python
"""Regenerate FIGURE_5 from timing_results.csv."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.visualization import figure_5_timing_comparison

df = pd.read_csv(ROOT / "results/lodo_evaluation/timing_results.csv")
out = ROOT / "results/figures/FIGURE_5_timing_comparison"
figure_5_timing_comparison(df, out)
agg = df.groupby("method").agg(
    source=("source_fit_time_sec", "mean") if "source_fit_time_sec" in df.columns else ("fit_time_sec", "mean"),
    calib=("calib_time_sec", "mean"),
    infer=("infer_time_per_1000_sec", "mean"),
)
print("\n=== FIGURE_5 aggregated timing ===")
print(agg.to_string())
print(f"\nWrote {out}.pdf and .png")

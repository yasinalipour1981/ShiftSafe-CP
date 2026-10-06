#!/usr/bin/env python
"""Export PNG previews of all figures (lighter than TIFF for quick viewing)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.pubstyle as pubstyle
from scripts.extension_e_reporting import filter_stale_extension_jk
from src.validity import annotate_guaranteed
from src import visualization as viz

FIG_DIR = ROOT / "results" / "figures"
CONF = ROOT / "results" / "FINAL_RESULTS_confirmatory_142-191.csv"
E_MERGE = ROOT / "results" / "extension_e" / "preview" / "EXTENSION_E_merged_preview.csv"
OUT = FIG_DIR / "png"


def _save_png(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df_c = annotate_guaranteed(pd.read_csv(CONF))
    df_e = pd.read_csv(E_MERGE) if E_MERGE.exists() else pd.DataFrame()
    sub_c = df_c[(df_c["K"] == 20) & (df_c["alpha"] == 0.10) & df_c["fold"].isin(["D1", "D2", "D4"])]
    if not df_e.empty:
        sub_e = filter_stale_extension_jk(df_e[(df_e["K"] == 20) & (df_e["alpha"] == 0.10)])
        df3 = pd.concat([sub_c, sub_e], ignore_index=True)
        df3 = annotate_guaranteed(df3)
    else:
        df3 = sub_c

    shift = ROOT / "results/shift_analysis/domain_classifier_auc.csv"
    timing = ROOT / "results/lodo_evaluation/timing_results.csv"
    sep = ROOT / "results/diagnostics/separability_summary.csv"
    diag = ROOT / "results/diagnostics"

    df_shift = pd.read_csv(shift) if shift.exists() else None
    df_timing = pd.read_csv(timing) if timing.exists() else None
    df_sep = pd.read_csv(sep) if sep.exists() else None

    jobs = [
        ("FIGURE_1_shift_heatmap", lambda: viz.figure_1_shift_heatmap(
            df_shift, OUT / "FIGURE_1_shift_heatmap", df_sep)),
        ("FIGURE_2_reliability_diagrams", lambda: viz.figure_2_reliability_diagrams(
            df_c, OUT / "FIGURE_2_reliability_diagrams")),
        ("FIGURE_3_coverage_width_pareto", lambda: viz.figure_3_coverage_width_pareto(
            df3, OUT / "FIGURE_3_coverage_width_pareto")),
        ("FIGURE_4_conditional_coverage", lambda: viz.figure_4_conditional_coverage(
            df_c, OUT / "FIGURE_4_conditional_coverage")),
        ("FIGURE_5_timing_comparison", lambda: viz.figure_5_timing_comparison(
            df_timing, OUT / "FIGURE_5_timing_comparison")),
        ("FIGURE_6_example_intervals", lambda: viz.figure_6_example_intervals(
            df_c, OUT / "FIGURE_6_example_intervals", diag)),
        ("FIGURE_7_ablation_breakdown", lambda: viz.figure_7_ablation_breakdown(
            df_c, OUT / "FIGURE_7_ablation_breakdown")),
        ("FIGURE_K_width_vs_K", lambda: viz.figure_k_width_vs_k(
            df_c, OUT / "FIGURE_K_width_vs_K")),
        ("FIGURE_COV_coverage_vs_K", lambda: viz.figure_coverage_vs_k(
            df_c, OUT / "FIGURE_COV_coverage_vs_K")),
        ("FIGURE_SELECT_candidate_histogram", lambda: viz.figure_selection_histogram(
            diag, OUT / "FIGURE_SELECT_candidate_histogram")),
        ("FLOWCHART_methodology", lambda: viz.draw_flowchart_methodology(
            OUT / "FLOWCHART_methodology")),
        ("FLOWCHART_lodo_pipeline", lambda: viz.draw_flowchart_lodo(
            OUT / "FLOWCHART_lodo_pipeline")),
    ]

    # Patch save_fig to write PNG into results/figures/png/
    orig = pubstyle.save_fig

    def patched(fig, name, formats=("png",)):
        p = Path(name)
        if not p.is_absolute():
            p = OUT / p.name
        _save_png(fig, p)

    pubstyle.save_fig = patched  # type: ignore[assignment]
    viz.save_fig = patched  # visualization binds save_fig at import time

    ok, skip = [], []
    for name, fn in jobs:
        if name.startswith("FIGURE_1") and df_shift is None:
            skip.append(name)
            continue
        if name == "FIGURE_5_timing_comparison" and (df_timing is None or df_timing.empty):
            skip.append(name)
            continue
        try:
            fn()
            ok.append(name)
            print(f"  {name}.png")
        except Exception as exc:
            skip.append(f"{name}: {exc}")
            print(f"  SKIP {name}: {exc}")

    pubstyle.save_fig = orig

    # Extension E smoke PNG (already exists; copy path note)
    smoke = ROOT / "results/extension_e/preview/SMOKE_extension_e.png"
    index = OUT / "ALL_CHARTS.md"
    lines = [
        "# All charts (PNG previews)",
        "",
        f"Generated PNGs in `{OUT.relative_to(ROOT)}`",
        "",
        "## Main figures",
    ]
    for n in ok:
        lines.append(f"- [{n}]({n}.png)")
    lines.extend([
        "",
        "## Extension E",
        f"- [SMOKE_extension_e](../extension_e/preview/SMOKE_extension_e.png)",
        "",
        "## PDF/TIFF (publication)",
        "Full resolution: `results/figures/*.pdf` and `*.tiff`",
        "",
        "## Skipped",
    ])
    for s in skip:
        lines.append(f"- {s}")
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nIndex: {index}")
    print(f"Smoke: {smoke}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

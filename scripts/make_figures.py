#!/usr/bin/env python
"""Single entry point to reproduce all figures from results CSV alone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import generate_all_figures  # noqa: E402
from src.validity import annotate_guaranteed  # noqa: E402


CAPTION_NOTES = """# Figure caption notes

## Operating regions (methods section)
- Jackknife+ arms (AdaptiveJackknifeCP): K >= 9 at alpha=0.10 (K >= ceil(1/alpha)-1).
- Split arms (TargetOnlyCP, AdaptiveSplitCP, TransferCal-*): K >= 18 at alpha=0.10 for a balanced fit/cal split with finite-sample guarantee.

## FIGURE_2
- K=20, alpha sweep across D1, D2, D4. **D3 excluded** from panels (K=10 only in protocol; no K=20 evaluation).
- SplitCP omitted (~5% coverage, off-scale).
- **A2 verdict:** TargetOnlyCP and AdaptiveSplitCP coincide only at K≤10 (local-ridge collapse at K_fit=1; 120/120 identical cells). At **K=20** mean curves differ by up to ~4.7 pp (D4); distinct markers (circle vs square) in figure.
- Grey dashed diagonal: ideal calibration (not a method).

## FIGURE_4
- D3 excluded from pooled Wilcoxon (n=25; K=10 only, below split-arm validity at alpha=0.10).

## FIGURE_5
- Hardware: NVIDIA RTX 4090.
- Source LGBM (~4 s/fold) attributed only to methods that consume it (not TargetOnlyCP).
- Target CP calib timed per method; AdaptiveJackknifeCP O(K) LOO refits dominate calib cost.

## FIGURE_8
- Dropped: v3 pipeline does not persist per-method interval arrays needed for EN 206 compliance heatmap.

## FIGURE_K / FIGURE_COV
- Open markers: no finite-sample guarantee (K below class minimum).
- Horizontal grey line (FIGURE_K only): test-set strength output range (max-min) per fold.
- FIGURE_COV: fold-faceted line plot (D1–D4), log K-axis; ±SEM across seeds; SplitCP annotated off-scale.
"""


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


FIGURE_MAP = {
    "FIGURE_1_shift_heatmap": "figure_1_shift_heatmap",
    "FIGURE_2_reliability_diagrams": "figure_2_reliability_diagrams",
    "FIGURE_3_coverage_width_pareto": "figure_3_coverage_width_pareto",
    "FIGURE_4_conditional_coverage": "figure_4_conditional_coverage",
    "FIGURE_5_timing_comparison": "figure_5_timing_comparison",
    "FIGURE_6_example_intervals": "figure_6_example_intervals",
    "FIGURE_7_ablation_breakdown": "figure_7_ablation_breakdown",
    "FIGURE_K_width_vs_K": "figure_k_width_vs_k",
    "FIGURE_COV_coverage_vs_K": "figure_coverage_vs_k",
    "FIGURE_SELECT_candidate_histogram": "figure_selection_histogram",
    "FLOWCHART_methodology": "draw_flowchart_methodology",
    "FLOWCHART_lodo_pipeline": "draw_flowchart_lodo",
}


def _load_auxiliary(output_dir: Path) -> dict:
    shift_path = output_dir / "shift_analysis" / "domain_classifier_auc.csv"
    timing_path = output_dir / "lodo_evaluation" / "timing_results.csv"
    comp_path = output_dir / "lodo_evaluation" / "TABLE_7_compliance_decisions.csv"
    sep_path = output_dir / "diagnostics" / "separability_summary.csv"

    comp_matrix = None
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        num_cols = comp_df.select_dtypes(include="number").columns.tolist()
        if "method" in comp_df.columns and "fold" in comp_df.columns and num_cols:
            val_col = num_cols[0]
            comp_matrix = comp_df.pivot_table(
                index="fold", columns="method", values=val_col, aggfunc="mean",
            )

    return {
        "shift_path": str(shift_path) if shift_path.exists() else None,
        "timing_path": str(timing_path) if timing_path.exists() else None,
        "comp_matrix": comp_matrix.to_dict() if comp_matrix is not None else None,
        "sep_path": str(sep_path) if sep_path.exists() else None,
        "diag_dir": str(output_dir / "diagnostics"),
        "fig_dir": str(output_dir / "figures"),
        "flow_dir": str(output_dir / "flowcharts"),
        "eval_dir": str(output_dir / "lodo_evaluation"),
    }


def _render_one(task: dict) -> str:
    """Worker: render a single figure (must be top-level for ProcessPoolExecutor)."""
    import pandas as pd
    from pathlib import Path

    root = Path(task["root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src import visualization as viz

    figure_id = task["figure_id"]
    df_results = pd.read_csv(task["results_path"])
    aux = task["aux"]
    fig_dir = Path(aux["fig_dir"])
    flow_dir = Path(aux["flow_dir"])
    diag_dir = Path(aux["diag_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    df_shift = pd.read_csv(aux["shift_path"]) if aux["shift_path"] else None
    df_timing = pd.read_csv(aux["timing_path"]) if aux["timing_path"] else None
    df_sep = pd.read_csv(aux["sep_path"]) if aux["sep_path"] else None
    comp_matrix = pd.DataFrame(aux["comp_matrix"]) if aux["comp_matrix"] else None

    if figure_id == "FIGURE_1_shift_heatmap" and df_shift is not None and len(df_shift):
        viz.figure_1_shift_heatmap(df_shift, fig_dir / "FIGURE_1_shift_heatmap", df_sep)
    elif figure_id == "FIGURE_2_reliability_diagrams":
        viz.figure_2_reliability_diagrams(df_results, fig_dir / "FIGURE_2_reliability_diagrams")
    elif figure_id == "FIGURE_3_coverage_width_pareto":
        viz.figure_3_coverage_width_pareto(df_results, fig_dir / "FIGURE_3_coverage_width_pareto")
    elif figure_id == "FIGURE_4_conditional_coverage":
        viz.figure_4_conditional_coverage(df_results, fig_dir / "FIGURE_4_conditional_coverage")
    elif figure_id == "FIGURE_5_timing_comparison" and df_timing is not None and len(df_timing):
        viz.figure_5_timing_comparison(df_timing, fig_dir / "FIGURE_5_timing_comparison")
    elif figure_id == "FIGURE_6_example_intervals":
        viz.figure_6_example_intervals(df_results, fig_dir / "FIGURE_6_example_intervals", diag_dir)
    elif figure_id == "FIGURE_7_ablation_breakdown":
        viz.figure_7_ablation_breakdown(df_results, fig_dir / "FIGURE_7_ablation_breakdown")
    elif figure_id == "FIGURE_K_width_vs_K" and "K" in df_results.columns:
        # Combined, plus a two-panel split for the printed page (Reviewer 2, c.9).
        viz.figure_k_width_vs_k(df_results, fig_dir / "FIGURE_K_width_vs_K")
        viz.emit_split_by_fold(
            viz.figure_k_width_vs_k, df_results, fig_dir / "FIGURE_K_width_vs_K", 2)
    elif figure_id == "FIGURE_COV_coverage_vs_K" and "K" in df_results.columns:
        viz.figure_coverage_vs_k(df_results, fig_dir / "FIGURE_COV_coverage_vs_K")
        viz.emit_split_by_fold(
            viz.figure_coverage_vs_k, df_results, fig_dir / "FIGURE_COV_coverage_vs_K", 2)
    elif figure_id == "FIGURE_SELECT_candidate_histogram":
        viz.figure_selection_histogram(diag_dir, fig_dir / "FIGURE_SELECT_candidate_histogram")
    elif figure_id == "FLOWCHART_methodology":
        viz.draw_flowchart_methodology(flow_dir / "FLOWCHART_methodology")
    elif figure_id == "FLOWCHART_lodo_pipeline":
        viz.draw_flowchart_lodo(flow_dir / "FLOWCHART_lodo_pipeline")
    else:
        return f"{figure_id}: skipped (missing inputs)"

    return f"{figure_id}: ok"


def _figures_to_render(df_results: pd.DataFrame, aux: dict, only: list[str] | None) -> list[str]:
    candidates = list(FIGURE_MAP)
    if only:
        candidates = [f for f in only if f in FIGURE_MAP]

    skip: set[str] = set()
    if aux["shift_path"] is None:
        skip.add("FIGURE_1_shift_heatmap")
    if aux["timing_path"] is None:
        skip.add("FIGURE_5_timing_comparison")
    skip.add("FIGURE_8_compliance_heatmap")  # dropped: no multi-method compliance intervals in v3
    if "K" not in df_results.columns:
        skip.update({"FIGURE_K_width_vs_K", "FIGURE_COV_coverage_vs_K"})

    return [f for f in candidates if f not in skip]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all figures from CSV")
    parser.add_argument("--results", default="results/FINAL_RESULTS_ALLSEEDS.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 4),
        help="Parallel worker count (default: min(8, cpu_count))",
    )
    parser.add_argument(
        "--figure", action="append", dest="figures",
        help="Render only named figure(s); repeatable",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Force sequential generation (legacy path)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    df_results = annotate_guaranteed(pd.read_csv(results_path))
    df_results.to_csv(results_path, index=False)
    aux = _load_auxiliary(output_dir)
    figure_list = _figures_to_render(df_results, aux, args.figures)

    caption_path = output_dir / "figures" / "CAPTION_NOTES.md"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(CAPTION_NOTES, encoding="utf-8")

    if args.sequential or args.jobs <= 1:
        shift_path = output_dir / "shift_analysis" / "domain_classifier_auc.csv"
        df_shift = pd.read_csv(shift_path) if shift_path.exists() else None
        timing_path = output_dir / "lodo_evaluation" / "timing_results.csv"
        df_timing = pd.read_csv(timing_path) if timing_path.exists() else None
        comp_matrix = pd.DataFrame(aux["comp_matrix"]) if aux["comp_matrix"] else None
        generate_all_figures(df_results, df_shift, df_timing, comp_matrix, output_dir)
        results = [f"{f}: ok" for f in figure_list]
    else:
        tasks = [
            {
                "figure_id": fig_id,
                "results_path": str(results_path.resolve()),
                "root": str(ROOT),
                "aux": aux,
            }
            for fig_id in figure_list
        ]
        results: list[str] = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_render_one, t): t["figure_id"] for t in tasks}
            for fut in as_completed(futures):
                fig_id = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append(f"{fig_id}: FAILED, {exc}")
                    print(f"ERROR {fig_id}: {exc}", file=sys.stderr)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_csv": str(results_path),
        "input_csv_hash": _file_hash(results_path),
        "parallel_jobs": 1 if args.sequential or args.jobs <= 1 else args.jobs,
        "figures": {
            name: {
                "function": func,
                "script": "scripts/make_figures.py",
            }
            for name, func in FIGURE_MAP.items()
        },
        "dropped_figures": {
            "FIGURE_8_compliance_heatmap": "v3 pipeline lacks multi-method stored intervals; see CAPTION_NOTES.md",
        },
        "render_log": sorted(results),
    }
    manifest_path = output_dir / "figures" / "FIGURE_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Rendered {len(results)} figure tasks with jobs={args.jobs}")
    for line in sorted(results):
        print(f"  {line}")
    print(f"Figures written to {output_dir / 'figures'}")
    print(f"Manifest: {manifest_path}")
    failed = [r for r in results if "FAILED" in r]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

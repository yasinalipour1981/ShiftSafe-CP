#!/usr/bin/env python
"""Extension E reporting: TABLE_E, Wilcoxon+Holm, figures, validity table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import holm_corrected_wilcoxon_tests
from src.pubstyle import METHOD_COLORS, save_fig
from src.validity import annotate_guaranteed

import matplotlib.pyplot as plt

N_EVAL = {"D1": 919, "D2": 83, "D4": 163}
PRIMARY_K = 20
PRIMARY_ALPHA = 0.10
E_BASELINES = (
    "CQR-target", "GPR-target", "GPR-transfer",
    "JKplus-source", "TabPFN-target", "WeightedCP-v2",
)


def filter_stale_extension_jk(df: pd.DataFrame) -> pd.DataFrame:
    """Drop JKplus-source only when Extension E rows look pre-fix (smoke artifact).

    Pre-fix smoke CSVs had ~41% D4 coverage from fake Jackknife+ intervals.
    Post-fix full grid (753/753) shows D4 mean ~0.70–0.75, keep those rows.
    """
    if df.empty or "method" not in df.columns:
        return df
    jk = df[df["method"] == "JKplus-source"]
    if jk.empty:
        return df
    d4 = jk[jk["fold"] == "D4"]
    if not d4.empty and float(d4["empirical_coverage"].mean()) >= 0.55:
        return df
    return df[df["method"] != "JKplus-source"].copy()


def _binomial_ci(coverage: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = coverage
    se = np.sqrt(p * (1 - p) / max(n, 1))
    return float(p - z * se), float(p + z * se)


def load_merged_primary(
    confirmatory_csv: Path,
    extension_csv: Path,
) -> pd.DataFrame:
    df_c = pd.read_csv(confirmatory_csv)
    df_e = pd.read_csv(extension_csv) if extension_csv.exists() else pd.DataFrame()
    df_c = df_c[(df_c["K"] == PRIMARY_K) & (df_c["alpha"] == PRIMARY_ALPHA)]
    df_c = df_c[df_c["fold"].isin(["D1", "D2", "D4"])]
    if not df_e.empty:
        df_e = df_e[(df_e["K"] == PRIMARY_K) & (df_e["alpha"] == PRIMARY_ALPHA)]
        df_e = filter_stale_extension_jk(df_e)
        df = pd.concat([df_c, df_e], ignore_index=True)
    else:
        df = df_c
    return annotate_guaranteed(df)


def build_table_e(df: pd.DataFrame, out_dir: Path) -> Path:
    """TABLE_E: all methods × coverage[CI], width, Winkler, guarantee_scope, time."""
    sub = df[(df["K"] == PRIMARY_K) & (df["alpha"] == PRIMARY_ALPHA)]
    rows = []
    for method in sorted(sub["method"].unique()):
        msub = sub[sub["method"] == method]
        cov = float(msub["empirical_coverage"].mean())
        # pooled n for CI: sum of per-fold eval counts weighted
        n_eff = sum(N_EVAL.get(f, 100) for f in msub["fold"].unique()) * max(
            1, msub["seed"].nunique() // len(msub["fold"].unique())
        )
        lo, hi = _binomial_ci(cov, n_eff)
        rows.append({
            "method": method,
            "coverage": round(cov, 4),
            "coverage_ci_lo": round(lo, 4),
            "coverage_ci_hi": round(hi, 4),
            "width_mpa": round(float(msub["mean_interval_width"].mean()), 2),
            "winkler": round(float(msub["winkler_score"].mean()), 2),
            "guarantee_scope": msub["guarantee_scope"].iloc[0]
            if "guarantee_scope" in msub.columns else "target",
            "guaranteed": bool(msub["guaranteed"].iloc[0])
            if "guaranteed" in msub.columns else True,
            "infer_time_sec": round(float(msub["inference_time_sec"].mean()), 4),
        })
    table = pd.DataFrame(rows).sort_values("winkler")
    path = out_dir / "TABLE_E.csv"
    table.to_csv(path, index=False)
    with open(out_dir / "TABLE_E.tex", "w") as f:
        f.write(table.to_latex(index=False, float_format="%.3f"))
    return path


def build_coverage_validity_table(df: pd.DataFrame, out_dir: Path) -> Path:
    """Coverage validity with binomial 95% CI per fold."""
    sub = df[(df["K"] == PRIMARY_K) & (df["alpha"] == PRIMARY_ALPHA)]
    rows = []
    for fold in ["D1", "D2", "D4"]:
        n = N_EVAL[fold]
        lo_nom, hi_nom = _binomial_ci(1 - PRIMARY_ALPHA, n)
        for method in sorted(sub["method"].unique()):
            msub = sub[(sub["method"] == method) & (sub["fold"] == fold)]
            if msub.empty:
                continue
            cov = float(msub["empirical_coverage"].mean())
            lo, hi = _binomial_ci(cov, n)
            rows.append({
                "fold": fold,
                "n_eval": n,
                "method": method,
                "coverage": round(cov, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "nominal_lo": round(lo_nom, 4),
                "nominal_hi": round(hi_nom, 4),
                "valid": lo >= lo_nom or cov >= 1 - PRIMARY_ALPHA - 0.05,
            })
    out = pd.DataFrame(rows)
    path = out_dir / "coverage_validity_table.csv"
    out.to_csv(path, index=False)
    return path


def build_wilcoxon_e(df: pd.DataFrame, out_dir: Path) -> Path:
    """Wilcoxon AdaptiveJackknifeCP vs each E-baseline + Holm."""
    sub = df[(df["K"] == PRIMARY_K) & (df["alpha"] == PRIMARY_ALPHA)]
    sub = sub[sub["fold"] != "D3"]
    pairs = [("AdaptiveJackknifeCP", b) for b in E_BASELINES if b in sub["method"].values]
    # Include original confirmatory pairs
    pairs.extend([
        ("AdaptiveJackknifeCP", "TargetOnlyCP"),
        ("AdaptiveJackknifeCP", "TransferCal-CP-affine-localsigma"),
    ])
    pairs = [(a, b) for a, b in pairs if a in sub["method"].values and b in sub["method"].values]
    pairs = list(dict.fromkeys(pairs))

    winkler = holm_corrected_wilcoxon_tests(sub, pairs, metric="winkler_score")
    width = holm_corrected_wilcoxon_tests(sub, pairs, metric="mean_interval_width")

    payload = {
        "primary_endpoint": "winkler_score",
        "K": PRIMARY_K,
        "alpha": PRIMARY_ALPHA,
        "excludes_D3": True,
        "winkler_tests": winkler,
        "width_tests": width,
    }
    path = out_dir / "wilcoxon_extension_e.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    rows = []
    for t in winkler:
        rows.append({
            "comparison": f"{t['method_a']} vs {t['method_b']}",
            "metric": "winkler",
            "p": t["p_value"],
            "holm_p": t.get("holm_p_value"),
            "effect_r": t.get("effect_size_rank_biserial_r"),
            "wins_a": t.get("wins_a"),
            "n_pairs": t.get("n_pairs"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "wilcoxon_extension_e_table.csv", index=False)
    return path


def make_smoke_figure(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[(df["alpha"] == PRIMARY_ALPHA) & (df["K"] == PRIMARY_K)]
    if sub.empty:
        return
    agg = sub.groupby("method").agg(
        cov=("empirical_coverage", "mean"),
        width=("mean_interval_width", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (method, row) in enumerate(agg.iterrows()):
        color = METHOD_COLORS.get(method, f"C{i}")
        marker = "o" if row.get("guaranteed", True) else "o"
        face = color if method not in ("GPR-target", "GPR-transfer") else "none"
        ax.scatter(row["width"], row["cov"], c=[color], s=80, marker=marker,
                   facecolors=face, edgecolors=color, linewidths=1.5, label=method)
    ax.axhline(0.90, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean width (MPa)")
    ax.set_ylabel("Coverage")
    ax.set_title("Extension E smoke (K=20, α=0.10)")
    ax.legend(fontsize=7, loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def regenerate_figure_3_5(
    df: pd.DataFrame,
    timing_csv: Path | None,
    figures_dir: Path,
) -> None:
    from src.visualization import figure_3_coverage_width_pareto, figure_5_timing_comparison

    sub = df[(df["K"] == PRIMARY_K) & (df["alpha"] == PRIMARY_ALPHA)]
    figure_3_coverage_width_pareto(sub, figures_dir / "FIGURE_3_coverage_width_pareto")
    if timing_csv and timing_csv.exists():
        df_t = pd.read_csv(timing_csv)
        if "K" in df_t.columns:
            k20 = df_t[df_t["K"] == 20]
            if len(k20):
                df_t = k20
        def _timing_extra_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
            t_extra = results_df.groupby("method").agg(
                fit_time_sec=("fit_time_sec", "mean"),
                calib_time_sec=("calib_time_sec", "mean"),
                infer_time_per_1000_sec=("inference_time_sec", "mean"),
            ).reset_index()
            if "source_fit_time_sec" in results_df.columns:
                src = results_df.groupby("method")["source_fit_time_sec"].mean().reset_index()
                t_extra = t_extra.merge(src, on="method", how="left")
            else:
                from src.pubstyle import method_uses_source_lgbm
                t_extra["source_fit_time_sec"] = t_extra.apply(
                    lambda r: float(r["fit_time_sec"]) if method_uses_source_lgbm(r["method"]) else 0.0,
                    axis=1,
                )
            return t_extra

        if "calib_time_sec" in df.columns and "calib_time_sec" not in df_t.columns:
            t_extra = _timing_extra_from_results(df)
            df_t = pd.concat([df_t, t_extra], ignore_index=True).drop_duplicates(
                subset=["method"], keep="last",
            )
        elif "calib_time_sec" in df.columns:
            t_extra = _timing_extra_from_results(df)
            df_t = pd.concat([df_t, t_extra], ignore_index=True).drop_duplicates(
                subset=["method"], keep="last",
            )
        figure_5_timing_comparison(df_t, figures_dir / "FIGURE_5_timing_comparison")


def update_figure_manifest(figures_dir: Path) -> None:
    manifest_path = figures_dir / "FIGURE_MANIFEST.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    manifest["git_commit"] = commit
    manifest["extension_e_note"] = (
        "FIGURE_3/5 regenerated with Extension E baselines; "
        "hollow markers = no finite-sample guarantee (GPR-target, GPR-transfer)."
    )
    manifest["provenance"] = manifest.get("provenance", {})
    manifest["provenance"]["aborted_confirmatory_runs"] = (
        "Seeds 142-169 partial run (ShiftSafeCP_20260711_181153.log) produced no CSV; "
        "confirmatory restarted 18:52 post-alpha-fix. Final 50/50 in ShiftSafeCP_20260711_185249.log."
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmatory", default="results/FINAL_RESULTS_confirmatory_142-191.csv")
    parser.add_argument("--extension", default="results/extension_e/EXTENSION_E_results.csv")
    parser.add_argument("--out", default="results/extension_e")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged_primary(Path(args.confirmatory), Path(args.extension))
    build_table_e(df, out_dir)
    build_coverage_validity_table(df, out_dir)
    build_wilcoxon_e(df, out_dir)

    figures_dir = Path("results/figures")
    timing = Path("results/lodo_evaluation/timing_results.csv")
    regenerate_figure_3_5(df, timing, figures_dir)
    update_figure_manifest(figures_dir)

    print(f"TABLE_E: {out_dir / 'TABLE_E.csv'}")
    print(f"Wilcoxon: {out_dir / 'wilcoxon_extension_e.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

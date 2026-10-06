"""Auto-generate LaTeX tables for paper."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _save_table(df: pd.DataFrame, output_dir: Path, name: str) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{name}.tex"
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    logger.info("Saved %s", tex_path)
    return df


def table_1_dataset_summary(
    datasets: dict[str, pd.DataFrame], output_dir: Path | str
) -> pd.DataFrame:
    rows = []
    for domain, df in datasets.items():
        n = len(df)
        pct_fa = (df["fly_ash"] > 0).sum() / max(n, 1) * 100
        rows.append({
            "Domain": domain,
            "n": n,
            "%FA": f"{pct_fa:.1f}",
            "w/cm": f"{df['w_cm'].mean():.2f}±{df['w_cm'].std():.2f}",
            "f/cm": f"{df['f_cm'].mean():.2f}±{df['f_cm'].std():.2f}",
            "Strength (MPa)": f"{df['strength_mpa'].mean():.1f}±{df['strength_mpa'].std():.1f}",
            "Ages (days)": ", ".join(map(str, sorted(df["age"].dropna().astype(int).unique()))),
        })
    return _save_table(pd.DataFrame(rows), Path(output_dir), "TABLE_1_dataset_summary")


def table_2_shift_quantification(
    df_shift: pd.DataFrame, output_dir: Path | str
) -> pd.DataFrame:
    return _save_table(df_shift.round(4), Path(output_dir), "TABLE_2_shift_quantification")


def table_3_lodo_results(
    df_lodo: pd.DataFrame, output_dir: Path | str
) -> pd.DataFrame:
    formatted = pd.DataFrame()
    for fold in sorted(df_lodo["fold"].unique()):
        for alpha in sorted(df_lodo["alpha"].unique()):
            sub = df_lodo[(df_lodo["fold"] == fold) & (df_lodo["alpha"] == alpha)]
            cov = sub.groupby("method")["empirical_coverage"].mean()
            width = sub.groupby("method")["mean_interval_width"].mean()
            formatted[f"{fold}_a{alpha:.2f}_cov"] = cov
            formatted[f"{fold}_a{alpha:.2f}_width"] = width
    return _save_table(formatted.round(3), Path(output_dir), "TABLE_3_lodo_coverage_width")


def table_4_ablation(
    df_lodo: pd.DataFrame, output_dir: Path | str, alpha: float = 0.10
) -> pd.DataFrame:
    ablation = [m for m in df_lodo["method"].unique() if "ShiftSafe" in m]
    df_a = df_lodo[(df_lodo["method"].isin(ablation)) & (df_lodo["alpha"] == alpha)]
    agg = df_a.groupby(["method", "fold"]).agg({
        "empirical_coverage": "mean",
        "mean_interval_width": "mean",
    }).reset_index()
    return _save_table(agg.round(3), Path(output_dir), "TABLE_4_ablation_shiftsafe")


def table_3v3_transfer(
    df_lodo: pd.DataFrame,
    output_dir: Path | str,
    K: int = 20,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """TABLE_3v3: v3 guaranteed-coverage methods at fixed K and alpha."""
    df = df_lodo[(df_lodo["alpha"] == alpha)]
    if "K" in df.columns:
        df = df[df["K"] == K]
    methods = [
        m for m in df["method"].unique()
        if m in (
            "SplitCP", "Jackknife+", "ShiftSafe-CP", "TargetOnlyCP",
            "TransferCal-CP", "TransferCal-CP-affine", "TransferCal-CP-boost",
        )
    ]
    rows = []
    for fold in sorted(df["fold"].unique()):
        for method in methods:
            sub = df[(df["fold"] == fold) & (df["method"] == method)]
            if len(sub) == 0:
                continue
            rows.append({
                "fold": fold,
                "method": method,
                "K": K,
                "coverage": sub["empirical_coverage"].mean(),
                "width": sub["mean_interval_width"].mean(),
            })
    return _save_table(pd.DataFrame(rows).round(3), Path(output_dir), "TABLE_3v3_transfer")


def table_5_point_accuracy(
    df_accuracy: pd.DataFrame, output_dir: Path | str
) -> pd.DataFrame:
    agg = df_accuracy.groupby("model").agg({
        "rmse": "mean", "mae": "mean", "r_squared": "mean"
    }).reset_index()
    agg.columns = ["Model", "RMSE", "MAE", "R2"]
    return _save_table(agg.round(3), Path(output_dir), "TABLE_5_point_accuracy")


def table_6_timing_memory(
    df_timing: pd.DataFrame, output_dir: Path | str
) -> pd.DataFrame:
    agg = df_timing.groupby("method").agg({
        "fit_time_sec": "mean",
        "calib_time_sec": "mean",
        "infer_time_per_1000_sec": "mean",
    }).reset_index()
    agg.columns = ["Method", "Fit (sec)", "Calibration (sec)", "Inference/1000 (sec)"]
    agg["Peak GPU Memory (MB)"] = "N/A"
    return _save_table(agg.round(3), Path(output_dir), "TABLE_6_timing_memory")


def generate_all_tables(
    datasets: dict[str, pd.DataFrame],
    df_lodo: pd.DataFrame,
    df_shift: pd.DataFrame | None,
    df_timing: pd.DataFrame | None,
    df_accuracy: pd.DataFrame | None,
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    table_1_dataset_summary(datasets, output_dir)
    if df_shift is not None:
        table_2_shift_quantification(df_shift, output_dir)
    table_3_lodo_results(df_lodo, output_dir)
    table_4_ablation(df_lodo, output_dir)
    if df_accuracy is not None and len(df_accuracy):
        table_5_point_accuracy(df_accuracy, output_dir)
    if df_timing is not None and len(df_timing):
        table_6_timing_memory(df_timing, output_dir)

"""Publication-quality figures for ShiftSafe-CP paper (journal standard)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import NullFormatter, ScalarFormatter
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.pubstyle import (
    FOLD_MARKERS,
    METHOD_COLORS,
    METHOD_ORDER,
    apply_pub_style,
    method_style,
    method_uses_source_lgbm,
    panel_label,
    save_fig,
    fs,
    TYPE_SCALE,
    display_name,
)
from src.validity import annotate_guaranteed, fold_output_range_mpa, is_guaranteed

logger = logging.getLogger(__name__)
apply_pub_style()

CANDIDATE_ORDER = [
    "local-ridge", "affine-transfer", "stacked", "local-gbm", "boost-transfer",
]

FIGURE_7_ARMS = [
    "TargetOnlyCP",
    "TransferCal-CP-affine-localsigma",
    "AdaptiveSplitCP",
    "AdaptiveJackknifeCP",
]


def _ordered_methods(methods) -> list[str]:
    known = [m for m in METHOD_ORDER if m in methods]
    rest = [m for m in sorted(methods) if m not in known]
    return known + rest


def _binomial_band(n: int, p: float, z: float = 1.96) -> tuple[float, float]:
    se = np.sqrt(p * (1 - p) / max(n, 1))
    return p - z * se, p + z * se


def _series_sem(values: pd.Series) -> float:
    n = int(values.count())
    if n <= 1:
        return 0.0
    std = float(values.std())
    if np.isnan(std):
        return 0.0
    return std / np.sqrt(n)


def _asymmetric_sem_yerr(
    means: np.ndarray, sems: np.ndarray,
    floor: float = 0.0, ceiling: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bar distances for mean ± SEM with bounded whiskers."""
    sems = np.nan_to_num(sems, nan=0.0)
    lower = np.maximum(floor, means - sems)
    if ceiling is not None:
        upper = np.minimum(ceiling, means + sems)
        return means - lower, upper - means
    return means - lower, sems


def _filter_primary(df: pd.DataFrame, alpha: float = 0.10, K: int = 20) -> pd.DataFrame:
    out = df[df["alpha"] == alpha].copy()
    if "K" in out.columns:
        out = out[out["K"] == K]
    return out


def _filter_k(df: pd.DataFrame, K: int = 20) -> pd.DataFrame:
    """Keep all alpha levels at fixed K (for reliability / alpha-sweep figures)."""
    out = df.copy()
    if "K" in out.columns:
        out = out[out["K"] == K]
    return out


def figure_1_shift_heatmap(
    df_shift: pd.DataFrame,
    output_path: Path,
    df_separability: pd.DataFrame | None = None,
) -> None:
    n_panels = 4 if df_separability is not None and len(df_separability) else 3
    # Four panels wrap to 2x2 rather than 1x4: a row of four is ~16 in wide and
    # loses three quarters of its type size when scaled to the printed column
    # (Reviewer 2, comment 9). Panels are addressed flat, so the shape is free.
    nrows, ncols = _panel_grid(n_panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=fs(4.0 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    cbar_labels = {
        "wasserstein_1": "W₁",
        "mmd_rbf": "MMD",
        "domain_classifier_auc": "AUC",
    }
    metrics = ["wasserstein_1", "mmd_rbf", "domain_classifier_auc"]
    domains = sorted(set(df_shift["domain_i"]) | set(df_shift["domain_j"]))
    flip_cells = {("D1", "D1"): (1.00, 0.70), ("D4", "D4"): (1.00, 0.66)}

    for ax, metric in zip(axes, metrics):
        matrix = np.zeros((len(domains), len(domains)))
        for _, row in df_shift.iterrows():
            i, j = domains.index(row["domain_i"]), domains.index(row["domain_j"])
            matrix[i, j] = row[metric]
            matrix[j, i] = row[metric]
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="RdYlGn_r",
            xticklabels=domains, yticklabels=domains, ax=ax,
            cbar_kws={"label": cbar_labels.get(metric, metric)},
            vmin=0, vmax=1 if "auc" in metric else None,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        for (di, dj), (shown, actual) in flip_cells.items():
            if di in domains and dj in domains:
                ii, jj = domains.index(di), domains.index(dj)
                ax.add_patch(plt.Rectangle(
                    (jj, ii), 1, 1, fill=False, edgecolor="black", linewidth=1.5,
                ))

    if df_separability is not None and len(df_separability) and n_panels == 4:
        ax = axes[3]
        sep = df_separability.set_index("fold")
        cols = [c for c in sep.columns if c.startswith("AUC")]
        sns.heatmap(sep[cols], annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, vmin=0.5, vmax=1.0)
        ax.set_xlabel("Feature set")
        ax.set_ylabel("Fold")
        panel_label(ax, "d")

    for i, ax in enumerate(axes[:3]):
        panel_label(ax, chr(ord("a") + i))

    plt.tight_layout()
    save_fig(fig, output_path)


def _ensure_guaranteed(df: pd.DataFrame) -> pd.DataFrame:
    if "guaranteed" in df.columns:
        return df
    return annotate_guaranteed(df)


def _reliability_marker(method: str) -> str:
    """Distinct markers so close split-arm curves remain readable at K=20."""
    if method == "AdaptiveSplitCP":
        return "s"
    if method == "TargetOnlyCP":
        return "o"
    if method == "TransferCal-CP-affine-localsigma":
        return "^"
    if method == "AdaptiveJackknifeCP":
        return "D"
    return "o"


def figure_2_reliability_diagrams(df_results: pd.DataFrame, output_path: Path) -> None:
    df = _ensure_guaranteed(_filter_k(df_results, K=20))
    domains = sorted(df["fold"].unique())
    n_panels = len(domains) + 1  # one per domain, plus a legend panel
    nrows, ncols = _panel_grid(n_panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=fs(4.0 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    methods = [m for m in _ordered_methods(df["method"].unique()) if m != "SplitCP"]
    n_eval = max(int(df.groupby("fold").size().max() / max(len(methods), 1) / 3), 20)

    xs = np.linspace(0.65, 1.0, 50)
    band_lo = np.array([_binomial_band(n_eval, p)[0] for p in xs])
    band_hi = np.array([_binomial_band(n_eval, p)[1] for p in xs])

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        df_d = df[df["fold"] == domain]
        for mi, method in enumerate(methods):
            df_m = df_d[df_d["method"] == method].groupby("alpha").agg(
                nom=("nominal_coverage", "mean"),
                emp=("empirical_coverage", "mean"),
            ).reset_index().sort_values("nom")
            if df_m.empty or len(df_m) < 2:
                continue
            st = method_style(method, mi)
            plot_kw = {k: v for k, v in st.items() if k != "label"}
            ax.plot(
                df_m["nom"], df_m["emp"],
                marker=_reliability_marker(method), markersize=4,
                linewidth=1.2, **plot_kw,
            )
        ax.fill_between(xs, band_lo, band_hi, alpha=0.12, color="gray", linewidth=0)
        ax.plot(xs, xs, "k--", alpha=0.4, linewidth=0.8, label="_nolegend_")
        ax.set_xlim(0.65, 1.0)
        ax.set_ylim(0.65, 1.0)
        ax.set_title(domain, fontsize=9)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        panel_label(ax, chr(ord("a") + idx))

    legend_ax = axes_flat[len(domains)]
    legend_ax.axis("off")
    legend_handles: list[Line2D] = [
        Line2D([0], [0], color="k", linestyle="--", alpha=0.4, linewidth=1.2, label="ideal"),
    ]
    for mi, method in enumerate(methods):
        st = method_style(method, mi)
        legend_handles.append(
            Line2D(
                [0], [0],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=_reliability_marker(method),
                markersize=5,
                linewidth=1.2,
                label=display_name(method),
            )
        )
    legend_ax.legend(
        handles=legend_handles,
        fontsize=7,
        loc="center",
        frameon=False,
        title="Methods (K=20, α sweep)",
    )
    legend_ax.text(
        0.5, 0.08,
        "SplitCP omitted (~5% cov, off-scale).\n"
        "D3 excluded (K=10 only; no K=20 eval).\n"
        "TargetOnlyCP vs AdaptiveSplitCP: distinct at K=20\n"
        "(coincide only at K≤10; local-ridge collapse).",
        transform=legend_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5 * TYPE_SCALE,
        color="#444444",
    )

    for j in range(len(domains) + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    save_fig(fig, output_path)


def figure_3_coverage_width_pareto(df_results: pd.DataFrame, output_path: Path) -> None:
    """Coverage–width Pareto at K=20, α=0.10 (primary panel).

    One point per method×fold (mean±SD across seeds); color=method, marker=fold.
    Log-scale x jitter separates overlapping fold markers within each method.
    """
    df = _filter_primary(df_results)
    if "fold" in df.columns:
        df = df[df["fold"].isin(["D1", "D2", "D4"])]
    if "guaranteed" not in df.columns:
        df = annotate_guaranteed(df)

    fig, ax = plt.subplots(figsize=fs(7.5, 5.5))
    methods = [m for m in _ordered_methods(df["method"].unique()) if m != "SplitCP"]
    fold_order = [f for f in ["D1", "D2", "D4"] if f in df["fold"].unique()]
    fold_jitter = {
        f: 10 ** (0.035 * (i - (len(fold_order) - 1) / 2))
        for i, f in enumerate(fold_order)
    }

    method_handles: list[Line2D] = []
    fold_handles: list[Line2D] = []
    hollow_proxy: Line2D | None = None

    for mi, method in enumerate(methods):
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        st = method_style(method, mi)
        color = st["color"]
        guaranteed = bool(sub["guaranteed"].iloc[0]) if "guaranteed" in sub.columns else True
        face = color if guaranteed else "none"

        for fold in fold_order:
            row = sub[sub["fold"] == fold]
            if row.empty:
                continue
            w_mean = float(row["mean_interval_width"].mean())
            c_mean = float(row["empirical_coverage"].mean())
            w_sd = float(row["mean_interval_width"].std()) if len(row) > 1 else 0.0
            c_sd = float(row["empirical_coverage"].std()) if len(row) > 1 else 0.0
            if np.isnan(w_sd):
                w_sd = 0.0
            if np.isnan(c_sd):
                c_sd = 0.0
            ax.errorbar(
                w_mean * fold_jitter.get(fold, 1.0), c_mean,
                xerr=w_sd, yerr=c_sd,
                fmt=FOLD_MARKERS.get(fold, "o"), capsize=2, markersize=5,
                color=color, linestyle="none",
                markerfacecolor=face, markeredgewidth=1.2,
            )
            if not guaranteed and hollow_proxy is None:
                hollow_proxy = Line2D(
                    [0], [0], marker="o", linestyle="none",
                    markerfacecolor="none", markeredgecolor="gray",
                    markeredgewidth=1.2, markersize=5,
                )

        method_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none", color=color,
                markerfacecolor=face, markeredgewidth=1.2, markersize=5,
                label=display_name(method),
            )
        )

    for fold in fold_order:
        fold_handles.append(
            Line2D(
                [0], [0], marker=FOLD_MARKERS.get(fold, "o"), linestyle="none",
                color="0.35", markerfacecolor="0.35", markersize=5, label=fold,
            )
        )

    ax.axhline(0.90, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(
        0.98, 0.905, "nominal 0.90", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="red",
    )
    split_sub = df[df["method"] == "SplitCP"]
    if not split_sub.empty:
        sc = float(split_sub["empirical_coverage"].mean())
        ax.annotate(
            f"SplitCP\n(~{sc:.0%} cov)",
            xy=(0.02, 0.05), xycoords=("axes fraction", "axes fraction"),
            fontsize=7, color="#666", ha="left",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean interval width (MPa)")
    ax.set_ylabel("Empirical coverage")
    ax.text(
        0.98, 0.98, "K=20, α=0.10", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.5, color="#444",
    )

    ncols = 1 if len(method_handles) <= 5 else 2
    leg_methods = ax.legend(
        handles=method_handles, fontsize=5.5, title="Method",
        loc="lower right", bbox_to_anchor=(0.98, 0.02),
        ncol=ncols, frameon=True, framealpha=0.92, borderaxespad=0.4,
    )
    ax.add_artist(leg_methods)
    fold_leg_handles = list(fold_handles)
    fold_leg_labels = [h.get_label() for h in fold_handles]
    if hollow_proxy is not None:
        fold_leg_handles.append(hollow_proxy)
        fold_leg_labels.append("open: no finite-sample guarantee")
    ax.legend(
        handles=fold_leg_handles, labels=fold_leg_labels,
        fontsize=6, title="Fold", loc="upper left", frameon=True, framealpha=0.92,
    )

    plt.tight_layout()
    save_fig(fig, output_path)


def figure_4_conditional_coverage(
    df_results: pd.DataFrame, output_path: Path, alpha: float = 0.10,
) -> None:
    df_a = _filter_primary(df_results, alpha=alpha)
    methods = _ordered_methods(df_a["method"].unique())
    fig, ax = plt.subplots(figsize=fs(9, 4))
    x = np.arange(len(methods))
    folds = sorted(df_a["fold"].unique())
    for i, fold in enumerate(folds):
        covs = [
            df_a[(df_a["method"] == m) & (df_a["fold"] == fold)]["empirical_coverage"].mean()
            for m in methods
        ]
        ax.bar(x + i * 0.12, covs, width=0.12, label=fold)
    lo, hi = _binomial_band(50, 1 - alpha)
    ax.axhline(1 - alpha, color="red", linestyle="--", alpha=0.5, linewidth=0.8,
               label=f"nominal {1-alpha:.2f}")
    ax.axhspan(lo, hi, alpha=0.10, color="gray", label="95% binomial band")
    ax.set_xticks(x + 0.12 * (len(folds) - 1) / 2)
    ax.set_xticklabels([display_name(m) for m in methods],
                       rotation=35, ha="right", fontsize=7 * TYPE_SCALE)
    ax.set_ylabel("Empirical coverage")
    for m_idx, m in enumerate(methods):
        if m == "SplitCP":
            sc = df_a[df_a["method"] == m]["empirical_coverage"].mean()
            if sc < 0.15:
                ax.annotate(f"{sc:.0%}", (m_idx, sc + 0.02), fontsize=6, ha="center", color="#666")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    save_fig(fig, output_path)


def figure_5_timing_comparison(df_timing: pd.DataFrame, output_path: Path) -> None:
    df = df_timing.copy()
    if "K" in df.columns:
        k20 = df[df["K"] == 20]
        if len(k20):
            df = k20
    if "source_fit_time_sec" not in df.columns:
        df["source_fit_time_sec"] = df.apply(
            lambda r: float(r["fit_time_sec"]) if method_uses_source_lgbm(r["method"]) else 0.0,
            axis=1,
        )
    agg = df.groupby("method").agg(
        source=("source_fit_time_sec", "mean"),
        calib=("calib_time_sec", "mean"),
        infer=("infer_time_per_1000_sec", "mean"),
    )
    methods = _ordered_methods(agg.index)
    fig, ax = plt.subplots(figsize=fs(10, 4.5))
    x = np.arange(len(methods))
    w = 0.25
    ax.bar(x - w, agg.loc[methods, "source"], w, label="Source LGBM (if used)", color="#BBBBBB")
    ax.bar(x, agg.loc[methods, "calib"], w, label="Target CP calib", color="#4477AA")
    ax.bar(x + w, agg.loc[methods, "infer"], w, label="Infer/1000 pts", color="#EE6677")
    ax.set_yscale("log")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in methods],
                       rotation=35, ha="right", fontsize=7 * TYPE_SCALE)
    ax.text(
        0.02, 0.98,
        "Source LGBM (~4 s/fold) attributed only to methods that use it.\n"
        "TargetOnlyCP: ridge on K target labels only (no source LGBM).",
        transform=ax.transAxes, fontsize=6, va="top", color="#444",
    )
    ax.text(
        0.98, 0.02, "NVIDIA RTX 4090", transform=ax.transAxes,
        fontsize=6, va="bottom", ha="right", style="italic", color="#888",
    )
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    save_fig(fig, output_path)


def figure_6_example_intervals(
    df_results: pd.DataFrame | None,
    output_path: Path,
    diagnostics_dir: Path | None = None,
) -> None:
    """ILLUSTRATIVE ONLY: plots generated points with each method's mean width.

    Not for publication. The manuscript's Figure 7 is produced from real
    evaluation mixes by scripts/make_example_intervals_figure.py.
    """
    if df_results is None or df_results.empty:
        return
    df = _filter_primary(df_results)
    d4 = df[df["fold"] == "D4"]
    if d4.empty:
        d4 = df[df["fold"] == df["fold"].iloc[-1]]

    fig, axes = plt.subplots(1, 2, figsize=fs(9, 4))
    specs = [("SplitCP", "SplitCP"), ("AdaptiveJackknifeCP", "AdaptiveJackknifeCP")]
    x = np.arange(10)
    rng = np.random.default_rng(42)
    y_true = 40.0 + rng.normal(0, 8, 10)
    ylims = (20, 60)

    for ax, (method, label) in zip(axes, specs):
        sub = d4[d4["method"] == method]
        cov = float(sub["empirical_coverage"].mean()) if not sub.empty else 0.5
        hw = float(sub["mean_interval_width"].mean()) / 2 if not sub.empty else 5.0
        y_pred = 40.0 + rng.normal(0, 3, 10)
        ax.errorbar(
            x, y_pred, yerr=hw, fmt="o", capsize=3,
            color=METHOD_COLORS.get(method, "gray"), alpha=0.85,
        )
        ax.scatter(x, y_true, marker="x", color="red", s=45, zorder=5, label="True")
        misses = int(round((1 - cov) * 10))
        miss_idx = list(range(min(misses, 10)))
        for j_idx, j in enumerate(miss_idx):
            ax.annotate(
                "miss", (x[j], y_true[j]), fontsize=6, color="red",
                xytext=(0, 10 + j_idx * 3), textcoords="offset points",
            )
        ax.annotate(
            "", xy=(x[5], y_pred[5] + hw), xytext=(x[5], y_pred[5] - hw),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8),
        )
        ax.text(x[5] + 0.3, y_pred[5], f"{2*hw:.1f} MPa", fontsize=6, va="center")
        ax.set_ylim(ylims)
        ax.set_xlabel("Test index")
        ax.set_ylabel("Strength (MPa)")
        ax.legend(fontsize=7, loc="upper right", title=f"cov={cov:.0%}")

    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    plt.tight_layout()
    save_fig(fig, output_path)


def figure_7_ablation_breakdown(df_results: pd.DataFrame, output_path: Path) -> None:
    """Ablation arms at K=20, α=0.10: coverage (±SD) and Winkler (±SEM, lower floored at 0)."""
    df_a = _filter_primary(df_results)
    arms = [m for m in FIGURE_7_ARMS if m in df_a["method"].unique()]
    if not arms:
        return
    sub = df_a[df_a["method"].isin(arms)]
    agg = sub.groupby("method").agg(
        cov_mean=("empirical_coverage", "mean"),
        cov_std=("empirical_coverage", "std"),
        wink_mean=("winkler_score", "mean"),
    )
    agg["wink_sem"] = sub.groupby("method")["winkler_score"].apply(_series_sem)
    agg = agg.reindex(arms)
    nominal = 0.90
    lo, hi = _binomial_band(80, nominal)

    fig, (ax_cov, ax_wink) = plt.subplots(1, 2, figsize=fs(9, 4))
    x = np.arange(len(arms))
    colors = [METHOD_COLORS.get(m, "gray") for m in arms]

    ax_cov.bar(x, agg["cov_mean"], yerr=agg["cov_std"].fillna(0), capsize=3, color=colors)
    ax_cov.axhline(nominal, color="red", linestyle="--", alpha=0.6, label="nominal 0.90")
    ax_cov.axhspan(lo, hi, alpha=0.10, color="gray", label="95% binomial band")
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels([display_name(a) for a in arms],
                           rotation=30, ha="right", fontsize=7 * TYPE_SCALE)
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.legend(fontsize=6, loc="lower right")
    panel_label(ax_cov, "a")

    wink_means = agg["wink_mean"].to_numpy(dtype=float)
    wink_sems = agg["wink_sem"].to_numpy(dtype=float)
    wink_yerr_lo, wink_yerr_hi = _asymmetric_sem_yerr(wink_means, wink_sems, floor=0.0)
    ax_wink.bar(
        x, wink_means, yerr=[wink_yerr_lo, wink_yerr_hi], capsize=3, color=colors,
    )
    ax_wink.set_ylim(bottom=0)
    ax_wink.set_xticks(x)
    ax_wink.set_xticklabels([display_name(a) for a in arms],
                            rotation=30, ha="right", fontsize=7 * TYPE_SCALE)
    ax_wink.set_ylabel("Winkler score")
    ax_wink.text(
        0.98, 0.02, "±SEM across seed×fold runs", transform=ax_wink.transAxes,
        ha="right", va="bottom", fontsize=6.5, color="#444",
    )
    panel_label(ax_wink, "b")

    plt.tight_layout()
    save_fig(fig, output_path)


def figure_8_compliance_heatmap(comp_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=fs(6, 5))
    sns.heatmap(comp_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    save_fig(fig, output_path)


def figure_k_width_vs_k(
    df_results: pd.DataFrame, output_path: Path, alpha: float = 0.10,
) -> None:
    if "K" not in df_results.columns:
        return
    df = _ensure_guaranteed(df_results[df_results["alpha"] == alpha])
    folds = sorted(df["fold"].unique())

    # Three panels in a row is ~11 in wide and loses two thirds of its type size
    # when placed at the 6.1 in text width (Reviewer 2, comment 9). Wrap to 2 x 2
    # and give the spare slot to the legend.
    nrows, ncols = _panel_grid(len(folds))
    fig, axes = plt.subplots(nrows, ncols, figsize=fs(3.8 * ncols, 3.5 * nrows), squeeze=False)
    split_w = df[df["method"] == "SplitCP"].groupby("K")["mean_interval_width"].mean()
    split_cov = df[df["method"] == "SplitCP"]["empirical_coverage"].mean()

    axes_flat = axes.flatten()
    hollow_proxy = None
    method_seen: dict[str, Line2D] = {}
    for ax, fold in zip(axes_flat, folds):
        sub = df[df["fold"] == fold]
        guar_sub = sub[sub["guaranteed"]] if "guaranteed" in sub.columns else sub
        y_cap = float(np.percentile(guar_sub["mean_interval_width"].dropna(), 99)) if len(guar_sub) else 100.0
        y_cap = max(y_cap, 20.0)
        out_rng = fold_output_range_mpa(df, fold)

        is_bar = len(sub["K"].unique()) <= 1
        n_annot = 0
        if is_bar:
            msub = sub.groupby("method")["mean_interval_width"].mean().sort_values()
            ax.bar(
                range(len(msub)), msub.values,
                color=[METHOD_COLORS.get(m, "gray") for m in msub.index],
            )
            ax.set_xticks(range(len(msub)))
            ax.set_xticklabels([display_name(m) for m in msub.index],
                               rotation=35, ha="right", fontsize=6 * TYPE_SCALE)
            ax.set_ylim(0, y_cap * 1.05)
        else:
            for mi, method in enumerate(_ordered_methods(sub["method"].unique())):
                if method == "SplitCP":
                    continue
                msub = sub[sub["method"] == method]
                st = method_style(method, mi)
                color = st["color"]
                if method not in method_seen:
                    method_seen[method] = Line2D(
                        [], [], color=color, marker="o", markersize=5,
                        linewidth=1.4, label=display_name(method),
                    )
                ks = sorted(msub["K"].unique())
                ys = [msub[msub["K"] == k]["mean_interval_width"].mean() for k in ks]
                gs = [
                    bool(msub[msub["K"] == k]["guaranteed"].all())
                    if "guaranteed" in msub.columns
                    else is_guaranteed(method, int(k), alpha)
                    for k in ks
                ]
                for k, y, g in zip(ks, ys, gs):
                    ax.errorbar(
                        k, min(y, y_cap * 1.02), fmt="o", markersize=5, color=color,
                        markerfacecolor=color if g else "none",
                        markeredgecolor=color, markeredgewidth=1.2 if g else 1.5,
                    )
                    if y > y_cap * 0.85:
                        # Stagger, or two out-of-range callouts overprint.
                        ax.annotate(
                            f"{y:.0f} MPa",
                            (k, y_cap * (0.92 - 0.07 * (n_annot % 2))),
                            fontsize=5.5 * TYPE_SCALE, ha="center", color=color,
                        )
                        n_annot += 1
                    if not g and hollow_proxy is None:
                        hollow_proxy = Line2D(
                            [], [], color="none", marker="o", markersize=6,
                            markerfacecolor="none", markeredgecolor="#444444",
                            markeredgewidth=1.5, linestyle="none",
                        )
                g_idx = [i for i, g in enumerate(gs) if g]
                ng_idx = [i for i, g in enumerate(gs) if not g]
                if len(g_idx) >= 2:
                    ax.plot([ks[i] for i in g_idx], [min(ys[i], y_cap) for i in g_idx],
                            ls="-", color=color, alpha=0.6, linewidth=1)
                if len(ng_idx) >= 2:
                    ax.plot([ks[i] for i in ng_idx], [min(ys[i], y_cap) for i in ng_idx],
                            ls=":", color=color, alpha=0.6, linewidth=1)
            if len(split_w):
                ax.axhline(
                    split_w.mean(), color="#999999", linestyle=":",
                    alpha=0.8, linewidth=1.0,
                )
                ax.text(sub["K"].max(), split_w.mean(), " SplitCP (invalid, ~5% cov)",
                        fontsize=6 * TYPE_SCALE, va="center", color="#666")
            if out_rng is not None:
                ax.axhline(out_rng, color="#AAAAAA", linestyle="-", linewidth=0.6, alpha=0.8)
                ax.text(sub["K"].min(), out_rng, " output range", fontsize=6 * TYPE_SCALE,
                        va="bottom", color="#888")
            ax.set_ylim(0, y_cap * 1.05)
        ax.set_title(fold, fontsize=9 * TYPE_SCALE)
        ax.set_xlabel("Method" if is_bar else "K")
        ax.set_ylabel("Width (MPa)")
        panel_label(ax, chr(ord("a") + list(folds).index(fold)))

    handles = list(method_seen.values())
    labels = [h.get_label() for h in handles]
    if hollow_proxy is not None:
        handles.append(hollow_proxy)
        labels.append("open marker: no finite-sample guarantee")
    spare = [axes_flat[j] for j in range(len(folds), len(axes_flat))]
    for ax_spare in spare:
        ax_spare.set_visible(False)
    if handles:
        if spare:  # park the key in the empty cell of the grid
            lax = spare[0]
            lax.set_visible(True)
            lax.axis("off")
            lax.legend(handles, labels, loc="center", frameon=False,
                       fontsize=8.0 * TYPE_SCALE, title="Methods")
            plt.tight_layout()
        else:      # full grid: put the key underneath
            plt.tight_layout(rect=[0, 0.11, 1, 1])
            fig.legend(
                handles, labels, loc="lower center",
                ncol=min(3, len(handles)), frameon=False,
                fontsize=8.0 * TYPE_SCALE, bbox_to_anchor=(0.5, 0.0),
            )
    else:
        plt.tight_layout()
    save_fig(fig, output_path)


def figure_coverage_vs_k(
    df_results: pd.DataFrame, output_path: Path, alpha: float = 0.10,
) -> None:
    """Coverage vs K: fold-faceted line plot (matches FIGURE_K layout).

    One panel per LODO fold; K on x-axis, methods as colored lines with ±SEM.
    Solid segments = finite-sample guarantee; dotted = no guarantee.
    SplitCP omitted from lines (~5% cov under shift).
    """
    if "K" not in df_results.columns:
        return
    nominal = 1 - alpha
    df = _ensure_guaranteed(df_results[df_results["alpha"] == alpha])
    folds = sorted(df["fold"].unique())
    # Approximate held-out test sizes for binomial reference band per fold.
    fold_n_test = {"D1": 80, "D2": 120, "D3": 25, "D4": 80}

    nrows, ncols = _panel_grid(len(folds))
    fig, axes = plt.subplots(nrows, ncols, figsize=fs(3.8 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for j in range(len(folds), len(axes_flat)):  # hide the unused slot
        axes_flat[j].set_visible(False)
    method_handles: list[Line2D] = []
    hollow_proxy: Line2D | None = None
    hatch_proxy: Patch | None = None
    ref_added = False

    for ax, fold in zip(axes_flat, folds):
        sub = df[df["fold"] == fold]
        n_eval = fold_n_test.get(fold, 80)
        lo, hi = _binomial_band(n_eval, nominal)
        ax.axhline(nominal, color="red", linestyle="--", alpha=0.5, linewidth=0.9)
        ax.axhspan(lo, hi, alpha=0.10, color="gray")
        if not ref_added:
            ax.axhline(nominal, color="red", linestyle="--", alpha=0.5, label="nominal 0.90")
            ax.axhspan(lo, hi, alpha=0.10, color="gray", label="binomial 95% band")
            ref_added = True

        split_cov = float(sub[sub["method"] == "SplitCP"]["empirical_coverage"].mean()) if (
            sub["method"] == "SplitCP"
        ).any() else None

        if len(sub["K"].unique()) <= 1:
            k_val = int(sorted(sub["K"].unique())[0])
            methods_bar = _ordered_methods(sub["method"].unique())
            for i, m in enumerate(methods_bar):
                msub_m = sub[sub["method"] == m]
                mean_b = float(msub_m["empirical_coverage"].mean())
                sem_b = _series_sem(msub_m["empirical_coverage"])
                guaranteed = (
                    bool(msub_m["guaranteed"].all())
                    if "guaranteed" in msub_m.columns
                    else is_guaranteed(m, k_val, alpha)
                )
                color = METHOD_COLORS.get(m, "gray")
                if guaranteed:
                    ax.bar(i, mean_b, yerr=sem_b, capsize=2, color=color, edgecolor=color)
                else:
                    ax.bar(
                        i, mean_b, yerr=sem_b, capsize=2,
                        facecolor="white", edgecolor=color, hatch="///", linewidth=1.0,
                    )
                    if hatch_proxy is None:
                        hatch_proxy = Patch(
                            facecolor="white", edgecolor="gray", hatch="///",
                            label="hatched: no finite-sample guarantee",
                        )
            ax.set_xticks(range(len(methods_bar)))
            ax.set_xticklabels([display_name(m) for m in methods_bar],
                               rotation=35, ha="right", fontsize=6 * TYPE_SCALE)
            ax.set_xlabel(f"Method (K={k_val})")
        else:
            for mi, method in enumerate(_ordered_methods(sub["method"].unique())):
                if method == "SplitCP":
                    continue
                msub = sub[sub["method"] == method]
                st = method_style(method, mi)
                color = st["color"]
                ks = sorted(msub["K"].unique())
                means = np.array(
                    [msub[msub["K"] == k]["empirical_coverage"].mean() for k in ks],
                    dtype=float,
                )
                sems = np.array(
                    [_series_sem(msub[msub["K"] == k]["empirical_coverage"]) for k in ks],
                    dtype=float,
                )
                gs = [
                    bool(msub[msub["K"] == k]["guaranteed"].all())
                    if "guaranteed" in msub.columns
                    else is_guaranteed(method, int(k), alpha)
                    for k in ks
                ]
                yerr_lo, yerr_hi = _asymmetric_sem_yerr(means, sems, floor=0.0, ceiling=1.0)
                for k, m, ylo, yhi, g in zip(ks, means, yerr_lo, yerr_hi, gs):
                    ax.errorbar(
                        k, m, yerr=[[ylo], [yhi]], fmt="o", markersize=5, capsize=2,
                        color=color,
                        markerfacecolor=color if g else "none",
                        markeredgecolor=color, markeredgewidth=1.2 if g else 1.5,
                        linestyle="none",
                    )
                    if not g and hollow_proxy is None:
                        hollow_proxy = Line2D(
                            [0], [0], marker="o", linestyle="none",
                            markerfacecolor="none", markeredgecolor="gray",
                            markeredgewidth=1.2, markersize=5,
                        )
                g_idx = [i for i, g in enumerate(gs) if g]
                ng_idx = [i for i, g in enumerate(gs) if not g]
                if len(g_idx) >= 2:
                    ax.plot(
                        [ks[i] for i in g_idx], [means[i] for i in g_idx],
                        ls="-", color=color, alpha=0.65, linewidth=1.2,
                    )
                if len(ng_idx) >= 2:
                    ax.plot(
                        [ks[i] for i in ng_idx], [means[i] for i in ng_idx],
                        ls=":", color=color, alpha=0.65, linewidth=1.2,
                    )
                if not any(h.get_label() == display_name(method) for h in method_handles):
                    method_handles.append(
                        Line2D(
                            [0], [0], marker="o", linestyle="-", color=color,
                            markerfacecolor=color, markeredgewidth=1.2, markersize=5,
                            label=display_name(method),
                        )
                    )

        if split_cov is not None:
            ax.text(
                0.02, 0.04, f"SplitCP ~{split_cov:.0%} cov (off-scale)",
                transform=ax.transAxes, fontsize=6 * TYPE_SCALE,
                va="bottom", color="#666",
            )
        ax.set_ylim(0.75, 1.0)
        if len(sub["K"].unique()) > 1:
            ax.set_xscale("log")
            ax.set_xticks(sorted(sub["K"].unique()))
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            # Suppress the log minor labels: they print "4 x 10^1" on top of
            # the 30 and 50 budget ticks.
            ax.get_xaxis().set_minor_formatter(NullFormatter())
            ax.set_xlabel("K")
        else:
            ax.set_xlabel(f"Method (K={int(sub['K'].iloc[0])})")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(fold, fontsize=9 * TYPE_SCALE)
        panel_label(ax, chr(ord("a") + list(folds).index(fold)))

    if len(axes_flat):
        ref_handles, ref_labels = axes_flat[0].get_legend_handles_labels()
        handles = method_handles + ref_handles
        labels = [h.get_label() for h in method_handles] + ref_labels
        if hollow_proxy is not None:
            handles.append(hollow_proxy)
            labels.append("open: no finite-sample guarantee")
        if hatch_proxy is not None:
            handles.append(hatch_proxy)
            labels.append(hatch_proxy.get_label())
        ncols = 3 if len(method_handles) > 4 else 2
        fig.legend(
            handles, labels, fontsize=6, ncol=ncols,
            loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False,
        )
    fig.text(
        0.99, 0.01, "±SEM across seeds; log K-axis",
        ha="right", va="bottom", fontsize=6, color="#666",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, output_path)


def figure_selection_histogram(diagnostics_dir: Path, output_path: Path) -> None:
    diag_dir = Path(diagnostics_dir)
    counts: dict[str, dict[str, int]] = {}
    total_loo = 0
    ks_seen: set[int] = set()
    for p in sorted(diag_dir.glob("fold_*_K*.json")):
        stem = p.stem  # fold_D1_seed142_K20
        k_part = stem.rsplit("_K", 1)[-1]
        try:
            k_val = int(k_part)
        except ValueError:
            continue
        if k_val < 20:
            continue
        ks_seen.add(k_val)
        d = json.loads(p.read_text())
        sel = d.get("adaptive_selection")
        if not sel:
            continue
        fold = d.get("fold", "?")
        if fold not in counts:
            counts[fold] = {k: 0 for k in CANDIDATE_ORDER}
        for k, v in sel.items():
            counts[fold][k] = counts[fold].get(k, 0) + v
            total_loo += v
    if not counts:
        logger.info("Skipping F_SELECT: no adaptive_selection diagnostics")
        return
    folds = sorted(counts)
    cands = [c for c in CANDIDATE_ORDER if any(counts[f].get(c, 0) for f in folds)]
    gbm_total = sum(counts[f].get("local-gbm", 0) for f in folds)
    fig, ax = plt.subplots(figsize=fs(7, 4))
    bottom = np.zeros(len(folds))
    for cand in cands:
        vals = np.array([counts[f].get(cand, 0) for f in folds], dtype=float)
        totals = np.array([sum(counts[f].values()) for f in folds], dtype=float)
        pct = np.where(totals > 0, 100 * vals / totals, 0)
        ax.bar(folds, pct, bottom=bottom, label=cand)
        bottom += pct
    n_per_fold = int(total_loo / max(len(folds), 1))
    k_label = ", ".join(str(k) for k in sorted(ks_seen))
    ax.text(
        0.02, 0.98, f"n≈{n_per_fold} LOO fits/fold (K={k_label})",
        transform=ax.transAxes, fontsize=7, va="top",
    )
    if gbm_total == 0:
        ax.text(
            0.02, 0.88,
            "local-gbm: gated at K≤15; available but never selected at K≥20.",
            transform=ax.transAxes, fontsize=6, va="top", color="#666",
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel("Selection proportion (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, title="Candidate")
    plt.tight_layout()
    save_fig(fig, output_path)


def _flow_box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str = "#2f2f2f",
    fontsize: float = 8.5,
    subtext: str | None = None,
    text_color: str = "#1a1a1a",
) -> tuple[float, float, float, float]:
    """Draw a rounded flowchart box; return (x, y, w, h) for connector routing."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(box)
    if subtext:
        ax.text(
            x + w / 2, y + h * 0.62, text,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=text_color, zorder=3,
        )
        ax.text(
            x + w / 2, y + h * 0.30, subtext,
            ha="center", va="center", fontsize=max(fontsize - 1.5, 6.5),
            color=text_color, zorder=3,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=text_color, zorder=3,
        )
    return x, y, w, h


def _flow_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#444444",
    style: str = "-|>",
    lw: float = 1.4,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    ))


def draw_flowchart_methodology(output_path: Path) -> None:
    """Colorized ShiftSafe-CP / AdaptiveJackknifeCP pipeline (journal-width layout)."""
    C = {
        "input": "#0072B2",
        "source": "#332288",
        "loo": "#882255",
        "select": "#E69F00",
        "scale": "#009E73",
        "quant": "#56B4E9",
        "out": "#D55E00",
        "cands": {
            "local-ridge": "#0072B2",
            "affine-transfer": "#009E73",
            "stacked": "#E69F00",
            "local-gbm": "#CC79A7",
            "boost-transfer": "#882255",
        },
    }

    # Sized for elsarticle figure* (~17 cm wide)
    fig_w, fig_h = 7.2, 3.35
    fig, ax = plt.subplots(figsize=fs(fig_w, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.35)
    ax.axis("off")
    ax.set_facecolor("#FFFFFF")

    def box(x, y, w, h, title, color, subtitle=None, fs=7.5, sub_fs=6.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.0, edgecolor="#333333", facecolor=color, zorder=2,
        ))
        if subtitle:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white", zorder=3)
            ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                    fontsize=sub_fs, color="white", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white", zorder=3)
        return x, y, w, h

    def arrow(p0, p1, color="#444444", rad=0.0, lw=1.2):
        style = "arc3,rad=0.0" if rad == 0 else f"arc3,rad={rad}"
        _flow_arrow(ax, p0, p1, color=color, lw=lw, connectionstyle=style)

    # --- Main row (compact, fits x in [0.2, 9.6]) ---
    y, h, w = 2.35, 0.82, 1.35
    xs = [0.20, 1.75, 3.30, 4.85, 6.40, 7.95]
    labels = [
        ("Target pool", r"$K$ labeled mixes"),
        ("Jackknife+", r"LOO on $K$ points"),
        ("Candidate\nselection", "min LOO MAE"),
        ("Local scale", r"$\hat\sigma(x)\geq 1$ MPa"),
        ("Conformal\nquantile", r"$q$ at $1-\alpha$"),
        ("Prediction\nintervals", r"$\hat\mu\pm q\hat\sigma$"),
    ]
    colors = [C["input"], C["loo"], C["select"], C["scale"], C["quant"], C["out"]]
    rects = [box(xs[i], y, w, h, labels[i][0], colors[i], labels[i][1]) for i in range(6)]

    for i in range(5):
        x0, _, w0, h0 = rects[i]
        x1, _, _, h1 = rects[i + 1]
        arrow((x0 + w0, y + h0 / 2), (x1, y + h1 / 2))

    # Source model → selection
    sx, sy, sw, sh = box(3.30, 3.25, 1.55, 0.52, "Source LightGBM", C["source"],
                         r"$\hat\mu_{\mathrm{src}}$ (other labs)", fs=7, sub_fs=6)
    arrow((sx + sw / 2, sy), (xs[2] + w / 2, y + h), color=C["source"], rad=-0.2)

    # --- Candidate strip (single bus arrow, no fan-in clutter) ---
    panel_y, panel_h = 0.18, 1.55
    ax.add_patch(FancyBboxPatch(
        (0.55, panel_y), 5.55, panel_h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        linewidth=0.9, edgecolor="#AAAAAA", facecolor="#F7F7F7",
        linestyle=(0, (4, 3)), zorder=0,
    ))
    ax.text(0.75, panel_y + panel_h - 0.18, "LOO candidate pool",
            fontsize=7, fontweight="bold", color="#444444", zorder=3)

    cand_specs = [
        ("local-ridge", "Ridge"),
        ("affine-transfer", r"$a\hat\mu_{\mathrm{src}}+b$"),
        ("stacked", "Stack"),
        ("local-gbm", "Local GBM"),
        ("boost-transfer", "Boost"),
    ]
    cw, ch = 0.95, 0.62
    cy = panel_y + 0.35
    for i, (key, short) in enumerate(cand_specs):
        cx = 0.75 + i * 1.05
        ax.add_patch(FancyBboxPatch(
            (cx, cy), cw, ch,
            boxstyle="round,pad=0.008,rounding_size=0.04",
            linewidth=0.8, edgecolor="#333", facecolor=C["cands"][key], zorder=2,
        ))
        ax.text(cx + cw / 2, cy + ch / 2, short, ha="center", va="center",
                fontsize=6.2, fontweight="bold", color="white", zorder=3)

    # One bus arrow from pool to selection
    bus_x = xs[2] + w / 2
    ax.plot([bus_x, bus_x], [panel_y + panel_h, y], color="#666666", lw=1.2, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (bus_x, y + 0.02), (bus_x, y + h * 0.15),
        arrowstyle="-|>", mutation_scale=10, linewidth=0, color="#666666", zorder=1,
    ))

    # Full-K refit note inside output box area (no floating overlap)
    ax.text(xs[5] + w / 2, y - 0.22, "Full-$K$ refit of LOO winner",
            ha="center", va="top", fontsize=6.2, color="#555555", zorder=3)

    # Legend: top-right, clear of candidate panel
    legend_items = [
        Patch(facecolor=C["input"], edgecolor="#333", label="Target"),
        Patch(facecolor=C["source"], edgecolor="#333", label="Source"),
        Patch(facecolor=C["loo"], edgecolor="#333", label="LOO"),
        Patch(facecolor=C["select"], edgecolor="#333", label="Selection"),
        Patch(facecolor=C["scale"], edgecolor="#333", label="Scale"),
        Patch(facecolor=C["quant"], edgecolor="#333", label="Quantile"),
        Patch(facecolor=C["out"], edgecolor="#333", label="Intervals"),
    ]
    ax.legend(
        handles=legend_items, loc="upper right", bbox_to_anchor=(0.995, 0.98),
        ncol=4, fontsize=5.8, framealpha=0.92, edgecolor="#CCCCCC",
        handlelength=1.0, columnspacing=0.8, handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fig(fig, output_path, formats=("pdf", "png", "tiff"))


def draw_flowchart_lodo(output_path: Path) -> None:
    """Colorized LODO experimental pipeline (journal-width, three-phase layout)."""
    C = {
        "data": "#0072B2",
        "harmonize": "#56B4E9",
        "shift": "#332288",
        "lodo": "#E69F00",
        "source": "#882255",
        "kdraw": "#009E73",
        "cp": "#D55E00",
        "eval": "#CC79A7",
        "stats": "#AA4499",
        "export": "#999999",
    }

    fig_w, fig_h = 7.2, 3.55
    fig, ax = plt.subplots(figsize=fs(fig_w, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.55)
    ax.axis("off")
    ax.set_facecolor("#FFFFFF")

    def box(x, y, w, h, title, color, subtitle=None, fs=7.2, sub_fs=6.2):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.0, edgecolor="#333333", facecolor=color, zorder=2,
        ))
        if subtitle:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white", zorder=3)
            ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                    fontsize=sub_fs, color="white", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white", zorder=3)
        return x, y, w, h

    def arrow(p0, p1, color="#444444", rad=0.0, lw=1.2):
        style = "arc3,rad=0.0" if rad == 0 else f"arc3,rad={rad}"
        _flow_arrow(ax, p0, p1, color=color, lw=lw, connectionstyle=style)

    def phase_label(x, y, text):
        ax.text(x, y, text, ha="left", va="center", fontsize=7,
                fontweight="bold", color="#555555", rotation=90, zorder=3)

    # Phase bands
    for y0, h0, face in [
        (2.45, 0.92, "#EEF4FB"),
        (1.25, 0.92, "#FFF6E8"),
        (0.18, 0.88, "#F4F0F5"),
    ]:
        ax.add_patch(FancyBboxPatch(
            (0.55, y0), 9.35, h0,
            boxstyle="round,pad=0.008,rounding_size=0.04",
            linewidth=0.6, edgecolor="#DDDDDD", facecolor=face, zorder=0,
        ))

    phase_label(0.22, 2.91, "Data")
    phase_label(0.22, 1.71, "LODO")
    phase_label(0.22, 0.62, "Output")

    # --- Row 1: Data preparation ---
    y1, h, w = 2.55, 0.72, 1.55
    r1 = [
        box(0.75, y1, w, h, "Load datasets", C["data"], subtitle="D1--D4 open labs"),
        box(2.55, y1, w, h, "Harmonize", C["harmonize"], subtitle="common feature space"),
        box(4.35, y1, w, h, "Shift analysis", C["shift"], subtitle=r"W$_1$, MMD, AUC"),
    ]
    for i in range(2):
        x0, _, w0, h0 = r1[i]
        x1, _, _, h1 = r1[i + 1]
        arrow((x0 + w0, y1 + h0 / 2), (x1, y1 + h1 / 2))

    # --- Row 2: LODO experiment ---
    y2 = 1.35
    r2 = [
        box(0.75, y2, w, h, "LODO split", C["lodo"], subtitle=r"hold out $D_t$"),
        box(2.55, y2, w, h, "Source LightGBM", C["source"], subtitle="fit on source domains"),
        box(4.35, y2, w, h, r"$K$-draw", C["kdraw"], subtitle="stratified target labels"),
    ]
    for i in range(2):
        x0, _, w0, h0 = r2[i]
        x1, _, _, h1 = r2[i + 1]
        arrow((x0 + w0, y2 + h0 / 2), (x1, y2 + h1 / 2))

    # Conformal methods panel (row 2, right)
    ax.add_patch(FancyBboxPatch(
        (6.15, y2 - 0.08), 3.55, 0.88,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=0.9, edgecolor="#AAAAAA", facecolor="#FFFFFF",
        linestyle=(0, (4, 3)), zorder=1,
    ))
    ax.text(6.30, y2 + 0.62, "Conformal arms (identical $K$-indices)",
            fontsize=6.5, fontweight="bold", color="#444444", zorder=3)
    methods = [
        ("AJK", C["cp"]),
        ("TargetOnly", C["data"]),
        ("TransferCal", C["kdraw"]),
        ("SplitCP", C["export"]),
        ("+ Ext. E", C["shift"]),
    ]
    mw, mh = 0.62, 0.38
    mx0 = 6.30
    for i, (lbl, col) in enumerate(methods):
        mx = mx0 + i * 0.68
        ax.add_patch(FancyBboxPatch(
            (mx, y2 + 0.12), mw, mh,
            boxstyle="round,pad=0.006,rounding_size=0.03",
            linewidth=0.7, edgecolor="#333", facecolor=col, zorder=2,
        ))
        ax.text(mx + mw / 2, y2 + 0.31, lbl, ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white", zorder=3)

    x2_last, _, w2_last, h2_last = r2[-1]
    arrow((x2_last + w2_last, y2 + h2_last / 2), (6.15, y2 + h2_last / 2))

    # Vertical connectors between phases
    arrow((1.52, y1), (1.52, y2 + h), color="#666666", rad=0.0)
    arrow((5.12, y1), (5.12, y2 + h), color="#666666", rad=0.0)

    # --- Row 3: Evaluation & export ---
    y3 = 0.28
    r3 = [
        box(0.75, y3, w, h, "Metrics", C["eval"], subtitle=r"Cov., width, Winkler"),
        box(2.55, y3, w, h, "Statistics", C["stats"], subtitle="Wilcoxon + Holm"),
        box(4.35, y3, w, h, "Export", C["export"], subtitle="CSV, figures, timing"),
    ]
    for i in range(2):
        x0, _, w0, h0 = r3[i]
        x1, _, _, h1 = r3[i + 1]
        arrow((x0 + w0, y3 + h0 / 2), (x1, y3 + h1 / 2))

    # From conformal panel down to metrics
    arrow((7.92, y2), (1.52, y3 + h), color=C["cp"], rad=0.18, lw=1.0)

    # Seeds annotation
    ax.text(6.20, 0.30, r"50 confirmatory seeds (142–191); primary cell $K{=}20$, $\alpha{=}0.10$",
            ha="left", va="center", fontsize=6.2, color="#555555", zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFAFA", edgecolor="#CCCCCC", linewidth=0.6))

    # Legend
    legend_items = [
        Patch(facecolor=C["data"], edgecolor="#333", label="Data"),
        Patch(facecolor=C["shift"], edgecolor="#333", label="Shift"),
        Patch(facecolor=C["lodo"], edgecolor="#333", label="LODO"),
        Patch(facecolor=C["source"], edgecolor="#333", label="Source fit"),
        Patch(facecolor=C["kdraw"], edgecolor="#333", label="$K$-draw"),
        Patch(facecolor=C["cp"], edgecolor="#333", label="Conformal"),
        Patch(facecolor=C["eval"], edgecolor="#333", label="Metrics"),
        Patch(facecolor=C["export"], edgecolor="#333", label="Export"),
    ]
    ax.legend(
        handles=legend_items, loc="upper right", bbox_to_anchor=(0.995, 0.995),
        ncol=4, fontsize=5.6, framealpha=0.92, edgecolor="#CCCCCC",
        handlelength=1.0, columnspacing=0.7, handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fig(fig, output_path, formats=("pdf", "png", "tiff"))


def _panel_grid(n: int) -> tuple[int, int]:
    """Rows and columns for `n` panels, capped at two columns.

    A row of three or four panels is 11-15 in wide and loses most of its type
    size at the 6.1 in text width of the journal template (Reviewer 2, c. 9).
    Two columns keep each panel wide enough to survive the reduction.
    """
    if n <= 2:
        return 1, max(n, 1)
    return (n + 1) // 2, 2


def emit_split_by_fold(fn, df_results, output_path, panels_per_figure: int = 2, **kw):
    """Emit a per-fold multi-panel figure as several smaller figures.

    Reviewer 2 (comment 9) notes that the four-domain panels of the K-sweep
    figures are unreadable at the journal's column width. Rather than change the
    plotting code, we call it once per group of `panels_per_figure` folds, so
    each panel gets roughly twice the width. The combined version is still
    emitted by the caller for readers who want it in one piece.
    """
    output_path = Path(output_path)
    folds = sorted(df_results["fold"].unique())
    if len(folds) <= panels_per_figure:
        fn(df_results, output_path, **kw)
        return [output_path]
    written = []
    for i in range(0, len(folds), panels_per_figure):
        chunk = folds[i : i + panels_per_figure]
        suffix = chr(ord("a") + i // panels_per_figure)
        part = output_path.with_name(f"{output_path.name}_{suffix}")
        fn(df_results[df_results["fold"].isin(chunk)], part, **kw)
        written.append(part)
    return written


def generate_all_figures(
    df_results: pd.DataFrame,
    df_shift: pd.DataFrame | None,
    df_timing: pd.DataFrame | None,
    comp_matrix: pd.DataFrame | None,
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    eval_dir = output_dir / "lodo_evaluation"
    fig_dir = output_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    flow_dir = output_dir / "flowcharts"
    flow_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"

    sep_path = diag_dir / "separability_summary.csv"
    df_sep = pd.read_csv(sep_path) if sep_path.exists() else None

    if df_shift is not None and len(df_shift):
        figure_1_shift_heatmap(df_shift, fig_dir / "FIGURE_1_shift_heatmap", df_sep)

    figure_2_reliability_diagrams(df_results, fig_dir / "FIGURE_2_reliability_diagrams")
    figure_3_coverage_width_pareto(df_results, fig_dir / "FIGURE_3_coverage_width_pareto")
    figure_4_conditional_coverage(df_results, fig_dir / "FIGURE_4_conditional_coverage")

    if df_timing is not None and len(df_timing):
        figure_5_timing_comparison(df_timing, fig_dir / "FIGURE_5_timing_comparison")

    figure_6_example_intervals(df_results, fig_dir / "FIGURE_6_example_intervals", diag_dir)
    figure_7_ablation_breakdown(df_results, fig_dir / "FIGURE_7_ablation_breakdown")

    if "K" in df_results.columns:
        # Combined versions, plus two-panel splits for the printed page (R2.9).
        figure_k_width_vs_k(df_results, fig_dir / "FIGURE_K_width_vs_K")
        figure_coverage_vs_k(df_results, fig_dir / "FIGURE_COV_coverage_vs_K")
        emit_split_by_fold(
            figure_k_width_vs_k, df_results, fig_dir / "FIGURE_K_width_vs_K", 2)
        emit_split_by_fold(
            figure_coverage_vs_k, df_results, fig_dir / "FIGURE_COV_coverage_vs_K", 2)

    figure_selection_histogram(diag_dir, fig_dir / "FIGURE_SELECT_candidate_histogram")

    # FIGURE_8 dropped: v3 pipeline does not store multi-method compliance intervals

    draw_flowchart_methodology(flow_dir / "FLOWCHART_methodology")
    draw_flowchart_lodo(flow_dir / "FLOWCHART_lodo_pipeline")

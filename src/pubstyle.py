"""Journal-grade matplotlib style for ShiftSafe-CP figures."""

from __future__ import annotations

import os as _os

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

METHOD_COLORS: dict[str, str] = {
    "SplitCP": "#999999",
    "TargetOnlyCP": "#0072B2",
    "TransferCal-CP": "#E69F00",
    "TransferCal-CP-affine": "#56B4E9",
    "TransferCal-CP-affine-localsigma": "#009E73",
    "TransferCal-CP-boost": "#CC79A7",
    "AdaptiveJackknifeCP": "#D55E00",
    "AdaptiveSplitCP": "#F0E442",
    "ShiftSafe-CP": "#000000",
    "Jackknife+": "#882255",
    "CQR": "#661100",
    "CQR-target": "#882255",
    "GPR-target": "#AA4499",
    "GPR-transfer": "#CC77AA",
    "JKplus-source": "#882255",
    "TabPFN-target": "#332288",
    "WeightedCP-v2": "#999999",
    "NGBoost": "#44AA99",
    "DeepEnsemble": "#117733",
}

METHOD_LINESTYLES: dict[str, str] = {
    "SplitCP": ":",
    "TargetOnlyCP": "-",
    "TransferCal-CP": "--",
    "TransferCal-CP-affine": "-.",
    "TransferCal-CP-affine-localsigma": "-",
    "TransferCal-CP-boost": "--",
    "AdaptiveSplitCP": "-.",
    "AdaptiveJackknifeCP": "-",
    "ShiftSafe-CP": "-",
    "Jackknife+": "-.",
    "CQR": ":",
}

METHOD_ORDER = [
    "SplitCP",
    "TargetOnlyCP",
    "TransferCal-CP",
    "TransferCal-CP-affine",
    "TransferCal-CP-affine-localsigma",
    "TransferCal-CP-boost",
    "AdaptiveSplitCP",
    "AdaptiveJackknifeCP",
    "CQR-target",
    "GPR-target",
    "GPR-transfer",
    "JKplus-source",
    "TabPFN-target",
    "WeightedCP-v2",
]

FOLD_MARKERS = {"D1": "o", "D2": "s", "D3": "^", "D4": "D"}

# Target-anchored arms that never consume the shared source LightGBM fit.
TARGET_ONLY_NO_SOURCE_LGBM = frozenset({
    "TargetOnlyCP",
    "CQR-target",
    "TabPFN-target",
    "GPR-target",
})


#: Names as they appear in figures. The internal identifiers are kept in the
#: result files and the code; the manuscript renamed the proposed method
#: (Reviewer 1, comment 6), and the figures must agree with the text.
METHOD_DISPLAY_NAMES: dict[str, str] = {
    "AdaptiveJackknifeCP": "TargetAnchoredJK+",
    "StackedOnlyCP": "StackedOnlyCP (ablation)",
    "TransferCal-CP-affine-localsigma": "TransferCal-CP (affine, local σ)",
    "TransferCal-CP-affine": "TransferCal-CP (affine)",
    "TransferCal-CP-boost": "TransferCal-CP (boost)",
    "SplitCP": "SplitCP (source-calibrated)",
    "JKplus-source": "Jackknife+ (source-calibrated)",
    "WeightedCP-v2": "Weighted CP",
}


def display_name(method: str) -> str:
    """Figure label for a method identifier."""
    return METHOD_DISPLAY_NAMES.get(method, method)


def method_uses_source_lgbm(method: str) -> bool:
    """True if method's pipeline uses the shared source-side LightGBM fit."""
    return method not in TARGET_ONLY_NO_SOURCE_LGBM


#: Readability controls (Reviewer 2, comment 9: multi-panel figures were
#: unreadable at the journal's printed width).
#:
#: What matters on the printed page is not the figure's own size but the type
#: size *after* the figure is scaled to the text width:
#:
#:     effective_pt = font_pt * page_width / figure_width
#:
#: Enlarging the canvas and the type together therefore changes nothing, and
#: enlarging the canvas alone makes matters worse. The fix is to raise type
#: relative to canvas, and to split figures that are too wide to survive the
#: reduction (see `emit_split_by_fold` in src/visualization.py).
#:
#: FIG_SCALE keeps the submitted geometry by default; TYPE_SCALE raises type.
FIG_SCALE = float(_os.environ.get("SHIFTSAFE_FIG_SCALE", "1.0"))
TYPE_SCALE = float(_os.environ.get("SHIFTSAFE_TYPE_SCALE", "1.30"))


def fs(w: float, h: float) -> tuple[float, float]:
    """Scale a figure size by :data:`FIG_SCALE`."""
    return (w * FIG_SCALE, h * FIG_SCALE)


def effective_pt(font_pt: float, fig_width_in: float, page_width_in: float = 6.5) -> float:
    """Type size as printed, once the figure is scaled to the text width."""
    return font_pt * page_width_in / fig_width_in


def apply_pub_style() -> None:
    t = TYPE_SCALE
    mpl.rcParams.update({
        "font.size": 10 * t,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "lines.linewidth": 1.2 * t,
        "axes.linewidth": 0.6 * t,
        "axes.labelsize": 10 * t,
        "axes.titlesize": 10 * t,
        "legend.fontsize": 8.5 * t,
        "legend.framealpha": 0.92,
        "xtick.labelsize": 9 * t,
        "ytick.labelsize": 9 * t,
        "xtick.major.width": 0.6 * t,
        "ytick.major.width": 0.6 * t,
    })


def panel_label(ax: plt.Axes, letter: str) -> None:
    # Sits clear of the top y-tick label, which a 2 x 2 grid brings close.
    ax.text(
        -0.20, 1.11, f"({letter})", transform=ax.transAxes,
        fontsize=10 * TYPE_SCALE, fontweight="bold", va="top",
    )


def method_style(method: str, idx: int = 0) -> dict:
    dash_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    return {
        "color": METHOD_COLORS.get(method, f"C{idx}"),
        "linestyle": METHOD_LINESTYLES.get(method, dash_cycle[idx % len(dash_cycle)]),
        "label": display_name(method),
    }


def save_fig(fig: plt.Figure, name, formats: tuple[str, ...] = ("pdf", "tiff")) -> None:
    """Save figure as vector PDF + 600 dpi TIFF with embedded Type-42 fonts."""
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem if path.suffix else path.name
    out_dir = path.parent
    for fmt in formats:
        out = out_dir / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", format=fmt, dpi=600 if fmt == "tiff" else None)
    plt.close(fig)


# Backward-compatible alias
save_figure = save_fig

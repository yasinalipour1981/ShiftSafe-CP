"""Figure 7 from real evaluation mixes.

Ten mixes are drawn at random from the evaluation set of one laboratory for one
calibration draw, and each panel shows the interval that the named method
actually produced for each of those mixes, with every "miss" computed from the
plotted interval and the measured strength. Nothing in the figure is simulated.

The previous implementation (``figure_6_example_intervals`` in
src/visualization.py) illustrated the contrast with generated points and should
not be used for the manuscript.

Output: results/revision/docx_figures/Fig7.png (+ .pdf) and a JSON sidecar with
the plotted values, so that the caption can quote them.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_cp import AdaptiveJackknifeCP  # noqa: E402
from src.base_models import fit_base_models  # noqa: E402
from src.conformal import (  # noqa: E402
    SplitConformal,
    physical_feature_indices,
    sample_k_indices,
)
from src.datasets import (  # noqa: E402
    align_dataframe_features,
    create_lodo_splits,
    get_feature_columns,
    load_all_datasets,
)
from src.pubstyle import METHOD_COLORS, apply_pub_style, fs, panel_label, TYPE_SCALE  # noqa: E402

FOLD, SEED, K, ALPHA, N_SHOW = "D4", 142, 20, 0.10, 10
OUT = ROOT / "results" / "revision" / "docx_figures"


def main() -> int:
    warnings.filterwarnings("ignore")
    apply_pub_style()
    cfg = yaml.safe_load(open(ROOT / "configs" / "final_confirmatory_revised.yaml",
                              encoding="utf-8"))
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_all_datasets(cfg, seed=SEED)
    tr, ca, te = (align_dataframe_features(d)
                  for d in create_lodo_splits(ds, 0.8, seed=SEED)[FOLD])
    cols = get_feature_columns(tr)
    X_tr, y_tr = tr[cols].values.astype(float), tr["strength_mpa"].values.astype(float)
    X_ca, y_ca = ca[cols].values.astype(float), ca["strength_mpa"].values.astype(float)
    X_te, y_te = te[cols].values.astype(float), te["strength_mpa"].values.astype(float)
    phys = physical_feature_indices(cols)

    base = fit_base_models(X_tr, y_tr, X_ca, y_ca, config=cfg,
                           device=torch.device(cfg["device"]), n_trials=0)["LightGBM"]

    sampler = lambda Xp, yp, k, sd: sample_k_indices(yp, k, seed=sd, scheme="random")  # noqa: E731
    tajk = AdaptiveJackknifeCP(base, alpha=ALPHA, K=K, seed=SEED, exact_jkp=True,
                               sampler=sampler)
    tajk.fit(X_te, y_te, physical_indices=phys)
    mask = np.ones(len(y_te), bool)
    mask[tajk.calib_target_indices] = False
    X_ev, y_ev = X_te[mask], y_te[mask]

    split = SplitConformal(base, X_ca, y_ca, ALPHA)
    split.fit()

    lo_s, hi_s = split.predict_intervals(X_ev)
    lo_t, hi_t = tajk.predict_intervals(X_ev)
    cov_s_all = float(np.mean((y_ev >= lo_s) & (y_ev <= hi_s)))
    cov_t_all = float(np.mean((y_ev >= lo_t) & (y_ev <= hi_t)))

    pick = np.sort(np.random.default_rng(SEED).choice(len(y_ev), N_SHOW, replace=False))
    y = y_ev[pick]
    panels = [
        ("SplitCP", "Source-calibrated split CP", lo_s[pick], hi_s[pick], cov_s_all),
        ("AdaptiveJackknifeCP", "TargetAnchoredJK+", lo_t[pick], hi_t[pick], cov_t_all),
    ]

    fig, axes = plt.subplots(1, 2, figsize=fs(9, 4), sharey=True)
    x = np.arange(1, N_SHOW + 1)
    record = {"fold": FOLD, "seed": SEED, "K": K, "alpha": ALPHA,
              "n_eval": int(len(y_ev)), "panels": {}}
    lo_all = min(float(np.min(p[2])) for p in panels)
    hi_all = max(float(np.max(p[3])) for p in panels)
    ymin = np.floor(min(lo_all, y.min()) / 5) * 5 - 2
    ymax = np.ceil(max(hi_all, y.max()) / 5) * 5 + 4
    for ax, (key, title, lo, hi, cov_all) in zip(axes, panels):
        miss = (y < lo) | (y > hi)
        col = METHOD_COLORS.get(key, "#555555")
        for xi, l, h, m in zip(x, lo, hi, miss):
            ax.vlines(xi, l, h, color=col, lw=2.2 * TYPE_SCALE, alpha=0.85)
            ax.hlines([l, h], xi - 0.18, xi + 0.18, color=col, lw=1.2 * TYPE_SCALE)
        ax.scatter(x[~miss], y[~miss], marker="o", s=36 * TYPE_SCALE, color="black",
                   zorder=5, label="measured strength, covered")
        ax.scatter(x[miss], y[miss], marker="x", s=52 * TYPE_SCALE, color="#C0392B",
                   linewidths=2, zorder=6, label="measured strength, missed")
        width = float(np.mean(hi - lo))
        ax.set_title(f"{title}\nmissed {int(miss.sum())} of {N_SHOW}; "
                     f"mean width {width:.1f} MPa", fontsize=9 * TYPE_SCALE)
        ax.set_xticks(x)
        ax.set_xlabel("Evaluation mix")
        ax.set_ylim(ymin, ymax)
        record["panels"][title] = {
            "missed": int(miss.sum()), "mean_width": width,
            "coverage_on_full_eval_set": cov_all,
            "y": y.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
        }
    axes[0].set_ylabel("Compressive strength (MPa)")
    axes[1].legend(loc="lower right", fontsize=7.5 * TYPE_SCALE, frameon=True)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Fig7.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Fig7.pdf", bbox_inches="tight")
    plt.close(fig)
    (OUT / "Fig7_values.json").write_text(json.dumps(record, indent=2))
    for t, p in record["panels"].items():
        print(f"{t:30} missed {p['missed']}/10, mean width {p['mean_width']:.1f} MPa, "
              f"coverage on all {record['n_eval']} evaluation mixes "
              f"{100 * p['coverage_on_full_eval_set']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

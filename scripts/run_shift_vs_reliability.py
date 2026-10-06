"""Shift magnitude versus reliability loss, and weighted-CP failure diagnostics.

Reviewer 2, comment 10 asks whether the loss of coverage suffered by
source-calibrated conformal prediction is systematically related to the measured
distribution shift, rather than the two being reported as parallel observations.
Reviewer 2, comment 12 asks for the weighted-conformal failure mechanism
(effective sample size, unbounded quantiles) to be tabulated per domain.

The leave-one-domain-out design yields only three usable transfer cases, which
cannot support a fitted relationship. We therefore enumerate all twelve ordered
single-source to single-target pairs among D1-D4, which multiplies the number of
transfer cases without inventing data, and report rank correlations as
descriptive evidence.

For each ordered pair (s -> t):
  * covariate shift : sliced Wasserstein-1, MMD-RBF, and domain-classifier AUC,
    on the full feature set and on the physical subspace separately;
  * response shift  : the mean signed residual of the source model on the target,
    standardised by the source calibration residual scale;
  * reliability     : empirical coverage and Winkler score of source-calibrated
    split conformal prediction on the target.

Outputs
  results/revision/SHIFT_vs_RELIABILITY_pairs.csv
  results/revision/WEIGHTEDCP_diagnostics.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.base_models import fit_base_models  # noqa: E402
from src.conformal import (  # noqa: E402
    ShiftSafeCP,
    SplitConformal,
    physical_feature_indices,
)
from src.datasets import (  # noqa: E402
    align_dataframe_features,
    get_feature_columns,
    load_all_datasets,
)
from src.metrics import empirical_coverage, mean_interval_width, winkler_score  # noqa: E402
from src.shift_analysis import ShiftAnalyzer  # noqa: E402

DOMAINS = ("D1", "D2", "D3", "D4")
ALPHA = 0.10


def _auc(analyzer, Xa, Xb) -> float:
    """Domain-classifier AUC between two covariate samples."""
    X = np.vstack([Xa, Xb])
    lab = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    return analyzer.domain_classifier_auc(X, lab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="seeds from 142 upward")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    with open(ROOT / "configs" / "final_confirmatory.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config["device"])

    pair_rows: list[dict] = []
    wcp_rows: list[dict] = []

    for seed in range(142, 142 + args.seeds):
        datasets = load_all_datasets(config, seed=seed)
        analyzer = ShiftAnalyzer(device=config["device"], seed=seed)
        rng = np.random.default_rng(seed)

        prepared: dict[str, dict] = {}
        for d in DOMAINS:
            df = align_dataframe_features(datasets[d])
            cols = get_feature_columns(df)
            prepared[d] = {
                "X": df[cols].values.astype(np.float64),
                "y": df["strength_mpa"].values.astype(np.float64),
                "cols": cols,
            }

        for src, tgt in permutations(DOMAINS, 2):
            S, T = prepared[src], prepared[tgt]
            cols = S["cols"]
            phys_idx = physical_feature_indices(cols)

            # 80/20 train/calibration split inside the source laboratory.
            n = len(S["y"])
            perm = rng.permutation(n)
            n_tr = max(5, int(0.8 * n))
            tr, ca = perm[:n_tr], perm[n_tr:]
            if len(ca) < 5:
                continue
            X_tr, y_tr = S["X"][tr], S["y"][tr]
            X_ca, y_ca = S["X"][ca], S["y"][ca]
            X_t, y_t = T["X"], T["y"]

            base = fit_base_models(
                X_tr, y_tr, X_ca, y_ca, config=config, device=device, n_trials=0,
            )["LightGBM"]

            # --- reliability of source-calibrated split conformal -----------
            cp = SplitConformal(base, X_ca, y_ca, ALPHA)
            cp.fit()
            lo, hi = cp.predict_intervals(X_t)
            cov = float(empirical_coverage(y_t, lo, hi))
            wink = float(winkler_score(y_t, lo, hi, alpha=ALPHA))
            width = float(mean_interval_width(lo, hi))

            # --- response (conditional) shift -------------------------------
            resid_src = y_ca - base.predict(X_ca)
            resid_tgt = y_t - base.predict(X_t)
            scale_src = float(np.std(resid_src)) or 1.0
            bias_mpa = float(np.mean(resid_tgt))
            std_bias = abs(bias_mpa) / scale_src
            resid_scale_ratio = float(np.std(resid_tgt)) / scale_src

            # --- covariate shift, full space and physical subspace ----------
            Xs_all, Xt_all = S["X"], T["X"]
            Xs_ph, Xt_ph = S["X"][:, phys_idx], T["X"][:, phys_idx]

            def _std(A, B):
                """Pooled standardisation. A feature that is constant in BOTH
                samples contributes nothing; one that is constant only in the
                source must not be divided by a near-zero source scale."""
                P = np.vstack([A, B])
                mu, sd = P.mean(0), P.std(0)
                sd = np.where(sd < 1e-8, 1.0, sd)
                return (A - mu) / sd, (B - mu) / sd

            Zs_all, Zt_all = _std(Xs_all, Xt_all)
            Zs_ph, Zt_ph = _std(Xs_ph, Xt_ph)
            w1_all = analyzer.wasserstein_distance(Zs_all, Zt_all)
            w1_ph = analyzer.wasserstein_distance(Zs_ph, Zt_ph)
            mmd_ph = analyzer.mmd_rbf(
                torch.tensor(Zs_ph, dtype=torch.float32, device=device),
                torch.tensor(Zt_ph, dtype=torch.float32, device=device),
            )
            auc_all = _auc(analyzer, Xs_all, Xt_all)
            auc_ph = _auc(analyzer, Xs_ph, Xt_ph)

            pair_rows.append(dict(
                seed=seed, source=src, target=tgt,
                n_source=n, n_target=len(y_t),
                w1_all=w1_all, w1_phys=w1_ph, mmd_phys=mmd_ph,
                auc_all=auc_all, auc_phys=auc_ph,
                bias_mpa=bias_mpa, std_bias=std_bias,
                resid_scale_ratio=resid_scale_ratio,
                splitcp_coverage=cov, splitcp_width=width, splitcp_winkler=wink,
                coverage_deficit=(1 - ALPHA) - cov,
            ))

        # --- weighted conformal failure diagnostics, LODO configuration -----
        for held in DOMAINS:
            srcs = [d for d in DOMAINS if d != held]
            cols = prepared[held]["cols"]
            phys_idx = physical_feature_indices(cols)
            Xs = np.vstack([prepared[d]["X"] for d in srcs])
            ys = np.concatenate([prepared[d]["y"] for d in srcs])
            Xt, yt = prepared[held]["X"], prepared[held]["y"]

            nn = len(ys)
            perm = rng.permutation(nn)
            n_tr = int(0.8 * nn)
            tr, ca = perm[:n_tr], perm[n_tr:]
            base = fit_base_models(
                Xs[tr], ys[tr], Xs[ca], ys[ca], config=config, device=device, n_trials=0,
            )["LightGBM"]
            f_cm_idx = cols.index("f_cm") if "f_cm" in cols else None
            wcp = ShiftSafeCP(
                base_model=base,
                X_calib=Xs[ca], y_calib=ys[ca],
                X_source_pool=Xs[tr], X_target_unlabeled=Xt,
                alpha=ALPHA, device=config["device"], seed=seed, layers=[1],
                f_cm_col_index=f_cm_idx, dr_feature_indices=phys_idx,
            )
            wcp.fit()
            lo, hi, diag = wcp.predict_intervals(Xt, return_diagnostics=True)
            wcp_rows.append(dict(
                seed=seed, held_out=held, n_calib=len(ca), n_target=len(yt),
                ess=wcp.diagnostics.get("ess"),
                ess_fraction=wcp.diagnostics.get("ess_fraction"),
                pct_infinite_quantile=diag.get("pct_infinite_quantile"),
                auc_phys=_auc(
                    ShiftAnalyzer(device=config["device"], seed=seed),
                    Xs[:, phys_idx], Xt[:, phys_idx],
                ),
                coverage=float(empirical_coverage(yt, lo, hi)),
                width=float(mean_interval_width(lo, hi)),
                winkler=float(winkler_score(yt, lo, hi, alpha=ALPHA)),
            ))
        print(f"seed {seed} done: {len(pair_rows)} pair rows, {len(wcp_rows)} wcp rows",
              flush=True)

    out = ROOT / "results" / "revision"
    out.mkdir(parents=True, exist_ok=True)
    dfp = pd.DataFrame(pair_rows)
    dfw = pd.DataFrame(wcp_rows)
    dfp.to_csv(out / "SHIFT_vs_RELIABILITY_pairs.csv", index=False)
    dfw.to_csv(out / "WEIGHTEDCP_diagnostics.csv", index=False)

    agg = dfp.groupby(["source", "target"]).mean(numeric_only=True).drop(columns=["seed"])
    print("\n=== Ordered transfer pairs, averaged over seeds ===")
    cols_show = ["w1_phys", "mmd_phys", "auc_phys", "auc_all", "std_bias",
                 "splitcp_coverage", "splitcp_winkler"]
    print(agg[cols_show].round(3).to_string())

    from scipy.stats import spearmanr
    print("\n=== Spearman rank correlation with SplitCP target coverage ===")
    print(f"{'metric':<16}{'rho':>8}{'p':>10}   (n = %d pairs)" % len(agg))
    for m in ["w1_all", "w1_phys", "mmd_phys", "auc_all", "auc_phys",
              "std_bias", "resid_scale_ratio"]:
        r, p = spearmanr(agg[m], agg["splitcp_coverage"])
        print(f"{m:<16}{r:>8.3f}{p:>10.4f}")
    print("\n=== Spearman rank correlation with SplitCP Winkler score ===")
    for m in ["w1_all", "w1_phys", "mmd_phys", "auc_all", "auc_phys",
              "std_bias", "resid_scale_ratio"]:
        r, p = spearmanr(agg[m], agg["splitcp_winkler"])
        print(f"{m:<16}{r:>8.3f}{p:>10.4f}")

    print("\n=== Weighted conformal prediction diagnostics (LODO) ===")
    print(dfw.groupby("held_out")[
        ["ess", "ess_fraction", "pct_infinite_quantile", "auc_phys",
         "coverage", "width", "winkler"]
    ].mean().round(3).to_string())
    print(f"\nWrote {out/'SHIFT_vs_RELIABILITY_pairs.csv'} and "
          f"{out/'WEIGHTEDCP_diagnostics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

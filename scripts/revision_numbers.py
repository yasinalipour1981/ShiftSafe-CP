"""Every number printed in the revised manuscript, computed from the result files.

Sources
  canonical   results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv
              run_all.py re-executed under the outcome-independent draw and the
              exact Barber construction: target-anchored arms, split CP, K-sweep.
  external    results/extension_e_revised/EXTENSION_E_results.csv
              CQR, TabPFN, GPR x2, source-side Jackknife+, weighted CP on the
              same seeds and the same draws.
  grid        results/revision/REVISION_draws_K20.csv
              the three draws, TargetOnlyJK+, StackedOnlyCP, exact vs single refit.
  shift       results/revision/SHIFT_vs_RELIABILITY_pairs.csv
  wcp         results/revision/WEIGHTEDCP_diagnostics.csv
  subspace    results/revision/SUBSPACE_sensitivity.csv

Nothing in the manuscript is typed by hand: `build_revised_docx.py` formats the
values returned by :func:`compute`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "results" / "FINAL_RESULTS_confirmatory_REVISED_142-191.csv"
EXT = ROOT / "results" / "extension_e_revised" / "EXTENSION_E_results.csv"
GRID = ROOT / "results" / "revision" / "REVISION_draws_K20.csv"
SHIFT = ROOT / "results" / "revision" / "SHIFT_vs_RELIABILITY_pairs.csv"
WCP = ROOT / "results" / "revision" / "WEIGHTEDCP_diagnostics.csv"
SUBSP = ROOT / "results" / "revision" / "SUBSPACE_sensitivity.csv"

FOLDS = ("D1", "D2", "D4")
K, ALPHA = 20, 0.10
RNG = np.random.default_rng(20260921)

# internal identifiers -> names used in the paper
NAME = {
    "AdaptiveJackknifeCP": "TargetAnchoredJK+",
    "TargetOnlyCP": "TargetOnlyCP",
    "TransferCal-CP-affine-localsigma": "TransferCal (affine, local $σ$)",
    "AdaptiveSplitCP": "AdaptiveSplitCP",
    "SplitCP": "SplitCP",
    "CQR-target": "CQR-target",
    "TabPFN-target": "TabPFN-target",
    "GPR-target": "GPR-target",
    "GPR-transfer": "GPR-transfer",
    "JKplus-source": "JKplus-source",
    "WeightedCP-v2": "WeightedCP-v2",
}
SCOPE = {
    "AdaptiveJackknifeCP": ("target", "on target, $≥1−2α$"),
    "TargetOnlyCP": ("target", "on target, $≥1−α$"),
    "TransferCal-CP-affine-localsigma": ("target", "on target, $≥1−α$"),
    "AdaptiveSplitCP": ("target", "on target, $≥1−α$"),
    "TabPFN-target": ("target", "on target, $≥1−α$"),
    "CQR-target": ("target", "on target, $≥1−α$"),
    "GPR-target": ("target", "none"),
    "GPR-transfer": ("target", "none"),
    "JKplus-source": ("source", "source only"),
    "WeightedCP-v2": ("source", "source only"),
    "SplitCP": ("source", "source only"),
}


def _boot(x, b=4000, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def _canon_primary(c):
    return c[(c.K == K) & np.isclose(c.alpha, ALPHA)]


def _ext_primary(e):
    return e[np.isclose(e.alpha, ALPHA)] if "alpha" in e.columns else e


def compute() -> dict:
    c = pd.read_csv(CANON)
    e = pd.read_csv(EXT) if EXT.exists() else None
    g = pd.read_csv(GRID)
    out: dict = {"n_seeds": int(c.seed.nunique())}

    cp = _canon_primary(c)
    cp = cp[cp.fold.isin(FOLDS)].rename(columns={
        "empirical_coverage": "coverage", "mean_interval_width": "width",
        "winkler_score": "winkler"})
    frames = [cp[["method", "fold", "seed", "coverage", "width", "winkler"]]]
    if e is not None:
        ep = _ext_primary(e)
        ep = ep[ep.fold.isin(FOLDS)].rename(columns={
            "empirical_coverage": "coverage", "mean_interval_width": "width",
            "winkler_score": "winkler"})
        frames.append(ep[["method", "fold", "seed", "coverage", "width", "winkler"]])
    allm = pd.concat(frames, ignore_index=True)
    out["allm"] = allm

    # ---- per fold (Table 2), with bootstrap CI over seeds -------------------
    per = []
    for (m, f), s in allm.groupby(["method", "fold"]):
        lo, hi = _boot(s.winkler)
        per.append(dict(method=m, fold=f, coverage=s.coverage.mean(),
                        width=s.width.mean(), winkler=s.winkler.mean(),
                        w_lo=lo, w_hi=hi, n=len(s)))
    out["per_fold"] = pd.DataFrame(per)

    # ---- pooled, each laboratory weighted equally (Table 3) -----------------
    def pooled(folds):
        s = allm[allm.fold.isin(folds)]
        return s.groupby(["method", "fold"])[["coverage", "width", "winkler"]] \
                .mean().groupby("method").mean()
    out["pooled"] = pooled(FOLDS)
    out["pooled_noD4"] = pooled(("D1", "D2"))
    out["lodo"] = {d: pooled(tuple(f for f in FOLDS if f != d)) for d in FOLDS}

    # ---- paired tests vs the proposed method (Appendix A.10) ----------------
    piv = allm.pivot_table(index=["fold", "seed"], columns="method", values="winkler")
    ref = "AdaptiveJackknifeCP"
    tests = []
    for m in piv.columns:
        if m == ref:
            continue
        d = (piv[m] - piv[ref]).dropna()
        if len(d) < 10:
            continue
        p = wilcoxon(d).pvalue
        # matched-pairs rank-biserial correlation
        ranks = pd.Series(np.abs(d.values)).rank().values
        rpos = ranks[d.values > 0].sum(); rneg = ranks[d.values < 0].sum()
        r = (rpos - rneg) / (rpos + rneg)
        tests.append(dict(method=m, n=len(d), p=p, r=r, mean_diff=d.mean()))
    t = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
    m_ = len(t)
    holm = []
    running = 0.0
    for i, p in enumerate(t.p):
        running = max(running, min(1.0, (m_ - i) * p))
        holm.append(running)
    t["p_holm"] = holm
    out["tests"] = t

    # ---- per-fold effect of the proposed method vs each comparator ---------
    eff = []
    for f in FOLDS:
        for m in ("TargetOnlyCP", "TransferCal-CP-affine-localsigma", "TabPFN-target"):
            if m not in piv.columns:
                continue
            d = (piv.loc[f][m] - piv.loc[f][ref]).dropna()
            lo, hi = _boot(d)
            eff.append(dict(fold=f, method=m, diff=d.mean(), lo=lo, hi=hi,
                            win=(d > 0).mean()))
    out["effects"] = pd.DataFrame(eff)

    # ---- grid: draws, decomposition, ablations, exact vs single refit ------
    def gpool(df, folds=FOLDS):
        s = df[df.fold.isin(folds)]
        return s.groupby(["method", "fold"])[["coverage", "width", "winkler"]] \
                .mean().groupby("method").mean()
    out["draws"] = {d: gpool(g[g.draw == d]) for d in ("random", "covariate",
                                                       "strength_quartile")}
    rnd = g[g.draw == "random"]
    dec = []
    for f in FOLDS:
        s = rnd[rnd.fold == f]
        gv = lambda m: s[s.method == m].set_index("seed")["winkler"]  # noqa: E731
        a, b, cc = gv("TargetOnlyCP"), gv("TargetOnlyJKplus-exactJK+"), \
            gv("AdaptiveJackknifeCP-exactJK+")
        i = a.index.intersection(b.index).intersection(cc.index)
        a, b, cc = a[i], b[i], cc[i]
        dec.append(dict(fold=f, toc=a.mean(), tojk=b.mean(), tajk=cc.mean(),
                        total=(a - cc).mean(), constr=(a - b).mean(),
                        transf=(b - cc).mean()))
    dec = pd.DataFrame(dec)
    out["decomp"] = dec
    out["pct_constr"] = 100 * dec.constr.sum() / dec.total.sum()
    out["pct_transf"] = 100 * dec.transf.sum() / dec.total.sum()
    d4 = dec.set_index("fold").loc["D4"]
    out["pct_transf_D4"] = 100 * d4.transf / d4.total

    # share of leave-one-out fits choosing the stacked centre at K >= 20, from the
    # per-fit selection counts the canonical run writes to results/diagnostics
    import collections, glob, json
    shares = []
    tot = collections.defaultdict(collections.Counter)
    for fpath in glob.glob(str(ROOT / "results" / "diagnostics" / "fold_*_K*.json")):
        dj = json.load(open(fpath))
        if not (142 <= int(dj.get("seed", 0)) <= 191) or int(dj.get("K", 0)) < 20:
            continue
        sc = dj.get("adaptive_selection")
        if isinstance(sc, dict):
            tot[(dj["fold"], int(dj["K"]))].update(sc)
    for key, cnt in tot.items():
        n = sum(cnt.values())
        if n:
            shares.append(100 * cnt.get("stacked", 0) / n)
    out["sel_lo"] = float(np.floor(min(shares))) if shares else float("nan")
    gp = gpool(rnd)
    out["exact_vs_refit"] = gp.loc[["AdaptiveJackknifeCP-exactJK+", "AdaptiveJackknifeCP"]]
    out["stacked"] = gp.loc[["AdaptiveJackknifeCP", "StackedOnlyCP"]]
    gpv = rnd.pivot_table(index=["fold", "seed"], columns="method", values="winkler")
    dd = (gpv["AdaptiveJackknifeCP"] - gpv["StackedOnlyCP"]).dropna()
    out["stacked_diff"] = float(dd.mean())
    out["stacked_ident"] = float(100 * (dd.abs() < 1e-4).mean())
    sel = rnd[rnd.method == "AdaptiveJackknifeCP"]["selected_candidate"]
    out["sel_stacked"] = float(100 * (sel == "stacked").mean())
    out["sel_by_fold"] = (rnd[rnd.method == "AdaptiveJackknifeCP"]
                          .groupby("fold")["selected_candidate"]
                          .apply(lambda s: 100 * (s == "stacked").mean()))

    # ---- shift vs reliability (Table 4) and weighted CP (Table 5) ----------
    sp = pd.read_csv(SHIFT).groupby(["source", "target"]).mean(numeric_only=True)
    rows = []
    for col, label in [("w1_all", "Sliced Wasserstein-1, full feature set"),
                       ("w1_phys", "Sliced Wasserstein-1, physical subspace"),
                       ("mmd_phys", "MMD (RBF), physical subspace"),
                       ("auc_all", "Domain-classifier AUC, full feature set"),
                       ("auc_phys", "Domain-classifier AUC, physical subspace"),
                       ("std_bias", "Standardized response bias")]:
        r1, p1 = spearmanr(sp[col], sp["splitcp_coverage"])
        r2, p2 = spearmanr(sp[col], sp["splitcp_winkler"])
        rows.append(dict(measure=label, rho_cov=r1, p_cov=p1, rho_w=r2, p_w=p2))
    out["shift_table"] = pd.DataFrame(rows)
    out["shift_pairs"] = sp
    out["wcp"] = pd.read_csv(WCP).groupby("held_out")[
        ["ess", "ess_fraction", "pct_infinite_quantile", "auc_phys", "coverage",
         "width", "winkler"]].mean()
    sub = pd.read_csv(SUBSP).pivot_table(index="subspace", columns="held_out",
                                         values="auc")
    out["subspace"] = sub
    rb = ROOT / "results" / "revision" / "LODO_response_bias.csv"
    if rb.exists():
        out["resp_bias"] = pd.read_csv(rb).groupby("held_out")["std_bias"].mean()

    # ---- K-sweep and timing (canonical) ------------------------------------
    ks = c[np.isclose(c.alpha, ALPHA)].groupby(["method", "fold", "K"])[
        ["empirical_coverage", "mean_interval_width"]].mean()
    out["ksweep"] = ks
    if "inference_time_sec" in c.columns:
        out["timing"] = cp.groupby("method")["inference_time_sec"].mean() \
            if "inference_time_sec" in cp.columns else None
    return out


if __name__ == "__main__":
    o = compute()
    pd.set_option("display.width", 160)
    print("seeds:", o["n_seeds"])
    print("\npooled (all three):\n", o["pooled"].round(3))
    print("\npooled (no D4):\n", o["pooled_noD4"].round(3))
    print("\ntests:\n", o["tests"].round(4))
    print("\neffects:\n", o["effects"].round(2))
    print("\ndecomposition:\n", o["decomp"].round(2),
          f"\n  construction {o['pct_constr']:.1f}%  transfer {o['pct_transf']:.1f}%")

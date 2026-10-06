"""Analysis of the revision grid.

Reframes the evidence as the reviewers ask:

  * Reviewer 1 comment 1 / Reviewer 2 comment 4 -- results are reported under an
    outcome-independent calibration draw, with the submitted strength-quartile
    draw retained only for comparison.
  * Reviewer 1 comment 3 -- decomposes the advantage over ``TargetOnlyCP`` into a
    calibration-construction component (split -> leave-one-out, measured by
    ``TargetOnlyJKplus``) and a source-transfer component.
  * Reviewer 1 comment 4 / Reviewer 2 comment 5 -- the unit of inference is the
    laboratory, not the seed. Seeds within a fold share one evaluation set and
    differ only in which mixes were drawn, so a bootstrap over seeds quantifies
    sensitivity to the calibration draw and nothing else. Between-laboratory
    uncertainty is reported as the spread over the three folds, and as a
    leave-one-domain-out recomputation of every pooled figure.
  * Reviewer 1 comment 5 / Reviewer 2 comment 2 -- every pooled summary is
    reported with and without the reconstructed domain D4.
  * Reviewer 1 comment 2 / Reviewer 2 comment 3 -- exact Barber Jackknife+ is
    reported beside the practical construction.

Output: results/revision/REVISION_SUMMARY.md plus supporting CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IN = ROOT / "results" / "revision" / "REVISION_draws_K20.csv"
OUT = ROOT / "results" / "revision"
PRIMARY_DRAW = "random"
B = 5000
RNG = np.random.default_rng(20260920)

ORDER = [
    "AdaptiveJackknifeCP",
    "AdaptiveJackknifeCP-exactJK+",
    "StackedOnlyCP",
    "TargetOnlyJKplus",
    "TargetOnlyJKplus-exactJK+",
    "TransferCal-CP-affine-localsigma",
    "AdaptiveSplitCP",
    "TargetOnlyCP",
    "SplitCP-source",
]


def boot_ci(x: np.ndarray, fn=np.mean, b: int = B, lo=2.5, hi=97.5) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic of one sample."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    stats = fn(x[idx], axis=1)
    return float(np.percentile(stats, lo)), float(np.percentile(stats, hi))


def per_domain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold means with a bootstrap-over-seeds interval on the Winkler score."""
    rows = []
    for (fold, method), g in df.groupby(["fold", "method"]):
        lo, hi = boot_ci(g["winkler"].to_numpy())
        rows.append(dict(
            fold=fold, method=method, n_seeds=len(g),
            coverage=g["coverage"].mean(),
            width=g["width"].mean(),
            winkler=g["winkler"].mean(),
            winkler_lo=lo, winkler_hi=hi,
        ))
    return pd.DataFrame(rows)


def paired_by_domain(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per-fold paired difference b - a in Winkler score, matched on seed."""
    rows = []
    for fold, g in df.groupby("fold"):
        wa = g[g.method == a].set_index("seed")["winkler"]
        wb = g[g.method == b].set_index("seed")["winkler"]
        common = wa.index.intersection(wb.index)
        d = (wb.loc[common] - wa.loc[common]).to_numpy()
        lo, hi = boot_ci(d)
        rows.append(dict(
            fold=fold, n=len(d), mean_diff=float(d.mean()),
            ci_lo=lo, ci_hi=hi,
            pct_reduction=100.0 * d.mean() / float(wb.loc[common].mean()),
            win_rate=float((d < 0).mean()),
        ))
    return pd.DataFrame(rows)


def pooled(df: pd.DataFrame, folds: tuple[str, ...]) -> pd.Series:
    """Fold-balanced pooled means (each laboratory weighted equally)."""
    sub = df[df.fold.isin(folds)]
    per_fold = sub.groupby(["method", "fold"])[
        ["coverage", "width", "winkler"]
    ].mean()
    return per_fold.groupby("method").mean()


def main() -> int:
    if not IN.exists():
        print(f"missing {IN}; run scripts/run_revision_experiments.py first")
        return 1
    df = pd.read_csv(IN)
    draws = [d for d in ("random", "covariate", "strength_quartile") if d in set(df.draw)]
    folds_all = tuple(sorted(df.fold.unique()))
    folds_nod4 = tuple(f for f in folds_all if f != "D4")
    seeds = sorted(df.seed.unique())

    L: list[str] = []
    w = L.append
    w("# Revision analysis, ShiftSafe-CP (26M-08-239)\n")
    w(f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), folds {', '.join(folds_all)}, "
      f"K = 20, alpha = 0.10.\n")
    w("All pooled figures weight each laboratory equally, so that the largest "
      "evaluation set does not dominate the mean.\n")

    # ---------------- 1. calibration draw ---------------------------------
    w("\n## 1. Effect of the calibration draw (R1.1, R2.4)\n")
    w("`strength_quartile` is the draw used in the submitted manuscript; it forms "
      "strata from the response and is therefore not implementable before the "
      "target specimens are tested. `random` is a simple random draw and is the "
      "revised primary analysis. `covariate` stratifies on water/cement ratio and "
      "age, both fixed at mix design.\n")
    for dr in draws:
        w(f"\n**draw = {dr}** (pooled over {', '.join(folds_all)})\n")
        t = pooled(df[df.draw == dr], folds_all).reindex(
            [m for m in ORDER if m in set(df.method)])
        w(t.round(3).to_markdown())
        w("")

    prim = df[df.draw == PRIMARY_DRAW]

    # ---------------- 2. per-domain ---------------------------------------
    w("\n## 2. Per-laboratory results under the primary (random) draw "
      "(R1.4, R2.5)\n")
    w("Intervals are 95% percentile bootstrap over the 50 seeds *within* a fold. "
      "They quantify sensitivity to the calibration draw only. They are not "
      "between-laboratory uncertainty: for that the sample size is three.\n")
    pdt = per_domain_table(prim)
    pdt.to_csv(OUT / "REVISION_per_domain.csv", index=False)
    for fold in folds_all:
        w(f"\n**{fold}**\n")
        t = (pdt[pdt.fold == fold]
             .set_index("method")
             .reindex([m for m in ORDER if m in set(pdt.method)])
             .drop(columns=["fold", "n_seeds"]))
        w(t.round(3).to_markdown())
        w("")

    # ---------------- 3. D4 sensitivity -----------------------------------
    w("\n## 3. Pooled summary with and without the reconstructed domain D4 "
      "(R1.5, R2.2)\n")
    a = pooled(prim, folds_all).reindex([m for m in ORDER if m in set(prim.method)])
    b = pooled(prim, folds_nod4).reindex([m for m in ORDER if m in set(prim.method)])
    comp = pd.DataFrame({
        "cov (all)": a["coverage"], "cov (no D4)": b["coverage"],
        "width (all)": a["width"], "width (no D4)": b["width"],
        "Winkler (all)": a["winkler"], "Winkler (no D4)": b["winkler"],
    })
    w(comp.round(3).to_markdown())
    comp.to_csv(OUT / "REVISION_d4_sensitivity.csv")
    w("")

    # leave-one-domain-out recomputation of the pooled ranking
    w("\n**Leave-one-domain-out recomputation of the pooled Winkler ranking.** "
      "Each column drops one laboratory; a claim that survives all three columns "
      "is not an artefact of any single laboratory.\n")
    lodo = {}
    for drop in folds_all:
        keep = tuple(f for f in folds_all if f != drop)
        lodo[f"without {drop}"] = pooled(prim, keep)["winkler"]
    lodo["all three"] = a["winkler"]
    lo_df = pd.DataFrame(lodo).reindex([m for m in ORDER if m in set(prim.method)])
    w(lo_df.round(2).to_markdown())
    lo_df.to_csv(OUT / "REVISION_lodo_pooled.csv")
    w("")

    # ---------------- 4. decomposition ------------------------------------
    w("\n## 4. Where the advantage over TargetOnlyCP comes from (R1.3)\n")
    w("`TargetOnlyCP` spends half of K on fitting and half on calibration. "
      "`TargetOnlyJKplus` spends all of K through the same leave-one-out "
      "construction as the proposed method but with a candidate pool restricted "
      "to target-only predictors, so the source model is never consulted. The "
      "difference between the two isolates the calibration construction; the "
      "remaining difference to `AdaptiveJackknifeCP` isolates source transfer.\n")
    dec_rows = []
    for fold in folds_all:
        g = prim[prim.fold == fold]
        get = lambda m: g[g.method == m].set_index("seed")["winkler"]  # noqa: E731
        toc, tojk, ajk = get("TargetOnlyCP"), get("TargetOnlyJKplus"), get("AdaptiveJackknifeCP")
        idx = toc.index.intersection(tojk.index).intersection(ajk.index)
        toc, tojk, ajk = toc.loc[idx], tojk.loc[idx], ajk.loc[idx]
        total = float((toc - ajk).mean())
        constr = float((toc - tojk).mean())
        transf = float((tojk - ajk).mean())
        dec_rows.append(dict(
            fold=fold,
            TargetOnlyCP=float(toc.mean()),
            TargetOnlyJKplus=float(tojk.mean()),
            AdaptiveJackknifeCP=float(ajk.mean()),
            total_gain=total,
            gain_from_LOO_construction=constr,
            gain_from_source_transfer=transf,
            pct_from_construction=100 * constr / total if total else np.nan,
            pct_from_transfer=100 * transf / total if total else np.nan,
        ))
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / "REVISION_decomposition.csv", index=False)
    w(dec.set_index("fold").round(2).to_markdown())
    w("")

    # ---------------- 5. exact vs practical, adaptivity -------------------
    w("\n## 5. Exact Barber Jackknife+ vs the practical construction (R1.2, R2.3)\n")
    pairs = [("AdaptiveJackknifeCP", "AdaptiveJackknifeCP-exactJK+"),
             ("TargetOnlyJKplus", "TargetOnlyJKplus-exactJK+")]
    ex_rows = []
    for practical, exact in pairs:
        if exact not in set(prim.method):
            continue
        for fold in list(folds_all) + ["pooled"]:
            g = prim if fold == "pooled" else prim[prim.fold == fold]
            for nm in (practical, exact):
                s = g[g.method == nm]
                if fold == "pooled":
                    s2 = s.groupby("fold")[["coverage", "width", "winkler"]].mean().mean()
                else:
                    s2 = s[["coverage", "width", "winkler"]].mean()
                ex_rows.append(dict(fold=fold, method=nm, **s2.round(4).to_dict()))
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(OUT / "REVISION_exact_vs_practical.csv", index=False)
    w(ex.pivot_table(index="fold", columns="method",
                     values=["coverage", "winkler"]).round(3).to_markdown())
    w("")

    w("\n## 6. Does adaptive candidate selection contribute? (R1.6, R2.6)\n")
    sel = prim[prim.method == "AdaptiveJackknifeCP"]["selected_candidate"].value_counts()
    w("Candidate selected by the full-K refit, primary draw:\n")
    w(sel.to_frame("n").to_markdown())
    ad = pooled(prim, folds_all).reindex(["AdaptiveJackknifeCP", "StackedOnlyCP"])
    w("\nAdaptive selection vs a fixed stacked centre:\n")
    w(ad.round(4).to_markdown())
    g = prim.pivot_table(index=["fold", "seed"], columns="method", values="winkler")
    if {"AdaptiveJackknifeCP", "StackedOnlyCP"} <= set(g.columns):
        d = (g["AdaptiveJackknifeCP"] - g["StackedOnlyCP"]).dropna()
        w(f"\nPaired difference (Adaptive − StackedOnly): mean {d.mean():.4f}, "
          f"max |diff| {d.abs().max():.4f}, identical in "
          f"{100 * (d.abs() < 1e-9).mean():.1f}% of the {len(d)} seed-by-fold cells.\n")

    (OUT / "REVISION_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {OUT/'REVISION_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

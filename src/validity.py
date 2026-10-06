"""Finite-sample validity regions for conformal arms."""

from __future__ import annotations

import math

import pandas as pd

from src.conformal import MinKViolation, min_k_for_alpha


SPLIT_TARGET_METHODS = frozenset({
    "TargetOnlyCP",
    "AdaptiveSplitCP",
    "TransferCalibratedCP",
    "TransferCal-CP",
    "TransferCal-CP-affine",
    "TransferCal-CP-affine-localsigma",
    "TransferCal-CP-boost",
    "CQR",
    "CQR-target",
    "TabPFN-target",
})

JACKKNIFE_METHODS = frozenset({
    "AdaptiveJackknifeCP",
    "JackknifeP",
    "Jackknife+",
    "JKplus-source",
})

SOURCE_GUARANTEE_METHODS = frozenset({
    "SplitCP",
    "WeightedCP-v2",
    "JKplus-source",
    "Jackknife+",
})

NONE_GUARANTEE_METHODS = frozenset({
    "GPR-target",
    "GPR-transfer",
    "GPR",
    "NGBoost",
    "DeepEnsemble",
})


def get_guarantee_scope(method: str) -> str:
    """Return guarantee_scope: 'target' | 'source' | 'none'."""
    if method in NONE_GUARANTEE_METHODS:
        return "none"
    if method in SOURCE_GUARANTEE_METHODS or method.startswith("ShiftSafe"):
        return "source"
    if method in SPLIT_TARGET_METHODS or method in JACKKNIFE_METHODS:
        if method in ("AdaptiveJackknifeCP", "CQR-target", "TabPFN-target", "TargetOnlyCP"):
            return "target"
        if method in ("JKplus-source", "Jackknife+", "JackknifeP"):
            return "source"
    if "Jackknife" in method and method != "AdaptiveJackknifeCP":
        return "source"
    if method.startswith("TransferCal") or method == "AdaptiveSplitCP":
        return "target"
    return "target"


def annotate_guaranteed(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `guaranteed` and str `guarantee_scope` columns."""
    out = df.copy()
    if "K" not in out.columns or "alpha" not in out.columns or "method" not in out.columns:
        out["guaranteed"] = True
        out["guarantee_scope"] = "target"
        return out
    out["guaranteed"] = [
        is_guaranteed(str(m), int(k), float(a))
        for m, k, a in zip(out["method"], out["K"], out["alpha"])
    ]
    out["guarantee_scope"] = [get_guarantee_scope(str(m)) for m in out["method"]]
    # GPR arms never have finite-sample guarantee under shift protocol
    gpr_mask = out["method"].isin(NONE_GUARANTEE_METHODS)
    out.loc[gpr_mask, "guaranteed"] = False
    return out


def split_k_cal(K: int, alpha: float) -> int:
    mk = min_k_for_alpha(alpha)
    return max(mk, K // 2)


def conformal_rank(K_cal: int, alpha: float) -> int:
    return int(math.ceil((K_cal + 1) * (1 - alpha)))


def min_k_jackknife(alpha: float) -> int:
    """Minimum K for Jackknife+ arms (uses all K labeled points)."""
    return min_k_for_alpha(alpha) + 1


def min_k_split_5050(alpha: float) -> int:
    """Minimum K for split arms with K_cal and K_fit each >= min_k."""
    return 2 * min_k_for_alpha(alpha)


def is_guaranteed(method: str, K: int, alpha: float) -> bool:
    """True iff (method, K, alpha) lies inside the class operating region."""
    if method in NONE_GUARANTEE_METHODS:
        return False
    if method == "SplitCP":
        return True
    if method in ("WeightedCP-v2", "JKplus-source", "Jackknife+"):
        return True
    if method in SPLIT_TARGET_METHODS or method.startswith("TransferCal"):
        if K < min_k_for_alpha(alpha) + 1:
            return False
        kcal = split_k_cal(K, alpha)
        if conformal_rank(kcal, alpha) > kcal:
            return False
        return K >= min_k_split_5050(alpha)
    if method == "CQR-target":
        K_cal = K - 10  # fixed K_fit=10 protocol
        if K_cal < min_k_for_alpha(alpha):
            return False
        return conformal_rank(K_cal, alpha) <= K_cal
    if method in JACKKNIFE_METHODS or "Jackknife" in method:
        return K >= min_k_jackknife(alpha)
    return True


def fold_output_range_mpa(df: pd.DataFrame, fold: str) -> float | None:
    """Estimate test-set strength span from normalized_width (width / norm_width)."""
    sub = df[(df["fold"] == fold) & (df["normalized_width"] > 1e-6)]
    if sub.empty:
        return None
    est = sub["mean_interval_width"] / sub["normalized_width"]
    return float(est.median())

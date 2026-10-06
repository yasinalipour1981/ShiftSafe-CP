"""Evaluation metrics for uncertainty quantification methods."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.validity import MinKViolation

logger = logging.getLogger(__name__)


def empirical_coverage(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> float:
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    return float(np.mean(y_upper - y_lower))


def normalized_mean_interval_width(
    y_lower: np.ndarray, y_upper: np.ndarray, y_true: np.ndarray
) -> float:
    y_range = np.max(y_true) - np.min(y_true)
    return float(np.mean(y_upper - y_lower) / (y_range + 1e-8))


def winkler_score(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, alpha: float
) -> float:
    widths = y_upper - y_lower
    below = np.maximum(0, y_lower - y_true)
    above = np.maximum(0, y_true - y_upper)
    scores = widths + (2 / alpha) * (below + above)
    return float(np.mean(scores))


def coverage_error(empirical_cov: float, nominal_cov: float) -> float:
    return float(np.abs(empirical_cov - nominal_cov))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def crps_gaussian(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> float:
    """CRPS for Gaussian predictive distribution."""
    z = (y_true - mean) / (std + 1e-8)
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def conditional_coverage_per_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    bins = np.linspace(np.min(y_pred), np.max(y_pred), n_bins + 1)
    bin_indices = np.digitize(y_pred, bins)

    rows = []
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if mask.sum() > 0:
            cov = empirical_coverage(y_true[mask], y_lower[mask], y_upper[mask])
            rows.append(
                {
                    "bin": b,
                    "range": f"[{bins[b-1]:.1f}, {bins[b]:.1f}]",
                    "n": int(mask.sum()),
                    "coverage": cov,
                }
            )
    return pd.DataFrame(rows)


def evaluate_method(
    predictor: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    method_name: str,
) -> dict[str, Any]:
    """Evaluate a single method at one alpha level."""
    start = time.time()
    lower, upper = predictor.predict_intervals(X_test, alpha=alpha)
    infer_time = time.time() - start

    if hasattr(predictor, "predict"):
        y_pred = predictor.predict(X_test)
    else:
        y_pred = (lower + upper) / 2

    cov = empirical_coverage(y_test, lower, upper)
    return {
        "method": method_name,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": cov,
        "coverage_error": coverage_error(cov, 1 - alpha),
        "mean_interval_width": mean_interval_width(lower, upper),
        "normalized_width": normalized_mean_interval_width(lower, upper, y_test),
        "winkler_score": winkler_score(y_test, lower, upper, alpha),
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "r_squared": r_squared(y_test, y_pred),
        "inference_time_sec": infer_time,
    }


def evaluate_all_methods(
    methods_dict: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_levels: list[float] | None = None,
) -> pd.DataFrame:
    alpha_levels = alpha_levels or [0.05, 0.10, 0.20]
    results = []
    for method_name, predictor in methods_dict.items():
        for alpha in alpha_levels:
            logger.info("Evaluating %s at alpha=%.2f...", method_name, alpha)
            try:
                row = evaluate_method(predictor, X_test, y_test, alpha, method_name)
            except MinKViolation as exc:
                logger.warning("Skip %s at alpha=%.2f: %s", method_name, alpha, exc)
                continue
            except ValueError as exc:
                if "too small for alpha" in str(exc):
                    logger.warning("Skip %s at alpha=%.2f: %s", method_name, alpha, exc)
                    continue
                raise
            results.append(row)
    return pd.DataFrame(results)


def rank_biserial_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation effect size for paired Wilcoxon (matched pairs)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    w_stat, _ = stats.wilcoxon(d, alternative="two-sided")
    # scipy returns sum of ranks for positive differences
    r = 1.0 - (2.0 * w_stat) / (n * (n + 1))
    return float(np.clip(r, -1.0, 1.0))


def statistical_significance_test(
    df_lodo_results: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "empirical_coverage",
    alpha_level: float = 0.10,
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test across LODO folds."""
    df_a = df_lodo_results[
        (df_lodo_results["method"] == method_a)
        & (df_lodo_results["alpha"] == alpha_level)
    ]
    df_b = df_lodo_results[
        (df_lodo_results["method"] == method_b)
        & (df_lodo_results["alpha"] == alpha_level)
    ]

    merged = df_a.merge(
        df_b, on=["fold", "seed"], suffixes=("_a", "_b"), how="inner"
    )
    if len(merged) < 3:
        return {
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "n_pairs": len(merged),
        }

    vals_a = merged[f"{metric}_a"].values
    vals_b = merged[f"{metric}_b"].values
    statistic, p_value = stats.wilcoxon(vals_a, vals_b)
    effect_r = rank_biserial_wilcoxon(vals_a, vals_b)
    wins_a = int(np.sum(vals_a < vals_b))

    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_rank_biserial_r": effect_r,
        "wins_a": wins_a,
        "n_pairs": len(merged),
        "significant_at_0.05": p_value < 0.05,
    }


def holm_corrected_wilcoxon_tests(
    df_lodo_results: pd.DataFrame,
    pairs: list[tuple[str, str]],
    metric: str = "mean_interval_width",
    alpha_level: float = 0.10,
) -> list[dict[str, Any]]:
    """Paired Wilcoxon tests with Holm-Bonferroni correction across pairs."""
    raw = [
        statistical_significance_test(df_lodo_results, a, b, metric, alpha_level)
        for a, b in pairs
    ]
    m = len(raw)
    if m == 0:
        return raw
    order = np.argsort([r["p_value"] for r in raw])
    for rank, idx in enumerate(order):
        raw[idx]["holm_p_value"] = min(1.0, raw[idx]["p_value"] * (m - rank))
        raw[idx]["significant_holm_0.05"] = raw[idx]["holm_p_value"] < 0.05
    return raw

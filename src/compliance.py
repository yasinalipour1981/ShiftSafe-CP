"""EN 206 / IS 456 compliance-probability linkage."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# IS 456 / EN 206 characteristic strength thresholds (MPa)
GRADE_THRESHOLDS: dict[str, float] = {
    "M15": 15.0,
    "M20": 20.0,
    "M25": 25.0,
    "M30": 30.0,
    "M35": 35.0,
    "M40": 40.0,
    "M45": 45.0,
    "M50": 50.0,
    "C20/25": 20.0,
    "C25/30": 25.0,
    "C30/37": 30.0,
    "C35/45": 35.0,
    "C40/50": 40.0,
}


def compliance_probability_from_interval(
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Estimate P(strength >= threshold) assuming uniform distribution in interval.
    """
    width = upper - lower
    width = np.maximum(width, 1e-8)
    prob = (upper - threshold) / width
    return np.clip(prob, 0.0, 1.0)


def compliance_decision(
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float,
    required_prob: float = 0.95,
) -> np.ndarray:
    """Accept if P(strength >= threshold) >= required_prob."""
    prob = compliance_probability_from_interval(lower, upper, threshold)
    return (prob >= required_prob).astype(int)


def compute_compliance_probabilities(
    methods_intervals: dict[str, tuple[np.ndarray, np.ndarray]],
    grade: str = "M25",
    required_prob: float = 0.95,
) -> pd.DataFrame:
    """
    Compute compliance probabilities and accept/reject decisions per method.

    methods_intervals: {method_name: (lower, upper)}
    """
    threshold = GRADE_THRESHOLDS.get(grade, 25.0)
    rows = []
    for method, (lower, upper) in methods_intervals.items():
        prob = compliance_probability_from_interval(lower, upper, threshold)
        decision = compliance_decision(lower, upper, threshold, required_prob)
        rows.append(
            {
                "method": method,
                "grade": grade,
                "threshold_mpa": threshold,
                "mean_compliance_prob": float(np.mean(prob)),
                "accept_rate": float(np.mean(decision)),
                "n_accept": int(np.sum(decision)),
                "n_total": len(decision),
            }
        )
    return pd.DataFrame(rows)


def compliance_agreement_matrix(
    methods_intervals: dict[str, tuple[np.ndarray, np.ndarray]],
    grade: str = "M25",
    required_prob: float = 0.95,
) -> pd.DataFrame:
    """Pairwise agreement on accept/reject decisions."""
    threshold = GRADE_THRESHOLDS.get(grade, 25.0)
    methods = list(methods_intervals.keys())
    decisions = {
        m: compliance_decision(lo, hi, threshold, required_prob)
        for m, (lo, hi) in methods_intervals.items()
    }

    n = len(methods)
    matrix = np.zeros((n, n))
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            agree = np.mean(decisions[mi] == decisions[mj])
            matrix[i, j] = agree

    return pd.DataFrame(matrix, index=methods, columns=methods)

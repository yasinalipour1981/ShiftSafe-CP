"""Tests for ShiftSafe-CP-final adaptive methods."""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive_cp import (
    GATE_FULL_N,
    SIGMA_MPA_FLOOR,
    AdaptiveJackknifeCP,
    AdaptiveSplitCP,
    _fit_candidates,
    _sigma_const_from_resid,
)
from src.conformal import min_k_for_alpha, stratified_sample_k_indices


class _LinModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return 30 + X[:, 0] * 5


def test_adaptive_jackknife_min_k():
    with pytest.raises(ValueError, match="min_k"):
        AdaptiveJackknifeCP(_LinModel(), alpha=0.10, K=5, seed=1).fit(
            np.random.randn(40, 6), np.random.randn(40), [0, 1, 2, 3],
        )


def test_adaptive_jackknife_disjoint():
    rng = np.random.default_rng(0)
    n = 80
    X = rng.standard_normal((n, 8))
    y = 30 + X[:, 0] * 4 + rng.normal(0, 2, n)
    ajk = AdaptiveJackknifeCP(_LinModel(), alpha=0.10, K=20, seed=42)
    ajk.fit(X, y, physical_indices=[0, 1, 2, 3])
    mask = np.ones(n, dtype=bool)
    mask[ajk.calib_target_indices] = False
    lo, hi = ajk.predict_intervals(X[mask])
    assert np.all(lo < hi)
    assert sum(ajk.diagnostics["selection_counts"].values()) == 20


def test_adaptive_selection_histogram_sums_to_k():
    rng = np.random.default_rng(1)
    n = 60
    X = rng.standard_normal((n, 6))
    y = 30 + X[:, 0] * 3 + rng.normal(0, 1.5, n)
    ajk = AdaptiveJackknifeCP(_LinModel(), alpha=0.10, K=15, seed=99)
    ajk.fit(X, y, [0, 1, 2, 3])
    assert sum(ajk.diagnostics["selection_counts"].values()) == 15
    assert ajk.diagnostics["winner_full"] in ajk.diagnostics["selection_counts"]
    rng = np.random.default_rng(2)
    n = 70
    X = rng.standard_normal((n, 6))
    y = 28 + X[:, 0] * 4 + rng.normal(0, 2, n)
    asc = AdaptiveSplitCP(_LinModel(), alpha=0.10, K=20, seed=7)
    asc.fit(X, y, physical_indices=[0, 1, 2, 3])
    assert asc.diagnostics["K_cal"] >= min_k_for_alpha(0.10)
    lo, hi = asc.predict_intervals(X[[5]])
    assert lo[0] < hi[0]


def test_sigma_const_floor_mpa():
    """final.1: sigma >= max(median|resid|, 0.5*MAD(y), 1.0 MPa)."""
    y = np.linspace(20.0, 40.0, 20)
    resid_tiny = np.full(20, 0.01)
    sigma = _sigma_const_from_resid(resid_tiny, y)
    assert sigma >= SIGMA_MPA_FLOOR
    mad_y = float(np.median(np.abs(y - np.median(y))))
    assert sigma >= 0.5 * mad_y
    assert sigma >= float(np.median(np.abs(resid_tiny)))


def test_candidate_gating_small_n():
    """n_fit < 15: only local-ridge, affine-transfer, stacked compete."""
    rng = np.random.default_rng(3)
    n = 14
    X = rng.standard_normal((n, 6))
    model = _LinModel()
    mu = model.predict(X)
    y = 30 + X[:, 0] * 3 + rng.normal(0, 2, n)
    fit = _fit_candidates(X, y, mu, [0, 1, 2, 3], seed=1)
    assert fit.name in {"local-ridge", "affine-transfer", "stacked"}
    ajk = AdaptiveJackknifeCP(model, alpha=0.10, K=n, seed=5)
    ajk.fit(X, y, physical_indices=[0, 1, 2, 3])
    counts = ajk.diagnostics["selection_counts"]
    assert counts.get("local-gbm", 0) == 0
    assert counts.get("boost-transfer", 0) == 0
    assert n == GATE_FULL_N - 1
    assert sum(counts.values()) == n


def test_candidate_gating_full_n():
    """n_fit >= 15: full candidate set available (tree learners permitted)."""
    rng = np.random.default_rng(4)
    n = GATE_FULL_N
    X = rng.standard_normal((n, 6))
    model = _LinModel()
    mu = model.predict(X)
    y = 30 + X[:, 0] * 3 + rng.normal(0, 2, n)
    fit = _fit_candidates(X, y, mu, [0, 1, 2, 3], seed=2)
    assert fit.name in {
        "local-ridge", "affine-transfer", "stacked", "local-gbm", "boost-transfer",
    }
    ajk = AdaptiveJackknifeCP(model, alpha=0.10, K=20, seed=8)
    pool_X = rng.standard_normal((60, 6))
    pool_y = 30 + pool_X[:, 0] * 3 + rng.normal(0, 2, 60)
    ajk.fit(pool_X, pool_y, physical_indices=[0, 1, 2, 3])
    assert ajk.diagnostics["candidate_gate_n"] == GATE_FULL_N


def test_selection_guard_fallback_to_stacked():
    """Interpolation (median|resid| < 0.5*MAD(y)) -> fallback to stacked."""
    rng = np.random.default_rng(99)
    n = GATE_FULL_N - 1  # gated set: no tree learners
    X = rng.standard_normal((n, 6))
    model = _LinModel()
    mu = model.predict(X)
    y = mu.copy()
    fit = _fit_candidates(X, y, mu, [0, 1, 2, 3], seed=1)
    assert fit.name == "stacked"
    ajk = AdaptiveJackknifeCP(model, alpha=0.10, K=20, seed=42)
    pool_X = rng.standard_normal((80, 6))
    pool_y = model.predict(pool_X)
    ajk.fit(pool_X, pool_y, physical_indices=[0, 1, 2, 3])
    assert ajk.diagnostics["winner_full"] == "stacked"

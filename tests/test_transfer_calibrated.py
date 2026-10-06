"""Tests for ShiftSafe-CP v3 transfer-calibrated conformal."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.conformal import (
    TargetOnlyCP,
    TransferCalibratedCP,
    min_k_for_alpha,
    stratified_sample_k_indices,
)


class _MeanModel:
    def __init__(self, value: float = 40.0):
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


class _ConstSigma:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 5.0)


def test_min_k_for_alpha():
    assert min_k_for_alpha(0.10) == 9
    assert min_k_for_alpha(0.05) == 19


def test_disjoint_calib_test_indices():
    rng = np.random.default_rng(0)
    y = rng.uniform(20, 60, 100)
    K = 20
    cal = stratified_sample_k_indices(y, K, seed=42)
    test = np.array([i for i in range(len(y)) if i not in set(cal)])
    assert set(cal).isdisjoint(set(test))
    assert len(cal) == K
    assert len(test) == len(y) - K


def test_transfer_calibrated_affine_mode():
    rng = np.random.default_rng(4)
    n = 80
    X = rng.standard_normal((n, 8))
    y = 35 + X[:, 0] * 5 + rng.normal(0, 3, n)
    K = 20
    tc = TransferCalibratedCP(
        _MeanModel(40.0), _ConstSigma(), 1.0,
        alpha=0.10, K=K, fine_tune_mode="affine", seed=42,
    )
    tc.fit(X, y)
    assert tc._affine_a is not None
    assert "bias_before_affine" in tc.diagnostics
    assert "bias_after_affine" in tc.diagnostics
    mask = np.ones(n, dtype=bool)
    mask[tc.calib_target_indices] = False
    lo, hi = tc.predict_intervals(X[mask], alpha=0.10)
    assert np.all(lo < hi)


def test_transfer_calibrated_coverage_smoke():
    rng = np.random.default_rng(1)
    n = 80
    X = rng.standard_normal((n, 8))
    y = 35 + X[:, 0] * 5 + rng.normal(0, 3, n)
    K = 20
    alpha = 0.10

    tc = TransferCalibratedCP(
        base_model=_MeanModel(40.0),
        sigma_model=_ConstSigma(),
        sigma_floor=1.0,
        alpha=alpha,
        K=K,
        fine_tune_mode="none",
        seed=42,
    )
    tc.fit(X, y)
    mask = np.ones(n, dtype=bool)
    mask[tc.calib_target_indices] = False
    lo, hi = tc.predict_intervals(X[mask], alpha=alpha)
    cov = float(np.mean((y[mask] >= lo) & (y[mask] <= hi)))
    assert cov >= 1 - alpha - 0.15  # binomial slack on small n
    assert tc.diagnostics["quantile_q"] > 0


def test_transfer_calibrated_min_k_assertion():
    with pytest.raises(ValueError, match="K too small"):
        TransferCalibratedCP(
            _MeanModel(), _ConstSigma(), 1.0, alpha=0.10, K=5, seed=42,
        ).fit(np.random.randn(50, 4), np.random.randn(50))


def test_target_only_cp_disjoint():
    rng = np.random.default_rng(2)
    n = 60
    X = rng.standard_normal((n, 8))
    y = 30 + X[:, 0] * 4 + rng.normal(0, 2, n)
    phys = [0, 1, 2, 3]
    toc = TargetOnlyCP(alpha=0.10, K=15, seed=7)
    toc.fit(X, y, physical_indices=phys)
    mask = np.ones(n, dtype=bool)
    mask[toc.calib_target_indices] = False
    assert mask.sum() == n - 15
    lo, hi = toc.predict_intervals(X[mask])
    assert np.all(lo < hi)


def test_target_only_k_fit_cal_disjoint():
    """mu/sigma fit on K_fit; scores only on K_cal (no label leakage)."""
    rng = np.random.default_rng(8)
    n = 80
    X = rng.standard_normal((n, 6))
    y = 30 + X[:, 0] * 3 + rng.normal(0, 2, n)
    phys = [0, 1, 2, 3]
    toc = TargetOnlyCP(alpha=0.10, K=20, seed=42)
    toc.fit(X, y, physical_indices=phys)
    fit_set = set(toc.diagnostics["fit_indices"])
    cal_set = set(toc.diagnostics["cal_indices"])
    assert fit_set.isdisjoint(cal_set)
    assert len(fit_set) + len(cal_set) == 20
    assert toc.diagnostics["K_cal"] >= min_k_for_alpha(0.10)
    assert toc.diagnostics["K_fit"] >= 1


def test_transfer_calibrated_affine_localsigma():
    rng = np.random.default_rng(9)
    n = 80
    X = rng.standard_normal((n, 8))
    y = 35 + X[:, 0] * 5 + rng.normal(0, 3, n)
    tc = TransferCalibratedCP(
        _MeanModel(40.0), _ConstSigma(), 1.0,
        alpha=0.10, K=20, fine_tune_mode="affine_localsigma", seed=42,
    )
    tc.fit(X, y)
    assert tc._sigma_local_const is not None
    assert tc._sigma_local_const > 0
    assert tc._sigma_scale_c is None
    sigma_test = tc._sigma(X[:5])
    assert np.allclose(sigma_test, tc._sigma_local_const)
    mask = np.ones(n, dtype=bool)
    mask[tc.calib_target_indices] = False
    lo, hi = tc.predict_intervals(X[mask])
    assert np.all(np.isfinite(lo))
    assert np.all(lo < hi)


def test_transfer_calibrated_perfect_affine_no_nan():
    """Perfect affine fit on K_ft must not produce NaN intervals."""
    constant_y = 42.0
    n = 50

    class _PerfectModel:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), constant_y)

    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, 4))
    y = np.full(n, constant_y)

    tc = TransferCalibratedCP(
        _PerfectModel(),
        _ConstSigma(),
        sigma_floor=1.0,
        alpha=0.10,
        K=20,
        fine_tune_mode="affine",
        seed=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tc.fit(X, y)
        mask = np.ones(n, dtype=bool)
        mask[tc.calib_target_indices] = False
        lo, hi = tc.predict_intervals(X[mask])

    assert tc._sigma_scale_c > 0
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    # With a perfect fit every calibration residual is zero, so the conformal
    # quantile is zero and the interval legitimately collapses to a point;
    # the test guards against NaN and divide-by-zero, not against zero width.
    assert np.all(lo <= hi)
    assert not any(
        "divide by zero" in str(w.message) for w in caught
    )


def test_v3_calib_never_in_test_set():
    """K calib points must be excluded from evaluation set."""
    rng = np.random.default_rng(3)
    n = 50
    X = rng.standard_normal((n, 6))
    y = rng.uniform(25, 55, n)
    K = 12
    tc = TransferCalibratedCP(
        _MeanModel(), _ConstSigma(), 1.0, alpha=0.10, K=K, seed=99,
    )
    tc.fit(X, y)
    eval_idx = np.array([i for i in range(n) if i not in tc.calib_target_indices])
    assert len(eval_idx) == n - K
    assert set(tc.calib_target_indices).isdisjoint(set(eval_idx))

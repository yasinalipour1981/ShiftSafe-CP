"""Tests for Extension E baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines_e import (
    CQRTargetCP,
    GPRTargetCP,
    GPRTransferCP,
    JKPlusSourceCP,
    WeightedCPv2,
)
from src.conformal import stratified_sample_k_indices


class _LinModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return 30 + X[:, 0] * 5


@pytest.fixture
def pool():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 8))
    y = 30 + X[:, 0] * 4 + rng.normal(0, 2, n)
    return X, y, [0, 1, 2, 3]


def test_cqr_target_intervals(pool):
    X, y, phys = pool
    k = stratified_sample_k_indices(y, 20, seed=7)
    cp = CQRTargetCP(alpha=0.10, K=20, seed=7, K_fit=10)
    cp.fit(X, y, k_indices=k)
    lo, hi = cp.predict_intervals(X[50:60], alpha=0.10)
    assert np.all(lo < hi)
    assert cp.guarantee_scope == "target"


def test_gpr_target_no_guarantee(pool):
    X, y, _ = pool
    k = stratified_sample_k_indices(y, 20, seed=7)
    cp = GPRTargetCP(alpha=0.10, K=20, seed=7)
    cp.fit(X, y, k_indices=k)
    lo, hi = cp.predict_intervals(X[50:55], alpha=0.10)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert cp.guarantee_scope == "none"


def test_gpr_transfer(pool):
    X, y, _ = pool
    k = stratified_sample_k_indices(y, 20, seed=7)
    cp = GPRTransferCP(_LinModel(), alpha=0.10, K=20, seed=7)
    cp.fit(X, y, k_indices=k)
    lo, hi = cp.predict_intervals(X[50:55])
    assert np.all(lo < hi)


def test_jkplus_source(pool):
    X, y, _ = pool
    n_cal = 40
    domains = np.array(["D1"] * 20 + ["D2"] * 20)
    cp = JKPlusSourceCP(
        _LinModel(), X[:n_cal], y[:n_cal], alpha=0.10, seed=1, max_calib_samples=30,
        held_out_domain="D4", calib_domains=domains, use_gpu=False,
    )
    cp.fit()
    lo, hi = cp.predict_intervals(X[50:55])
    assert np.all(lo < hi)
    assert cp.guarantee_scope == "source"
    assert cp.diagnostics["n_models_trained"] == 30
    assert cp._jk is not None and len(cp._jk.loo_models) == 30


def test_jkplus_source_rejects_held_out_leakage(pool):
    X, y, _ = pool
    domains = np.array(["D1"] * 20 + ["D4"] * 20)  # D4 = held-out leaked into calib
    with pytest.raises(AssertionError, match="LEAKAGE"):
        JKPlusSourceCP(
            _LinModel(), X[:40], y[:40], alpha=0.10, seed=1, max_calib_samples=30,
            held_out_domain="D4", calib_domains=domains, use_gpu=False,
        )


def test_jkplus_source_y_test_poison_unused(pool):
    """Predict path must ignore poisoned y_test (never consumed)."""
    X, y, _ = pool
    cp = JKPlusSourceCP(
        _LinModel(), X[:40], y[:40], alpha=0.10, seed=2, max_calib_samples=25,
        use_gpu=False,
    )
    cp.fit()
    y_poison = np.full(5, np.nan)
    _ = y_poison  # deliberately unused, intervals must not need y_test
    lo, hi = cp.predict_intervals(X[50:55])
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    # True Jackknife+ must not equal base_model ± single residual quantile
    base = _LinModel().predict(X[50:55])
    R = np.sort(cp._jk.scores_calib)
    q = R[int(np.ceil((len(R) + 1) * 0.9)) - 1]
    split_style_lo, split_style_hi = base - q, base + q
    assert not (
        np.allclose(lo, split_style_lo) and np.allclose(hi, split_style_hi)
    ), "JKplus collapsed to SplitCP-style base_model ± q (bug regression)"


def test_k_indices_match_adaptive(pool):
    """Same seed -> same K draw as stratified_sample_k_indices."""
    X, y, phys = pool
    from src.adaptive_cp import AdaptiveJackknifeCP

    ajk = AdaptiveJackknifeCP(_LinModel(), alpha=0.10, K=20, seed=99)
    ajk.fit(X, y, physical_indices=phys)
    k2 = stratified_sample_k_indices(y, 20, seed=99)
    assert np.array_equal(np.sort(ajk.calib_target_indices), np.sort(k2))


def test_weighted_cp_v2_leakage(pool):
    X, y, _ = pool
    n_tr, n_cal = 30, 20
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_cal, y_cal = X[n_tr : n_tr + n_cal], y[n_tr : n_tr + n_cal]
    X_te = X[50:60]
    y_te_poison = np.full(len(X_te), np.nan)
    _ = y_te_poison
    cp = WeightedCPv2(
        _LinModel(), X_tr, y_tr, X_cal, y_cal, X_te,
        alpha=0.10, seed=1, device="cpu",
    )
    cp.fit()
    lo, hi = cp.predict_intervals(X_te)
    assert np.all(np.isfinite(lo))

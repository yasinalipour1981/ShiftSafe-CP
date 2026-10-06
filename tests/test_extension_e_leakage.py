"""Leakage tests for Extension E baseline classes."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines_e import (
    CQRTargetCP,
    GPRTargetCP,
    GPRTransferCP,
    JKPlusSourceCP,
    TabPFNTargetCP,
    WeightedCPv2,
)


class _LinModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return 30 + X[:, 0] * 5


@pytest.fixture
def lodo():
    rng = np.random.default_rng(11)
    n_tr, n_cal, n_te, d = 50, 30, 40, 8
    X_tr = rng.standard_normal((n_tr, d))
    X_cal = rng.standard_normal((n_cal, d))
    X_te = rng.standard_normal((n_te, d)) + 0.5
    coef = rng.uniform(1, 2, d)
    y_tr = X_tr @ coef + 30
    y_cal = X_cal @ coef + 30
    y_te = X_te @ coef + 30
    y_te_poison = np.full_like(y_te, np.nan)
    k = np.arange(20)
    return X_tr, y_tr, X_cal, y_cal, X_te, y_te, y_te_poison, k


@pytest.mark.parametrize("cls,kwargs", [
    (CQRTargetCP, {"K_fit": 10}),
    (GPRTargetCP, {}),
    (GPRTransferCP, {"base_model": _LinModel()}),
])
def test_target_baselines_ignore_y_test_poison(lodo, cls, kwargs):
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, y_poison, k = lodo
    _ = y_poison
    cp = cls(alpha=0.10, K=20, seed=5, **kwargs)
    if cls is GPRTransferCP:
        cp.fit(X_te, y_te, k_indices=k)
    else:
        cp.fit(X_te, y_te, k_indices=k)
    lo, hi = cp.predict_intervals(X_te[20:])
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_jkplus_source_poison_and_source_only(lodo):
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, y_poison, k = lodo
    _ = k
    domains = np.array(["D1_uci"] * len(X_cal))
    cp = JKPlusSourceCP(
        _LinModel(), X_cal, y_cal, alpha=0.10, seed=4, max_calib_samples=20,
        held_out_domain="D4_mondal", calib_domains=domains, use_gpu=False,
    )
    cp.fit()
    assert cp.diagnostics["n_models_trained"] == 20
    _ = y_poison
    lo, hi = cp.predict_intervals(X_te)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

    with pytest.raises(AssertionError, match="LEAKAGE"):
        JKPlusSourceCP(
            _LinModel(), X_cal, y_cal, alpha=0.10, seed=4, max_calib_samples=20,
            held_out_domain="D4_mondal",
            calib_domains=np.array(["D4_mondal"] * len(X_cal)),
            use_gpu=False,
        )

def test_weighted_cp_v2_no_y_test_in_fit(lodo):
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, y_poison, _ = lodo
    _ = y_poison
    cp = WeightedCPv2(
        _LinModel(), X_tr, y_tr, X_cal, y_cal, X_te,
        alpha=0.10, seed=3, device="cpu",
    )
    cp.fit()
    lo, hi = cp.predict_intervals(X_te)
    assert np.all(np.isfinite(lo))


@pytest.mark.skipif(True, reason="TabPFN optional dependency")
def test_tabpfn_target_leakage(lodo):
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, y_poison, k = lodo
    _ = y_poison
    cp = TabPFNTargetCP(alpha=0.10, K=20, seed=2)
    cp.fit(X_te, y_te, k_indices=k)
    lo, hi = cp.predict_intervals(X_te[20:])
    assert np.all(np.isfinite(lo))

"""Alpha-sensitivity tests for conformal predictors."""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive_cp import AdaptiveJackknifeCP, AdaptiveSplitCP
from src.conformal import (
    CQR,
    JackknifeP,
    SplitConformal,
    TargetOnlyCP,
    TransferCalibratedCP,
)
from src.metrics import mean_interval_width


class _LinModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return 30 + X[:, 0] * 5


class _ConstSigma:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 5.0)


def _synthetic_pool(seed: int = 0, n: int = 80):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    y = 30 + X[:, 0] * 4 + rng.normal(0, 2, n)
    phys = [0, 1, 2, 3]
    return X, y, phys


def _mean_width(predictor, X, alphas: list[float]) -> dict[float, float]:
    widths = {}
    for a in alphas:
        lo, hi = predictor.predict_intervals(X, alpha=a)
        widths[a] = mean_interval_width(lo, hi)
    return widths


@pytest.mark.parametrize("name,factory", [
    ("SplitConformal", "split"),
    ("CQR", "cqr"),
    ("TargetOnlyCP", "toc"),
    ("AdaptiveSplitCP", "asc"),
    ("AdaptiveJackknifeCP", "ajk"),
    ("TransferCalibratedCP", "tc"),
])
def test_alpha_monotonicity(name: str, factory: str):
    """Higher alpha -> wider intervals: width(0.30) > width(0.20) > width(0.10)."""
    X, y, phys = _synthetic_pool(seed=11)
    alphas = [0.10, 0.20, 0.30]
    X_eval = X[50:]

    if factory == "split":
        pred = SplitConformal(_LinModel(), X[:40], y[:40], alpha=0.10)
        pred.fit()
    elif factory == "cqr":
        rng = np.random.default_rng(99)
        n_cal = 60
        X_cal = rng.standard_normal((n_cal, 8))
        y_cal = 30 + X_cal[:, 0] * 4 + rng.normal(0, 3, n_cal)
        pred = CQR(_LinModel(), X_cal, y_cal, alpha=0.10)
        pred.fit()
        X_eval = X[50:]
    elif factory == "toc":
        pred = TargetOnlyCP(alpha=0.10, K=20, seed=42)
        pred.fit(X, y, physical_indices=phys)
    elif factory == "asc":
        pred = AdaptiveSplitCP(_LinModel(), alpha=0.10, K=20, seed=42)
        pred.fit(X, y, physical_indices=phys)
    elif factory == "ajk":
        pred = AdaptiveJackknifeCP(_LinModel(), alpha=0.10, K=20, seed=42)
        pred.fit(X, y, physical_indices=phys)
    else:
        pred = TransferCalibratedCP(
            _LinModel(), _ConstSigma(), 1.0, alpha=0.10, K=20, seed=42,
        )
        pred.fit(X, y)

    widths = _mean_width(pred, X_eval, alphas)
    assert widths[0.30] < widths[0.20] < widths[0.10], (
        f"{name}: widths at alpha 0.10/0.20/0.30 = "
        f"{widths[0.10]:.3f}/{widths[0.20]:.3f}/{widths[0.30]:.3f}"
    )


@pytest.mark.parametrize("factory", ["toc", "asc", "tc"])
def test_no_cached_quantile(factory: str):
    """predict_intervals with different alphas must yield different outputs."""
    X, y, phys = _synthetic_pool(seed=22)
    X_eval = X[[0, 1, 2]]

    if factory == "toc":
        pred = TargetOnlyCP(alpha=0.10, K=20, seed=42)
        pred.fit(X, y, physical_indices=phys)
    elif factory == "asc":
        pred = AdaptiveSplitCP(_LinModel(), alpha=0.10, K=20, seed=42)
        pred.fit(X, y, physical_indices=phys)
    else:
        pred = TransferCalibratedCP(
            _LinModel(), _ConstSigma(), 1.0, alpha=0.10, K=20, seed=42,
        )
        pred.fit(X, y)

    lo_a, hi_a = pred.predict_intervals(X_eval, alpha=0.10)
    lo_b, hi_b = pred.predict_intervals(X_eval, alpha=0.20)
    assert not (np.allclose(lo_a, lo_b) and np.allclose(hi_a, hi_b)), (
        f"{factory}: intervals identical across alpha=0.10 vs 0.20"
    )


def test_method_outputs_distinct():
    """TargetOnlyCP and AdaptiveSplitCP must not produce identical intervals at K=20."""
    X, y, phys = _synthetic_pool(seed=33)
    X_eval = X[50:60]

    toc = TargetOnlyCP(alpha=0.10, K=20, seed=42)
    toc.fit(X, y, physical_indices=phys)
    asc = AdaptiveSplitCP(_LinModel(), alpha=0.10, K=20, seed=42)
    asc.fit(X, y, physical_indices=phys)

    lo_t, hi_t = toc.predict_intervals(X_eval, alpha=0.10)
    lo_a, hi_a = asc.predict_intervals(X_eval, alpha=0.10)

    assert not (np.allclose(lo_t, lo_a) and np.allclose(hi_t, hi_a)), (
        "TargetOnlyCP and AdaptiveSplitCP produced identical intervals at K=20"
    )


def test_k10_coincidence_documented():
    """At K=10 local-ridge collapse may make intervals identical: frozen explanation."""
    X, y, phys = _synthetic_pool(seed=99)
    X_eval = X[50:55]
    toc = TargetOnlyCP(alpha=0.10, K=10, seed=42)
    toc.fit(X, y, physical_indices=phys)
    asc = AdaptiveSplitCP(_LinModel(), alpha=0.10, K=10, seed=42)
    asc.fit(X, y, physical_indices=phys)
    lo_t, hi_t = toc.predict_intervals(X_eval, alpha=0.10)
    lo_a, hi_a = asc.predict_intervals(X_eval, alpha=0.10)
    assert np.allclose(lo_t, lo_a) and np.allclose(hi_t, hi_a)


def test_jackknife_alpha_monotonicity():
    """Jackknife+ recomputes quantile per alpha (small subsample for speed)."""
    rng = np.random.default_rng(44)
    n = 25
    X = rng.standard_normal((n, 5))
    y = 30 + X[:, 0] * 3 + rng.normal(0, 1, n)
    jkp = JackknifeP(_LinModel(), X, y, alpha=0.10, max_calib_samples=20, use_gpu=False, seed=1)
    jkp.fit()
    X_eval = X[:5]
    widths = _mean_width(jkp, X_eval, [0.10, 0.20, 0.30])
    assert widths[0.30] < widths[0.20] < widths[0.10]

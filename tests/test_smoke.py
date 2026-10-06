"""Smoke tests for ShiftSafe-CP pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.conformal import SplitConformal, ShiftSafeCP, conformal_quantile
from src.datasets import generate_derived_features, create_lodo_splits
from src.metrics import empirical_coverage, mean_interval_width, rmse


class DummyModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 40.0)


def _make_synthetic_dataset(n: int = 50, domain: str = "D1") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "cement": rng.uniform(200, 500, n),
        "slag": rng.uniform(0, 100, n),
        "fly_ash": rng.uniform(0, 150, n),
        "water": rng.uniform(120, 220, n),
        "superplasticizer": rng.uniform(0, 30, n),
        "coarse_agg": rng.uniform(800, 1100, n),
        "fine_agg": rng.uniform(600, 900, n),
        "age": rng.choice([7, 28, 90], n),
        "strength_mpa": rng.uniform(20, 60, n),
        "domain_id": domain,
    })
    return generate_derived_features(df)


def test_derived_features():
    df = _make_synthetic_dataset()
    assert "w_cm" in df.columns
    assert "f_cm" in df.columns
    assert "interaction_w_cm_f_cm" in df.columns
    assert len(df) > 0


def test_lodo_splits():
    d1 = _make_synthetic_dataset(40, "D1")
    d2 = _make_synthetic_dataset(30, "D2")
    splits = create_lodo_splits({"D1": d1, "D2": d2}, seed=42)
    assert "D1" in splits
    assert "D2" in splits
    train, calib, test = splits["D1"]
    assert len(test) == len(d1)
    assert len(train) + len(calib) > 0


def test_conformal_quantile():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = conformal_quantile(scores, alpha=0.2)
    assert q >= 1.0


def test_split_conformal_intervals():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 10 + 30 + rng.normal(0, 1, n)
    model = DummyModel()

    cp = SplitConformal(model, X[:40], y[:40], alpha=0.1)
    cp.fit()
    lower, upper = cp.predict_intervals(X[40:], alpha=0.1)

    assert len(lower) == 20
    assert np.all(lower < upper)
    cov = empirical_coverage(y[40:], lower, upper)
    assert 0.0 <= cov <= 1.0


def test_shiftsafe_cp_fit_predict():
    rng = np.random.default_rng(1)
    n = 80
    X = rng.standard_normal((n, 6))
    y = X[:, 0] * 5 + 35 + rng.normal(0, 2, n)
    model = DummyModel()

    X_train, y_train = X[:30], y[:30]
    X_calib, y_calib = X[30:50], y[30:50]
    X_test = X[50:]

    ss = ShiftSafeCP(
        model, X_calib, y_calib,
        X_source_pool=np.vstack([X_train, X_calib]),
        X_target_unlabeled=X_test,
        layers=[1, 3], device="cpu", seed=42,
    )
    ss.fit()
    lower, upper = ss.predict_intervals(X_test, alpha=0.1)
    assert len(lower) == 30
    assert np.all(lower < upper)
    assert "domain_auc" in ss.diagnostics
    assert "ess" in ss.diagnostics


def test_metrics_bounds():
    y = np.array([30.0, 35.0, 40.0])
    lo = np.array([25.0, 30.0, 35.0])
    hi = np.array([35.0, 40.0, 45.0])
    assert 0.0 <= empirical_coverage(y, lo, hi) <= 1.0
    assert mean_interval_width(lo, hi) > 0
    assert rmse(y, y) == 0.0

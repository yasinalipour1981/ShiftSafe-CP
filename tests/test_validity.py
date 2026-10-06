"""Tests for finite-sample validity regions."""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive_cp import AdaptiveSplitCP
from src.conformal import MinKViolation, TargetOnlyCP, quantile_from_calib_scores
from src.validity import annotate_guaranteed, is_guaranteed, min_k_split_5050


def test_is_guaranteed_split_k10_alpha010():
    assert not is_guaranteed("TargetOnlyCP", 10, 0.10)
    assert not is_guaranteed("AdaptiveSplitCP", 15, 0.10)
    assert is_guaranteed("TargetOnlyCP", 20, 0.10)
    assert is_guaranteed("AdaptiveJackknifeCP", 10, 0.10)


def test_min_k_split_5050_alpha010():
    assert min_k_split_5050(0.10) == 18


def test_quantile_raises_min_k_violation():
    scores = np.array([1.0, 2.0, 3.0])
    with pytest.raises(MinKViolation):
        quantile_from_calib_scores(scores, alpha=0.05, K_cal=3, context="test")


def test_annotate_guaranteed_column():
    import pandas as pd

    df = pd.DataFrame([
        {"method": "TargetOnlyCP", "K": 10, "alpha": 0.10},
        {"method": "TargetOnlyCP", "K": 20, "alpha": 0.10},
    ])
    out = annotate_guaranteed(df)
    assert "guaranteed" in out.columns
    assert out.iloc[0]["guaranteed"] is np.bool_(False) or out.iloc[0]["guaranteed"] is False
    assert bool(out.iloc[1]["guaranteed"])


def test_k10_targetonly_adaptive_split_coincide():
    """Document degenerate K=10 ridge-collapse: identical intervals."""
    rng = np.random.default_rng(99)
    n = 80
    X = rng.standard_normal((n, 8))
    y = 30 + X[:, 0] * 4 + rng.normal(0, 2, n)
    phys = [0, 1, 2, 3]
    X_eval = X[50:55]

    class _Lin:
        def predict(self, Xa):
            return 30 + Xa[:, 0] * 5

    toc = TargetOnlyCP(alpha=0.10, K=10, seed=42)
    toc.fit(X, y, physical_indices=phys)
    asc = AdaptiveSplitCP(_Lin(), alpha=0.10, K=10, seed=42)
    asc.fit(X, y, physical_indices=phys)

    lo_t, hi_t = toc.predict_intervals(X_eval, alpha=0.10)
    lo_a, hi_a = asc.predict_intervals(X_eval, alpha=0.10)
    assert np.allclose(lo_t, lo_a) and np.allclose(hi_t, hi_a)


def test_k20_targetonly_adaptive_split_distinct():
    X, y, phys = _synthetic_pool_k20()
    X_eval = X[50:60]

    class _Lin:
        def predict(self, Xa):
            return 30 + Xa[:, 0] * 5

    toc = TargetOnlyCP(alpha=0.10, K=20, seed=42)
    toc.fit(X, y, physical_indices=phys)
    asc = AdaptiveSplitCP(_Lin(), alpha=0.10, K=20, seed=42)
    asc.fit(X, y, physical_indices=phys)

    lo_t, hi_t = toc.predict_intervals(X_eval, alpha=0.10)
    lo_a, hi_a = asc.predict_intervals(X_eval, alpha=0.10)
    assert not (np.allclose(lo_t, lo_a) and np.allclose(hi_t, hi_a))


def _synthetic_pool_k20():
    rng = np.random.default_rng(33)
    X = rng.standard_normal((80, 8))
    y = 30 + X[:, 0] * 4 + rng.normal(0, 2, 80)
    return X, y, [0, 1, 2, 3]

"""Leakage guard: ShiftSafe-CP must never touch target-domain labels."""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal import ShiftSafeCP


class _LinearModel:
    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef = coef
        self.intercept = intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept


@pytest.fixture
def lodo_fixture():
    rng = np.random.default_rng(7)
    n_tr, n_cal, n_te, d = 120, 60, 40, 8

    X_tr = rng.standard_normal((n_tr, d))
    X_cal = rng.standard_normal((n_cal, d))
    # Shifted target domain
    X_te = rng.standard_normal((n_te, d)) + 0.8

    coef = rng.uniform(1, 3, d)
    y_tr = X_tr @ coef + 30 + rng.normal(0, 2, n_tr)
    y_cal = X_cal @ coef + 30 + rng.normal(0, 2, n_cal)
    y_te = X_te @ coef + 30 + rng.normal(0, 2, n_te)

    base = _LinearModel(coef, 30.0)
    f_idx = 0
    return X_tr, y_tr, X_cal, y_cal, X_te, y_te, base, f_idx


def test_shiftsafe_never_touches_target_labels(lodo_fixture):
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, base, f_idx = lodo_fixture
    # If any fit/calibration path reads y_test, NaNs will propagate and break
    y_te_poisoned = np.full_like(y_te, np.nan)
    _ = y_te_poisoned  # available but must never be passed anywhere

    m = ShiftSafeCP(
        base, X_cal, y_cal,
        np.vstack([X_tr, X_cal]), X_te, f_cm_col_index=f_idx,
        device="cpu",
    )
    m.fit()
    lo, hi = m.predict_intervals(X_te, alpha=0.10)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_shiftsafe_rejects_label_vector_as_target(lodo_fixture):
    X_tr, y_tr, X_cal, y_cal, X_te, y_te, base, f_idx = lodo_fixture
    # Passing a 1-D label array where covariates belong must fail loudly
    with pytest.raises(AssertionError):
        ShiftSafeCP(
            base, X_cal, y_cal,
            np.vstack([X_tr, X_cal]), y_te, f_cm_col_index=f_idx,
            device="cpu",
        )

"""Conformal prediction methods including ShiftSafe-CP v2.

ShiftSafe-CP v2, corrected three-layer conformal predictor.

Layer 1: Density-ratio weighting, source vs UNLABELED TARGET (Tibshirani et al. 2019)
Layer 2: Mondrian per-point group lookup with min-count fallback
Layer 3: Cross-fitted heteroscedastic scale sigma_hat(x)

ASSUMPTION (paper §3.2): ShiftSafe-CP requires access to the unlabeled
covariates X of the target domain (the mix designs, NOT the strengths).
This is standard in weighted conformal prediction and realistic: a lab
knows its mix proportions before casting cubes. No target labels (y_test)
may ever enter any fit/calibration path.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.base_models import clone_lgb_model

logger = logging.getLogger(__name__)


class MinKViolation(ValueError):
    """Raised when conformal rank exceeds K_cal (no finite-sample guarantee)."""


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conservative conformal quantile: rank ceil((n+1)(1-alpha))."""
    n = len(scores)
    if n == 0:
        return 0.0
    q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_idx = min(max(q_idx, 0), n - 1)
    return float(np.sort(scores)[q_idx])


def quantile_from_calib_scores(
    scores: np.ndarray,
    alpha: float,
    K_cal: int,
    *,
    context: str = "",
) -> float:
    """Recompute conformal quantile at predict time; never cache across alphas."""
    min_k = min_k_for_alpha(alpha)
    rank = int(np.ceil((K_cal + 1) * (1 - alpha)))
    if rank > K_cal:
        msg = f"K_cal={K_cal} too small for alpha={alpha}: need K_cal >= {min_k}"
        logger.warning("Skip %s: %s", context or "predict_intervals", msg)
        raise MinKViolation(msg)
    sorted_scores = np.sort(scores)
    q_idx = min(max(rank - 1, 0), len(sorted_scores) - 1)
    return float(sorted_scores[q_idx])


PHYSICAL_SUBSPACE_COLS = ["w_cm", "f_cm", "age", "binder_total"]


def min_k_for_alpha(alpha: float) -> int:
    """Minimum labeled target points for finite split-conformal intervals."""
    import math
    return int(math.ceil(1.0 / alpha) - 1)


def split_k_fit_cal_indices(
    all_idx: np.ndarray,
    K: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """50/50 split of K labeled target indices into fit vs calibration sets.

    K_cal >= min_k_for_alpha(alpha) is enforced. Returns (fit_idx, cal_idx, K_fit, K_cal).
    """
    min_k = min_k_for_alpha(alpha)
    K_cal = max(min_k, K // 2)
    K_fit = K - K_cal
    if K_fit < 1:
        raise ValueError(
            f"K={K} too small for fit/cal split at alpha={alpha}: "
            f"need K >= {min_k + 1}, got {K}"
        )
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(K)
    fit_idx = all_idx[perm[:K_fit]]
    cal_idx = all_idx[perm[K_fit:]]
    assert len(cal_idx) >= min_k, f"K_cal={len(cal_idx)} < min_k={min_k}"
    return fit_idx, cal_idx, K_fit, K_cal


def physical_feature_indices(feature_cols: list[str]) -> list[int]:
    """Column indices for density-ratio physical subspace."""
    return [feature_cols.index(c) for c in PHYSICAL_SUBSPACE_COLS if c in feature_cols]


def stratified_sample_k_indices(
    y: np.ndarray, K: int, seed: int = 42, n_quartiles: int = 4
) -> np.ndarray:
    """Stratified random sample of K indices by strength quartile."""
    n = len(y)
    if K > n:
        raise ValueError(f"K={K} exceeds target pool size n={n}")
    rng = np.random.default_rng(seed)
    quartiles = np.quantile(y, np.linspace(0, 1, n_quartiles + 1))
    quartiles[0] -= 1e-6
    strata = np.digitize(y, quartiles[1:-1])
    selected: list[int] = []
    per_stratum = max(1, K // n_quartiles)
    remainder = K
    for s in range(n_quartiles):
        pool = np.where(strata == s)[0]
        if len(pool) == 0:
            continue
        take = min(len(pool), per_stratum if s < n_quartiles - 1 else remainder)
        if take > 0:
            chosen = rng.choice(pool, size=take, replace=False)
            selected.extend(chosen.tolist())
            remainder -= take
    if len(selected) < K:
        remaining = list(set(range(n)) - set(selected))
        extra = rng.choice(remaining, size=K - len(selected), replace=False)
        selected.extend(extra.tolist())
    return np.array(selected[:K], dtype=int)


def fit_source_sigma_predictor(
    base_model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    sigma_cv_folds: int = 5,
    seed: int = 42,
    use_gpu: bool = True,
) -> tuple[Any, float]:
    """
    Cross-fitted sigma model on SOURCE calibration residuals.
    Returns (lgb Booster, sigma_floor).
    """
    n = len(X_calib)
    y_pred = base_model.predict(X_calib)
    abs_resid = np.abs(y_calib - y_pred)

    sigma_oof = np.zeros(n)
    kf = KFold(n_splits=sigma_cv_folds, shuffle=True, random_state=seed)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "seed": seed,
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    if use_gpu:
        try:
            lgb.train(
                {**lgb_params, "device_type": "gpu"},
                lgb.Dataset(X_calib[:8], label=abs_resid[:8]),
                num_boost_round=1,
            )
            lgb_params["device_type"] = "gpu"
        except Exception:
            pass

    for tr_idx, te_idx in kf.split(X_calib):
        m = lgb.train(
            lgb_params,
            lgb.Dataset(X_calib[tr_idx], label=abs_resid[tr_idx]),
            num_boost_round=300,
        )
        sigma_oof[te_idx] = m.predict(X_calib[te_idx])

    sigma_floor = max(0.25, float(np.quantile(abs_resid, 0.05)))
    sigma_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_calib, label=abs_resid),
        num_boost_round=300,
    )
    return sigma_model, sigma_floor


def _sigma_predict(sigma_model: Any, sigma_floor: float, X: np.ndarray) -> np.ndarray:
    return np.clip(sigma_model.predict(X), sigma_floor, None)


def _domain_classifier_auc(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int = 42,
) -> tuple[float, LogisticRegression, list[int]]:
    """Train domain classifier; return AUC and top-5 |coef| feature indices."""
    X_dom = np.vstack([X_source, X_target])
    y_dom = np.hstack([np.zeros(len(X_source)), np.ones(len(X_target))])
    scaler = StandardScaler().fit(X_dom)
    X_s = scaler.transform(X_dom)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(clf, X_s, y_dom, cv=kf, method="predict_proba")[:, 1]
    clf.fit(X_s, y_dom)
    auc = float(roc_auc_score(y_dom, oof))
    top5 = np.argsort(np.abs(clf.coef_[0]))[::-1][:5].tolist()
    return auc, clf, top5


class ConformalPredictor(ABC):
    """Abstract base for conformal methods."""

    def __init__(
        self,
        base_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float = 0.1,
    ):
        self.base_model = base_model
        self.X_calib = X_calib
        self.y_calib = y_calib
        self.alpha = alpha
        self.fitted = False
        self.scores_calib: np.ndarray = np.array([])

    @abstractmethod
    def fit(self) -> None: ...

    @abstractmethod
    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.base_model.predict(X_test)


class SplitConformal(ConformalPredictor):
    """Standard split conformal prediction."""

    def fit(self) -> None:
        y_pred = self.base_model.predict(self.X_calib)
        self.scores_calib = np.abs(self.y_calib - y_pred)
        self.fitted = True
        logger.info("Split Conformal fitted on %d calib samples", len(self.scores_calib))

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            self.fit()
        alpha = alpha if alpha is not None else self.alpha
        y_pred = self.base_model.predict(X_test)
        q = conformal_quantile(self.scores_calib, alpha)
        return y_pred - q, y_pred + q


class JackknifeP(ConformalPredictor):
    """True Jackknife+ (Barber et al. 2021) with LOO LightGBM refits.

    Fit trains n LOO models and stores them. Predict builds per-test-point
    intervals from {μ_{-i}(x) ± R_i^{LOO}}, NOT base_model ± residual quantile.
    """

    def __init__(
        self,
        base_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float = 0.1,
        max_calib_samples: int = 500,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        super().__init__(base_model, X_calib, y_calib, alpha)
        self.max_calib_samples = max_calib_samples
        self.use_gpu = use_gpu
        self.seed = seed
        self.loo_preds: np.ndarray = np.array([])
        self.loo_models: list[Any] = []
        self.n_models_trained: int = 0
        self._subsample_idx: np.ndarray | None = None

    def fit(self) -> None:
        n = len(self.X_calib)
        if n > self.max_calib_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, self.max_calib_samples, replace=False)
            self._subsample_idx = idx
            X_cal = self.X_calib[idx]
            y_cal = self.y_calib[idx]
            logger.warning(
                "Jackknife+ subsampled calib from %d to %d (source-only pool)",
                n,
                self.max_calib_samples,
            )
        else:
            self._subsample_idx = np.arange(n)
            X_cal, y_cal = self.X_calib, self.y_calib

        n = len(X_cal)
        self.loo_preds = np.zeros(n)
        self.loo_models = []
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbose": -1,
            "seed": self.seed,
        }
        if self.use_gpu:
            lgb_params["device_type"] = "gpu"

        logger.info("Jackknife+: training %d LOO models (genuine refits)...", n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = clone_lgb_model(
                lgb_params, X_cal[mask], y_cal[mask], num_boost_round=200, use_gpu=self.use_gpu
            )
            self.loo_models.append(model)
            self.loo_preds[i] = model.predict(X_cal[i : i + 1])[0]

        self.n_models_trained = len(self.loo_models)
        if self.n_models_trained != n:
            raise RuntimeError(
                f"Jackknife+ LOO incomplete: trained {self.n_models_trained} != n={n}"
            )

        self.scores_calib = np.abs(y_cal - self.loo_preds)
        self.X_calib = X_cal
        self.y_calib = y_cal
        self.fitted = True
        logger.info(
            "Jackknife+ fitted: n_models_trained=%d mean_R=%.3f",
            self.n_models_trained,
            float(np.mean(self.scores_calib)),
        )

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            self.fit()
        if not self.loo_models:
            raise RuntimeError(
                "Jackknife+ has no stored LOO models, refusing SplitCP-style fallback"
            )
        alpha = alpha if alpha is not None else self.alpha
        X_test = np.asarray(X_test, dtype=np.float64)
        n = len(self.loo_models)
        R = self.scores_calib
        # Barber et al. 2021:
        # lower = floor(α(n+1))-th smallest of {μ_{-i}(x) - R_i}  (−∞ if floor=0)
        # upper = ceil((1−α)(n+1))-th smallest of {μ_{-i}(x) + R_i}
        lo_k = int(np.floor(alpha * (n + 1)))  # 1-indexed count; 0 => −∞
        hi_k = int(np.ceil((1.0 - alpha) * (n + 1)))  # 1-indexed
        hi_rank = min(hi_k - 1, n - 1)

        mu_loo = np.vstack([m.predict(X_test) for m in self.loo_models])
        lower_cands = np.sort(mu_loo - R[:, None], axis=0)
        upper_cands = np.sort(mu_loo + R[:, None], axis=0)
        if lo_k <= 0:
            lower = np.full(X_test.shape[0], -np.inf)
        else:
            lower = lower_cands[lo_k - 1]
        upper = upper_cands[hi_rank]
        return lower.astype(np.float64), upper.astype(np.float64)

    def release_loo_models(self) -> None:
        """Free LOO boosters after intervals are computed (memory hygiene)."""
        self.loo_models = []
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass



class CQR(ConformalPredictor):
    """Conformalized Quantile Regression (Romano et al. 2019)."""

    def __init__(
        self,
        base_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float = 0.1,
        quantiles: tuple[float, float] = (0.05, 0.95),
    ):
        super().__init__(base_model, X_calib, y_calib, alpha)
        self.quantiles = quantiles
        self.q_lower: QuantileRegressor | None = None
        self.q_upper: QuantileRegressor | None = None
        self.calib_scores: np.ndarray = np.array([])

    def fit(self) -> None:
        q_lo, q_hi = self.quantiles
        self.q_lower = QuantileRegressor(quantile=q_lo, alpha=0.0, solver="highs")
        self.q_upper = QuantileRegressor(quantile=q_hi, alpha=0.0, solver="highs")
        self.q_lower.fit(self.X_calib, self.y_calib)
        self.q_upper.fit(self.X_calib, self.y_calib)

        lo = self.q_lower.predict(self.X_calib)
        hi = self.q_upper.predict(self.X_calib)
        self.calib_scores = np.maximum(lo - self.y_calib, self.y_calib - hi)
        self.fitted = True
        logger.info(
            "CQR fitted; n_calib=%d", len(self.calib_scores),
        )

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            self.fit()
        alpha = alpha if alpha is not None else self.alpha
        assert self.q_lower is not None and self.q_upper is not None
        correction = conformal_quantile(self.calib_scores, alpha)
        lo = self.q_lower.predict(X_test) - correction
        hi = self.q_upper.predict(X_test) + correction
        return lo, hi


class ShiftSafeCP(ConformalPredictor):
    """
    Corrected ShiftSafe-CP (v2).

    Layer 1 (FIX B1): density ratio w(x) = p_target(x)/p_source(x) trained on
        SOURCE POOL (train+calib) vs UNLABELED TARGET covariates, not
        train-vs-calib, which made the ratio ~1 everywhere.
    Layer 2 (FIX B4): Mondrian cells with PER-POINT lookup and effective-mass
        fallback to the global quantile, not averaged across groups.
    Layer 3 (FIX B3): cross-fitted sigma_hat(x); calibration scores use only
        out-of-fold sigma predictions.
    FIX B2: proper per-test-point weighted quantile (Tibshirani et al. 2019)
        including the +infinity mass term w_test[j], vectorized in torch.

    Parameters
    ----------
    base_model : fitted regressor with .predict(X) -> np.ndarray
    X_calib, y_calib : calibration data (from SOURCE domains)
    X_source_pool : full source covariates (train + calib), used ONLY for
                    density-ratio estimation
    X_target_unlabeled : covariates of the held-out target domain.
                         MUST NOT contain or be derived from y_test.
    weight_clip : (low, high) clip for density ratios. Default (0.05, 20).
                  DO NOT tune this to fix coverage; only widen if ESS
                  diagnostics show collapse.
    sigma_cv_folds : K for cross-fitted sigma_hat. Default 5.
    mondrian_min_count : min effective calib mass per cell before falling
                         back to the global quantile. Default 30.
    layers : subset of [1, 2, 3] for ablations. Default all three.
    f_cm_col_index : explicit column index of the f_cm feature; no guessing.
    """

    STRENGTH_EDGES = np.array([-np.inf, 25.0, 40.0, 55.0, np.inf])

    def __init__(
        self,
        base_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        X_source_pool: np.ndarray,
        X_target_unlabeled: np.ndarray,
        weight_clip: tuple[float, float] = (0.05, 20.0),
        sigma_cv_folds: int = 5,
        mondrian_min_count: int = 30,
        alpha: float = 0.10,
        device: str = "cuda",
        seed: int = 42,
        f_cm_col_index: int | None = None,
        layers: list[int] | None = None,
        dr_feature_indices: list[int] | None = None,
    ):
        super().__init__(
            base_model,
            np.asarray(X_calib, dtype=np.float64),
            np.asarray(y_calib, dtype=np.float64),
            alpha,
        )
        self.X_source_pool = np.asarray(X_source_pool, dtype=np.float64)
        self.X_target = np.asarray(X_target_unlabeled, dtype=np.float64)
        # Leakage guard: target input must be a 2-D covariate matrix with the
        # same feature dimension as calibration data: never a label vector.
        assert self.X_target.ndim == 2 and self.X_target.shape[1] == self.X_calib.shape[1], (
            "X_target_unlabeled must be a 2-D covariate matrix matching X_calib "
            "features (never labels / y_test)"
        )
        self.weight_clip = weight_clip
        self.sigma_cv_folds = sigma_cv_folds
        self.mondrian_min_count = mondrian_min_count
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.f_cm_col_index = f_cm_col_index
        self.layers = layers if layers is not None else [1, 2, 3]
        self.dr_feature_indices = dr_feature_indices
        self.diagnostics: dict[str, Any] = {}

    def _select_dr_features(self, X: np.ndarray) -> np.ndarray:
        if self.dr_feature_indices is not None:
            return X[:, self.dr_feature_indices]
        return X

    # ------------------------------------------------------------------ #
    # LAYER 1 (FIX B1): density ratio  w(x) = p_target(x) / p_source(x)  #
    # ------------------------------------------------------------------ #
    def _fit_density_ratio(self) -> None:
        """
        Domain classifier on source vs unlabeled target.
        Uses physical subspace when dr_feature_indices is set (v3 fix).
        """
        if self.dr_feature_indices is not None:
            auc_all, _, _ = _domain_classifier_auc(
                self.X_source_pool, self.X_target, seed=self.seed
            )
            self.diagnostics["domain_auc_all_features"] = auc_all

        X_src = self._select_dr_features(self.X_source_pool)
        X_tgt = self._select_dr_features(self.X_target)
        X_dom = np.vstack([X_src, X_tgt])
        y_dom = np.hstack([np.zeros(len(X_src)), np.ones(len(X_tgt))])

        self._dr_scaler = StandardScaler().fit(X_dom)
        X_dom_s = self._dr_scaler.transform(X_dom)

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_prob = cross_val_predict(
            clf, X_dom_s, y_dom, cv=kf, method="predict_proba"
        )[:, 1]
        self._dr_clf = clf.fit(X_dom_s, y_dom)
        self._dr_iso = IsotonicRegression(
            out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4
        ).fit(oof_prob, y_dom)

        self.diagnostics["domain_auc"] = float(roc_auc_score(y_dom, oof_prob))
        if self.dr_feature_indices is not None:
            logger.info(
                "  [L1] AUC all-features=%.3f -> physical-subspace=%.3f",
                self.diagnostics.get("domain_auc_all_features", float("nan")),
                self.diagnostics["domain_auc"],
            )
        else:
            logger.info(
                "  [L1] domain-classifier AUC (source vs target) = %.3f",
                self.diagnostics["domain_auc"],
            )

    def _density_ratio(self, X: np.ndarray) -> np.ndarray:
        """Return clipped w(x) for rows of X (ones if Layer 1 disabled)."""
        if 1 not in self.layers:
            return np.ones(len(X))
        X_dr = self._select_dr_features(X)
        p = self._dr_clf.predict_proba(self._dr_scaler.transform(X_dr))[:, 1]
        p = self._dr_iso.predict(p)
        w = p / (1.0 - p)
        return np.clip(w, self.weight_clip[0], self.weight_clip[1])

    # ------------------------------------------------------------------ #
    # LAYER 3 (FIX B3): cross-fitted heteroscedastic scale sigma_hat(x)  #
    # ------------------------------------------------------------------ #
    def _fit_sigma_crossfit(self) -> None:
        """
        K-fold cross-fitting on the calibration set: for each fold k, fit the
        sigma-model on calib \\ fold_k and predict on fold_k. Calibration
        scores use ONLY out-of-fold sigma predictions. A final sigma model on
        ALL calib data is kept for test-time prediction.
        """
        n = len(self.X_calib)
        y_pred = self.base_model.predict(self.X_calib)
        abs_resid = np.abs(self.y_calib - y_pred)

        sigma_oof = np.zeros(n)
        kf = KFold(n_splits=self.sigma_cv_folds, shuffle=True, random_state=self.seed)
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "seed": self.seed,
            "num_leaves": 31,
            "learning_rate": 0.05,
        }
        # GPU if available; silently fall back to CPU if GPU build absent
        try:
            lgb.train(
                {**lgb_params, "device_type": "gpu"},
                lgb.Dataset(self.X_calib[:8], label=abs_resid[:8]),
                num_boost_round=1,
            )
            lgb_params["device_type"] = "gpu"
        except Exception:
            logger.warning("  [L3] LightGBM GPU unavailable for sigma model; CPU fallback")

        for tr_idx, te_idx in kf.split(self.X_calib):
            m = lgb.train(
                lgb_params,
                lgb.Dataset(self.X_calib[tr_idx], label=abs_resid[tr_idx]),
                num_boost_round=300,
            )
            sigma_oof[te_idx] = m.predict(self.X_calib[te_idx])

        # Floor sigma at a small quantile of |resid| to avoid divide-by-tiny
        self._sigma_floor = max(0.25, float(np.quantile(abs_resid, 0.05)))
        self._sigma_oof = np.clip(sigma_oof, self._sigma_floor, None)

        # Test-time sigma model on all calib data
        self._sigma_model = lgb.train(
            lgb_params,
            lgb.Dataset(self.X_calib, label=abs_resid),
            num_boost_round=300,
        )
        logger.info(
            "  [L3] cross-fitted sigma: floor=%.2f, median oof sigma=%.2f",
            self._sigma_floor,
            float(np.median(self._sigma_oof)),
        )

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        if 3 not in self.layers:
            return np.ones(len(X))
        return np.clip(self._sigma_model.predict(X), self._sigma_floor, None)

    # ------------------------------------------------------------------ #
    # LAYER 2 (FIX B4): Mondrian cells with per-point lookup             #
    # ------------------------------------------------------------------ #
    def _cell_id(self, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Cell = (predicted-strength bin) x (fly-ash present)."""
        s_bin = np.digitize(y_pred, self.STRENGTH_EDGES[1:-1])  # 0..3
        if self.f_cm_col_index is not None:
            fa = (X[:, self.f_cm_col_index] > 0.01).astype(int)
        else:
            fa = np.zeros(len(X), dtype=int)  # degrade gracefully, log it
            logger.warning(
                "  [L2] f_cm_col_index not provided; Mondrian uses strength bins only"
            )
        return s_bin * 2 + fa  # integer cell id 0..7

    # ------------------------------------------------------------------ #
    # FIT                                                                #
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        logger.info("Fitting ShiftSafe-CP v2 (layers=%s)...", self.layers)
        if 1 in self.layers:
            self._fit_density_ratio()
        if 3 in self.layers:
            self._fit_sigma_crossfit()

        # Nonconformity scores with OUT-OF-FOLD sigma (fix B3)
        y_pred_calib = self.base_model.predict(self.X_calib)
        abs_resid = np.abs(self.y_calib - y_pred_calib)
        if 3 in self.layers:
            self._scores = abs_resid / self._sigma_oof
        else:
            self._scores = abs_resid
        self.scores_calib = self._scores

        # Precompute calib weights and Mondrian cells
        self._w_calib = self._density_ratio(self.X_calib)
        self._cells_calib = self._cell_id(self.X_calib, y_pred_calib)

        # ESS diagnostic (weight degeneracy check)
        w = self._w_calib
        self.diagnostics["ess"] = float((w.sum() ** 2) / (w**2).sum())
        self.diagnostics["ess_fraction"] = self.diagnostics["ess"] / len(w)
        logger.info(
            "  [L1] ESS = %.1f (%.1f%% of %d calib points)",
            self.diagnostics["ess"],
            100 * self.diagnostics["ess_fraction"],
            len(w),
        )
        if self.diagnostics["ess_fraction"] < 0.10:
            logger.warning(
                "  [L1] ESS < 10%%, weights degenerate. Report this; "
                "do NOT silently retune clip range."
            )
        self.fitted = True
        logger.info("ShiftSafe-CP v2 fitted")

    # ------------------------------------------------------------------ #
    # WEIGHTED QUANTILE (FIX B2): Tibshirani et al. 2019, per test point #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _weighted_quantile_batch(
        scores_sorted: torch.Tensor,
        w_sorted_cum: torch.Tensor,
        w_calib_sum: torch.Tensor,
        w_test: torch.Tensor,
        level: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized per-test-point weighted quantile on GPU.

        scores_sorted : (m,) ascending calib scores        [torch, device]
        w_sorted_cum  : (m,) cumulative weights in score order [torch]
        w_calib_sum   : scalar, sum of calib weights
        w_test        : (n_test,) weights of test points   [torch]
        level         : 1 - alpha

        For test point j the +infinity mass is w_test[j]; the quantile is the
        smallest score s_(k) with
            w_sorted_cum[k] / (w_calib_sum + w_test[j]) >= level.
        If no k satisfies it, the quantile is +inf (interval = whole line);
        we return a large sentinel and flag it.
        """
        denom = w_calib_sum + w_test  # (n_test,)
        threshold = level * denom  # (n_test,)
        idx = torch.searchsorted(w_sorted_cum, threshold)  # (n_test,)
        inf_mask = idx >= len(scores_sorted)
        idx_clamped = torch.clamp(idx, max=len(scores_sorted) - 1)
        q = scores_sorted[idx_clamped]
        q = torch.where(
            inf_mask,
            torch.full_like(q, float(scores_sorted[-1] * 10)),
            q,
        )
        return q, inf_mask

    def predict_intervals(
        self,
        X_test: np.ndarray,
        alpha: float | None = None,
        return_diagnostics: bool = False,
    ):
        if alpha is None:
            alpha = self.alpha
        if not self.fitted:
            self.fit()
        level = 1.0 - alpha

        X_test = np.asarray(X_test, dtype=np.float64)
        y_pred = self.base_model.predict(X_test)
        sigma_t = self._sigma(X_test)
        w_test = self._density_ratio(X_test)
        cells_t = self._cell_id(X_test, y_pred)

        n_test = len(X_test)
        q_out = np.empty(n_test)
        inf_flags = np.zeros(n_test, dtype=bool)

        # ---- global (all-calib) weighted quantile, GPU, computed once ----
        order = np.argsort(self._scores)
        s_sorted = torch.tensor(self._scores[order], device=self.device)
        w_sorted = torch.tensor(self._w_calib[order], device=self.device)
        w_cum = torch.cumsum(w_sorted, dim=0)
        w_sum = w_cum[-1]
        w_test_t = torch.tensor(w_test, device=self.device)
        q_global, inf_global = self._weighted_quantile_batch(
            s_sorted, w_cum, w_sum, w_test_t, level
        )
        q_global = q_global.cpu().numpy()
        inf_global = inf_global.cpu().numpy()

        q_out[:] = q_global
        inf_flags[:] = inf_global

        # ---- per-cell quantiles where the cell has enough EFFECTIVE mass ----
        if 2 in self.layers:
            for cell in np.unique(cells_t):
                test_mask = cells_t == cell
                calib_mask = self._cells_calib == cell
                # effective count = ESS within cell, not raw count
                w_cell = self._w_calib[calib_mask]
                ess_cell = (
                    (w_cell.sum() ** 2) / ((w_cell**2).sum() + 1e-12)
                    if calib_mask.sum() > 0
                    else 0.0
                )
                if ess_cell >= self.mondrian_min_count:
                    o = np.argsort(self._scores[calib_mask])
                    s_c = torch.tensor(
                        self._scores[calib_mask][o], device=self.device
                    )
                    w_c = torch.tensor(w_cell[o], device=self.device)
                    w_c_cum = torch.cumsum(w_c, dim=0)
                    q_c, inf_c = self._weighted_quantile_batch(
                        s_c,
                        w_c_cum,
                        w_c_cum[-1],
                        torch.tensor(w_test[test_mask], device=self.device),
                        level,
                    )
                    q_out[test_mask] = q_c.cpu().numpy()
                    inf_flags[test_mask] = inf_c.cpu().numpy()
                # else: keep global quantile fallback (FIX B4)

        lower = y_pred - q_out * sigma_t
        upper = y_pred + q_out * sigma_t

        diag = {
            "pct_infinite_quantile": float(inf_flags.mean()),
            "mean_w_test": float(w_test.mean()),
            "cells_used": {
                int(c): int((cells_t == c).sum()) for c in np.unique(cells_t)
            },
        }
        self.diagnostics["last_predict"] = diag
        if diag["pct_infinite_quantile"] > 0.05:
            logger.warning(
                "  %.1f%% of test points hit the +inf quantile branch, shift is "
                "severe there; intervals are maximally wide.",
                100 * diag["pct_infinite_quantile"],
            )
        if return_diagnostics:
            return lower, upper, diag
        return lower, upper


class TransferCalibratedCP:
    """
    ShiftSafe-CP v3: source-informed efficiency, target-anchored validity.

    GUARANTEE: marginal coverage >= 1 - alpha on the target domain because
    calibration scores come exclusively from K labeled target samples.
    """

    def __init__(
        self,
        base_model: Any,
        sigma_model: Any,
        sigma_floor: float,
        alpha: float = 0.10,
        K: int = 20,
        fine_tune_mode: str = "none",
        seed: int = 42,
        sampler: Any | None = None,
    ):
        self.base_model = base_model
        self.sigma_model = sigma_model
        self.sigma_floor = sigma_floor
        self.alpha = alpha
        self.K = K
        self.fine_tune_mode = fine_tune_mode
        self.seed = seed
        # Optional `sampler(X_pool, y_pool, K, seed) -> indices`; defaults to the
        # originally submitted strength-quartile draw (2026-09 revision).
        self.sampler = sampler
        self.fitted = False
        self.calib_target_indices: np.ndarray = np.array([], dtype=int)
        self.calib_scores: np.ndarray = np.array([])
        self.residual_booster: Any | None = None
        self.diagnostics: dict[str, Any] = {}
        self._K_cal = 0
        self._affine_a: float | None = None
        self._affine_b: float | None = None
        self._sigma_scale_c: float | None = None
        self._sigma_local_const: float | None = None

    def _sigma_raw(self, X: np.ndarray) -> np.ndarray:
        return _sigma_predict(self.sigma_model, self.sigma_floor, X)

    def _sigma(self, X: np.ndarray) -> np.ndarray:
        if self._sigma_local_const is not None:
            return np.full(len(X), self._sigma_local_const)
        s = self._sigma_raw(X)
        if self._sigma_scale_c is not None:
            s = s * self._sigma_scale_c
        return np.maximum(s, self.sigma_floor)

    def _predict_mu_raw(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict(X)

    def _predict_mu(self, X: np.ndarray) -> np.ndarray:
        mu = self._predict_mu_raw(X)
        if self._affine_a is not None:
            mu = self._affine_a * mu + self._affine_b
        if self.residual_booster is not None:
            mu = mu + self.residual_booster.predict(X)
        return mu

    def _needs_k_split(self) -> bool:
        return self.fine_tune_mode in (
            "affine", "residual_boost", "affine_localsigma",
        )

    def _split_k_ft_cal(
        self, all_idx: np.ndarray, min_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        fit_idx, cal_idx, _, _ = split_k_fit_cal_indices(
            all_idx, self.K, self.alpha, self.seed,
        )
        return fit_idx, cal_idx

    def _bias_stats(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> dict[str, float]:
        resid = y - mu
        med_sigma = float(np.median(sigma))
        return {
            "mean_bias": float(np.mean(resid)),
            "median_abs_resid": float(np.median(np.abs(resid))),
            "median_sigma": med_sigma,
            "resid_sigma_ratio": float(np.median(np.abs(resid)) / max(med_sigma, 1e-8)),
        }

    def _fit_affine(self, X_ft: np.ndarray, y_ft: np.ndarray) -> None:
        mu_raw = self._predict_mu_raw(X_ft)
        design = np.column_stack([mu_raw, np.ones(len(mu_raw))])
        coef, _, _, _ = np.linalg.lstsq(design, y_ft, rcond=None)
        self._affine_a = float(coef[0])
        self._affine_b = float(coef[1])
        mu_prime = self._affine_a * mu_raw + self._affine_b
        resid_abs = np.abs(y_ft - mu_prime)
        sigma_raw = self._sigma_raw(X_ft)
        med_resid = float(np.median(resid_abs))
        if med_resid < 1e-8:
            resid_signed = y_ft - mu_prime
            std_resid = float(np.std(resid_signed))
            mad_resid = float(
                np.median(np.abs(resid_signed - np.median(resid_signed)))
            )
            med_resid = max(std_resid, mad_resid * 1.4826, self.sigma_floor)
        med_sigma_raw = float(np.median(sigma_raw))
        scale_denom = max(med_sigma_raw, self.sigma_floor)
        if self.fine_tune_mode == "affine_localsigma":
            self._sigma_local_const = max(med_resid, self.sigma_floor)
            self._sigma_scale_c = None
        else:
            self._sigma_local_const = None
            self._sigma_scale_c = max(
                med_resid / scale_denom,
                self.sigma_floor / scale_denom,
            )

    def _fit_residual_boost(self, X_ft: np.ndarray, y_ft: np.ndarray) -> None:
        mu_affine = self._predict_mu(X_ft)
        resid = y_ft - mu_affine
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "seed": self.seed,
            "num_leaves": 8,
            "max_depth": 3,
            "learning_rate": 0.05,
        }
        self.residual_booster = lgb.train(
            lgb_params,
            lgb.Dataset(X_ft, label=resid),
            num_boost_round=100,
        )

    def fit(self, X_target_pool: np.ndarray, y_target_pool: np.ndarray) -> None:
        X_target_pool = np.asarray(X_target_pool, dtype=np.float64)
        y_target_pool = np.asarray(y_target_pool, dtype=np.float64)
        n = len(y_target_pool)
        min_k = min_k_for_alpha(self.alpha)

        if self.K > n:
            raise ValueError(f"K={self.K} exceeds target pool n={n}")
        if not self._needs_k_split() and self.K < min_k:
            raise ValueError(
                f"K too small for alpha={self.alpha}: need K >= {min_k}, got {self.K}"
            )

        all_idx = (
            np.asarray(self.sampler(X_target_pool, y_target_pool, self.K, self.seed), dtype=int)
            if getattr(self, "sampler", None) is not None
            else stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        )
        self.calib_target_indices = all_idx
        X_k = X_target_pool[all_idx]
        y_k = y_target_pool[all_idx]

        mu_k_before = self._predict_mu_raw(X_k)
        sigma_k_before = self._sigma_raw(X_k)
        self.diagnostics["bias_before_affine"] = self._bias_stats(
            y_k, mu_k_before, sigma_k_before
        )

        if self._needs_k_split():
            ft_idx, cal_idx = self._split_k_ft_cal(all_idx, min_k)
            self._K_cal = len(cal_idx)
            X_ft, y_ft = X_target_pool[ft_idx], y_target_pool[ft_idx]
            self._fit_affine(X_ft, y_ft)
            if self.fine_tune_mode == "residual_boost":
                self._fit_residual_boost(X_ft, y_ft)
            mu_k_after = self._predict_mu(X_k)
            sigma_k_after = self._sigma(X_k)
            self.diagnostics["bias_after_affine"] = self._bias_stats(
                y_k, mu_k_after, sigma_k_after
            )
            self.diagnostics["affine_params"] = {
                "a": self._affine_a,
                "b": self._affine_b,
                "sigma_scale_c": self._sigma_scale_c,
                "sigma_local_const": self._sigma_local_const,
            }
            logger.info(
                "  [bias] before: mean=%.2f ratio=%.2f | after affine: mean=%.2f ratio=%.2f",
                self.diagnostics["bias_before_affine"]["mean_bias"],
                self.diagnostics["bias_before_affine"]["resid_sigma_ratio"],
                self.diagnostics["bias_after_affine"]["mean_bias"],
                self.diagnostics["bias_after_affine"]["resid_sigma_ratio"],
            )
        else:
            cal_idx = all_idx
            self._K_cal = len(cal_idx)
            self.diagnostics["bias_after_affine"] = self.diagnostics["bias_before_affine"]

        X_cal = X_target_pool[cal_idx]
        y_cal = y_target_pool[cal_idx]
        mu_cal = self._predict_mu(X_cal)
        sigma_cal = np.maximum(self._sigma(X_cal), self.sigma_floor)
        self.calib_scores = np.abs(y_cal - mu_cal) / sigma_cal

        q_fit = quantile_from_calib_scores(
            self.calib_scores, self.alpha, self._K_cal, context="TransferCalibratedCP.fit",
        )
        self.fitted = True
        self.diagnostics.update({
            "K": self.K,
            "K_cal": self._K_cal,
            "quantile_q": q_fit,
            "fine_tune_mode": self.fine_tune_mode,
        })
        logger.info(
            "TransferCalibratedCP fitted: K=%d K_cal=%d q=%.3f mode=%s",
            self.K, self._K_cal, q_fit, self.fine_tune_mode,
        )

    def predict_intervals(
        self,
        X_test: np.ndarray,
        alpha: float | None = None,
        return_diagnostics: bool = False,
    ):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_intervals")
        alpha = alpha if alpha is not None else self.alpha
        q = quantile_from_calib_scores(
            self.calib_scores, alpha, self._K_cal, context="TransferCalibratedCP",
        )

        X_test = np.asarray(X_test, dtype=np.float64)
        mu = self._predict_mu(X_test)
        sigma = self._sigma(X_test)
        lower = mu - q * sigma
        upper = mu + q * sigma
        diag = {"quantile_q": q, "K_cal": self._K_cal}
        if return_diagnostics:
            return lower, upper, diag
        return lower, upper

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._predict_mu(X_test)


class TargetOnlyCP:
    """
    Target-only split conformal on K labeled target points (width ablation).

    mu_hat and sigma are fit ONLY on K_fit; conformal scores use ONLY K_cal
    (50/50 split, K_cal >= ceil(1/alpha)-1).
    """

    PHYSICAL_COLS = PHYSICAL_SUBSPACE_COLS

    def __init__(
        self, alpha: float = 0.10, K: int = 20, seed: int = 42, sampler: Any | None = None,
    ):
        self.alpha = alpha
        self.K = K
        self.seed = seed
        self.sampler = sampler
        self.fitted = False
        self.calib_target_indices: np.ndarray = np.array([], dtype=int)
        self.calib_scores: np.ndarray = np.array([])
        self.sigma_const: float = 1.0
        self.diagnostics: dict[str, Any] = {}
        self._ridge: Any | None = None
        self._phys_indices: list[int] | None = None
        self._K_fit = 0
        self._K_cal = 0

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        physical_indices: list[int],
    ) -> None:
        from sklearn.linear_model import Ridge

        X_target_pool = np.asarray(X_target_pool, dtype=np.float64)
        y_target_pool = np.asarray(y_target_pool, dtype=np.float64)
        min_k = min_k_for_alpha(self.alpha)
        if self.K < min_k + 1:
            raise ValueError(
                f"K={self.K} too small for fit/cal split at alpha={self.alpha}: "
                f"need K >= {min_k + 1}"
            )

        all_idx = (
            np.asarray(self.sampler(X_target_pool, y_target_pool, self.K, self.seed), dtype=int)
            if getattr(self, "sampler", None) is not None
            else stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        )
        self.calib_target_indices = all_idx
        self._phys_indices = physical_indices

        fit_idx, cal_idx, self._K_fit, self._K_cal = split_k_fit_cal_indices(
            all_idx, self.K, self.alpha, self.seed,
        )

        X_ft = X_target_pool[fit_idx][:, physical_indices]
        y_ft = y_target_pool[fit_idx]
        self._ridge = Ridge(alpha=1.0).fit(X_ft, y_ft)
        resid_ft = np.abs(y_ft - self._ridge.predict(X_ft))
        self.sigma_const = max(0.25, float(np.median(resid_ft)))

        X_cal = X_target_pool[cal_idx][:, physical_indices]
        y_cal = y_target_pool[cal_idx]
        mu_cal = self._ridge.predict(X_cal)
        self.calib_scores = np.abs(y_cal - mu_cal) / self.sigma_const

        q_fit = quantile_from_calib_scores(
            self.calib_scores, self.alpha, self._K_cal, context="TargetOnlyCP.fit",
        )
        self.fitted = True
        self.diagnostics = {
            "K": self.K,
            "K_fit": self._K_fit,
            "K_cal": self._K_cal,
            "quantile_q": q_fit,
            "fit_indices": fit_idx.tolist(),
            "cal_indices": cal_idx.tolist(),
        }
        logger.info(
            "TargetOnlyCP fitted: K=%d K_fit=%d K_cal=%d q=%.3f",
            self.K, self._K_fit, self._K_cal, q_fit,
        )

    def predict_intervals(self, X_test: np.ndarray, alpha: float | None = None):
        if not self.fitted or self._ridge is None or self._phys_indices is None:
            raise RuntimeError("Call fit() before predict_intervals")
        alpha = alpha if alpha is not None else self.alpha
        q = quantile_from_calib_scores(
            self.calib_scores, alpha, self._K_cal, context="TargetOnlyCP",
        )
        X_phys = np.asarray(X_test, dtype=np.float64)[:, self._phys_indices]
        mu = self._ridge.predict(X_phys)
        lower = mu - q * self.sigma_const
        upper = mu + q * self.sigma_const
        return lower, upper

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._ridge is None or self._phys_indices is None:
            raise RuntimeError("Call fit() first")
        return self._ridge.predict(X_test[:, self._phys_indices])


def build_shiftsafe_ablations(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_target_unlabeled: np.ndarray,
    f_cm_col_index: int | None = None,
    dr_feature_indices: list[int] | None = None,
    config: dict[str, Any] | None = None,
    alpha: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, ShiftSafeCP]:
    """Build corrected ShiftSafe-CP ablation variants.

    X_target_unlabeled must contain ONLY covariates of the held-out target
    domain, never y_test.
    """
    config = config or {}
    ss_cfg = config.get("methods", {}).get("shiftsafe_cp", {})
    common = dict(
        base_model=base_model,
        X_calib=X_calib,
        y_calib=y_calib,
        X_source_pool=np.vstack([X_train, X_calib]),
        X_target_unlabeled=X_target_unlabeled,
        weight_clip=tuple(ss_cfg.get("density_ratio_clip", [0.05, 20.0])),
        sigma_cv_folds=ss_cfg.get("sigma_cv_folds", 5),
        mondrian_min_count=ss_cfg.get("mondrian_min_count", 30),
        alpha=alpha,
        device=device,
        seed=seed,
        f_cm_col_index=f_cm_col_index,
        dr_feature_indices=dr_feature_indices,
    )
    return {
        "ShiftSafe-CP": ShiftSafeCP(**common, layers=[1, 2, 3]),
        "ShiftSafe-CP-L1": ShiftSafeCP(**common, layers=[1]),
        "ShiftSafe-CP-L1-L2": ShiftSafeCP(**common, layers=[1, 2]),
        "ShiftSafe-CP-L1-L3": ShiftSafeCP(**common, layers=[1, 3]),
    }


def fit_conformal_method(
    name: str,
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    config: dict[str, Any] | None = None,
    alpha: float = 0.1,
    use_gpu: bool = True,
    X_target_unlabeled: np.ndarray | None = None,
    f_cm_col_index: int | None = None,
    seed: int = 42,
) -> ConformalPredictor:
    """Factory for conformal predictors."""
    config = config or {}
    if name == "SplitCP":
        cp = SplitConformal(base_model, X_calib, y_calib, alpha)
    elif name == "Jackknife+":
        max_cal = config.get("methods", {}).get("jackknife_plus", {}).get(
            "max_calib_samples", 500
        )
        cp = JackknifeP(
            base_model, X_calib, y_calib, alpha, max_calib_samples=max_cal, use_gpu=use_gpu
        )
    elif name == "CQR":
        cp = CQR(base_model, X_calib, y_calib, alpha)
    elif name.startswith("ShiftSafe"):
        if X_target_unlabeled is None:
            raise ValueError(
                "ShiftSafe-CP requires X_target_unlabeled (target covariates)"
            )
        ablations = build_shiftsafe_ablations(
            base_model, X_train, y_train, X_calib, y_calib,
            X_target_unlabeled=X_target_unlabeled,
            f_cm_col_index=f_cm_col_index,
            config=config, alpha=alpha,
            device="cuda" if use_gpu else "cpu", seed=seed,
        )
        cp = ablations.get(name, ablations["ShiftSafe-CP"])
    else:
        raise ValueError(f"Unknown conformal method: {name}")

    cp.fit()
    return cp


# ---------------------------------------------------------------------------
# Revision (2026-09): outcome-independent calibration draws.
#
# Reviewer 1 (comment 1) and Reviewer 2 (comment 4) correctly observe that
# `stratified_sample_k_indices` forms strata from the response y, which is not
# observable before the target specimens are tested. The samplers below are
# outcome-independent and are used for the revised primary analysis.
# `stratified_sample_k_indices` is retained unchanged so that the originally
# submitted results remain exactly reproducible.
# ---------------------------------------------------------------------------

#: Pre-test observables used for covariate-stratified draws. All four are fixed
#: at mix-design time and are known before any specimen is crushed.
PRETEST_STRATA_COLS = ["w_cm", "age"]


def random_sample_k_indices(y: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """Simple random sample of K indices, independent of the response.

    This is the deployment-realistic draw: a laboratory selects K mixes to send
    for testing without knowing their strengths.
    """
    n = len(y)
    if K > n:
        raise ValueError(f"K={K} exceeds target pool size n={n}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=K, replace=False)).astype(int)


def covariate_stratified_sample_k_indices(
    X: np.ndarray,
    K: int,
    strata_indices: list[int],
    seed: int = 42,
    n_bins: int = 2,
) -> np.ndarray:
    """Stratified sample of K indices using only pre-test covariates.

    Strata are the cross-product of `n_bins` quantile bins of each covariate in
    `strata_indices` (e.g. water/cement ratio and age). No response value is
    used. Note (second-round revision): because the selection depends on the
    covariates, the calibration and evaluation sets differ in covariate
    distribution by design, so the standard exchangeability argument is stated
    for the simple random draw only; this draw is reported as an empirical
    comparison (manuscript Section 4.4 and Appendix A.7).
    """
    n = X.shape[0]
    if K > n:
        raise ValueError(f"K={K} exceeds target pool size n={n}")
    rng = np.random.default_rng(seed)
    if not strata_indices:
        return random_sample_k_indices(np.zeros(n), K, seed=seed)

    codes = np.zeros(n, dtype=int)
    for j in strata_indices:
        col = np.asarray(X[:, j], dtype=np.float64)
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))[1:-1]
        codes = codes * n_bins + np.digitize(col, edges)

    uniq, counts = np.unique(codes, return_counts=True)
    # Proportional allocation with a floor of one per non-empty stratum.
    alloc = np.maximum(1, np.floor(K * counts / n).astype(int))
    alloc = np.minimum(alloc, counts)
    while alloc.sum() > K:  # trim from the largest strata first
        alloc[np.argmax(alloc)] -= 1
    selected: list[int] = []
    for code, take in zip(uniq, alloc):
        pool = np.where(codes == code)[0]
        selected.extend(rng.choice(pool, size=int(take), replace=False).tolist())
    if len(selected) < K:  # top up at random from what is left
        remaining = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=int))
        selected.extend(rng.choice(remaining, size=K - len(selected), replace=False).tolist())
    return np.sort(np.asarray(selected[:K], dtype=int))


def sample_k_indices(
    y: np.ndarray,
    K: int,
    seed: int = 42,
    scheme: str = "strength_quartile",
    X: np.ndarray | None = None,
    strata_indices: list[int] | None = None,
) -> np.ndarray:
    """Dispatch a calibration draw by scheme name.

    ``strength_quartile`` reproduces the originally submitted (outcome-dependent)
    draw; ``random`` and ``covariate`` are outcome-independent.
    """
    if scheme == "strength_quartile":
        return stratified_sample_k_indices(y, K, seed=seed)
    if scheme == "random":
        return random_sample_k_indices(y, K, seed=seed)
    if scheme == "covariate":
        if X is None:
            raise ValueError("scheme='covariate' requires X")
        return covariate_stratified_sample_k_indices(
            X, K, strata_indices or [], seed=seed,
        )
    raise ValueError(f"unknown sampling scheme: {scheme!r}")

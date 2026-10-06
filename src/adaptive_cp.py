"""ShiftSafe-CP-final: adaptive target-anchored Jackknife+ and split CP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.conformal import (
    conformal_quantile,
    min_k_for_alpha,
    quantile_from_calib_scores,
    split_k_fit_cal_indices,
    stratified_sample_k_indices,
)

logger = logging.getLogger(__name__)

CANDIDATE_NAMES = (
    "local-ridge",
    "local-gbm",
    "affine-transfer",
    "stacked",
    "boost-transfer",
)

SIGMA_MPA_FLOOR = 1.0
GATE_FULL_N = 15


@dataclass
class _CandidateFit:
    name: str
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    sigma_const: float
    loo_mae: float = 0.0
    median_abs_resid: float = 0.0


def _phys_matrix(X: np.ndarray, phys_indices: list[int]) -> np.ndarray:
    return np.asarray(X[:, phys_indices], dtype=np.float64)


def _mad(y: np.ndarray) -> float:
    med = float(np.median(y))
    return float(np.median(np.abs(y - med)))


def _sigma_const_from_resid(resid: np.ndarray, y_fit: np.ndarray) -> float:
    """final.1 guard: max(median|resid|, 0.5*MAD(y), 1.0 MPa)."""
    med_res = float(np.median(np.abs(resid)))
    return max(med_res, 0.5 * _mad(y_fit), SIGMA_MPA_FLOOR)


def _ridge_mae(X: np.ndarray, y: np.ndarray) -> tuple[Ridge, float, np.ndarray]:
    m = Ridge(alpha=1.0).fit(X, y)
    preds = m.predict(X)
    return m, float(np.mean(np.abs(y - preds))), preds


def _affine_fit(mu: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    design = np.column_stack([mu, np.ones(len(mu))])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    preds = a * mu + b
    return a, b, preds


def _stacked_fit(Xp: np.ndarray, mu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([mu, Xp, np.ones(len(mu))])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coef, design @ coef


def _gbm_cv_mae(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_rounds: int = 50,
    max_depth: int = 2,
) -> tuple[Any, float, np.ndarray]:
    n = len(y)
    params = {
        "objective": "regression", "verbose": -1, "seed": seed,
        "num_leaves": 4, "max_depth": max_depth, "learning_rate": 0.1,
    }
    if n < 6:
        m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=n_rounds)
        preds = m.predict(X)
        return m, float(np.mean(np.abs(y - preds))), preds
    kf = KFold(n_splits=min(3, n), shuffle=True, random_state=seed)
    oof = np.zeros(n)
    for tr, va in kf.split(X):
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=n_rounds)
        oof[va] = m.predict(X[va])
    full = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=n_rounds)
    return full, float(np.mean(np.abs(y - oof))), full.predict(X)


def _build_stacked(
    Xp: np.ndarray, mu_src: np.ndarray, y: np.ndarray, phys_indices: list[int],
) -> _CandidateFit:
    coef, pred_s = _stacked_fit(Xp, mu_src, y)

    def pred_stacked(Xt: np.ndarray, mut: np.ndarray, c=coef, pi=phys_indices) -> np.ndarray:
        Xpt = _phys_matrix(Xt, pi)
        return np.column_stack([mut, Xpt, np.ones(len(mut))]) @ c

    resid = y - pred_s
    return _CandidateFit(
        "stacked", pred_stacked,
        _sigma_const_from_resid(resid, y),
        float(np.mean(np.abs(resid))),
        float(np.median(np.abs(resid))),
    )


def _fit_candidates(
    X: np.ndarray,
    y: np.ndarray,
    mu_src: np.ndarray,
    phys_indices: list[int],
    seed: int,
    fixed_candidate: str | None = None,
    allowed: tuple[str, ...] | None = None,
) -> _CandidateFit:
    """Fit the candidate pool and return the LOO-MAE winner.

    ``allowed`` restricts the pool to a subset of :data:`CANDIDATE_NAMES`. It is
    used by :class:`LocalOnlyJackknifeCP` to build a target-only comparator that
    shares every other element of the calibration machinery (added in the 2026-09
    revision for Reviewer 1, comment 3).
    """
    n_fit = len(y)
    Xp = _phys_matrix(X, phys_indices)
    if fixed_candidate == "stacked":
        fit = _build_stacked(Xp, mu_src, y, phys_indices)
        fit.loo_mae = float(np.mean(np.abs(y - fit.predict_fn(X, mu_src))))
        return fit

    candidates: list[_CandidateFit] = []
    y_mad = _mad(y)

    ridge, mae_r, pred_r = _ridge_mae(Xp, y)

    def pred_ridge(Xt: np.ndarray, mut: np.ndarray, m=ridge, pi=phys_indices) -> np.ndarray:
        return m.predict(_phys_matrix(Xt, pi))

    resid_r = y - pred_r
    candidates.append(_CandidateFit(
        "local-ridge", pred_ridge,
        _sigma_const_from_resid(resid_r, y), mae_r,
        float(np.median(np.abs(resid_r))),
    ))

    a, b, pred_a = _affine_fit(mu_src, y)

    def pred_affine(Xt: np.ndarray, mut: np.ndarray, aa=a, bb=b) -> np.ndarray:
        return aa * mut + bb

    resid_a = y - pred_a
    candidates.append(_CandidateFit(
        "affine-transfer", pred_affine,
        _sigma_const_from_resid(resid_a, y),
        float(np.mean(np.abs(resid_a))),
        float(np.median(np.abs(resid_a))),
    ))

    candidates.append(_build_stacked(Xp, mu_src, y, phys_indices))

    if n_fit >= GATE_FULL_N:
        gbm, mae_g, pred_g = _gbm_cv_mae(Xp, y, seed, n_rounds=50)

        def pred_gbm(Xt: np.ndarray, mut: np.ndarray, m=gbm, pi=phys_indices) -> np.ndarray:
            return m.predict(_phys_matrix(Xt, pi))

        resid_g = y - pred_g
        candidates.append(_CandidateFit(
            "local-gbm", pred_gbm,
            _sigma_const_from_resid(resid_g, y), mae_g,
            float(np.median(np.abs(resid_g))),
        ))

        mu_aff = a * mu_src + b
        resid_boost = y - mu_aff
        boost, mae_b, pred_bt = _gbm_cv_mae(Xp, resid_boost, seed + 1, n_rounds=30, max_depth=2)

        def pred_boost(Xt: np.ndarray, mut: np.ndarray, aa=a, bb=b, bm=boost, pi=phys_indices) -> np.ndarray:
            return aa * mut + bb + bm.predict(_phys_matrix(Xt, pi))

        resid_bt = y - pred_bt
        candidates.append(_CandidateFit(
            "boost-transfer", pred_boost,
            _sigma_const_from_resid(resid_bt, y), mae_b,
            float(np.median(np.abs(resid_bt))),
        ))

    if allowed is not None:
        candidates = [c for c in candidates if c.name in allowed]
        if not candidates:
            raise ValueError(f"no candidate in {allowed} could be fitted (n_fit={n_fit})")

    winner = min(candidates, key=lambda c: c.loo_mae)

    # Selection guard: near-interpolation -> most regularised available fallback.
    # Default fallback is the stacked transfer fit; when the candidate pool is
    # restricted (e.g. local-only), fall back to the ridge fit instead.
    if winner.median_abs_resid < 0.5 * y_mad:
        fallback = next(
            (c for c in candidates if c.name == "stacked"),
            next((c for c in candidates if c.name == "local-ridge"), winner),
        )
        logger.debug(
            "Selection guard: %s median|resid|=%.3f < 0.5*MAD=%.3f -> %s",
            winner.name, winner.median_abs_resid, 0.5 * y_mad, fallback.name,
        )
        winner = fallback

    return winner


class AdaptiveJackknifeCP:
    """ShiftSafe-CP-final: Jackknife+ with inner per-LOO candidate selection."""

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.10,
        K: int = 20,
        seed: int = 42,
        exact_jkp: bool = False,
        sigma_floor: float = SIGMA_MPA_FLOOR,
        fixed_candidate: str | None = None,
        sampler: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray] | None = None,
        allowed_candidates: tuple[str, ...] | None = None,
    ):
        self.base_model = base_model
        self.alpha = alpha
        self.K = K
        self.seed = seed
        self.exact_jkp = exact_jkp
        self.sigma_floor = sigma_floor
        self.fixed_candidate = fixed_candidate
        # `sampler(X_pool, y_pool, K, seed) -> indices`. Defaults to the
        # originally submitted strength-quartile draw so existing results stay
        # reproducible; the revised primary analysis passes an
        # outcome-independent sampler instead.
        self.sampler = sampler
        self.allowed_candidates = allowed_candidates
        self.calib_target_indices: np.ndarray = np.array([], dtype=int)
        self.fitted = False
        self.diagnostics: dict[str, Any] = {}
        self._phys_indices: list[int] | None = None
        self._loo_scores: np.ndarray = np.array([])
        self._full_fit: _CandidateFit | None = None
        self._loo_fits: list[_CandidateFit] = []

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        physical_indices: list[int],
    ) -> None:
        min_k = min_k_for_alpha(self.alpha)
        if self.K < min_k:
            raise ValueError(f"K={self.K} < min_k={min_k} for alpha={self.alpha}")

        all_idx = (
            self.sampler(X_target_pool, y_target_pool, self.K, self.seed)
            if self.sampler is not None
            else stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        )
        self.calib_target_indices = np.asarray(all_idx, dtype=int)
        all_idx = self.calib_target_indices
        self._phys_indices = physical_indices

        X_k = np.asarray(X_target_pool[all_idx], dtype=np.float64)
        y_k = np.asarray(y_target_pool[all_idx], dtype=np.float64)
        mu_k = self.base_model.predict(X_k)

        scores: list[float] = []
        loo_fits: list[_CandidateFit] = []
        sel_counts: dict[str, int] = {n: 0 for n in CANDIDATE_NAMES}
        guard_hits = 0

        for i in range(self.K):
            mask = np.ones(self.K, dtype=bool)
            mask[i] = False
            fit = _fit_candidates(
                X_k[mask], y_k[mask], mu_k[mask], physical_indices, self.seed + i,
                fixed_candidate=self.fixed_candidate,
                allowed=self.allowed_candidates,
            )
            sel_counts[fit.name] += 1
            loo_fits.append(fit)
            mu_i = fit.predict_fn(X_k[i : i + 1], mu_k[i : i + 1])[0]
            sigma_i = max(fit.sigma_const, self.sigma_floor)
            s_i = abs(y_k[i] - mu_i) / sigma_i
            scores.append(s_i)

        self._loo_scores = np.array(scores)
        self._loo_fits = loo_fits
        self._full_fit = _fit_candidates(
            X_k, y_k, mu_k, physical_indices, self.seed + 999,
            fixed_candidate=self.fixed_candidate,
            allowed=self.allowed_candidates,
        )
        self.fitted = True

        q = conformal_quantile(self._loo_scores, self.alpha)
        self.diagnostics = {
            "K": self.K,
            "quantile_q": q,
            "sigma_const": self._full_fit.sigma_const,
            "winner_full": self._full_fit.name,
            "selection_counts": sel_counts,
            "worst_case_coverage": 1.0 - 2 * self.alpha,
            "sigma_floor_mpa": SIGMA_MPA_FLOOR,
            "candidate_gate_n": GATE_FULL_N,
            "exact_jkp": bool(self.exact_jkp),
            "fixed_candidate": self.fixed_candidate,
            "allowed_candidates": self.allowed_candidates,
            "calibration_draw": "custom_sampler" if self.sampler is not None else "strength_quartile",
            "interval_construction": (
                "barber_jkp_exact" if self.exact_jkp else "practical_fullfit_plus_loo_quantile"
            ),
        }
        logger.info(
            "AdaptiveJackknifeCP: K=%d q=%.3f sigma=%.2f winner=%s sel=%s",
            self.K, q, self._full_fit.sigma_const, self._full_fit.name, sel_counts,
        )

    def predict_intervals(
        self,
        X_test: np.ndarray,
        alpha: float | None = None,
        return_diagnostics: bool = False,
    ):
        if not self.fitted or self._full_fit is None:
            raise RuntimeError("Call fit() first")
        alpha = alpha if alpha is not None else self.alpha
        X_test = np.asarray(X_test, dtype=np.float64)
        mu_src = self.base_model.predict(X_test)

        if self.exact_jkp and self._loo_fits:
            # True Barber et al. 2021 Jackknife+ (asymmetric):
            # lower = floor(α(n+1))-th smallest of {μ_{-i}(x) - R_i}
            # upper = ceil((1-α)(n+1))-th smallest of {μ_{-i}(x) + R_i}
            n = self.K
            lo_rank = int(np.floor(alpha * (n + 1))) - 1  # -1 => -inf
            hi_rank = min(int(np.ceil((1.0 - alpha) * (n + 1))) - 1, n - 1)
            # Batched over test points: each leave-one-out predictor scores the
            # whole test matrix once (K predict calls instead of K * n_test).
            # Arithmetic is identical to the per-point loop.
            mu_loo = np.vstack([
                np.asarray(fit.predict_fn(X_test, mu_src), dtype=np.float64)
                for fit in self._loo_fits
            ])                                                     # (K, n_test)
            radii = np.asarray([
                s * max(fit.sigma_const, self.sigma_floor)
                for fit, s in zip(self._loo_fits, self._loo_scores)
            ], dtype=np.float64)[:, None]                          # (K, 1)
            lo_sorted = np.sort(mu_loo - radii, axis=0)
            hi_sorted = np.sort(mu_loo + radii, axis=0)
            lower = (
                np.full(len(X_test), -np.inf) if lo_rank < 0 else lo_sorted[lo_rank]
            )
            upper = hi_sorted[hi_rank]
        else:
            # Practical shortcut (confirmatory default): single full-K refit center
            # ± conformal quantile of LOO normalized residuals. NOT exact Barber JK+.
            mu = self._full_fit.predict_fn(X_test, mu_src)
            sigma = max(self._full_fit.sigma_const, self.sigma_floor)
            q = conformal_quantile(self._loo_scores, alpha)
            lower = mu - q * sigma
            upper = mu + q * sigma

        if return_diagnostics:
            return lower, upper, {"quantile_q": float(conformal_quantile(self._loo_scores, alpha))}
        return lower, upper

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._full_fit is None:
            raise RuntimeError("Call fit() first")
        return self._full_fit.predict_fn(X_test, self.base_model.predict(X_test))


class StackedOnlyCP(AdaptiveJackknifeCP):
    """Ablation arm: stacked candidate only, same LOO conformal calibration."""

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.10,
        K: int = 20,
        seed: int = 42,
        exact_jkp: bool = False,
        sigma_floor: float = SIGMA_MPA_FLOOR,
        sampler: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray] | None = None,
    ):
        super().__init__(
            base_model=base_model,
            alpha=alpha,
            K=K,
            seed=seed,
            exact_jkp=exact_jkp,
            sigma_floor=sigma_floor,
            fixed_candidate="stacked",
            sampler=sampler,
        )


class LocalOnlyJackknifeCP(AdaptiveJackknifeCP):
    """Target-only Jackknife+ comparator.

    Identical to :class:`AdaptiveJackknifeCP` except that the candidate pool is
    restricted to predictors fitted on the K labeled target points alone
    (``local-ridge``, ``local-gbm``); the source model is never consulted. It
    therefore isolates the contribution of *source transfer* from that of
    spending the whole budget of K through a leave-one-out construction rather
    than a fit/calibration split, which is what ``TargetOnlyCP`` does.
    """

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.10,
        K: int = 20,
        seed: int = 42,
        exact_jkp: bool = False,
        sigma_floor: float = SIGMA_MPA_FLOOR,
        sampler: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray] | None = None,
    ):
        super().__init__(
            base_model=base_model,
            alpha=alpha,
            K=K,
            seed=seed,
            exact_jkp=exact_jkp,
            sigma_floor=sigma_floor,
            sampler=sampler,
            allowed_candidates=("local-ridge", "local-gbm"),
        )


class AdaptiveSplitCP:
    """Adaptive candidate selection + split conformal (K_fit / K_cal ablation)."""

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.10,
        K: int = 20,
        seed: int = 42,
        sigma_floor: float = SIGMA_MPA_FLOOR,
        sampler: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray] | None = None,
    ):
        self.base_model = base_model
        self.alpha = alpha
        self.K = K
        self.seed = seed
        self.sigma_floor = sigma_floor
        self.sampler = sampler
        self.calib_target_indices: np.ndarray = np.array([], dtype=int)
        self.fitted = False
        self.diagnostics: dict[str, Any] = {}
        self._fit_result: _CandidateFit | None = None
        self.calib_scores: np.ndarray = np.array([])
        self._K_cal = 0

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        physical_indices: list[int],
    ) -> None:
        min_k = min_k_for_alpha(self.alpha)
        if self.K < min_k + 1:
            raise ValueError(f"K={self.K} too small for split at alpha={self.alpha}")

        all_idx = (
            self.sampler(X_target_pool, y_target_pool, self.K, self.seed)
            if self.sampler is not None
            else stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        )
        all_idx = np.asarray(all_idx, dtype=int)
        self.calib_target_indices = all_idx
        fit_idx, cal_idx, _, self._K_cal = split_k_fit_cal_indices(
            all_idx, self.K, self.alpha, self.seed,
        )

        X_ft = np.asarray(X_target_pool[fit_idx], dtype=np.float64)
        y_ft = y_target_pool[fit_idx]
        mu_ft = self.base_model.predict(X_ft)
        self._fit_result = _fit_candidates(X_ft, y_ft, mu_ft, physical_indices, self.seed)

        X_cal = np.asarray(X_target_pool[cal_idx], dtype=np.float64)
        y_cal = y_target_pool[cal_idx]
        mu_cal = self._fit_result.predict_fn(X_cal, self.base_model.predict(X_cal))
        sigma = max(self._fit_result.sigma_const, self.sigma_floor)
        self.calib_scores = np.abs(y_cal - mu_cal) / sigma
        q_fit = quantile_from_calib_scores(
            self.calib_scores, self.alpha, self._K_cal, context="AdaptiveSplitCP.fit",
        )
        self.fitted = True
        self.diagnostics = {
            "K": self.K,
            "K_cal": self._K_cal,
            "winner": self._fit_result.name,
            "quantile_q": q_fit,
            "sigma_const": self._fit_result.sigma_const,
        }

    def predict_intervals(self, X_test: np.ndarray, alpha: float | None = None):
        if not self.fitted or self._fit_result is None:
            raise RuntimeError("Call fit() first")
        alpha = alpha if alpha is not None else self.alpha
        q = quantile_from_calib_scores(
            self.calib_scores, alpha, self._K_cal, context="AdaptiveSplitCP",
        )
        X_test = np.asarray(X_test, dtype=np.float64)
        mu_src = self.base_model.predict(X_test)
        mu = self._fit_result.predict_fn(X_test, mu_src)
        sigma = max(self._fit_result.sigma_const, self.sigma_floor)
        return mu - q * sigma, mu + q * sigma

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Call fit() first")
        return self._fit_result.predict_fn(X_test, self.base_model.predict(X_test))

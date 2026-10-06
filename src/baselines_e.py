"""Extension E external SOTA baselines (post-confirmatory comparators)."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import gpytorch
import lightgbm as lgb
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from src.base_models import clone_lgb_model
from src.conformal import (
    JackknifeP,
    ShiftSafeCP,
    conformal_quantile,
    min_k_for_alpha,
    quantile_from_calib_scores,
    split_k_fit_cal_indices,
    stratified_sample_k_indices,
)

logger = logging.getLogger(__name__)

E_METHOD_NAMES = (
    "CQR-target",
    "GPR-target",
    "GPR-transfer",
    "JKplus-source",
    "TabPFN-target",
    "WeightedCP-v2",
)


class EBaselinePredictor(ABC):
    """Common interface for Extension E baselines."""

    method_name: str = "EBaseline"
    guarantee_scope: str = "none"  # 'target' | 'source' | 'none'

    def __init__(self, alpha: float = 0.10, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.fitted = False
        self.calib_target_indices: np.ndarray = np.array([], dtype=int)
        self.diagnostics: dict[str, Any] = {}

    @abstractmethod
    def fit(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_intervals(X_test)
        return (lo + hi) / 2


class CQRTargetCP(EBaselinePredictor):
    """E1a: CQR on K labeled target points (K_fit=10 / K_cal=10 at K=20)."""

    method_name = "CQR-target"
    guarantee_scope = "target"

    def __init__(
        self,
        alpha: float = 0.10,
        K: int = 20,
        seed: int = 42,
        K_fit: int = 10,
        quantiles: tuple[float, float] = (0.05, 0.95),
    ):
        super().__init__(alpha, seed)
        self.K = K
        self.K_fit = K_fit
        self.quantiles = quantiles
        self.q_lower: QuantileRegressor | None = None
        self.q_upper: QuantileRegressor | None = None
        self.calib_scores: np.ndarray = np.array([])
        self._K_cal = 0

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        k_indices: np.ndarray | None = None,
    ) -> None:
        X_target_pool = np.asarray(X_target_pool, dtype=np.float64)
        y_target_pool = np.asarray(y_target_pool, dtype=np.float64)
        if k_indices is None:
            k_indices = stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        self.calib_target_indices = np.asarray(k_indices, dtype=int)
        if len(self.calib_target_indices) != self.K:
            raise ValueError(f"Expected K={self.K} indices, got {len(self.calib_target_indices)}")

        fit_idx, cal_idx, self.K_fit, self._K_cal = split_k_fit_cal_indices(
            self.calib_target_indices, self.K, self.alpha, self.seed,
        )
        if self.K_fit != 10 or self._K_cal != 10:
            logger.warning(
                "CQR-target: K_fit=%d K_cal=%d (expected 10/10 at K=20)",
                self.K_fit, self._K_cal,
            )

        X_fit, y_fit = X_target_pool[fit_idx], y_target_pool[fit_idx]
        X_cal, y_cal = X_target_pool[cal_idx], y_target_pool[cal_idx]

        q_lo, q_hi = self.quantiles
        self.q_lower = QuantileRegressor(quantile=q_lo, alpha=0.0, solver="highs")
        self.q_upper = QuantileRegressor(quantile=q_hi, alpha=0.0, solver="highs")
        self.q_lower.fit(X_fit, y_fit)
        self.q_upper.fit(X_fit, y_fit)

        lo = self.q_lower.predict(X_cal)
        hi = self.q_upper.predict(X_cal)
        self.calib_scores = np.maximum(lo - y_cal, y_cal - hi)
        self.fitted = True
        self.diagnostics = {
            "K": self.K,
            "K_fit": self.K_fit,
            "K_cal": self._K_cal,
            "fit_indices": fit_idx.tolist(),
            "cal_indices": cal_idx.tolist(),
        }
        logger.info("CQR-target fitted: K_fit=%d K_cal=%d", self.K_fit, self._K_cal)

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.q_lower is None or self.q_upper is None:
            raise RuntimeError("Call fit() first")
        alpha = alpha if alpha is not None else self.alpha
        correction = quantile_from_calib_scores(
            self.calib_scores, alpha, self._K_cal, context="CQR-target",
        )
        lo = self.q_lower.predict(X_test) - correction
        hi = self.q_upper.predict(X_test) + correction
        return lo, hi


class _GPyTorchExactGP:
    """Exact GP with RBF+ARD kernel (GPU when available)."""

    def __init__(self, seed: int = 42, training_iters: int = 150):
        self.seed = seed
        self.training_iters = training_iters
        self.scaler = StandardScaler()
        self.model: Any = None
        self.likelihood: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s = self.scaler.fit_transform(X)
        train_x = torch.tensor(X_s, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(y, dtype=torch.float32, device=self.device)
        d = X_s.shape[1]

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.cov_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=d)
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.cov_module(x)
                )

        torch.manual_seed(self.seed)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(train_x, train_y, self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.08)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(self.training_iters):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        self.model.eval()
        self.likelihood.eval()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("GP not fitted")
        X_s = self.scaler.transform(X)
        test_x = torch.tensor(X_s, dtype=torch.float32, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(test_x))
            mean = pred.mean.cpu().numpy()
            std = pred.stddev.cpu().numpy()
        return mean, std


class GPRTargetCP(EBaselinePredictor):
    """E1b: GPyTorch exact GP RBF+ARD on K target labels; no cal split."""

    method_name = "GPR-target"
    guarantee_scope = "none"

    def __init__(self, alpha: float = 0.10, K: int = 20, seed: int = 42):
        super().__init__(alpha, seed)
        self.K = K
        self._gp = _GPyTorchExactGP(seed=seed)
        self._sklearn_fallback = False
        self._scaler: StandardScaler | None = None
        self._sklearn_gpr: Any = None

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        k_indices: np.ndarray | None = None,
    ) -> None:
        if k_indices is None:
            k_indices = stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        self.calib_target_indices = np.asarray(k_indices, dtype=int)
        X_k = np.asarray(X_target_pool[self.calib_target_indices], dtype=np.float64)
        y_k = np.asarray(y_target_pool[self.calib_target_indices], dtype=np.float64)
        try:
            self._gp.fit(X_k, y_k)
            self._sklearn_fallback = False
        except Exception as exc:
            logger.warning("GPyTorch GPR-target failed (%s); sklearn fallback", exc)
            self._scaler = StandardScaler().fit(X_k)
            X_s = self._scaler.transform(X_k)
            kernel = RBF(length_scale=np.ones(X_k.shape[1]))
            self._sklearn_gpr = GaussianProcessRegressor(
                kernel=kernel, random_state=self.seed,
            )
            self._sklearn_gpr.fit(X_s, y_k)
            self._sklearn_fallback = True
        self.fitted = True
        self.diagnostics = {"K": self.K, "guaranteed": False}

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        alpha = alpha if alpha is not None else self.alpha
        z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)))
        if self._sklearn_fallback:
            assert self._scaler is not None and self._sklearn_gpr is not None
            X_s = self._scaler.transform(X_test)
            mean, std = self._sklearn_gpr.predict(X_s, return_std=True)
        else:
            mean, std = self._gp.predict(X_test)
        return mean - z * std, mean + z * std


class GPRTransferCP(EBaselinePredictor):
    """E1c: GP on target residuals y - mu_hat(x) with source LightGBM mu_hat."""

    method_name = "GPR-transfer"
    guarantee_scope = "none"

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.10,
        K: int = 20,
        seed: int = 42,
    ):
        super().__init__(alpha, seed)
        self.base_model = base_model
        self.K = K
        self._gp = _GPyTorchExactGP(seed=seed)
        self._use_sklearn = False
        self._sklearn_gpr: Any = None
        self._scaler: StandardScaler | None = None

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        k_indices: np.ndarray | None = None,
    ) -> None:
        if k_indices is None:
            k_indices = stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        self.calib_target_indices = np.asarray(k_indices, dtype=int)
        X_k = np.asarray(X_target_pool[self.calib_target_indices], dtype=np.float64)
        y_k = np.asarray(y_target_pool[self.calib_target_indices], dtype=np.float64)
        mu_k = self.base_model.predict(X_k)
        resid = y_k - mu_k
        try:
            self._gp.fit(X_k, resid)
        except Exception as exc:
            logger.warning("GPyTorch GPR-transfer failed (%s); sklearn fallback", exc)
            self._scaler = StandardScaler().fit(X_k)
            X_s = self._scaler.transform(X_k)
            kernel = RBF(length_scale=np.ones(X_k.shape[1]))
            self._sklearn_gpr = GaussianProcessRegressor(
                kernel=kernel, random_state=self.seed,
            )
            self._sklearn_gpr.fit(X_s, resid)
            self._use_sklearn = True
        self.fitted = True
        self.diagnostics = {"K": self.K, "guaranteed": False}

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        alpha = alpha if alpha is not None else self.alpha
        z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)))
        mu = self.base_model.predict(X_test)
        if self._use_sklearn:
            assert self._scaler is not None and self._sklearn_gpr is not None
            X_s = self._scaler.transform(X_test)
            r_mean, r_std = self._sklearn_gpr.predict(X_s, return_std=True)
        else:
            r_mean, r_std = self._gp.predict(X_test)
        return mu + r_mean - z * r_std, mu + r_mean + z * r_std


class JKPlusSourceCP(EBaselinePredictor):
    """E1d: LightGBM Jackknife+ on source cal (subsample 300); K-independent."""

    method_name = "JKplus-source"
    guarantee_scope = "source"

    def __init__(
        self,
        base_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        alpha: float = 0.10,
        seed: int = 42,
        max_calib_samples: int = 300,
        use_gpu: bool = True,
        held_out_domain: str | None = None,
        calib_domains: np.ndarray | None = None,
    ):
        super().__init__(alpha, seed)
        self.base_model = base_model
        self.X_calib = np.asarray(X_calib, dtype=np.float64)
        self.y_calib = np.asarray(y_calib, dtype=np.float64)
        self.max_calib_samples = max_calib_samples
        self.use_gpu = use_gpu
        self.held_out_domain = held_out_domain
        self.calib_domains = None if calib_domains is None else np.asarray(calib_domains)
        self._jk: JackknifeP | None = None
        self._assert_source_only_pool()

    def _assert_source_only_pool(self) -> None:
        """Hard assert: calib pool has ZERO rows from the held-out fold."""
        if self.held_out_domain is None or self.calib_domains is None:
            return
        held = str(self.held_out_domain)
        leaked = np.array(
            [
                (str(d) == held) or str(d).startswith(held.split("_")[0] + "_") or str(d).startswith(held)
                for d in self.calib_domains
            ]
        )
        # Also catch fold key vs full domain_id (D4 vs D4_mondal)
        fold_key = held.split("_")[0]
        leaked |= np.array([str(d).startswith(fold_key) for d in self.calib_domains])
        n_leak = int(np.sum(leaked))
        if n_leak > 0:
            raise AssertionError(
                f"JKplus-source LEAKAGE: {n_leak}/{len(self.calib_domains)} calib rows "
                f"belong to held-out domain '{self.held_out_domain}'"
            )

    def fit(self) -> None:
        self._assert_source_only_pool()
        # Subsample from SOURCE calib only, before any target merge (pool is already source-only)
        self._jk = JackknifeP(
            self.base_model,
            self.X_calib,
            self.y_calib,
            alpha=self.alpha,
            max_calib_samples=self.max_calib_samples,
            use_gpu=self.use_gpu,
            seed=self.seed,
        )
        self._jk.fit()
        self.fitted = True
        self.diagnostics = {
            "max_calib_samples": self.max_calib_samples,
            "n_calib_pool": len(self.X_calib),
            "n_calib_used": len(self._jk.X_calib),
            "n_models_trained": self._jk.n_models_trained,
            "held_out_domain": self.held_out_domain,
            "guaranteed": True,
            "guarantee_scope": "source",
        }
        logger.info(
            "JKplus-source fit: n_models_trained=%d n_calib_used=%d/%d held_out=%s",
            self._jk.n_models_trained,
            len(self._jk.X_calib),
            len(self.X_calib),
            self.held_out_domain,
        )

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._jk is None:
            raise RuntimeError("Call fit() first")
        # Poison guard: prediction must not consume y_test
        return self._jk.predict_intervals(X_test, alpha=alpha)


class TabPFNTargetCP(EBaselinePredictor):
    """E1e: TabPFN v2 + split conformal on K_cal."""

    method_name = "TabPFN-target"
    guarantee_scope = "target"

    def __init__(self, alpha: float = 0.10, K: int = 20, seed: int = 42):
        super().__init__(alpha, seed)
        self.K = K
        self._regressor: Any = None
        self.calib_scores: np.ndarray = np.array([])
        self._K_cal = 0
        self._max_rows = 1024

    def fit(
        self,
        X_target_pool: np.ndarray,
        y_target_pool: np.ndarray,
        k_indices: np.ndarray | None = None,
    ) -> None:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "TabPFN not installed; pip install tabpfn"
            ) from exc

        X_target_pool = np.asarray(X_target_pool, dtype=np.float64)
        y_target_pool = np.asarray(y_target_pool, dtype=np.float64)
        if k_indices is None:
            k_indices = stratified_sample_k_indices(y_target_pool, self.K, seed=self.seed)
        self.calib_target_indices = np.asarray(k_indices, dtype=int)

        fit_idx, cal_idx, _, self._K_cal = split_k_fit_cal_indices(
            self.calib_target_indices, self.K, self.alpha, self.seed,
        )
        X_fit = X_target_pool[fit_idx]
        y_fit = y_target_pool[fit_idx]
        X_cal = X_target_pool[cal_idx]
        y_cal = y_target_pool[cal_idx]

        if len(X_fit) > self._max_rows:
            rng = np.random.default_rng(self.seed)
            sub = rng.choice(len(X_fit), self._max_rows, replace=False)
            X_fit, y_fit = X_fit[sub], y_fit[sub]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Prefer ungated TabPFN v2 checkpoints: v2.5/v2.6/v3 require PriorLabs
        # license acceptance beyond a valid TABPFN_TOKEN. Fall back to default
        # constructor only if create_default_for_version is unavailable.
        try:
            from tabpfn.constants import ModelVersion  # type: ignore[import-untyped]

            self._regressor = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, device=device, random_state=self.seed,
            )
            tabpfn_version = "v2"
        except Exception:
            self._regressor = TabPFNRegressor(device=device, random_state=self.seed)
            tabpfn_version = "default"
        self._regressor.fit(X_fit, y_fit)

        mu_cal = self._regressor.predict(X_cal)
        resid = np.abs(y_cal - mu_cal)
        sigma = max(float(np.median(resid)), 0.25)
        self.calib_scores = resid / sigma
        self._sigma = sigma
        self.fitted = True
        self.diagnostics = {
            "K": self.K,
            "K_cal": self._K_cal,
            "sigma": sigma,
            "device": device,
            "tabpfn_version": tabpfn_version,
            "tabpfn_token_env": bool(os.environ.get("TABPFN_TOKEN")),
        }

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self._regressor is None:
            raise RuntimeError("Call fit() first")
        alpha = alpha if alpha is not None else self.alpha
        q = quantile_from_calib_scores(
            self.calib_scores, alpha, self._K_cal, context="TabPFN-target",
        )
        mu = self._regressor.predict(X_test)
        return mu - q * self._sigma, mu + q * self._sigma


class WeightedCPv2(EBaselinePredictor):
    """E1f: ShiftSafe-CP v2 Layer-1 weighted conformal (source validity only)."""

    method_name = "WeightedCP-v2"
    guarantee_scope = "source"

    def __init__(
        self,
        base_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        X_target_unlabeled: np.ndarray,
        dr_feature_indices: list[int] | None = None,
        f_cm_col_index: int | None = None,
        alpha: float = 0.10,
        seed: int = 42,
        device: str = "cuda",
    ):
        super().__init__(alpha, seed)
        self._cp = ShiftSafeCP(
            base_model=base_model,
            X_calib=X_calib,
            y_calib=y_calib,
            X_source_pool=np.vstack([X_train, X_calib]),
            X_target_unlabeled=X_target_unlabeled,
            layers=[1],
            dr_feature_indices=dr_feature_indices,
            f_cm_col_index=f_cm_col_index,
            alpha=alpha,
            seed=seed,
            device=device,
        )

    def fit(self) -> None:
        self._cp.fit()
        self.fitted = True
        self.diagnostics = {"layers": [1], "ess": self._cp.diagnostics.get("ess")}

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self._cp.predict_intervals(X_test, alpha=alpha)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._cp.predict(X_test)


def fit_e_baseline(
    name: str,
    *,
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_target_pool: np.ndarray,
    y_target_pool: np.ndarray,
    X_eval: np.ndarray,
    k_indices: np.ndarray,
    phys_indices: list[int],
    dr_feature_indices: list[int] | None = None,
    f_cm_col_index: int | None = None,
    alpha: float = 0.10,
    K: int = 20,
    seed: int = 42,
    use_gpu: bool = True,
) -> EBaselinePredictor:
    """Factory for Extension E baselines."""
    _ = phys_indices  # reserved for future Mondrian variants
    if name == "CQR-target":
        cp: EBaselinePredictor = CQRTargetCP(alpha=alpha, K=K, seed=seed, K_fit=10)
        cp.fit(X_target_pool, y_target_pool, k_indices=k_indices)
    elif name == "GPR-target":
        cp = GPRTargetCP(alpha=alpha, K=K, seed=seed)
        cp.fit(X_target_pool, y_target_pool, k_indices=k_indices)
    elif name == "GPR-transfer":
        cp = GPRTransferCP(base_model, alpha=alpha, K=K, seed=seed)
        cp.fit(X_target_pool, y_target_pool, k_indices=k_indices)
    elif name == "JKplus-source":
        cp = JKPlusSourceCP(
            base_model, X_calib, y_calib, alpha=alpha, seed=seed, use_gpu=use_gpu,
            held_out_domain=None, calib_domains=None,
        )
        cp.fit()
    elif name == "TabPFN-target":
        cp = TabPFNTargetCP(alpha=alpha, K=K, seed=seed)
        cp.fit(X_target_pool, y_target_pool, k_indices=k_indices)
    elif name == "WeightedCP-v2":
        cp = WeightedCPv2(
            base_model, X_train, y_train, X_calib, y_calib, X_eval,
            dr_feature_indices=dr_feature_indices,
            f_cm_col_index=f_cm_col_index,
            alpha=alpha, seed=seed,
            device="cuda" if use_gpu else "cpu",
        )
        cp.fit()
    else:
        raise ValueError(f"Unknown E baseline: {name}")
    return cp

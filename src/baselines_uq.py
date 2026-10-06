"""Uncertainty quantification baselines: GPR, NGBoost, Deep Ensemble."""

from __future__ import annotations

import logging
from typing import Any

import gpytorch
import numpy as np
import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.base_models import MLPEnsembleWrapper

logger = logging.getLogger(__name__)


class IntervalPredictor:
    """Unified interface for UQ baselines."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_intervals(X_test)
        return (lo + hi) / 2


class GPRModel(IntervalPredictor):
    """Gaussian Process Regression intervals."""

    def __init__(
        self,
        alpha: float = 0.1,
        max_samples: int = 2000,
        use_gpytorch: bool = True,
        seed: int = 42,
    ):
        super().__init__(alpha)
        self.max_samples = max_samples
        self.use_gpytorch = use_gpytorch
        self.seed = seed
        self.model: Any = None
        self.scaler = StandardScaler()
        self.y_train: np.ndarray = np.array([])
        self._gpytorch_model: Any = None
        self._likelihood: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n = len(X_train)
        if n > self.max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, self.max_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            logger.warning("GPR subsampled to %d samples", self.max_samples)

        X_s = self.scaler.fit_transform(X_train)
        self.y_train = y_train

        if self.use_gpytorch and len(X_train) <= self.max_samples:
            try:
                self._fit_gpytorch(X_s, y_train)
                self.fitted = True
                return
            except Exception as e:
                logger.warning("GPyTorch GPR failed (%s); falling back to sklearn", e)

        kernel = RBF() + WhiteKernel(noise_level=1.0)
        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2, random_state=self.seed
        )
        self.model.fit(X_s, y_train)
        self.fitted = True

    def _fit_gpytorch(self, X: np.ndarray, y: np.ndarray) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(y, dtype=torch.float32)

        class ExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.cov_module(x)
                )

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._gpytorch_model = ExactGP(train_x, train_y, self._likelihood)
        self._gpytorch_model = self._gpytorch_model.to(device)
        self._likelihood = self._likelihood.to(device)
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        self._gpytorch_model.train()
        self._likelihood.train()
        optimizer = torch.optim.Adam(self._gpytorch_model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._gpytorch_model)

        for _ in range(100):
            optimizer.zero_grad()
            output = self._gpytorch_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        self._gpytorch_model.eval()
        self._likelihood.eval()

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        alpha = alpha if alpha is not None else self.alpha
        z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)))

        X_s = self.scaler.transform(X_test)

        if self._gpytorch_model is not None:
            device = next(self._gpytorch_model.parameters()).device
            test_x = torch.tensor(X_s, dtype=torch.float32).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = self._likelihood(self._gpytorch_model(test_x))
                mean = pred.mean.cpu().numpy()
                std = pred.stddev.cpu().numpy()
            return mean - z * std, mean + z * std

        mean, std = self.model.predict(X_s, return_std=True)
        return mean - z * std, mean + z * std


class NGBoostModel(IntervalPredictor):
    """NGBoost distributional prediction intervals."""

    def __init__(self, alpha: float = 0.1, n_estimators: int = 500, seed: int = 42):
        super().__init__(alpha)
        self.n_estimators = n_estimators
        self.seed = seed
        self.model: NGBRegressor | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = NGBRegressor(
            Dist=Normal,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            verbose=False,
        )
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        alpha = alpha if alpha is not None else self.alpha
        dist = self.model.pred_dist(X_test)
        lo = dist.ppf(alpha / 2)
        hi = dist.ppf(1 - alpha / 2)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class DeepEnsemblePredictor(IntervalPredictor):
    """Deep MLP ensemble mean ± k*std intervals."""

    def __init__(self, ensemble: MLPEnsembleWrapper, alpha: float = 0.1):
        super().__init__(alpha)
        self.ensemble = ensemble
        self.fitted = True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict_intervals(
        self, X_test: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = alpha if alpha is not None else self.alpha
        z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)))
        preds = self.ensemble.predict_ensemble(X_test)
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        return mean - z * std, mean + z * std

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.ensemble.predict(X_test)


def fit_uq_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ensemble: MLPEnsembleWrapper | None = None,
    alpha: float = 0.1,
    seed: int = 42,
) -> dict[str, IntervalPredictor]:
    """Fit all UQ baseline models."""
    models: dict[str, IntervalPredictor] = {}

    gpr = GPRModel(alpha=alpha, seed=seed)
    try:
        gpr.fit(X_train, y_train)
        models["GPR"] = gpr
    except Exception as e:
        logger.warning("GPR fit failed: %s", e)

    ngb = NGBoostModel(alpha=alpha, seed=seed)
    try:
        ngb.fit(X_train, y_train)
        models["NGBoost"] = ngb
    except Exception as e:
        logger.warning("NGBoost fit failed: %s", e)

    if ensemble is not None:
        models["DeepEnsemble"] = DeepEnsemblePredictor(ensemble, alpha=alpha)

    return models

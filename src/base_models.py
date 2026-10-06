"""GPU-optimized base learners with Optuna hyperparameter tuning."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BasePredictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class LightGBMTuner:
    """Tune and fit LightGBM with Optuna."""

    def __init__(
        self,
        device: str = "gpu",
        seed: int = 42,
        n_trials: int = 100,
        use_gpu: bool = True,
    ):
        self.device = device
        self.seed = seed
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.model: lgb.Booster | None = None
        self.best_params: dict[str, Any] = {}

    def _device_params(self) -> dict[str, Any]:
        if self.use_gpu:
            return {"device_type": "gpu", "gpu_device_id": 0}
        return {"device_type": "cpu"}

    def objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
            "seed": self.seed,
            **self._device_params(),
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        return float(np.sqrt(np.mean((preds - y_val) ** 2)))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> lgb.Booster:
        logger.info("Tuning LightGBM (%d trials)...", self.n_trials)
        study = optuna.create_study(direction="minimize", pruner=MedianPruner())

        if self.n_trials > 0:
            study.optimize(
                lambda t: self.objective(t, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )
            self.best_params = study.best_params
            logger.info("  Best RMSE: %.4f", study.best_value)
        else:
            self.best_params = {
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.0,
                "lambda_l2": 0.0,
                "min_child_samples": 20,
            }

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "seed": self.seed,
            **self.best_params,
            **self._device_params(),
        }

        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        train_data = lgb.Dataset(X_full, label=y_full)
        self.model = lgb.train(params, train_data, num_boost_round=500)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)


class XGBoostTuner:
    """Tune and fit XGBoost with Optuna."""

    def __init__(self, seed: int = 42, n_trials: int = 100, use_gpu: bool = True):
        self.seed = seed
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.model: xgb.XGBRegressor | None = None
        self.best_params: dict[str, Any] = {}

    def _tree_method(self) -> str:
        if not self.use_gpu:
            return "hist"
        try:
            import xgboost as xgb
            ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
            if ver >= (2, 0):
                return "hist"  # GPU via device='cuda' in XGB 2.x+
        except Exception:
            pass
        return "hist"

    def _xgb_params(self, trial_params: dict | None = None) -> dict:
        p = {
            "n_estimators": 500,
            "early_stopping_rounds": 50,
            "random_state": self.seed,
            "verbosity": 0,
            "tree_method": self._tree_method(),
        }
        if self.use_gpu:
            p["device"] = "cuda"
        if trial_params:
            p.update(trial_params)
        return p

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> xgb.XGBRegressor:
        logger.info("Tuning XGBoost (%d trials)...", self.n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = self._xgb_params({
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            })
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return float(np.sqrt(np.mean((preds - y_val) ** 2)))

        if self.n_trials > 0:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            self.best_params = study.best_params
        else:
            self.best_params = {"max_depth": 6, "learning_rate": 0.1}

        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        final_params = self._xgb_params(self.best_params)
        final_params.pop("early_stopping_rounds", None)
        self.model = xgb.XGBRegressor(**final_params)
        self.model.fit(X_full, y_full)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)


class CatBoostTuner:
    """Tune and fit CatBoost with Optuna."""

    def __init__(self, seed: int = 42, n_trials: int = 100, use_gpu: bool = True):
        self.seed = seed
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.model: cb.CatBoostRegressor | None = None
        self.best_params: dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> cb.CatBoostRegressor:
        logger.info("Tuning CatBoost (%d trials)...", self.n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
                "iterations": 500,
                "early_stopping_rounds": 50,
                "random_seed": self.seed,
                "task_type": "GPU" if self.use_gpu else "CPU",
                "verbose": False,
            }
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            preds = model.predict(X_val)
            return float(np.sqrt(np.mean((preds - y_val) ** 2)))

        if self.n_trials > 0:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            self.best_params = study.best_params
        else:
            self.best_params = {"depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0}

        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        self.model = cb.CatBoostRegressor(
            **self.best_params,
            iterations=500,
            random_seed=self.seed,
            task_type="GPU" if self.use_gpu else "CPU",
            verbose=False,
        )
        self.model.fit(X_full, y_full)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)


class DeepMLPEnsemble(nn.Module):
    """Ensemble of neural networks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        n_models: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        hidden_dims = hidden_dims or [256, 128, 64]
        self.models = nn.ModuleList()
        for m in range(n_models):
            torch.manual_seed(seed + m)
            layers: list[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout_rate)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.models.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(x) for m in self.models], dim=1)


class MLPEnsembleWrapper:
    """Sklearn-like wrapper for DeepMLPEnsemble."""

    def __init__(self, model: DeepMLPEnsemble, scaler: StandardScaler, device: torch.device):
        self.model = model
        self.scaler = scaler
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(X_scaled).to(self.device)
            preds = self.model(t).mean(dim=1).cpu().numpy()
        return preds

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_models) predictions."""
        self.model.eval()
        X_scaled = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(X_scaled).to(self.device)
            return self.model(t).cpu().numpy()


def fit_ensemble_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device | None = None,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    hidden_dims: list[int] | None = None,
    n_models: int = 5,
    patience: int = 30,
) -> MLPEnsembleWrapper:
    """Train deep MLP ensemble."""
    logger.info("Training Deep MLP Ensemble...")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_s),
            torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_s),
            torch.from_numpy(y_val.astype(np.float32)).reshape(-1, 1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = DeepMLPEnsemble(
        X_train_s.shape[1], hidden_dims=hidden_dims, n_models=n_models, seed=seed
    ).to(device)

    use_amp = device.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(xb)
                loss = criterion(preds, yb.repeat(1, preds.shape[1]))
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb.repeat(1, preds.shape[1])).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    return MLPEnsembleWrapper(model, scaler, device)


def clone_lgb_model(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    num_boost_round: int = 300,
    use_gpu: bool = True,
) -> lgb.Booster:
    """Clone and retrain LightGBM (for Jackknife+)."""
    p = copy.deepcopy(params)
    p.update({"verbose": -1})
    if use_gpu:
        p.setdefault("device_type", "gpu")
    else:
        p["device_type"] = "cpu"
    data = lgb.Dataset(X, label=y)
    return lgb.train(p, data, num_boost_round=num_boost_round)


def fit_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    device: torch.device | None = None,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Fit all enabled base learners."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    models_cfg = config.get("models", {})
    n_trials = n_trials if n_trials is not None else config.get("optuna", {}).get("n_trials", 100)

    models: dict[str, Any] = {}

    if models_cfg.get("lightgbm", {}).get("enabled", True):
        lgb_tuner = LightGBMTuner(
            n_trials=models_cfg.get("lightgbm", {}).get("n_trials", n_trials),
            use_gpu=use_gpu,
        )
        models["LightGBM"] = lgb_tuner.fit(X_train, y_train, X_val, y_val)

    if models_cfg.get("xgboost", {}).get("enabled", True):
        xgb_tuner = XGBoostTuner(
            n_trials=models_cfg.get("xgboost", {}).get("n_trials", n_trials),
            use_gpu=use_gpu,
        )
        models["XGBoost"] = xgb_tuner.fit(X_train, y_train, X_val, y_val)

    if models_cfg.get("catboost", {}).get("enabled", True):
        cb_tuner = CatBoostTuner(
            n_trials=models_cfg.get("catboost", {}).get("n_trials", n_trials),
            use_gpu=use_gpu,
        )
        models["CatBoost"] = cb_tuner.fit(X_train, y_train, X_val, y_val)

    if models_cfg.get("deep_mlp", {}).get("enabled", True):
        mlp_cfg = models_cfg.get("deep_mlp", {})
        models["DeepMLP"] = fit_ensemble_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            epochs=mlp_cfg.get("epochs", 200),
            batch_size=mlp_cfg.get("batch_size", 32),
            lr=mlp_cfg.get("lr", 1e-3),
            hidden_dims=mlp_cfg.get("hidden_dims", [256, 128, 64]),
            n_models=mlp_cfg.get("n_ensemble", 5),
            patience=mlp_cfg.get("patience", 30),
        )

    return models


def predict_base_models(models: dict[str, Any], X: np.ndarray) -> dict[str, np.ndarray]:
    """Predict with all base models."""
    preds = {}
    for name, model in models.items():
        if hasattr(model, "predict"):
            preds[name] = model.predict(X)
        elif isinstance(model, lgb.Booster):
            preds[name] = model.predict(X)
    return preds


def save_best_hyperparams(
    tuners: dict[str, Any],
    output_path: Path | str,
) -> None:
    """Save best hyperparameters from tuners."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = {}
    for name, obj in tuners.items():
        if hasattr(obj, "best_params"):
            params[name] = obj.best_params
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

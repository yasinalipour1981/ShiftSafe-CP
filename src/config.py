"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    alpha_levels: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20])
    lodo_n_seeds: int = 10
    cv_folds: int = 5
    train_calib_split: float = 0.8
    protocol: str = "v2"
    K_sweep: list[int] = Field(default_factory=lambda: [20])
    max_target_calib_fraction: float = 0.30
    primary_K: int = 20
    held_out_folds: list[str] | None = None


class FeatureEngineeringConfig(BaseModel):
    derive_ratios: bool = True
    derive_interaction: bool = True
    scale_features: str = "standardize"
    handle_missing: str = "mean"
    filter_outliers: bool = True
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42
    cuda_seed: int = 42
    torch_deterministic: bool = True
    torch_benchmark: bool = False
    device: str = "cuda"
    gpu_id: int = 0
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    results_dir: str = "./results"
    config_dir: str = "./configs"
    log_level: str = "INFO"
    log_dir: str = "./results/logs"
    datasets: dict[str, Any] = Field(default_factory=dict)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    n_optuna_trials: int = 100

    @property
    def alpha_levels(self) -> list[float]:
        return self.evaluation.alpha_levels

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def raw_data_path(self) -> Path:
        return Path(self.raw_data_dir)

    @property
    def processed_data_path(self) -> Path:
        return Path(self.processed_data_dir)

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    base_path: str | Path = "configs/base.yaml",
    extra_paths: list[str | Path] | None = None,
) -> AppConfig:
    """Load base.yaml, merge stage configs, then overlay the requested config file."""
    base_path = Path(base_path)
    config_dir = base_path.parent

    root_base = config_dir / "base.yaml"
    merged: dict = load_yaml(root_base) if root_base.exists() else {}

    default_extras = [
        "data_harmonization.yaml",
        "base_learners.yaml",
        "conformal.yaml",
        "lodo_experiment.yaml",
    ]
    paths = extra_paths if extra_paths is not None else default_extras

    for name in paths:
        path = Path(name) if Path(name).is_absolute() else config_dir / name
        if path.exists():
            merged = _deep_merge(merged, load_yaml(path))

    # User-selected config (e.g. quick_test.yaml) overrides everything
    if base_path.resolve() != root_base.resolve() and base_path.exists():
        merged = _deep_merge(merged, load_yaml(base_path))

    return AppConfig.model_validate(merged)


def config_to_dict(config: AppConfig) -> dict:
    return config.model_dump()

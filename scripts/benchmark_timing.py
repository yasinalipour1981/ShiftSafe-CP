#!/usr/bin/env python
"""Benchmark per-method timing at primary cell (K=20, alpha=0.10, seed 142)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_to_dict, load_config
from src.datasets import create_lodo_splits, load_all_datasets
from src.experiments import LODOExperiment
from src.utils import set_all_seeds, setup_logging


def main() -> int:
    set_all_seeds(142)
    setup_logging("TimingBenchmark", Path("results/logs"))

    config = config_to_dict(load_config("configs/final_confirmatory.yaml"))
    config["device"] = "cuda"
    config["n_optuna_trials"] = 0
    config.setdefault("evaluation", {})
    config["evaluation"]["K_sweep"] = [20]
    config["evaluation"]["alpha_levels"] = [0.10]
    config["evaluation"]["held_out_folds"] = ["D1", "D2", "D4"]

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    methods = config.get("methods_to_evaluate", [])

    datasets = load_all_datasets(config, seed=142)
    ratio = config["evaluation"].get("train_calib_split", 0.8)
    splits = create_lodo_splits(datasets, ratio, seed=142)
    splits = {k: v for k, v in splits.items() if k in config["evaluation"]["held_out_folds"]}

    exp = LODOExperiment(config, seed=142)
    exp.all_results = []
    exp.timing_results = []
    exp._run_lodo_v3(splits, [0.10], 0, use_gpu, device, methods)

    out = Path("results/lodo_evaluation/timing_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(exp.timing_results)
    df.to_csv(out, index=False)

    print(f"\nWrote {out} ({len(df)} rows)")
    summary = df.groupby("method").agg(
        source=("source_fit_time_sec", "mean"),
        calib=("calib_time_sec", "mean"),
        infer=("infer_time_per_1000_sec", "mean"),
    ).round(4)
    print("\n=== Timing summary (K=20, seed=142) ===")
    print(summary.to_string())
    calib_spread = summary["calib"].max() - summary["calib"].min()
    print(f"\nCP calib spread across methods: {calib_spread:.4f} s")
    if calib_spread < 1e-6:
        print("WARNING: calib times still identical, check timing instrumentation")
    else:
        print("OK: per-method calib times differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

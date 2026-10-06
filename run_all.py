#!/usr/bin/env python
"""Master script to run full ShiftSafe-CP pipeline end-to-end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.config import config_to_dict, load_config
from src.datasets import load_all_datasets
from src.experiments import LODOExperiment
from src.paper_tables import generate_all_tables
from src.utils import (
    append_execution_log,
    create_results_manifest,
    get_device,
    set_all_seeds,
    setup_logging,
)
from src.visualization import generate_all_figures


def main() -> int:
    parser = argparse.ArgumentParser(description="ShiftSafe-CP Full Pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials override")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    logger, log_file = setup_logging("ShiftSafeCP", Path("results/logs"))
    logger.info("=" * 80)
    logger.info("ShiftSafe-CP: Cross-Laboratory Calibrated Prediction Intervals")
    logger.info("=" * 80)
    logger.info("Config: %s", args.config)
    logger.info("Seed: %d, N seeds: %d, Device: %s", args.seed, args.n_seeds, args.device)
    logger.info("Log file: %s", log_file)

    app_config = load_config(args.config)
    config = config_to_dict(app_config)
    config["device"] = args.device
    if args.n_trials is not None:
        config["n_optuna_trials"] = args.n_trials

    device = get_device(prefer_cuda=args.device == "cuda")
    logger.info("Using device: %s", device)

    results_dir = Path(config["results_dir"])
    all_results: list[pd.DataFrame] = []

    for seed_idx in range(args.n_seeds):
        run_seed = args.seed + seed_idx
        set_all_seeds(run_seed)
        logger.info("\n%s", "=" * 80)
        logger.info("SEED %d/%d (run_seed=%d)", seed_idx + 1, args.n_seeds, run_seed)
        logger.info("%s", "=" * 80)

        experiment = LODOExperiment(config, seed=run_seed)
        df_results = experiment.run_full_pipeline()
        all_results.append(df_results)

    df_final = pd.concat(all_results, ignore_index=True)
    eval_dir = results_dir / "lodo_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(results_dir / "FINAL_RESULTS_ALLSEEDS.csv", index=False)
    df_final.to_csv(eval_dir / "LODO_all_results.csv", index=False)

    if config.get("evaluation", {}).get("protocol") in ("v3", "v_final"):
        v3_exp = LODOExperiment(config, seed=args.seed)
        v3_exp._write_v3_tables(df_final, eval_dir)
        v3_exp._run_v3_stat_tests(df_final, eval_dir)

    # Load auxiliary data for figures/tables
    datasets = load_all_datasets(config, seed=args.seed)
    shift_path = results_dir / "shift_analysis" / "domain_classifier_auc.csv"
    df_shift = pd.read_csv(shift_path) if shift_path.exists() else None
    timing_path = eval_dir / "timing_results.csv"
    df_timing = pd.read_csv(timing_path) if timing_path.exists() else None
    accuracy_path = eval_dir / "point_accuracy.csv"
    df_accuracy = pd.read_csv(accuracy_path) if accuracy_path.exists() else None

    logger.info("\n[STAGE 6] Generating figures and tables...")
    generate_all_figures(df_final, df_shift, df_timing, None, results_dir)
    generate_all_tables(datasets, df_final, df_shift, df_timing, df_accuracy, eval_dir)

    create_results_manifest(
        results_dir,
        config,
        stage="full_pipeline",
        output_files=["FINAL_RESULTS_ALLSEEDS.csv"],
    )
    append_execution_log(f"Full pipeline complete: {args.n_seeds} seed(s)")

    summary = df_final.groupby("method").agg({
        "empirical_coverage": ["mean", "std"],
        "mean_interval_width": ["mean", "std"],
        "rmse": ["mean", "std"],
    }).round(4)
    logger.info("\nFINAL SUMMARY:\n%s", summary)
    logger.info("Pipeline complete. Results saved to %s", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

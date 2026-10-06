#!/usr/bin/env python
"""Run Extension E external baselines with progress tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_to_dict, load_config
from src.extension_e import run_extension_e
from src.utils import append_execution_log, set_all_seeds, setup_logging


def _print_smoke_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("SMOKE: no results")
        return
    sub = df[(df["alpha"] == 0.10) & (df["K"] == 20)]
    agg = sub.groupby("method").agg(
        coverage=("empirical_coverage", "mean"),
        width=("mean_interval_width", "mean"),
        winkler=("winkler_score", "mean"),
        scope=("guarantee_scope", "first"),
    ).round(3)
    print("\n=== SMOKE TABLE (one row per method) ===")
    print(agg.to_string())
    print("=========================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extension E baselines")
    parser.add_argument("--config", default="configs/extension_e.yaml")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="One seed, fold D4 only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--skip-gpr", action="store_true")
    parser.add_argument("--skip-jk", action="store_true", help="Skip JKplus-source (slow LOO)")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override output CSV path (default: results/extension_e/EXTENSION_E_results.csv)",
    )
    args = parser.parse_args()

    set_all_seeds(args.seed)
    logger, log_file = setup_logging("ExtensionE", Path("results/logs"))
    logger.info("Extension E run: config=%s smoke=%s", args.config, args.smoke)

    app_config = load_config(args.config)
    config = config_to_dict(app_config)
    config["device"] = args.device

    seeds = [args.seed] if args.smoke else [args.seed + i for i in range(args.n_seeds)]
    folds = ("D4",) if args.smoke else ("D1", "D2", "D4")
    logger.info(
        "Extension E grid: n_seeds=%d seeds=%s..%s folds=%s methods_cfg=%d smoke=%s tabpfn_token=%s",
        len(seeds), seeds[0], seeds[-1], folds, len(config.get("methods_to_evaluate", [])),
        args.smoke, bool(__import__("os").environ.get("TABPFN_TOKEN")),
    )

    methods = list(config.get("methods_to_evaluate", []))
    if args.skip_tabpfn and "TabPFN-target" in methods:
        methods.remove("TabPFN-target")
    if args.skip_gpr:
        for m in ("GPR-target", "GPR-transfer"):
            if m in methods:
                methods.remove(m)

    try:
        df = run_extension_e(
            config,
            seeds=seeds,
            folds=folds,
            methods=tuple(methods),
            smoke=args.smoke,
            skip_jk=args.skip_jk,
            output_csv=args.output_csv,
        )
    except Exception:
        logger.exception("Extension E run failed (partial results may be in EXTENSION_E_results.csv)")
        raise
    _print_smoke_table(df)

    out_dir = Path(config.get("results_dir", "results")) / "extension_e"
    if args.smoke and not df.empty:
        try:
            from scripts.extension_e_reporting import make_smoke_figure
            make_smoke_figure(df, out_dir / "figures" / "SMOKE_extension_e.png")
            logger.info("Smoke figure: %s", out_dir / "figures" / "SMOKE_extension_e.png")
        except Exception as exc:
            logger.warning("Smoke figure failed: %s", exc)

    append_execution_log(
        f"Extension E {'smoke' if args.smoke else 'full'} complete: "
        f"{len(df)} rows, seeds={seeds[0]}-{seeds[-1]}"
    )
    logger.info("Done. Progress: %s", out_dir / "progress.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

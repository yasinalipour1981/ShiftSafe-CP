"""Logging, reproducibility, and GPU utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> int:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return seed


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available; CPU otherwise."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA available: {name} ({mem_gb:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("CUDA not available; using CPU")
    return device


def setup_logging(
    name: str,
    log_dir: Path | str = Path("results/logs"),
    level: int | str = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Configure reproducible logging to file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file


def log_gpu_memory(logger: logging.Logger, tag: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"[GPU {tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )


def get_git_hash() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_results_manifest(
    output_dir: Path,
    config: dict[str, Any],
    stage: str,
    output_files: list[str] | None = None,
    git_hash: str | None = None,
) -> Path:
    """Record experiment provenance in results/RESULTS_MANIFEST.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "RESULTS_MANIFEST.json"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "git_commit": git_hash or get_git_hash(),
        "config_snapshot": config,
        "output_files": output_files or [],
    }

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"runs": []}

    manifest["runs"].append(entry)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def append_execution_log(
    message: str,
    log_path: Path | str = "EXECUTION_LOG.md",
) -> None:
    """Append a timestamped line to EXECUTION_LOG.md."""
    log_path = Path(log_path)
    timestamp = datetime.now().isoformat()
    line = f"- [{timestamp}] {message}\n"
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
    else:
        content = "# ShiftSafe-CP Execution Log\n\n"
    log_path.write_text(content + line, encoding="utf-8")

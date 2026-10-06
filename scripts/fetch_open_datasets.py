#!/usr/bin/env python
"""Download and cache open concrete datasets used by ShiftSafe-CP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UCI_182_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "concrete/slump/slump_test.data"
)


def fetch_uci_182(out_dir: Path = Path("data/raw/uci_182")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slump_test.csv"
    df = pd.read_csv(UCI_182_URL)
    df.to_csv(out_path, index=False)
    print(f"UCI #182: saved {len(df)} rows -> {out_path}")
    return out_path


def main():
    fetch_uci_182()
    print("Done. Re-run pipeline to include refreshed open datasets.")


if __name__ == "__main__":
    main()

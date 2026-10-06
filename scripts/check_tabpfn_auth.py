"""Fit TabPFN v2 (ungated) to confirm auth/download works. Never prints token."""
from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    if not os.environ.get("TABPFN_TOKEN"):
        print("NO_TOKEN")
        return 2
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 5))
    y = 30 + X[:, 0] * 3
    reg = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=0,
    )
    reg.fit(X[:14], y[:14])
    pred = reg.predict(X[14:])
    assert np.all(np.isfinite(pred))
    print(f"AUTH_OK_V2: fit+predict n={len(pred)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

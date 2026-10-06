"""Sensitivity of the shift diagnostics to the definition of the physical subspace.

Reviewer 2, comment 8 asks whether the variables defining the physical subspace
were fixed a priori on engineering grounds or chosen after inspecting domain
separability, since the latter would make the reported shift diagnostics
partly exploratory.

The subspace used in the paper is {w/cm, fly-ash fraction, binder content, age}.
No search over subsets was carried out. This script supplies the corresponding
evidence: it recomputes the domain-classifier AUC for every held-out laboratory
under a range of alternative subspace definitions, including the minimal
Abrams-law pair, the full mix-proportion set, and every leave-one-variable-out
version of the published subspace. If the qualitative verdict -- D2 and D3
genuinely disjoint, D1 and D4 separable only through provenance markers -- is
stable across these definitions, it does not depend on the particular subset
chosen.

Output: results/revision/SUBSPACE_sensitivity.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    align_dataframe_features,
    get_feature_columns,
    load_all_datasets,
)
from src.shift_analysis import ShiftAnalyzer  # noqa: E402

DOMAINS = ("D1", "D2", "D3", "D4")

#: Published subspace, chosen on engineering grounds: Abrams' law (w/cm),
#: supplementary-cementitious replacement (f/cm), paste content (binder_total)
#: and maturity (age).
PUBLISHED = ["w_cm", "f_cm", "age", "binder_total"]

VARIANTS: dict[str, list[str]] = {
    "published (w/cm, f/cm, age, binder)": PUBLISHED,
    "Abrams minimal (w/cm, age)": ["w_cm", "age"],
    "w/cm, f/cm only": ["w_cm", "f_cm"],
    "published + superplasticizer": PUBLISHED + ["superplasticizer"],
    "published + slag fraction": PUBLISHED + ["s_cm"],
    "mix proportions (all 8 raw)": [
        "cement", "slag", "fly_ash", "water", "superplasticizer",
        "coarse_agg", "fine_agg", "age",
    ],
}
for drop in PUBLISHED:  # leave-one-variable-out versions of the published set
    VARIANTS[f"published minus {drop}"] = [c for c in PUBLISHED if c != drop]


def main() -> int:
    warnings.filterwarnings("ignore")
    with open(ROOT / "configs" / "final_confirmatory.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for seed in range(142, 152):
        datasets = load_all_datasets(config, seed=seed)
        analyzer = ShiftAnalyzer(device=config["device"], seed=seed)
        prep = {}
        for d in DOMAINS:
            df = align_dataframe_features(datasets[d])
            cols = get_feature_columns(df)
            prep[d] = (df, cols)

        for held in DOMAINS:
            srcs = [d for d in DOMAINS if d != held]
            for name, wanted in VARIANTS.items():
                cols = [c for c in wanted if c in prep[held][1]]
                if not cols:
                    continue
                Xs = np.vstack([prep[d][0][cols].values.astype(float) for d in srcs])
                Xt = prep[held][0][cols].values.astype(float)
                X = np.vstack([Xs, Xt])
                lab = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
                rows.append(dict(
                    seed=seed, held_out=held, subspace=name,
                    n_vars=len(cols), auc=analyzer.domain_classifier_auc(X, lab),
                ))
        print(f"seed {seed} done", flush=True)

    out = ROOT / "results" / "revision"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "SUBSPACE_sensitivity.csv", index=False)

    piv = df.pivot_table(index="subspace", columns="held_out", values="auc")
    order = list(VARIANTS.keys())
    piv = piv.reindex([o for o in order if o in piv.index])
    print("\n=== Domain-classifier AUC by physical-subspace definition "
          "(mean over 10 seeds) ===")
    print(piv.round(3).to_string())
    print("\nVerdict per definition (threshold 0.90 = near-disjoint support):")
    verdict = (piv > 0.90).replace({True: "disjoint", False: "overlapping"})
    print(verdict.to_string())
    print(f"\nWrote {out/'SUBSPACE_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

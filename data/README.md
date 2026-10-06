# Dataset Registry

Open and manual datasets used for cross-laboratory LODO evaluation.

| ID | Name | n | Source | Role in shift study |
|----|------|---|--------|---------------------|
| D1 | UCI #165 Concrete Compressive Strength | 1030 | [ucimlrepo](https://archive.ics.uci.edu/dataset/165) / OpenML #4353 | Large diverse lab mixes, ages 1–365 d |
| D2 | UCI #182 Concrete Slump Test | 103 | [UCI archive CSV](https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data) | Different lab protocol (slump/flow), fixed age 28 d |
| D3 | Zviazhynski et al. 2025 | 35 | Supplementary PDF / parsed CSV | Rheology-focused lab, GGBS blends |
| D4 | Mondal 2026 Fly-Ash Concrete | 165* | Reconstructed or manual CSV | Controlled fly-ash lab (7/28/90 d) |
| D5 | User extra | varies | `data/raw/extra/*.csv` | Optional additional domains |

\*165 rows = 55 mixes × 3 ages when reconstructed.

## Why these datasets?

- **D1 vs D2**: Same author (Yeh), different experimental setups, good baseline domain shift without leaving public data.
- **D3**: Independent 2025 lab data with rheology metadata.
- **D4**: Fly-ash controlled design; strongest shift signal for Mondrian `f_cm` conditioning.

## UCI #182 notes

`ucimlrepo` id=182 is currently **blocked** on many installs. The loader uses:

1. Local cache: `data/raw/uci_182/slump_test.csv` (recommended, offline)
2. Live download from UCI archive URL (CSV **with header**)
3. Synthetic fallback (last resort only)

Refresh cache:

```bash
python scripts/fetch_open_datasets.py
```

## UCI #165 fallback

If `ucimlrepo` id=165 fails, OpenML dataset **#4353** is used automatically (identical 1030-row dataset).

## Other useful open sources (optional, via D5)

Place harmonized CSVs in `data/raw/extra/`:

| Dataset | URL | Notes |
|---------|-----|-------|
| UCI #182 (raw) | https://archive.ics.uci.edu/dataset/182/concrete+slump+test | Already integrated as D2 |
| OpenML Concrete Data | https://www.openml.org/d/4353 | Duplicate of D1 |
| Kaggle Concrete Strength | Various mirrors | Usually duplicate of UCI #165, skip unless verified distinct |

## Canonical schema

All datasets map to: `cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age, strength_mpa`

See `configs/data_harmonization.yaml` for column aliases.

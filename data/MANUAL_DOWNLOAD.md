# Manual Dataset Downloads

## UCI #182 Concrete Slump Test (D2)

**Source:** UCI ML Repository #182, [Concrete Slump Test](https://archive.ics.uci.edu/dataset/182/concrete+slump+test)  
**DOI:** 10.24432/C5FG7D  
**n:** 103 mixes, 28-day compressive strength (MPa)

### Status

Cached locally at:

```
data/raw/uci_182/slump_test.csv
```

`ucimlrepo` import for id=182 is currently unavailable on many systems. The pipeline loads the cached CSV or downloads:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data
```

Refresh cache:

```bash
python scripts/fetch_open_datasets.py
```

### Why D2 is useful

- Same canonical mix features as UCI #165 but **different lab protocol** (slump/flow testing)
- Fixed age (28 d) vs D1's variable ages → realistic cross-lab shift for LODO
- Small n (103) tests calibration under limited target-domain data

---

## Zviazhynski et al. 2025 (D3)

**Source:** Data-Centric Engineering  
**DOI:** https://doi.org/10.1017/dce.2025.10018

### Status

Dataset extracted from supplementary PDF (Table 1, 35 mixes) and saved to:

```
data/raw/zviazhynski_2025/zviazhynski_supplementary.csv
```

Source: Zviazhynski et al., Data-Centric Engineering (DOI: 10.1017/dce.2025.10018)

Columns mapped from supplementary Table 1:

| PDF column | CSV / canonical |
|---|---|
| Cement (kg/m³) | `cement` |
| GGBS (kg/m³) | `slag` |
| Fine / coarse aggregate | `fine_agg`, `coarse_agg` |
| Water (kg/m³) | `water` |
| Water/binder ratio | `w_b_ratio` (cross-check: ≈ `w_cm`) |
| 28-day cube strength | `strength_mpa` (age = 28) |
| 56-day cube strength | `strength_56d_mpa` (optional) |

**Note:** If exporting manually from the PDF, do **not** label the cement column as `GGBS`, that was a common export error. Use the canonical headers in the CSV or run `python scripts/parse_zviazhynski_pdf.py`.

To re-parse from PDF: `python scripts/parse_zviazhynski_pdf.py`

### Manual download (if re-fetching)

1. Download the supplementary CSV from the paper's data repository.
2. Save it as:
   ```
   data/raw/zviazhynski_2025/zviazhynski_supplementary.csv
   ```
3. Expected columns (any of these naming conventions are accepted):
   - `cement`, `slag`, `fly_ash`, `water`, `superplasticizer`
   - `coarse_agg`, `fine_agg`, `age`, `strength_mpa`

### If Missing

The pipeline runs without D3. A warning is logged and LODO folds use D1, D2, and D4 only.

## User-Provided Datasets (D5)

Place additional fly-ash concrete CSVs in:

```
data/raw/extra/*.csv
```

Each file should follow the canonical column schema documented in `configs/data_harmonization.yaml`.

## Mondal 2026 (D4)

If `data/raw/mondal_2026/mondal_2026_55mixes.csv` is missing or invalid (e.g. all strengths = 100 MPa), the loader **automatically reconstructs** 61 mixes × 3 ages (7, 28, 90 d) from the published experimental design using Mondal Eq. 5.11/5.12 (natural-log Abrams form).

Replace with real supplementary data when available.

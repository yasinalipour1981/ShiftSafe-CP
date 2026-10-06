# Manifest of archived result files

Every table and figure of the paper is regenerated from the files below; nothing in
the manuscript is typed by hand. Row counts exclude the header. Seeds 142–191 are
the registered confirmatory block; the primary cell is K = 20, α = 0.10.

## Runs reported in the revised manuscript

| File | Rows | Description |
|------|------|-------------|
| `results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv` | 12350 | Confirmatory grid: simple random calibration draw, exact jackknife-plus; folds D1–D4, K-sweep {10, 15, 20, 30, 50}, α ∈ {0.05, 0.10, 0.20, 0.30}; `configs/final_confirmatory_revised.yaml` |
| `results/extension_e_revised/EXTENSION_E_results.csv` | 900 | External comparators (CQR, GPR ×2, source-side Jackknife+, TabPFN, weighted CP) on the same draws; `configs/extension_e_revised.yaml` |
| `results/revision/REVISION_draws_K20.csv` | 4050 | Supplementary grid at the primary cell: three calibration draws × nine arms (TargetAnchoredJK+ exact and single refit, TargetOnlyJK+, StackedOnlyCP, TargetOnlyCP, AdaptiveSplitCP, TransferCal, source split CP); `scripts/run_revision_experiments.py` |
| `results/revision/REVISION_*.csv`, `REVISION_SUMMARY.md` | n/a | Summaries of the grid above (`scripts/analyze_revision.py`) |
| `results/revision/SHIFT_vs_RELIABILITY_pairs.csv` | 120 | Twelve ordered source→target pairs × ten seeds: shift measures and source-calibrated coverage (Table 4) |
| `results/revision/WEIGHTEDCP_diagnostics.csv` | 40 | Weighted-CP diagnostics per held-out laboratory (Table 5) |
| `results/revision/SUBSPACE_sensitivity.csv` | 400 | Domain-classifier AUC under ten definitions of the physical subspace (Table A.9) |
| `results/shift_analysis/` | n/a | Pairwise shift diagnostics (Fig. 1) |
| `results/lodo_evaluation/timing_results.csv` | n/a | Per-method timing (Section 6.7) |
| `results/diagnostics/fold_{D}_seed{seed}_K{K}.json` | 950 files | Calibration-index draws of the confirmatory runs |
| `results/extension_e_revised/k_indices/` | 150 files | Calibration-index draws of the external benchmark |
| `results/RESULTS_MANIFEST.json` | n/a | Provenance of every pipeline run (timestamp, stage, configuration snapshot) |

## Runs of the originally submitted version (kept for the development history)

| File | Rows | Description |
|------|------|-------------|
| `results/FINAL_RESULTS_confirmatory_142-191.csv` | 12350 | Confirmatory grid with the strength-quartile draw and the single-refit construction |
| `results/extension_e/EXTENSION_E_results.csv` | 900 | External comparators on those draws |
| `results/ablation/STACKED_ONLY_confirmatory_K20.csv` | 300 | StackedOnlyCP ablation, first round |
| `results/ablation/EXACT_JKP_sensitivity_confirmatory_K20.csv` | 300 | Exact vs single-refit jackknife-plus, first round |
| `results/FINAL_RESULTS_exploratory_42-51.csv` | 2470 | Exploratory seeds used for method development only |
| `results/FINAL_RESULTS_exploratory_pre_final1.csv`, `results/appendix/` | n/a | Pre-/post-safeguard comparison on exploratory seeds |

## Regeneration

```bash
python run_all.py --config configs/final_confirmatory_revised.yaml --seed 142 --n-seeds 50 --device cuda
python scripts/run_extension_e.py --config configs/extension_e_revised.yaml --seed 142 --n-seeds 50 --device cuda
python scripts/run_revision_experiments.py && python scripts/analyze_revision.py
python scripts/run_shift_vs_reliability.py
python scripts/run_subspace_sensitivity.py
python scripts/make_figures.py --results results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv
python scripts/make_example_intervals_figure.py
```

`scripts/revision_numbers.py` computes every number printed in the paper from the
files in the first table (bootstrap intervals, equally weighted pooled means,
leave-one-laboratory-out figures, Holm-adjusted Wilcoxon tests).

## Raw data

| Domain | Path | Provenance |
|--------|------|------------|
| D1 UCI #165 | fetched at run time via `ucimlrepo` (OpenML #4353 fallback) | Yeh (1998) |
| D2 UCI #182 | `data/raw/uci_182/slump_test.csv` | UCI Machine Learning Repository, CC BY 4.0 |
| D3 Zviazhynski 2025 | `data/raw/zviazhynski_2025/zviazhynski_supplementary.csv` | parsed from the open-access supplementary table, doi:10.1017/dce.2025.10018 |
| D4 Mondal 2026 | `data/raw/mondal_2026/mondal_2026_55mixes.csv` | reconstructed by `src/datasets.py` from published mix designs (not measured) |

## Contact

Corresponding author: Javad Ghasemian, Damghan University.

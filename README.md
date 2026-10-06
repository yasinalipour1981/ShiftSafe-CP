# ShiftSafe-CP / TargetAnchoredJK+

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23195294.svg)](https://doi.org/10.5281/zenodo.23195294)

Code, configuration files and archived results for

> J. Ghasemian, Y. Alipour, M. H. Talebpour, **Reliable Cross-Laboratory Prediction
> Intervals for Concrete Compressive Strength under Distribution Shift**,
> *Computers and Concrete* (manuscript 26M-08-239, under review).

The paper asks what happens to conformal prediction intervals for concrete
compressive strength when a model calibrated in one laboratory is used in another.
Source-calibrated intervals collapse (about 4% coverage at a nominal 90%, with
intervals under 3 MPa wide); **TargetAnchoredJK+** restores validity by anchoring the
conformal quantile in the receiving laboratory: a small budget of K labeled target
mixes, drawn by simple random sampling, is spent on an exact jackknife-plus
construction whose guarantee does not depend on the severity of the shift. Every
number, table and figure in the paper is regenerated from the result files archived
here by the scripts listed below.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Library: data harmonization and the leave-one-domain-out protocol, the conformal methods, external comparators, metrics, shift diagnostics, figures |
| `configs/` | YAML configurations; `final_confirmatory_revised.yaml` and `extension_e_revised.yaml` define the runs reported in the paper |
| `scripts/` | Experiment, analysis and figure scripts (see the reproduction table below) |
| `run_all.py` | End-to-end pipeline (data, shift analysis, LODO evaluation, figures and tables) |
| `results/` | Archived result files from which every table and figure is derived |
| `data/` | Cached open datasets and their documentation (`data/README.md`, `data/MANUAL_DOWNLOAD.md`) |
| `tests/` | pytest suite (validity, leakage, operating-region and smoke tests) |
| `docs/method_evolution.md` | How the method lineage evolved, including the changes made for the revised manuscript |
| `REPOSITORY.md` | Manifest of the primary result files and how to regenerate them |

### Names used in the code and in the paper

| Code | Paper |
|---|---|
| `AdaptiveJackknifeCP` (`exact_jkp=True`) | TargetAnchoredJK+ (exact jackknife-plus, primary) |
| `AdaptiveJackknifeCP` (`exact_jkp=False`) | single-refit approximation (Appendix A.3) |
| `LocalOnlyJackknifeCP` / `TargetOnlyJKplus` | TargetOnlyJK+ |
| `StackedOnlyCP` | StackedOnlyCP ablation (Appendix A.4) |
| `AdaptiveSplitCP` | AdaptiveSplitCP |
| `TargetOnlyCP` | TargetOnlyCP (fair local baseline) |
| `TransferCal-CP-affine-localsigma` | TransferCal (affine, local σ) |
| `SplitCP` | source-calibrated split conformal prediction |
| `JKplus-source`, `WeightedCP-v2`, `CQR-target`, `GPR-target`, `GPR-transfer`, `TabPFN-target` | external comparators of Section 4.5 |

## Installation

Python 3.11 or 3.12 and an NVIDIA GPU are assumed (all timings in the paper are
for a single RTX 4090; the code also runs on CPU, more slowly).

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q tests          # 56 tests, about 15 s
```

`TabPFN-target` needs a TabPFN token in the environment variable `TABPFN_TOKEN`;
pass `--skip-tabpfn` to `scripts/run_extension_e.py` to run the external benchmark
without it.

## Data

| ID | Source | How it is obtained |
|---|---|---|
| D1 | UCI #165, Yeh (1998) | fetched at run time through `ucimlrepo` (OpenML #4353 as fallback) |
| D2 | UCI #182, concrete slump test | cached in `data/raw/uci_182/slump_test.csv` (CC BY 4.0) |
| D3 | Zviazhynski et al. (2025), Data-Centric Engineering | parsed from the open-access supplementary table into `data/raw/zviazhynski_2025/zviazhynski_supplementary.csv` |
| D4 | Mondal (2026) fly-ash mixes | **reconstructed**, not measured: strengths are generated from the published mix designs with age-specific Abrams-law equations and seed-controlled Gaussian noise (σ = 0.8 MPa) by `ConcreteDatasetLoader.load_mondal` in `src/datasets.py`; the generated file is `data/raw/mondal_2026/mondal_2026_55mixes.csv` |

The paper treats D4 as a reconstructed stress-test domain and reports every pooled
summary with and without it as a target laboratory. See `data/README.md` for the
canonical schema and `data/MANUAL_DOWNLOAD.md` for provenance details.

## Reproducing the paper

Seeds 42–51 were used for exploratory development; seeds 142–191 (fifty seeds) are
the registered confirmatory block on which every primary claim rests. The primary
cell is K = 20 labeled target mixes at α = 0.10.

```bash
# smoke test (~1 min)
python run_all.py --config configs/quick_test.yaml --seed 42 --n-seeds 1 --device cuda --n-trials 0

# confirmatory grid: random draw, exact jackknife-plus, K-sweep, four nominal levels (~35 min)
python run_all.py --config configs/final_confirmatory_revised.yaml --seed 142 --n-seeds 50 --device cuda

# external comparators on the same draws (~20 min without TabPFN)
python scripts/run_extension_e.py --config configs/extension_e_revised.yaml --seed 142 --n-seeds 50 --device cuda

# supplementary grid: three calibration draws, TargetOnlyJK+, StackedOnlyCP, exact vs single refit (~30 min)
python scripts/run_revision_experiments.py
python scripts/analyze_revision.py

# shift diagnostics versus reliability, weighted-CP diagnostics, physical-subspace sensitivity
python scripts/run_shift_vs_reliability.py
python scripts/run_subspace_sensitivity.py

# figures
python scripts/make_figures.py --results results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv
python scripts/make_example_intervals_figure.py
python scripts/benchmark_timing.py
```

| Paper | Result file(s) | Produced by |
|---|---|---|
| Tables 2, 3, A.1; Appendix A.9 (Table A.7); Appendix A.10 (Table A.8) | `results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv`, `results/extension_e_revised/EXTENSION_E_results.csv` | `run_all.py`, `scripts/run_extension_e.py`; summarised by `scripts/revision_numbers.py` |
| Table A.2 (operating regions) | analytical, Eq. (4) | `src/validity.py` |
| Tables A.3, A.4, A.5, A.6 (exact vs single refit, StackedOnlyCP, calibration draws, decomposition) | `results/revision/REVISION_draws_K20.csv` and the `REVISION_*.csv` summaries | `scripts/run_revision_experiments.py`, `scripts/analyze_revision.py` |
| Tables 4 and 5 (shift measures vs coverage loss; weighted-CP diagnostics) | `results/revision/SHIFT_vs_RELIABILITY_pairs.csv`, `results/revision/WEIGHTEDCP_diagnostics.csv` | `scripts/run_shift_vs_reliability.py` |
| Table A.9 (definition of the physical subspace) | `results/revision/SUBSPACE_sensitivity.csv` | `scripts/run_subspace_sensitivity.py` |
| Table A.10 (chronology) | `results/RESULTS_MANIFEST.json`, `docs/method_evolution.md` | written by the pipeline / project notes |
| Appendix A.5 (D4 reconstruction diagnostic) | n/a | `scripts/check_mondal_artifact.py` |
| Fig. 1 (pairwise shift) | `results/shift_analysis/` | `run_all.py` stage 2, `scripts/make_figures.py` |
| Figs. 2–6 | `results/figures/` | `scripts/make_figures.py` |
| Fig. 7 (example intervals) | n/a | `scripts/make_example_intervals_figure.py` |
| Section 6.7 (timing) | `results/lodo_evaluation/timing_results.csv` | `scripts/benchmark_timing.py` |
| First-round ablations (strength-quartile draw) | `results/ablation/` | `scripts/run_stacked_only_ablation.py`, `scripts/run_exact_jkp_sensitivity.py` |

`scripts/revision_numbers.py` is the single place where the numbers printed in the
paper are computed from the archived CSVs (per-laboratory means with percentile
bootstrap intervals over the fifty draws, equally weighted pooled means, leave-one-
laboratory-out pooled figures, Holm-adjusted Wilcoxon tests with rank-biserial
effect sizes).

### Runs of the originally submitted version

`results/FINAL_RESULTS_confirmatory_142-191.csv` and `results/extension_e/` are the
runs of the originally submitted version (strength-quartile draw, single-refit
construction); they are kept because the development history in
`docs/method_evolution.md` refers to them.

## Reproducibility notes

- Every script seeds NumPy, PyTorch and the tree learners; the calibration draw of a
  given seed and fold is deterministic and shared by every method (asserted in code).
- GPU training of LightGBM is not bit-reproducible across driver versions; results
  regenerated on other hardware agree to the precision reported in the paper, and the
  archived CSVs are the reference.
- The 900 per-seed calibration-index files in `results/diagnostics/` and
  `results/extension_e*/k_indices/` record exactly which target mixes were labeled in
  each run.

## License

The code is released under the MIT License (see `LICENSE`). The datasets cached under
`data/raw/` keep the licences of their sources (see `data/README.md`): UCI Machine
Learning Repository #182 (CC BY 4.0) and the open-access supplementary table of
Zviazhynski et al. (2025, *Data-Centric Engineering*, CC BY 4.0). The D4 file is a
reconstruction generated by this code from published mix designs, not measured data.

## Citation

The repository is archived at Zenodo under the concept DOI https://doi.org/10.5281/zenodo.23195294,
which always resolves to the latest version. See `CITATION.cff`. Until the article
appears, please cite it as

> Ghasemian, J., Alipour, Y., Talebpour, M. H. (2026). Reliable Cross-Laboratory
> Prediction Intervals for Concrete Compressive Strength under Distribution Shift.
> *Computers and Concrete*, under review.

## Contact

Corresponding author: Javad Ghasemian, Damghan University. Code: Yasin Alipour
(yasin.alipour.1981@gmail.com).

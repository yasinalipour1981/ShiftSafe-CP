# Method evolution: ShiftSafe-CP

## v1 (pre-patch)
Weighted Mondrian conformal with inert density-ratio (train vs calib).
Coverage collapsed to ~3–5% under shift.

## v2 (ShiftSafe-CP)
Three-layer wrapper: density-ratio reweighting (source vs unlabeled target),
Mondrian cells, heteroscedastic σ̂(x). Corrected B1–B4 bugs.
Still fails under near-disjoint support (ESS < 10%, infinite quantiles).

## v3 (TransferCal-CP)
Target-anchored split conformal: quantile computed exclusively on K labeled
target points. Source model provides μ̂ and σ̂ for efficiency only.
Guaranteed ≥ 1−α coverage on target (Vovk; Lei et al. 2018).

## v3.2 (fairness audit)
TargetOnlyCP corrected to K_fit/K_cal split (was in-sample).
Primary endpoint switched to Winkler score.

## final (AdaptiveJackknifeCP)
Per-LOO candidate selection among local-ridge, affine-transfer, stacked,
local-gbm, boost-transfer. final.1 guards: σ floor, n_fit gating (≥15 for trees),
interpolation fallback to stacked.

### Interval construction (originally submitted confirmatory run)
**Confirmatory primary used `exact_jkp=False` (default)** in
`src/experiments.py` (`AdaptiveJackknifeCP(..., K=K, seed=...)`, flag omitted).

That path is the **practical shortcut**, not Barber et al. 2021 Jackknife+:
- Fit: K leave-one-out candidate fits → LOO residual scores `R_i`.
- Predict: single full-K refit center `μ_full(x)` ± `q·σ`, with
  `q = conformal_quantile({R_i}, α)`.

The exact Barber path (`exact_jkp=True`) is implemented and verified:
- lower = ⌊α(K+1)⌋-th smallest of `{μ_{-i}(x) − R_i}`
- upper = ⌈(1−α)(K+1)⌉-th smallest of `{μ_{-i}(x) + R_i}`
(asymmetric; no `base_model ± q` fallback). Because K≤50 this is cheap;
appendix / D4 smoke reports exact vs practical side-by-side. **Primary tables
remain the practical construction** that confirmatory seeds 142–191 used.

## Extension E (external SOTA, post-confirmatory)
Added **external** comparators on identical K-draws (seeds 142–191, D1/D2/D4):

| Method | Scope | Guarantee |
|--------|-------|-----------|
| CQR-target | target K_fit/K_cal=10/10 | target |
| GPR-target | K labels, GP posterior | none |
| GPR-transfer | GP on source μ̂ residuals | none |
| JKplus-source | **true Barber JK+** on source cal (300 cap, 10 reps/fold) | source |
| TabPFN-target | TabPFN v2 + split conformal | target |
| WeightedCP-v2 | ShiftSafe L1 density-ratio only | source |

**JKplus-source bug (2026-07-11):** v1 `JackknifeP.predict_intervals` discarded
LOO models and used `base_model(x) ± residual_quantile` (SplitCP hybrid) →
spurious ~41% coverage on D4 smoke. Fixed to retain LOO models and emit
asymmetric Barber intervals. Source-only domain asserts + poison tests added.

**Skipped:** NGBoost (compute cost, redundant with GPR); Mondrian ShiftSafe v2
(retained as historical ablation only, superseded by v3/final).

**Primary E-endpoint:** Winkler at K=20, α=0.10, AdaptiveJackknifeCP vs each
E-baseline, paired Wilcoxon, Holm across E-family. Original confirmatory
AdaptiveJackknifeCP vs TargetOnlyCP stands independently.

## Revision after first-round review (2026-09): what the paper now reports

The section above describes the originally submitted run. For the revised
manuscript the confirmatory grid was re-executed with two changes, both set in
`configs/final_confirmatory_revised.yaml`:

- `calibration_draw: random`, the K labeled target mixes are drawn by simple
  random sampling, without reference to their strengths (the submitted version
  used a strength-quartile draw, which presupposes the outcome).
- `exact_jackknife_plus: true`, the exact Barber et al. (2021) jackknife-plus is
  the primary construction for every confirmatory result; the single-refit form is
  reported only as an empirical approximation (manuscript Appendix A.3).

Archived output: `results/FINAL_RESULTS_confirmatory_REVISED_142-191.csv`; the
external comparators were re-run on the same draws
(`configs/extension_e_revised.yaml`, `results/extension_e_revised/`). The paper
names the method **TargetAnchoredJK+** (class `AdaptiveJackknifeCP` in the code).
Two comparators were added for the review: **TargetOnlyJK+** (`LocalOnlyJackknifeCP`,
same leave-one-out construction, target-only candidate pool) and **StackedOnlyCP**
(fixed stacked centre), both in `scripts/run_revision_experiments.py` with output
`results/revision/REVISION_draws_K20.csv`. The statement above that "primary tables
remain the practical construction" therefore applies to the submitted version only.

## Second-round review (2026-10)

Wording clarifications only (nominal vs. empirical vs. guaranteed coverage; the
formal guarantee restricted to the simple random draw; "without D4" meaning D4
withheld as a *target* laboratory while it remains in the source pool of the other
folds).


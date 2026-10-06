# Revision analysis: ShiftSafe-CP (26M-08-239)

Seeds 142–191 (50), folds D1, D2, D4, K = 20, alpha = 0.10.

All pooled figures weight each laboratory equally, so that the largest evaluation set does not dominate the mean.


## 1. Effect of the calibration draw (R1.1, R2.4)

`strength_quartile` is the draw used in the submitted manuscript; it forms strata from the response and is therefore not implementable before the target specimens are tested. `random` is a simple random draw and is the revised primary analysis. `covariate` stratifies on water/cement ratio and age, both fixed at mix design.


**draw = random** (pooled over D1, D2, D4)

| method                           |   coverage |   width |   winkler |
|:---------------------------------|-----------:|--------:|----------:|
| AdaptiveJackknifeCP              |      0.91  |  29.1   |    37.918 |
| AdaptiveJackknifeCP-exactJK+     |      0.906 |  27.693 |    36.526 |
| StackedOnlyCP                    |      0.91  |  29.03  |    37.853 |
| TargetOnlyJKplus                 |      0.916 |  36.743 |    50.148 |
| TargetOnlyJKplus-exactJK+        |      0.903 |  34.603 |    49.49  |
| TransferCal-CP-affine-localsigma |      0.909 |  37.901 |    46.391 |
| AdaptiveSplitCP                  |      0.915 |  52.31  |    69.671 |
| TargetOnlyCP                     |      0.91  |  53.188 |    70.361 |
| SplitCP-source                   |      0.044 |   2.998 |   240.755 |


**draw = covariate** (pooled over D1, D2, D4)

| method                           |   coverage |   width |   winkler |
|:---------------------------------|-----------:|--------:|----------:|
| AdaptiveJackknifeCP              |      0.919 |  28.173 |    37.572 |
| AdaptiveJackknifeCP-exactJK+     |      0.916 |  27.411 |    36.738 |
| StackedOnlyCP                    |      0.919 |  28.17  |    37.605 |
| TargetOnlyJKplus                 |      0.916 |  35.857 |    48.593 |
| TargetOnlyJKplus-exactJK+        |      0.908 |  34.537 |    48.184 |
| TransferCal-CP-affine-localsigma |      0.916 |  37.36  |    44.898 |
| AdaptiveSplitCP                  |      0.915 |  50.853 |    63.012 |
| TargetOnlyCP                     |      0.914 |  49.447 |    65.134 |
| SplitCP-source                   |      0.044 |   2.998 |   240.954 |


**draw = strength_quartile** (pooled over D1, D2, D4)

| method                           |   coverage |   width |   winkler |
|:---------------------------------|-----------:|--------:|----------:|
| AdaptiveJackknifeCP              |      0.924 |  30.722 |    39.489 |
| AdaptiveJackknifeCP-exactJK+     |      0.916 |  28.611 |    37.91  |
| StackedOnlyCP                    |      0.924 |  30.459 |    39.267 |
| TargetOnlyJKplus                 |      0.93  |  39.011 |    50.405 |
| TargetOnlyJKplus-exactJK+        |      0.917 |  35.988 |    48.992 |
| TransferCal-CP-affine-localsigma |      0.927 |  39.628 |    45.704 |
| AdaptiveSplitCP                  |      0.914 |  52.695 |    69.191 |
| TargetOnlyCP                     |      0.92  |  56.95  |    71.77  |
| SplitCP-source                   |      0.044 |   2.998 |   240.887 |


## 2. Per-laboratory results under the primary (random) draw (R1.4, R2.5)

Intervals are 95% percentile bootstrap over the 50 seeds *within* a fold. They quantify sensitivity to the calibration draw only. They are not between-laboratory uncertainty: for that the sample size is three.


**D1**

| method                           |   coverage |   width |   winkler |   winkler_lo |   winkler_hi |
|:---------------------------------|-----------:|--------:|----------:|-------------:|-------------:|
| AdaptiveJackknifeCP              |      0.93  |  53.708 |    68.179 |       63.258 |       73.821 |
| AdaptiveJackknifeCP-exactJK+     |      0.919 |  50.359 |    65.224 |       60.203 |       70.758 |
| StackedOnlyCP                    |      0.93  |  53.509 |    68.013 |       62.87  |       73.6   |
| TargetOnlyJKplus                 |      0.918 |  58.615 |    84.704 |       75.553 |       95.726 |
| TargetOnlyJKplus-exactJK+        |      0.9   |  54.435 |    82.98  |       74.545 |       92.902 |
| TransferCal-CP-affine-localsigma |      0.902 |  47.6   |    57.821 |       55.089 |       60.803 |
| AdaptiveSplitCP                  |      0.921 | 110.38  |   151.288 |      118.727 |      187.278 |
| TargetOnlyCP                     |      0.906 |  94.5   |   132.083 |      107.277 |      160.052 |
| SplitCP-source                   |      0.075 |   2.584 |   176.956 |      175.434 |      178.374 |


**D2**

| method                           |   coverage |   width |   winkler |   winkler_lo |   winkler_hi |
|:---------------------------------|-----------:|--------:|----------:|-------------:|-------------:|
| AdaptiveJackknifeCP              |      0.897 |  19.224 |    26.949 |       24.874 |       29.738 |
| AdaptiveJackknifeCP-exactJK+     |      0.894 |  18.731 |    25.91  |       24.734 |       27.354 |
| StackedOnlyCP                    |      0.897 |  19.212 |    26.92  |       24.907 |       29.729 |
| TargetOnlyJKplus                 |      0.913 |  20     |    25.657 |       24.861 |       26.536 |
| TargetOnlyJKplus-exactJK+        |      0.903 |  18.828 |    25.06  |       24.443 |       25.719 |
| TransferCal-CP-affine-localsigma |      0.909 |  27.153 |    33.716 |       32.198 |       35.583 |
| AdaptiveSplitCP                  |      0.915 |  25.992 |    32.054 |       29.365 |       35.085 |
| TargetOnlyCP                     |      0.897 |  21.519 |    27.951 |       26.766 |       29.28  |
| SplitCP-source                   |      0.001 |   3.205 |   337.276 |      333.356 |      340.985 |


**D4**

| method                           |   coverage |   width |   winkler |   winkler_lo |   winkler_hi |
|:---------------------------------|-----------:|--------:|----------:|-------------:|-------------:|
| AdaptiveJackknifeCP              |      0.903 |  14.369 |    18.627 |       17.693 |       19.587 |
| AdaptiveJackknifeCP-exactJK+     |      0.904 |  13.989 |    18.443 |       17.484 |       19.467 |
| StackedOnlyCP                    |      0.903 |  14.369 |    18.627 |       17.697 |       19.566 |
| TargetOnlyJKplus                 |      0.917 |  31.613 |    40.083 |       38.549 |       41.721 |
| TargetOnlyJKplus-exactJK+        |      0.906 |  30.544 |    40.429 |       38.912 |       42.16  |
| TransferCal-CP-affine-localsigma |      0.915 |  38.949 |    47.637 |       44.9   |       50.895 |
| AdaptiveSplitCP                  |      0.909 |  20.559 |    25.671 |       23.105 |       28.47  |
| TargetOnlyCP                     |      0.928 |  43.545 |    51.049 |       46.808 |       56.319 |
| SplitCP-source                   |      0.055 |   3.207 |   208.034 |      202.966 |      213.022 |


## 3. Pooled summary with and without the reconstructed domain D4 (R1.5, R2.2)

| method                           |   cov (all) |   cov (no D4) |   width (all) |   width (no D4) |   Winkler (all) |   Winkler (no D4) |
|:---------------------------------|------------:|--------------:|--------------:|----------------:|----------------:|------------------:|
| AdaptiveJackknifeCP              |       0.91  |         0.914 |        29.1   |          36.466 |          37.918 |            47.564 |
| AdaptiveJackknifeCP-exactJK+     |       0.906 |         0.907 |        27.693 |          34.545 |          36.526 |            45.567 |
| StackedOnlyCP                    |       0.91  |         0.913 |        29.03  |          36.361 |          37.853 |            47.467 |
| TargetOnlyJKplus                 |       0.916 |         0.916 |        36.743 |          39.307 |          50.148 |            55.18  |
| TargetOnlyJKplus-exactJK+        |       0.903 |         0.901 |        34.603 |          36.632 |          49.49  |            54.02  |
| TransferCal-CP-affine-localsigma |       0.909 |         0.906 |        37.901 |          37.377 |          46.391 |            45.768 |
| AdaptiveSplitCP                  |       0.915 |         0.918 |        52.31  |          68.186 |          69.671 |            91.671 |
| TargetOnlyCP                     |       0.91  |         0.901 |        53.188 |          58.009 |          70.361 |            80.017 |
| SplitCP-source                   |       0.044 |         0.038 |         2.998 |           2.894 |         240.755 |           257.116 |


**Leave-one-domain-out recomputation of the pooled Winkler ranking.** Each column drops one laboratory; a claim that survives all three columns is not an artefact of any single laboratory.

| method                           |   without D1 |   without D2 |   without D4 |   all three |
|:---------------------------------|-------------:|-------------:|-------------:|------------:|
| AdaptiveJackknifeCP              |        22.79 |        43.4  |        47.56 |       37.92 |
| AdaptiveJackknifeCP-exactJK+     |        22.18 |        41.83 |        45.57 |       36.53 |
| StackedOnlyCP                    |        22.77 |        43.32 |        47.47 |       37.85 |
| TargetOnlyJKplus                 |        32.87 |        62.39 |        55.18 |       50.15 |
| TargetOnlyJKplus-exactJK+        |        32.74 |        61.7  |        54.02 |       49.49 |
| TransferCal-CP-affine-localsigma |        40.68 |        52.73 |        45.77 |       46.39 |
| AdaptiveSplitCP                  |        28.86 |        88.48 |        91.67 |       69.67 |
| TargetOnlyCP                     |        39.5  |        91.57 |        80.02 |       70.36 |
| SplitCP-source                   |       272.65 |       192.49 |       257.12 |      240.76 |


## 4. Where the advantage over TargetOnlyCP comes from (R1.3)

`TargetOnlyCP` spends half of K on fitting and half on calibration. `TargetOnlyJKplus` spends all of K through the same leave-one-out construction as the proposed method but with a candidate pool restricted to target-only predictors, so the source model is never consulted. The difference between the two isolates the calibration construction; the remaining difference to `AdaptiveJackknifeCP` isolates source transfer.

| fold   |   TargetOnlyCP |   TargetOnlyJKplus |   AdaptiveJackknifeCP |   total_gain |   gain_from_LOO_construction |   gain_from_source_transfer |   pct_from_construction |   pct_from_transfer |
|:-------|---------------:|-------------------:|----------------------:|-------------:|-----------------------------:|----------------------------:|------------------------:|--------------------:|
| D1     |         132.08 |              84.7  |                 68.18 |        63.9  |                        47.38 |                       16.53 |                   74.14 |               25.86 |
| D2     |          27.95 |              25.66 |                 26.95 |         1    |                         2.29 |                       -1.29 |                  229.13 |             -129.13 |
| D4     |          51.05 |              40.08 |                 18.63 |        32.42 |                        10.97 |                       21.46 |                   33.82 |               66.18 |


## 5. Exact Barber Jackknife+ vs the practical construction (R1.2, R2.3)

| fold   |   ('coverage', 'AdaptiveJackknifeCP') |   ('coverage', 'AdaptiveJackknifeCP-exactJK+') |   ('coverage', 'TargetOnlyJKplus') |   ('coverage', 'TargetOnlyJKplus-exactJK+') |   ('winkler', 'AdaptiveJackknifeCP') |   ('winkler', 'AdaptiveJackknifeCP-exactJK+') |   ('winkler', 'TargetOnlyJKplus') |   ('winkler', 'TargetOnlyJKplus-exactJK+') |
|:-------|--------------------------------------:|-----------------------------------------------:|-----------------------------------:|--------------------------------------------:|-------------------------------------:|----------------------------------------------:|----------------------------------:|-------------------------------------------:|
| D1     |                                 0.93  |                                          0.919 |                              0.918 |                                       0.9   |                               68.179 |                                        65.224 |                            84.704 |                                     82.98  |
| D2     |                                 0.897 |                                          0.894 |                              0.913 |                                       0.903 |                               26.949 |                                        25.91  |                            25.656 |                                     25.06  |
| D4     |                                 0.903 |                                          0.904 |                              0.917 |                                       0.906 |                               18.627 |                                        18.443 |                            40.083 |                                     40.429 |
| pooled |                                 0.91  |                                          0.906 |                              0.916 |                                       0.903 |                               37.918 |                                        36.526 |                            50.148 |                                     49.49  |


## 6. Does adaptive candidate selection contribute? (R1.6, R2.6)

Candidate selected by the full-K refit, primary draw:

| selected_candidate   |   n |
|:---------------------|----:|
| stacked              | 147 |
| local-ridge          |   3 |

Adaptive selection vs a fixed stacked centre:

| method              |   coverage |   width |   winkler |
|:--------------------|-----------:|--------:|----------:|
| AdaptiveJackknifeCP |     0.91   | 29.1004 |   37.9182 |
| StackedOnlyCP       |     0.9098 | 29.0303 |   37.8534 |

Paired difference (Adaptive − StackedOnly): mean 0.0649, max |diff| 4.9282, identical in 96.7% of the 150 seed-by-fold cells.

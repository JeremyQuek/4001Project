# Logistic-regression detector baselines

Eight fitted detectors are saved. Each StandardScaler and classifier was fitted only on the training split. Validation selected C, class weighting and provisional abstention thresholds. The test split was not opened by this training/reporting stage.

Target: 1 = the original frozen VLM answer disagrees with VSR truth; 0 = it agrees. Outputs are error-risk rankings, not assumed calibrated correctness probabilities.

## Protocol

C grid: 0.001, 0.01, 0.1, 1, 10. Class weighting: none or balanced. Eight experiments, 80 candidate fits. L2-regularized logistic regression, LBFGS, at most 5,000 iterations, seed 42, one numerical thread. Selection: validation AUROC, then average precision, then stronger regularization and unweighted fits for exact ties. All selected fits converged and reproduced predictions after save/reload.

The VLM and feature-extraction configuration are unchanged. Baselines in each cohort use identical training/validation examples. Do not compare the 148-row and 95-row cohorts as if they were the same evaluation set.

## All examples

Training: 659 pairs. Validation: 148 pairs, 30 errors. Average-precision reference prevalence: 0.203.

| Detector | Validation AUROC | Average precision | 95% AUROC interval |
|---|---:|---:|---|
| Raw VLM confidence | 0.716 | 0.416 | 0.608–0.823 |
| LR: confidence features | 0.742 | 0.430 | 0.638–0.839 |
| LR: 4 core features | 0.744 | 0.447 | 0.639–0.844 |
| LR: layer 9 hidden state | 0.569 | 0.247 | 0.442–0.682 |
| LR: layer 18 hidden state | 0.575 | 0.236 | 0.464–0.677 |
| LR: layer 36 hidden state | 0.716 | 0.335 | 0.623–0.810 |

## Consistency-eligible vertical subset

Training: 436 pairs. Validation: 95 pairs, 21 errors. Average-precision reference prevalence: 0.221.

| Detector | Validation AUROC | Average precision | 95% AUROC interval |
|---|---:|---:|---|
| Raw VLM confidence | 0.760 | 0.462 | 0.632–0.865 |
| LR: confidence features | 0.786 | 0.483 | 0.672–0.881 |
| LR: 4 core features | 0.792 | 0.493 | 0.680–0.882 |
| LR: 7 behavioral features | 0.786 | 0.530 | 0.674–0.885 |

## Interpretation and limits

The core detector is the practical starting baseline on all examples. Its validation AUROC is only slightly above the learned confidence-only baseline. The seven-feature detector increases average precision on the consistency subset but does not increase AUROC over its four-feature counterpart. These results do not establish that the extra diagnostic calls improve detection.

The final-layer linear hidden-state probe performs better than the early/middle probes here. The probes have 2,048 input dimensions and only 659 training examples; consult generalization_gaps.csv for train–validation gaps. This is not evidence that a particular layer contains or loses the correct spatial answer.

All reported intervals resample images and condition on the selected models. They do not account for hyperparameter-selection optimism on the same validation set. No final generalization claim should be made before the later untouched-test comparison.

## Provisional abstention policy

Thresholds target at most 80% validation coverage, accepting whole tied-score groups. They measure retained accuracy of the original VLM answers, not classification accuracy of the error detector. For the full-cohort core baseline:

- Original VLM validation accuracy: 79.7%.
- Retained: 118/148 (79.7% coverage).
- Retained VLM errors: 16; retained accuracy: 86.4%.
- Selected risk threshold: 0.336856.

Class-weighted models especially should not be presented as calibrated error probabilities. A later calibrated-probability claim requires separate validation/calibration design.

## Files and next step

- models/*.joblib: fitted scaler + classifier pipelines; JSON sidecars document inputs and settings.
- protocol.json: frozen search and data fingerprints.
- hyperparameter_search.csv: all 80 candidate validation results.
- validation_metrics.csv and validation_predictions.csv: selected-model validation outputs.
- generalization_gaps.csv: training versus validation performance.
- paired_comparisons.json: matched incremental AUROC comparisons.
- summary.json: metrics, intervals and coverage thresholds.

Next: train a small MLP with the same feature sets and cohorts, using training-only preprocessing and validation-based selection. Freeze the comparison and then evaluate the selected LR/MLP models once on test. Keep these LR artifacts unchanged for that comparison.

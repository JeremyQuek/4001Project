# Frozen MLP versus logistic-regression comparison

Final held-out test evaluation. Hyperparameters, checkpoints, primary seed 42, and abstention thresholds were frozen before test inference. No test tuning or seed selection was performed.

The primary comparisons do not establish an MLP advantage. On all relations, the core MLP scored 0.746 AUROC versus 0.755 for logistic regression (paired difference −0.010, 95% interval [−0.034, +0.012]). On the vertical consistency subset, the behavior MLP scored 0.810 versus 0.802 (difference +0.008, interval [−0.026, +0.043]). Both intervals include zero. Keep the practical logistic core baseline; retain the MLP as a comparison rather than claiming a demonstrated improvement.

The more promising signal is consistency information: on the same vertical subset, MLP core/behavior AUROC was 0.774/0.810 and LR core/behavior AUROC was 0.761/0.802. The MLP increment remains uncertain (paired interval [−0.025, +0.107]); a larger independent evaluation is needed. Hidden-state MLPs did not beat the practical core baseline on the primary seed; early/middle probes had large train/test gaps. These results support testing behavioral features further, rather than adding model capacity alone.

## Method

Two learned layers: standardized inputs → Linear(32) → ReLU → Dropout(0.2) → Linear(1). Sigmoid converts the output logit to an error-risk score. The VLM and its extracted features were unchanged.

Scalers and weights used training only. Adam, learning rate 0.001, batch size 32, up to 200 epochs; early stopping after 15 epochs without validation-BCE improvement, restoring the best checkpoint. A fixed search compared weight decay 0.0001/0.01 and unweighted/training-balanced BCE. Settings were selected by mean validation AUROC across seeds 42/43/44, then mean AP. Seed 42 was predeclared for the primary test comparison; the three runs are not an ensemble.

Eight matched feature/cohort experiments, 32 candidate configurations, 96 seed fits; training took 194.8 seconds on CPU. No new VLM calls were needed.

Target: `1` means the original VLM answer disagrees with the accepted VSR label; `0` means it agrees. This measures answer error, not intent to deceive. Inputs exclude labels, image IDs, and dataset split.

AUROC measures error ranking (0.5 random). AP is average precision for the error class; its chance reference is the cohort error prevalence. Higher AUROC/AP is better. Brier scores are descriptive and do not establish probability calibration.

## All relations

Test: 134 pairs across 119 images, 25 VLM errors (18.7%). Unfiltered VLM accuracy: 81.3%.

| Inputs | LR validation AUROC | MLP validation mean ± SD | LR test AUROC | MLP test AUROC (seed 42) | LR / MLP test AP | MLP test mean ± SD (3 seeds) |
|---|---:|---:|---:|---:|---:|---:|
| Raw confidence risk | 0.716 | — | 0.718 | — | 0.361 / — | — |
| Confidence (2) | 0.742 | 0.740 ± 0.001 | 0.745 | 0.742 | 0.382 / 0.377 | 0.742 ± 0.001 |
| Core (4) | 0.744 | 0.747 ± 0.001 | 0.755 | 0.746 | 0.398 / 0.394 | 0.754 ± 0.007 |
| Hidden layer 9 (2048) | 0.569 | 0.576 ± 0.025 | 0.599 | 0.559 | 0.242 / 0.222 | 0.539 ± 0.018 |
| Hidden layer 18 (2048) | 0.575 | 0.618 ± 0.011 | 0.483 | 0.549 | 0.192 / 0.228 | 0.498 ± 0.073 |
| Hidden layer 36 (2048) | 0.716 | 0.753 ± 0.012 | 0.708 | 0.743 | 0.380 / 0.386 | 0.757 ± 0.017 |

### Frozen validation-threshold abstention policy

Each threshold targeted at most 80% coverage on validation, keeping whole ties. These unchanged thresholds can yield different coverage on test; operating-point accuracies are therefore not matched-coverage comparisons.

| Model | Test answered | Test coverage | Errors retained | Accuracy on answered | Errors caught / total |
|---|---:|---:|---:|---:|---:|
| Raw confidence | 104/134 | 77.6% | 14 | 86.5% | 11/25 |
| Logistic regression | 103/134 | 76.9% | 10 | 90.3% | 15/25 |
| MLP, seed 42 | 107/134 | 79.9% | 16 | 85.0% | 9/25 |

### Paired AUROC differences

Image-cluster bootstrap, 2,000 resamples. A 95% interval spanning zero does not establish a reliable improvement. Intervals condition on the frozen fits; seed variability is separate. The core full-cohort and behavior subset comparisons were predeclared primary; other feature comparisons are exploratory, without multiplicity adjustment.

| MLP minus matched LR | Difference | 95% paired interval |
|---|---:|---:|
| Confidence (2) | -0.003 | [-0.011, +0.000] |
| Core (4) | -0.010 | [-0.034, +0.012] |
| Hidden layer 9 (2048) | -0.040 | [-0.124, +0.056] |
| Hidden layer 18 (2048) | +0.066 | [-0.026, +0.142] |
| Hidden layer 36 (2048) | +0.035 | [-0.025, +0.099] |

### MLP generalization and selected settings

| Inputs | Train AUROC (seed 42) | Validation AUROC (seed 42) | Test AUROC (seed 42) | Selected decay | Class weights | Best epoch |
|---|---:|---:|---:|---:|---|---:|
| Confidence (2) | 0.769 | 0.741 | 0.742 | 0.01 | None | 67 |
| Core (4) | 0.775 | 0.746 | 0.746 | 0.0001 | None | 17 |
| Hidden layer 9 (2048) | 0.940 | 0.590 | 0.559 | 0.01 | None | 11 |
| Hidden layer 18 (2048) | 0.866 | 0.616 | 0.549 | 0.0001 | None | 5 |
| Hidden layer 36 (2048) | 0.883 | 0.744 | 0.743 | 0.01 | None | 3 |

## Vertical consistency subset

Test: 87 pairs across 80 images, 17 VLM errors (19.5%). Unfiltered VLM accuracy: 80.5%.

| Inputs | LR validation AUROC | MLP validation mean ± SD | LR test AUROC | MLP test AUROC (seed 42) | LR / MLP test AP | MLP test mean ± SD (3 seeds) |
|---|---:|---:|---:|---:|---:|---:|
| Raw confidence risk | 0.760 | — | 0.713 | — | 0.364 / — | — |
| Confidence (2) | 0.786 | 0.782 ± 0.001 | 0.749 | 0.758 | 0.388 / 0.444 | 0.760 ± 0.003 |
| Core (4) | 0.792 | 0.789 ± 0.007 | 0.761 | 0.774 | 0.414 / 0.478 | 0.774 ± 0.000 |
| Behavior (7) | 0.786 | 0.791 ± 0.005 | 0.802 | 0.810 | 0.494 / 0.539 | 0.804 ± 0.007 |

### Frozen validation-threshold abstention policy

Each threshold targeted at most 80% coverage on validation, keeping whole ties. These unchanged thresholds can yield different coverage on test; operating-point accuracies are therefore not matched-coverage comparisons.

| Model | Test answered | Test coverage | Errors retained | Accuracy on answered | Errors caught / total |
|---|---:|---:|---:|---:|---:|
| Raw confidence | 68/87 | 78.2% | 9 | 86.8% | 8/17 |
| Logistic regression | 64/87 | 73.6% | 5 | 92.2% | 12/17 |
| MLP, seed 42 | 61/87 | 70.1% | 4 | 93.4% | 13/17 |

### Paired AUROC differences

Image-cluster bootstrap, 2,000 resamples. A 95% interval spanning zero does not establish a reliable improvement. Intervals condition on the frozen fits; seed variability is separate. The core full-cohort and behavior subset comparisons were predeclared primary; other feature comparisons are exploratory, without multiplicity adjustment.

| MLP minus matched LR | Difference | 95% paired interval |
|---|---:|---:|
| Confidence (2) | +0.009 | [-0.004, +0.026] |
| Core (4) | +0.013 | [+0.003, +0.028] |
| Behavior (7) | +0.008 | [-0.026, +0.043] |

Adding the three consistency features to the MLP on this same subset changes test AUROC by +0.036, paired 95% interval [-0.025, +0.107].

### MLP generalization and selected settings

| Inputs | Train AUROC (seed 42) | Validation AUROC (seed 42) | Test AUROC (seed 42) | Selected decay | Class weights | Best epoch |
|---|---:|---:|---:|---:|---|---:|
| Confidence (2) | 0.798 | 0.783 | 0.758 | 0.01 | None | 54 |
| Core (4) | 0.798 | 0.788 | 0.774 | 0.0001 | None | 31 |
| Behavior (7) | 0.830 | 0.797 | 0.810 | 0.0001 | None | 73 |

## Interpretation limits and next steps

The primary comparison is seed 42, fixed before test. Three-seed mean ± sample SD describes initialization/training variability and is not a confidence interval or an ensemble. Do not choose a seed or layer using the test table. Small test cohorts yield substantial sampling uncertainty.

Consistency features are evaluated only on the vertical subset (above/below). Its metrics cannot be compared directly with the all-relations cohort. Input dimensionality is large relative to the number of training errors for hidden-state probes; train/validation gaps and seed sensitivity matter. Probe performance alone does not show that a model knew the correct answer internally.

Risk scores rank likely answer errors; they are not established calibrated probabilities. Abstention improves reliability by declining questions, rather than correcting answers. Images were split without overlap, but this is one VSR-derived sample and one fixed VLM/configuration. External validity is still untested.

This held-out split has now been used. Future architectural or feature tuning needs a fresh holdout or an external benchmark such as a separately prepared R-Bench evaluation. Retain the frozen logistic baseline, quantify gains on that new evaluation, and calibrate/choose any deployment policy on validation data.

## Saved artifacts and usage

`models/{experiment}_seed42.pt` is the primary MLP; `models/{experiment}_scaler.joblib` is its training-fitted scaler. Seeds 43 and 44 are separately saved. See `validation_selection.json` for checkpoint metadata and unchanged validation thresholds. Never fit another scaler at inference.

For `all_core`, input order is `[p_original, confidence, p_blind, image_sensitivity]`, with `confidence=max(p_original,1-p_original)` and `image_sensitivity=abs(p_original-p_blind)`. The behavior input appends converse disagreement, exclusive violation, and mirror disagreement, on eligible vertical examples only. Hidden models take the original-prompt 2,048-value vector at their named layer.

```python
import sys, joblib, numpy as np
sys.path.insert(0, 'scripts')
from train_mlp_detectors import load_checkpoint, risk_scores
scaler = joblib.load('outputs/mlp_comparison/models/all_core_scaler.joblib')
model, _ = load_checkpoint('outputs/mlp_comparison/models/all_core_seed42.pt')
X = np.asarray([[p_original, confidence, p_blind, image_sensitivity]], dtype=np.float32)
risk = risk_scores(model, scaler.transform(X).astype(np.float32))
```

Reproduction entry points (training deliberately refuses to run after test exposure):

```powershell
.\.venv\Scripts\python.exe scripts\train_mlp_detectors.py
.\.venv\Scripts\python.exe scripts\evaluate_detector_comparison.py
.\.venv\Scripts\python.exe scripts\report_mlp_comparison.py
```

The evaluator checks frozen file hashes and returns cached results after successful completion. The report reads cached predictions only. Preserve `frozen_comparison.json`, `test_evaluation_started.json`, `test_summary.json`, `test_predictions.csv` and model files together. The source LR report remains a historical validation-only report.

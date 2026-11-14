# Detector training data

Raw inputs ready for detector training. All 941 original VSR statements and their source labels are retained. This array bundle contains no fitted preprocessing or models; training-run artifacts and status are stored separately in `outputs/logistic_baselines/`.

## Dataset and targets

| Split | Rows | Images | Consistency-eligible rows |
|---|---:|---:|---:|
| train | 659 | 596 | 436 |
| validation | 148 | 140 | 95 |
| test | 134 | 119 | 87 |

Training targets: 536 correct answers (0), 123 incorrect answers (1). Validation/test accuracy is not analysed here.

`y_error = int(vlm_prediction != truth_label)`. Original predictions are Yes when P(Yes) >= 0.5. Source truth labels and original predictions are metadata, never part of X.

## Arrays in each split NPZ

| Key | Dimensions | Meaning |
|---|---|---|
| X_confidence | N × 2 | p_original, confidence |
| X_core | N × 4 | p_original, confidence, p_blind, image_sensitivity |
| X_behavior | N × 7 | Core plus converse disagreement, exclusive violation, mirror disagreement |
| H_layer9 / H_layer18 / H_layer36 | N × 2048 each | Original-prompt hidden state at each selected decoder layer |
| y_error | N | Target: 1 = original answer incorrect |
| consistency_mask | N | Which rows support the 7-feature comparison |
| example_ids / image_ids / relations | N | Alignment and metadata |

The hidden state is saved at the final prompt token before answering, for every image–statement pair. Arrays are float32 conversions of captured bfloat16 vectors. No feature standardization has been fitted.

## Consistency eligibility

All originals retained. Vertical consistency rules plus explicit human approvals; other auxiliary features unavailable. No numeric reference-frame codes inferred.

This gives 618 consistency-eligible rows (vertical relations), while all 941 rows support core and hidden-state detectors. Unavailable auxiliary columns are NaN, not zero. Use consistency_mask or the loader below; do not feed the full NaN-containing X_behavior array directly to a classifier. No earlier agent quality flag removes an original row. Exclusive statements have no automatically assigned truth label.

## Load data without training

From the project root:

```python
import sys
sys.path.insert(0, "scripts")
from load_detector_data import load_detector_data

X_train, y_train, info = load_detector_data("train", "core")
H_train, y_hidden, _ = load_detector_data("train", "hidden", layer=18)

# Compare consistency features and confidence on exactly the same rows:
B_train, y_behavior, b_info = load_detector_data(
    "train", "behavior", cohort="consistency")
C_train, y_confidence, c_info = load_detector_data(
    "train", "confidence", cohort="consistency")
assert (b_info["example_ids"] == c_info["example_ids"]).all()
```

Feature sets: confidence (2), core (4), behavior (7), hidden (2048), combined_core (2052), combined_behavior (2055). The last combines seven features and one chosen layer, and requires cohort="consistency". Use cohort="consistency" for every baseline in that ablation, including hidden-only comparisons.

Fit StandardScaler and logistic regression using training data only. Choose regularization, hidden layer, feature groups and abstention threshold with validation data. Keep the test split for the final detector evaluation. The files preserve image separation and metadata row order; index JSON files link rows to original statements.

## Reproduce / resume

```powershell
.\.venv\Scripts\python.exe scripts\extract_vsr_features.py
.\.venv\Scripts\python.exe scripts\package_vsr_features.py
.\.venv\Scripts\python.exe scripts\load_detector_data.py
```

Raw per-pair observations and individual hidden-state files are in outputs/vsr_full/. The signature pins model revision, processing settings, package versions and the scoring function. Complete cached records are validated before resumption. metadata.json lists artifact checksums; verification.json records the final integrity checks.

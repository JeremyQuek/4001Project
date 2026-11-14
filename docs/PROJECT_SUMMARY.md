# Frozen-VLM spatial answer-error detection: completed project snapshot

Snapshot date: 14 November 2026. Dataset expansion has been discussed but has not
been executed. This snapshot records the completed 941-example experiment.

## Goal and scope

Predict whether a frozen vision-language model's original spatial Yes/No answer
is incorrect. A separate small supervised detector observes answer scores,
behavior under controlled input changes, or saved hidden states. It returns a
scalar error-risk score. A validation-selected threshold can abstain on risky
questions. The detector does not correct the answer or update the VLM.

The accepted VSR True/False label defines answer correctness. Here, the term
hallucination is operationalized as a wrong binary answer, rather than a claim
about deception, explanation faithfulness, or unrestricted generated captions.

## Models and environment

| Component | Configuration |
|---|---|
| Frozen VLM | Qwen/Qwen2.5-VL-3B-Instruct |
| VLM revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Quantization | 4-bit NF4, double quantization; visual module and lm_head excluded |
| Computation | bfloat16 on NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB; SDPA; batch size 1 |
| Image budget | max_pixels 200,704, approximately 256 visual tokens; min_pixels 3,136 |
| LR detectors | Training-fitted StandardScaler followed by L2 logistic regression |
| MLP detectors | Training-fitted StandardScaler; Linear(input,32), ReLU, dropout 0.2, Linear(32,1); sigmoid at inference |
| Detector compute | CPU; no additional VLM calls needed for fitting detectors |

The local environment used Python 3.12.5, torch 2.9.1+cu126, torchvision
0.24.1+cu126, transformers 4.57.6, accelerate 1.12.0, bitsandbytes 0.49.2,
Pillow 12.3.0 and huggingface-hub 0.36.2. Analysis used scikit-learn 1.9.1,
NumPy 2.5.4 and matplotlib 3.11.2. Both environment snapshots and installation
scripts are included. The VLM weights are downloaded separately into the ignored
local Hugging Face cache; no VLM fine-tuning was performed.

## Data and splitting

VSR metadata came from `cambridgeltl/visual-spatial-reasoning` at revision
`b27a0af0ee1462d2b6b92c8c83e869d9254a241a`. The local metadata has 10,972
statements across 6,259 images and 66 relation labels. The completed experiment
filters to above (341), below (277), left of (210), and right of (113):
941 statements across 855 images. No further examples for these four relations
remain in this cached release.

The source random splits shared 1,400 image identities. All unique image IDs
were therefore assigned globally to 70/15/15 train/validation/test groups with
seed 42 before relation filtering. Every statement for an image inherits its
image's assignment. The resulting filtered counts need not be exactly 70/15/15.

| Split | Statements | Images | Original VLM errors | Consistency-eligible statements |
|---|---:|---:|---:|---:|
| Training | 659 | 596 | 123 | 436 |
| Validation | 148 | 140 | 30 | 95 |
| Test | 134 | 119 | 25 | 87 |
| Total | 941 | 855 | 178 | 618 |

Original source labels were retained for all examples, with the user's explicit
batch acceptance. The initial 100-example agent-assisted visual audit and the
human-review worksheet are recorded separately. They do not represent independent
human validation of every image or transformed statement. Existing human review
decisions were not fabricated or overwritten.

The final seven-feature cohort consists of the 618 above/below examples.
Perspective-dependent horizontal transformations were not automatically certified.
All 941 originals remain available to the core and hidden-state detectors.

## Completed steps

### 1. Set up and check local GPU inference

Installed the pinned CUDA/PyTorch and VLM dependencies, checked the GPU, and ran
a controlled shapes example before using research images. VLM weights are put
in evaluation mode with gradients disabled. The image processor and model
revision are pinned in the experiment configuration.

### 2. Prepare metadata and download selected images

Validated source checksums, labels, image URLs and example identities, parsed
the four directional relation types, and generated image-disjoint manifests.
Downloaded only the selected 855 unique COCO images. Downloads are decoded,
verified and hashed, with retries and resumable reports.

### 3. Run a training-only pilot and freeze extraction settings

Ran 100 training examples across 98 images. The original VLM answer matched
82/100 VSR labels. A 73-example provisional consistency subset had 65/73
matches; it is not the same cohort as the final 618-example vertical subset.
The pilot performed 419 scoring observations, took 274.2 seconds, and peaked at
3.281 GiB PyTorch allocated GPU memory. The matched 512-token image-budget check
also scored 82/100, fixing two errors and introducing two others, so the smaller
256-token budget was retained. This was a training-pilot choice, not test tuning.

The blank-image pilot answered No on 99/100 statements. Consequently, agreement
with a blank-image answer is not by itself proof of language-prior reliance.

### 4. Extract VLM observations and hidden states

The exact original prompt is:

```text
Statement: "{statement}"
Is this statement true in the image? Answer Yes or No.
```

At the first answer position, the scoring code verifies that canonical Yes and
No are each one token in the actual prompt context, then computes
`p_yes = softmax([logit_Yes, logit_No])[0]`. This is relative Yes/No support,
not a calibrated probability of answer correctness. Full-vocabulary Yes/No mass
and leading unrestricted tokens are also recorded for diagnostics.

All originals receive their original-image and same-size grey-image scores.
Eligible vertical examples additionally receive a converse statement, an
exclusive-relation statement, and a horizontally mirrored image with its
equivalent statement. These auxiliary calls diagnose consistency; exclusive
statements do not receive automatically fabricated ground-truth labels.

Original-prompt hidden vectors at decoder layers 9, 18 and 36 are captured at
the final prompt token before answering. Each vector has 2,048 values. This
captures the model's representation, not an interpretable thought or proof of
internal knowledge. Capturing the vectors uses the original forward pass.

Extraction was organized into 11 resumable batches. Completed observations,
image hashes, configuration signatures and hidden files are checked before reuse.
The final bundle contains 941 original records, 3,736 scoring observations and
2,823 hidden vectors. All 100 cached pilot originals/hidden records were reused
and verified. Session-specific timing files distinguish new work from reuse;
the last progress record alone is not the total historical GPU runtime.

### 5. Generate detector inputs and targets

For each image-statement pair:

```text
vlm_prediction = 1 if p_original >= 0.5 else 0
y_error = int(vlm_prediction != VSR_truth_label)
```

Thus error class 1 means the original answer was incorrect, and class 0 means
it matched VSR. Original truth and prediction are metadata; neither is included
as a detector feature. IDs and split membership are also excluded from inputs.

| Input set | Exact values, in order |
|---|---|
| Confidence, 2 | p_original; confidence = max(p_original, 1-p_original) |
| Core, 4 | Confidence inputs; p_blind; image_sensitivity = abs(p_original-p_blind) |
| Behavior, 7 | Core inputs; abs(p_original-p_converse); max(0,p_original+p_exclusive-1); abs(p_original-p_mirrored) |
| Hidden, 2048 | Original-prompt vector from one named layer: 9, 18 or 36 |

Unavailable behavioral columns are NaN with an explicit eligibility mask; they
are not filled with zero. Every behavioral comparison uses the same vertical
cohort for both models. The loader also supports concatenated core/behavior plus
hidden inputs, but those combined feature models were not trained in this study.

Raw arrays are saved unstandardized in separate train/validation/test NPZ files.
The loader validates checksums, input finiteness, target values and alignment.
Hidden vectors were converted from captured bfloat16 to float32 for packaging.

### 6. Train logistic-regression baselines

Eight experiments cover all-cohort confidence, core and layers 9/18/36, plus
vertical-cohort confidence, core and behavior features. Each used C in
`[0.001,0.01,0.1,1,10]` and class weighting none/balanced: 80 candidate fits.
LR used LBFGS, max_iter 5,000, tolerance 1e-6, seed 42. All selected fits converged.

Scaling and fitting used training only. Validation selected AUROC, then AP,
then smaller C and unweighted fitting for exact ties. Models reproduced their
predictions after save/reload. The test remained closed during this stage.

### 7. Train and select small MLPs

Used Adam with learning rate 0.001, batch size 32, maximum 200 epochs and
early-stopping patience 15. Checkpoints minimize validation BCE under a fixed
1e-5 improvement rule. Balanced BCE candidates derive the positive-class weight
from training labels only.

The fixed search tested weight decay 0.0001/0.01 and unweighted/balanced BCE,
using seeds 42/43/44 for every candidate in the same eight experiments:
32 configurations and 96 seed fits. Settings were selected by mean validation
AUROC, then mean AP, larger decay and unweighted fitting for exact ties.
All selected MLP configurations were unweighted. Training took 194.8 seconds.
Checkpoints and training-fitted scalers are saved separately and round-trip
predictions were checked exactly.

Seed 42 was predeclared for the primary comparison. Seeds 43 and 44 measure
training variability; no best-seed choice or ensemble was used. Hidden unit
count, regularization search and early-stopping rules were not tuned on test.

### 8. Freeze and evaluate once on held-out test

Model/scaler hashes, hyperparameters, primary seed and validation thresholds
were frozen before test inference. The evaluator checks those bytes and image
separation. It writes an exposure marker, saves per-example test predictions,
and returns cached results after completion. The MLP training script refuses
to retrain this comparison after test exposure.

Metrics include AUROC, error-class average precision (AP), descriptive Brier
score, risk-coverage curves and fixed-threshold abstention. Paired 95% AUROC
intervals use 2,000 image-cluster bootstrap resamples with identical draws for
the compared models. They condition on frozen fits; seed variability is separate.
Core full-cohort and behavior vertical-cohort LR/MLP comparisons were primary;
other comparisons are exploratory without multiplicity adjustment.

## Held-out results

Higher AUROC and AP are better. AUROC 0.5 is random ranking. AP's chance
reference is the cohort error prevalence: 0.187 for all relations and 0.195
for the vertical subset. These values evaluate detection of wrong answers,
not the accuracy of generated replacement answers.

### All four relation types: 134 statements, 25 errors

| Inputs | LR test AUROC | MLP test AUROC, seed 42 | LR test AP | MLP test AP |
|---|---:|---:|---:|---:|
| Raw confidence risk, 1-confidence | 0.718 (heuristic) | — | 0.361 | — |
| Confidence, 2 | 0.745 | 0.742 | 0.382 | 0.377 |
| Core, 4 | 0.755 | 0.746 | 0.398 | 0.394 |
| Hidden layer 9 | 0.599 | 0.559 | 0.242 | 0.222 |
| Hidden layer 18 | 0.483 | 0.549 | 0.192 | 0.228 |
| Hidden layer 36 | 0.708 | 0.743 | 0.380 | 0.386 |

Core MLP minus LR AUROC: -0.010; paired 95% interval [-0.034,+0.012].
Across the three MLP seeds, core test AUROC averaged 0.754 with sample SD 0.007.
The mean is not an ensemble, and the SD is not a confidence interval.

### Vertical consistency subset: 87 statements, 17 errors

| Inputs | LR test AUROC | MLP test AUROC, seed 42 | LR test AP | MLP test AP |
|---|---:|---:|---:|---:|
| Raw confidence risk | 0.713 (heuristic) | — | 0.364 | — |
| Confidence, 2 | 0.749 | 0.758 | 0.388 | 0.444 |
| Core, 4 | 0.761 | 0.774 | 0.414 | 0.478 |
| Behavior, 7 | 0.802 | 0.810 | 0.494 | 0.539 |

Behavior MLP minus LR AUROC: +0.008; paired 95% interval [-0.026,+0.043].
Across three seeds, behavior MLP test AUROC averaged 0.804 with sample SD 0.007.
For the MLP, adding consistency features to core on this same subset changes
AUROC by +0.036, with paired interval [-0.025,+0.107]. The exploratory vertical
core MLP/LR difference was +0.013, interval [+0.003,+0.028]; it is not a
multiplicity-adjusted primary finding.

### Abstention at unchanged validation thresholds

Each threshold targeted at most 80% validation coverage, retaining whole score
ties. Actual test coverage differs between models. These are practical frozen
policies, not matched-coverage accuracy comparisons.

| Cohort/policy | Questions answered | Coverage | Errors retained | Answered-question VLM accuracy |
|---|---:|---:|---:|---:|
| All, no abstention | 134/134 | 100% | 25 | 81.3% |
| All, raw confidence | 104/134 | 77.6% | 14 | 86.5% |
| All, core LR | 103/134 | 76.9% | 10 | 90.3% |
| All, core MLP | 107/134 | 79.9% | 16 | 85.0% |
| Vertical, no abstention | 87/87 | 100% | 17 | 80.5% |
| Vertical, behavior LR | 64/87 | 73.6% | 5 | 92.2% |
| Vertical, behavior MLP | 61/87 | 70.1% | 4 | 93.4% |

## Analysis and implications

1. The predeclared primary comparisons do not establish an MLP advantage over LR:
   both paired intervals include zero. Keeping the simple core LR baseline is
   reasonable while retaining all frozen MLP results for comparison.
2. Behavioral consistency is promising on the vertical subset, but the added-feature
   increment is uncertain. Subset metrics must not be compared directly with
   full-cohort metrics, since their examples differ.
3. Hidden-state probes are data hungry. The layer-9 MLP had train/validation/test
   AUROC 0.940/0.590/0.559. Large generalization gaps and seed sensitivity limit
   conclusions; these results do not establish where the model lost information
   or that it internally knew the correct answer.
4. With only 123 training errors and 25 full-cohort test errors, this is a useful
   pilot with limited power to compare architectures. More independent images
   and errors are a higher priority than increasing model size.
5. Risk is an uncalibrated ranking score. Brier scores alone do not validate
   calibration, particularly for class-weighted LR fits. Abstention improves
   retained reliability by declining questions rather than fixing answers.
6. The test has now been exposed. Future tuning requires a new evaluation design
   or independent benchmark; R-Bench evaluation has not been performed. The
   current conclusions concern one VLM, image budget and VSR-derived sample.

## What is in Git and how to use the snapshot

Included: source code, configs, tests, environment snapshots, review UI, source
metadata with URLs/checksums, processed and extraction manifests, batch approval,
raw prepared train/validation/test feature arrays, extracted hidden states,
resumable observation records, pilot/audit artifacts, LR pipelines, 24 selected
MLP seed checkpoints and eight scalers, predictions, frozen protocols and reports.
Pre-existing planning PDFs and Claude planning material are retained.

Excluded: `.venv/`, `.cache/` (including downloaded VLM weights), original
`data/vsr/images/` photos, incidental smoke-test scratch files and credentials.
Contact sheets in the historical audit are included; they do not replace original
images for future VLM inference. Exact artifact bytes and line endings are
preserved in Git because checksums and evaluation protocols depend on them.

Useful entry points:

| Location | Purpose |
|---|---|
| `scripts/setup_vlm.ps1` | Create the GPU environment and install dependencies |
| `scripts/check_gpu.py`, `scripts/run_vlm.py` | GPU/model installation checks and single-example scoring |
| `scripts/prepare_vsr.py` | Cached metadata preparation and image-disjoint candidate manifests |
| `scripts/download_vsr_images.py` | Download/verify only manifest-selected originals |
| `review/`, `REVIEW_GUIDE.md` | Manual-review UI and instructions |
| `scripts/extract_vsr_features.py` | Resume frozen VLM extraction from approved manifests |
| `scripts/load_detector_data.py` | Load aligned, checksum-verified feature inputs and error labels |
| `data/vsr/detector_ready/` | Prepared arrays, row indexes, dataset card and verification |
| `outputs/logistic_baselines/` | Historical LR training/validation outputs and fitted pipelines |
| `outputs/mlp_comparison/REPORT.md` | Final matched test comparison and inference example |
| `outputs/mlp_comparison/test_comparison.png` | AUROC intervals and risk-coverage plots |
| `outputs/mlp_comparison/models/` | Training-fitted MLP scalers and seed checkpoints |

With the pinned environment, results can be inspected without downloading photos
or loading the VLM. The following commands run checks and rebuild reports from
the saved artifacts, without model fitting:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_project_snapshot.py
.\.venv\Scripts\python.exe scripts\load_detector_data.py
.\.venv\Scripts\python.exe scripts\evaluate_detector_comparison.py
.\.venv\Scripts\python.exe scripts\report_mlp_comparison.py
```

The evaluator returns cached results after verifying the frozen hashes. Preserve
its source bytes and frozen artifacts. Do not overwrite the LR or MLP models by
rerunning training against this exposed test. Historical reports and preparation
flags intentionally describe their original stage and are not the current global
training status. The final comparison report and this summary give current status.

To restore original photos for further VLM work:

```powershell
.\.venv\Scripts\python.exe scripts\download_vsr_images.py --manifest data/vsr/processed/directional_images.csv --report outputs/directional_image_download.json
```

The source metadata URLs and pinned revision are in `data/vsr/raw/sources.json`.
No Visual Genome `relationships.json` was used in the completed experiment.

## Verification and unfinished work

The final automated suite passed all 40 tests, covering review validation,
feature/cohort alignment, corrupted-artifact rejection, hidden-state checks,
training-only preprocessing, risk orientation, saved-model round trips,
dropout-disabled inference, threshold tie handling and paired bootstrap behavior.
Prepared-data verification also covered all 2,823 hidden vectors and 20 real-data
loader scenarios. Full per-file checksums and evaluation markers are included.

Dataset expansion, new relation rules, independent directional annotations,
R-Bench evaluation, probability calibration, deployment and VLM fine-tuning have
not been executed. The proposed 10,681-row broader-relation expansion is only a
plan. It would contain 807 existing training/validation directional examples;
the old 134 directional test examples would be historical diagnostics. It cannot
provide additional above/below/left/right examples from this cached VSR release.

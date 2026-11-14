# Option 1: VLM error-detector dataset

For the complete current project record, read
[docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md): goal, exact VLM configuration,
data and labeling, extraction and training steps, final test results, limitations,
saved artifacts and reproduction instructions. Dataset expansion is a plan only.

This branch includes prepared feature arrays, hidden states, trained detectors
and completed research reports. Original image downloads, VLM weight caches and
the local virtual environment are excluded. See the summary for restoration steps.

This project runs a frozen vision-language model and extracts Yes/No scores.
Preparation and feature extraction are complete for all 941 image–statement pairs
across 855 images. The user's batch approval accepts every original VSR label.
Eight logistic-regression baselines and eight matched MLP variants are trained.
The frozen comparison has now been evaluated on the held-out test split.

## Completed MLP comparison

Read `outputs/mlp_comparison/REPORT.md` for methods, matched comparisons,
uncertainty intervals and abstention results. Plots are saved as
`test_comparison.png` and `test_comparison.pdf`. Per-example predictions and full
metrics are in `test_predictions.csv` and `test_summary.json`.

The MLP has two learned layers (32 ReLU hidden units, dropout 0.2, one output
logit). Scaling and weight fitting used training only; validation selected
settings and checkpoints. Four configurations across three seeds and eight
feature/cohort experiments required 96 fits (194.8 seconds on CPU). Seed 42 was
predeclared for primary test results; other seeds measure training variability.
All 40 automated tests passed. No additional VLM inference was required.

On 134 test pairs (25 original VLM errors), the core MLP scored 0.746 AUROC versus
0.755 for logistic regression and 0.718 for raw confidence. The MLP minus LR
paired 95% image-bootstrap interval was [-0.034, +0.012]. On the separate 87-row
vertical subset, behavior-feature MLP/LR AUROC was 0.810/0.802, with difference
interval [-0.026, +0.043]. These primary comparisons do not establish an MLP gain.
The three-seed MLP means were 0.754 ± 0.007 (all core) and 0.804 ± 0.007
(vertical behavior); these are sample SDs, not confidence intervals.

Saved MLP checkpoints and training-fitted scalers are under
`outputs/mlp_comparison/models/`. The report includes an inference example.
Validation thresholds were applied unchanged on test, so actual test coverage
differs between models. Future tuning needs a new holdout or external evaluation;
this test split must not be reused for model selection.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detector_comparison.py
.\.venv\Scripts\python.exe scripts\report_mlp_comparison.py
```

The evaluator returns cached results after completion; the report reads cached
predictions only. MLP training deliberately refuses to run after test exposure.
Preserve the frozen models and comparison files. The LR section below records
the earlier validation stage; its reserved-test statements are historical.

## Trained logistic-regression baselines

Results and fitted pipelines are in `outputs/logistic_baselines/`. Read
`outputs/logistic_baselines/REPORT.md`. The practical full-cohort baseline is
`models/all_core.joblib`: it takes p_original, confidence, p_blind and
image_sensitivity in that order, and returns risk of error class 1. The pipeline
includes the training-fitted StandardScaler. Use float32 inputs to match extraction.

On 148 validation pairs, AUROC was 0.716 for raw confidence, 0.742 for learned
confidence features and 0.744 for four core features. Hidden-layer probes at layers
9/18/36 scored 0.569/0.575/0.716 and showed larger train–validation gaps. On the
separate 95-row vertical validation cohort, core and seven-feature models scored
0.792 and 0.786 respectively. These are validation-selected results, not final
test performance; bootstrap intervals do not remove selection optimism.

All scaling/model fits used training only; validation selected regularization,
class weighting and provisional abstention thresholds. Eight experiments used
80 candidate fits. Selected models converged and reproduced predictions after
save/reload. All 32 automated tests passed. The raw prepared arrays below remain
unchanged and unstandardized; their preparation-stage flags describe that bundle,
not the separately saved trained models.

```powershell
.\.venv\Scripts\python.exe scripts\train_logistic_baselines.py
.\.venv\Scripts\python.exe scripts\report_logistic_baselines.py
```

The search protocol is fixed in `configs/logistic_baselines.json`. Summary metrics,
all candidate results, validation risks, model hashes, train–validation gaps and
paired comparisons are saved alongside the report. The matched MLP and final
test comparison are now complete, as documented above.

## Begin detector training

Read `data/vsr/detector_ready/DATASET_CARD.md`. The split NPZ files, feature CSVs,
row indexes, metadata and verification report are in `data/vsr/detector_ready/`.

| Split | Rows | Full consistency-feature rows |
|---|---:|---:|
| Training | 659 | 436 |
| Validation | 148 | 95 |
| Test | 134 | 87 |

All 941 rows contain original/blank-image scores, error labels, and original-prompt
hidden states at layers 9, 18 and 36 (2,048 values per layer). The core input has
four features; the extended behavioral input has seven. For the 323 horizontal
cases, perspective-dependent consistency features are unavailable under the
conservative rule, while core/hidden inputs remain usable. Do not replace those
missing values with zero or compare models on different cohorts.

Use `scripts/load_detector_data.py` to load aligned inputs/targets. It excludes
truth labels and identifiers from X, checks artifact hashes, and requires explicit
cohort selection for consistency experiments. Training targets contain 536 correct
answers (0) and 123 errors (1). Scale and fit on training data only; select settings
with validation and reserve the test set for final evaluation.

Verification passed for all source labels, score probabilities, 2,823 hidden
vectors, array round-trips, and split separation. Twenty real-data loader scenarios
and all 27 automated tests passed. The 100 cached pilot originals/hidden states
were reused and checked. The prepared array bundle remains raw; fitted models and
their preprocessing are stored separately under `outputs/logistic_baselines/`.

Extraction provenance and resumable per-pair records are in `outputs/vsr_full/`.
Batch approval, the frozen input manifest and 11-batch plan are in
`data/vsr/extraction/`. The earlier manual-review worksheet below is optional
history; its per-item gate was superseded for original labels by the user's
explicit chat batch approval. No human review decisions were forged or overwritten.

```powershell
.\.venv\Scripts\python.exe scripts\load_detector_data.py
```

This loads the training arrays and prints their shapes; it does not fit a model.

## Review the larger dataset

Open **http://127.0.0.1:8766/**. The worksheet presents all 941 statements in 19
batches, with original/mirrored images, source labels, transformation proposals,
reasons, notes, keyboard shortcuts and persistent local saving. Follow
**REVIEW_GUIDE.md** for the exact review and greenlight procedure.

| Larger candidate dataset | Statements | Images |
|---|---:|---:|
| Training | 659 | 596 |
| Validation | 148 | 140 |
| Test | 134 | 119 |
| Total | 941 | 855 |

There are 501 true and 440 false source statements. All images passed file
integrity and pixel-decoding checks: 757 newly downloaded plus 98 cached pilot
images, 132,961,210 bytes total. The full download report is
`outputs/directional_image_download.json`.

Start or restart the local-only server from PowerShell if the page is unavailable:

```powershell
.\.venv\Scripts\python.exe scripts\serve_vsr_review.py
```

Keep that terminal open. The server binds only to `127.0.0.1`, serves only the
review assets and selected dataset images, and persists decisions in
`data/vsr/review/human_decisions.json`. No model predictions or earlier agent
approvals appear in the worksheet. All human decisions start unreviewed.

**Approve all checks** includes all five diagnostic calls. **Original only**
includes original/grey calls without consistency features. **Set aside** preserves
the source record but excludes it from the reviewed extraction dataset. Source
labels and image-based splits stay unchanged. Approval requires a reviewer name,
all 941 decisions, confirmation of consistent criteria, and included examples in
each split. The greenlight validates metadata/image checksums and creates versioned
manifests under `data/vsr/review/approved/`; it does not launch a GPU job. A later
saved edit invalidates the active greenlight.

To rebuild the review dataset after image preparation:

```powershell
.\.venv\Scripts\python.exe scripts\build_vsr_review.py
```

Recorded decisions are bound to the dataset fingerprint; a changed dataset cannot
silently reuse them. Data/approval tests use disposable fixtures, not real review
decisions. Run them with `python -m unittest discover -s tests -p test_vsr_review.py`.

## This laptop

- Windows; Python 3.12.5.
- NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, driver 566.07.
- CUDA-enabled PyTorch 2.9.1 (`cu126`), torchvision 0.24.1.
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Inference: bitsandbytes 4-bit NF4 with double quantization; vision and output head kept
  at bfloat16, batch size 1, SDPA attention, at most 256 image tokens initially.

Verified locally on 13 November 2026: CUDA matrix multiplication passed, the model
loaded, all three labelled shapes checks were answered correctly, and hidden-state
vectors were saved. The first forward pass took 0.967 seconds; subsequent demo
passes took 0.274–0.284 seconds. Peak PyTorch allocated GPU memory was 3.256 GiB.
These timings exclude model loading, image preprocessing and tokenization. They
are installation measurements on a small synthetic image, not full VSR estimates.

Quantization changes the numerical model. Keep this configuration fixed for VSR
feature extraction, verifier training labels, and later transfer evaluation.
No VLM fine-tuning occurs. The low image-token budget is an initial memory setting;
choose the final resolution during the pilot and freeze it before collecting data.

## Run the installation check

Open PowerShell in this project. No virtual-environment activation is necessary:

```powershell
Set-Location C:\Users\Isaac\4001Project
.\.venv\Scripts\python.exe scripts\check_gpu.py
.\.venv\Scripts\python.exe scripts\run_vlm.py --suite --save-hidden
```

`check_gpu.py` performs an actual GPU matrix multiplication and prints GPU and
software details. `run_vlm.py` downloads the public model on its first run and
reuses the cache thereafter. Internet access is required for that initial download.
The model weights have already been downloaded on this machine (approximately
7 GiB). To run entirely from the cache, set `$env:HF_HUB_OFFLINE = '1'` in your
PowerShell window before running the script. Use `Remove-Item Env:HF_HUB_OFFLINE`
to allow downloads again in that window.

The controlled demo draws a red square to the left of a blue circle, then scores:

1. The true left-of statement on the original image.
2. The false right-of statement on the original image.
3. The true right-of statement on the mirrored image.
4. The original statement on a blank grey image (no expected answer).

These examples check installation and score extraction. Their accuracy is not a
benchmark result, and a VLM answering incorrectly does not mean setup failed.

Files created under `outputs/`:

- `demo_shapes.png`: the controlled input image.
- `smoke_test.json`: scores, token IDs, timing, peak allocated VRAM, model revision,
  package versions and image processing settings.
- `smoke_test.hidden.pt`: selected early/middle/final decoder states at the last
  prompt token, before an answer is supplied, when `--save-hidden` is requested.
  Indices are stored explicitly; index 0 would be embeddings and is not saved.

## Ask about your own image

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm.py --image "C:\path\to\photo.jpg" --statement "The cat is under the table." --output outputs\my_question.json
```

Use a statement, rather than a question: the script wraps it in the same fixed
prompt each time. It prints the chosen binary answer and P(Yes). Your image is
processed locally; it is not sent to Hugging Face.

Scores use the final prompt position's next-token logits. The script verifies
that canonical `Yes` and `No` are single tokens in that exact answer context,
then applies softmax to those two logits only. This is relative binary preference,
not a calibrated probability of correctness. The JSON also includes the full
vocabulary mass of those choices and the unrestricted top five tokens, so we can
detect whether other spellings or responses dominate. Free-form text is not generated.

## Recreate the environment

The packages are isolated inside `.venv`. Model files and download caches are
inside `.cache`; both directories, plus `outputs`, are excluded from Git.
Reserve roughly 15–25 GB for the environment, original model files and caches.
Loading in 4-bit still downloads the original model weights before quantizing them.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_vlm.ps1
```

The execution-policy flag applies only to that process; it does not change the
machine's policy. The setup script installs the official CUDA 12.6 PyTorch wheels
first, then the remaining packages in `requirements-vlm.txt`. It changes neither
system Python packages nor the NVIDIA driver. The runtime is included with the
PyTorch distribution; a separate CUDA Toolkit installation is not needed here.

To reproduce a recorded model version, copy `model_revision` from the saved JSON:

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm.py --suite --revision COMMIT_HASH --output outputs\rerun.json
```

`requirements-vlm.lock.txt` records the complete installed environment. To recreate
that exact environment, install CUDA PyTorch first as in the setup script, then
install that lock file instead of `requirements-vlm.txt`.

## Running comfortably on a laptop

Plug in the charger and use the laptop's normal performance mode. Close GPU-heavy
games or applications. Keep the vents clear. Watch memory and utilisation from a
second PowerShell window with `nvidia-smi -l 2` (Ctrl+C stops the monitor).

If a run exhausts VRAM, first close other GPU applications, omit `--save-hidden`,
or try `--max-pixels 100352` (128 image tokens). Changing this setting changes the
experiment, so record the new value. The program explicitly loads onto GPU 0 and
will fail if it cannot fit, rather than silently moving layers to CPU.

## VSR metadata preparation

The required inputs are the official VSR `data/splits/random/train.jsonl`,
`dev.jsonl`, and `test.jsonl` files. `relationships.json` belongs to Visual Genome
and is not an input to this pipeline. Each VSR line already contains a statement,
True/False label, relation, image filename, direct COCO image URL, and annotation
metadata. The three files together occupy 3,374,912 bytes.

The metadata is already downloaded. Rebuild the processed manifests locally:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_vsr.py
```

On another machine, use `--download-metadata` to retrieve missing files. Downloads
are pinned to official repository commit
`b27a0af0ee1462d2b6b92c8c83e869d9254a241a`; URLs and SHA-256 checksums are recorded
in `data/vsr/raw/sources.json`. No new packages are required for preprocessing.
The script downloads no images and never runs the VLM.

Outputs under `data/vsr/processed/`:

| File | Purpose |
|---|---|
| `all.jsonl` | 10,972 normalized statements across 6,259 images; preserve this full pool |
| `directional_candidates.jsonl` | 941 left-of/right-of/above/below candidate statements across 855 images |
| `pilot_candidates.jsonl` | 100 training statements across 98 images for a first visual audit and pilot |
| `all_images.csv` | One download entry per image in the full pool |
| `directional_images.csv` | One download entry per image in the directional pool |
| `pilot_images.csv` | The 98 image URLs needed for the initial pilot |
| `summary.json` | Dataset counts, label counts, splits, relation counts and checks |

Unique images are shuffled once globally using seed 42 and allocated approximately
70/15/15 to training/validation/test. Every subset inherits that allocation. The
original VSR split name is preserved as `source_split`, but the custom `split`
field is used for the verifier. This is a custom image-disjoint evaluation: the
official random splits have 1,400 images appearing across multiple splits.

| Directional candidate split | Statements | Images |
|---|---:|---:|
| Training | 659 | 596 |
| Validation | 148 | 140 |
| Test | 134 | 119 |

Preprocessing validates labels, filenames, image hosts, duplicate statements,
conflicting labels, parsed subject/object phrases, and image separation. No exact
duplicate image/statement pairs were found. The directional pool contains 501 true
and 440 false statements. These are **truth labels**, not verifier error labels.
Error labels are only created after comparing the VLM's original answer with truth.

The directional records are candidates, not a completed semantic audit. Left/right
can describe the viewer's perspective or an object's intrinsic perspective, which
affects whether swapping objects or mirroring preserves truth. Those rows carry
`requires_reference_frame_review`; the original reference-frame field is preserved
without guessing what numeric codes mean. Missing reference-frame metadata alone
is not grounds for dropping a row. Do not map opposite relations to `1 - truth_label`.

Download the 98 unique pilot images and verify that they decode with:

```powershell
.\.venv\Scripts\python.exe scripts\download_vsr_images.py
```

This reads `pilot_images.csv`, downloads into `data/vsr/images/`, and verifies
both image-file integrity and full pixel decoding. It retries failures, reuses
valid existing files, and saves dimensions, SHA-256 checksums and failures in
`outputs/pilot_image_download.json`. Rerunning the command retries failed or
invalid images while reusing valid ones. No GPU is used for this step.

Pilot download completed on 13 November 2026: 98/98 images verified, zero failures,
15,503,368 bytes total (approximately 15.5 MB).

The first-pass visual review is completed: 100 statements across 98 images were
inspected by Codex, using 10 contact sheets and closer inspection of selected
originals and mirror comparisons. There are 73 provisional consistency-ready
examples and 27 held cases. This is agent-assisted review, not independent human
annotation; no VSR truth labels were changed and no VLM predictions were used.

Review artifacts are in `outputs/vsr_audit/`: `report.md`, `review.csv`,
`reviewed_pilot.jsonl` (all 100), `consistency_ready_pilot.jsonl` (73),
`flagged_pilot.jsonl` (27), and numbered contact sheets. Uncertain cases include
cropped or unclear referents, vertical overlap/depth ambiguity, and perspective
issues in horizontal relations. Keep source labels for original-answer evaluation
on all 100; compare consistency feature ablations on the same eligible subset.
The reviewed subset is exploratory and does not replace the official source data.

To reproduce the visual-review artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\build_vsr_audit.py
.\.venv\Scripts\python.exe scripts\finalize_vsr_audit.py
```

`build_vsr_audit.py` rebuilds contact sheets and provisional text transformations;
it does not perform a new visual review. The recorded review decisions and notes
are in `configs/vsr_pilot_audit.json`, bound to the pilot manifest's SHA-256 hash.
`finalize_vsr_audit.py` applies those decisions and preserves original labels.

The GPU pilot is now complete. Reproduce or resume it with:

```powershell
.\.venv\Scripts\python.exe scripts\run_vsr_pilot.py --save-hidden
```

The runner uses only cached model files, pins the model revision, saves each
completed example immediately, and checks the manifest/configuration before
reusing results. Completed runs preserve their original timings. All 100 examples
get original and grey-image scores; 73 eligible examples also get converse,
exclusive and mirrored-equivalent scores (419 calls total). No verifier is trained.

Measured results: 82/100 original answers matched VSR; the reviewed subset had
65/73 matches. The run took 274.2 seconds and peaked at 3.281 GiB PyTorch allocated
GPU memory. Original hidden states at decoder layers 9, 18 and 36 were saved for
all 100 examples. A matched 512-token-budget run also scored 82/100: two errors
were fixed and two introduced. The main configuration retains the 256-token cap.

Pilot outputs are in `outputs/vsr_pilot_256/`: `report.md`, `results.jsonl`,
`features.csv`, `hidden/`, `analysis.json`, `pilot_results.png`, and
`resolution_comparison.json`. `configs/vsr_experiment.json` records the selected
configuration. These are training-pilot findings, not held-out test results.

To regenerate the plots and diagnostic metrics:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe scripts\analyze_vsr_pilot.py
```

AUROC and average precision describe individual untuned risk signals. Bootstrap
intervals resample images; risk–coverage values describe pilot rankings and do
not set a deployment threshold. On the same 73 examples, confidence AUROC was
0.808 and converse-disagreement AUROC 0.841, but the paired improvement interval
included zero. The blank-image model answered No on 99/100 statements, so blank
agreement is not proof of language-prior reliance. Keep predicted-answer and
relation confound baselines in the later verifier study.

Next, complete the human review in the local worksheet and greenlight its manifests,
then extract the larger dataset's features with the frozen configuration. Keep original image files;
make mirrored and grey versions in memory during feature extraction. Keep all
original metadata so additional vetted relation rules can expand the pool if the
pilot yields too few error examples. Image failures should be logged and retried;
do not replace them based on VLM performance.

## Official references

- Model and usage: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- PyTorch CUDA 12.6 wheels: https://pytorch.org/get-started/previous-versions/#v291
- Windows bitsandbytes support: https://huggingface.co/docs/bitsandbytes/en/installation
- Transformers quantization: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- VSR data format and image URLs: https://github.com/cambridgeltl/visual-spatial-reasoning/blob/master/data/README.md

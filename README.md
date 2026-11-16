# Detecting and reducing spatial-relation errors of a frozen VLM

SC4001 Neural Networks and Deep Learning, Group Project E (Hallucination and Reliability
in Vision Language Models). This is the code for the Visual Genome experiment in the report.

## What the code does

1. Builds 8,000 Yes/No questions about spatial relations from Visual Genome annotations.
   Each annotated relationship gives four statements: the original and its converse (true),
   and a swapped and an opposite version (false).
2. Asks a frozen vision language model (Qwen2.5-VL-3B-Instruct, 4-bit) each question and
   saves its Yes/No probability and its hidden state at layers 9, 18 and 36.
3. Trains small detectors (logistic regression and a one-hidden-layer MLP) to predict when
   the model's answer is wrong, from either the confidence or one hidden layer.
4. Evaluates the detectors once on a held-out test split.
5. Compares three ways of using the signals on the validation split: abstention,
   aggregating the four statements of one fact, and a cascade that verifies only the
   answers a detector flags.

## Folder layout

```
README.md              this file
ASSUMPTIONS.md         every design decision and its reason (labels A1, A2, ... are used in code comments)
requirements.txt       pinned Python packages
scripts/               the pipeline, one numbered script per stage (see "Files" below)
tests/                 unit tests
data/vg/processed/     filtered relationships, the sampled questions (queries.csv), image list
data/vg/detector_ready/  row tables, checksums and dataset card for the detector inputs
data/vsr/processed/    image list of the group's VSR experiment (those images are left out here)
outputs/               saved results: REPORT.md, results.json, aggregation.json, figures,
                       trained detectors (outputs/detectors/), logs
claude/                instructions given to the AI coding assistant (see the AI Use Statement)
```

Not included because of their size, and rebuilt by the scripts: the Visual Genome archives
(130 MB), the 1,929 images (250 MB), the model weights (7.5 GB, downloaded from Hugging
Face on first use), the per-question VLM records (220 MB) and the hidden-state arrays
`data/vg/detector_ready/{train,val,test}.npz` (190 MB).

## Requirements

- Linux with an NVIDIA GPU of at least 6 GB (developed on an RTX 4060 Laptop, 8 GB; peak
  use 3.3 GiB) and CUDA 12.6 drivers
- Python 3.12
- About 10 GB of free disk and an internet connection (model weights, metadata, images)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

With [uv](https://docs.astral.sh/uv/) the equivalent is:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

## Option A: regenerate the tables and figures from the saved results (no GPU, 1 minute)

The submission ships the test results, the trained detectors and the validation analysis.
These two commands rebuild `outputs/REPORT.md` and the three figures from them:

```bash
.venv/bin/python -I scripts/09_report.py
.venv/bin/python -I scripts/12_report_figures.py
```

## Option B: reproduce everything from scratch (GPU, about 1.5 hours)

The saved results act as a lock: once the test split has been evaluated, the training
script refuses to run again, so that nothing can be tuned on test. To start afresh, first
remove the saved results:

```bash
rm -rf outputs/detectors outputs/results.json outputs/test_evaluation_started.json \
       outputs/test_metrics.csv outputs/test_predictions.csv outputs/aggregation.json
```

Then run the stages in this order from the project root. `-I` runs Python in isolated
mode, which is used for every stage that reads downloaded files.

```bash
# Data preparation
.venv/bin/python scripts/vlm.py                          # check: load the VLM, score one synthetic image
.venv/bin/python -I scripts/01_download_metadata.py      # Visual Genome metadata from the official site
.venv/bin/python -I scripts/02_parse_filter.py           # keep the eight spatial predicates (3 minutes)
.venv/bin/python -I scripts/03_split_sample.py           # split by image, sample 250 relationships per predicate
.venv/bin/python -I scripts/04_download_images.py        # download the 1,929 images that are needed
#   If any image failed, run 03 and 04 again: failed images are replaced.

# VLM features
.venv/bin/python -I scripts/05_extract.py                # 8,000 forward passes, about 45 minutes; safe to interrupt and re-run
.venv/bin/python -I scripts/06_build_npz.py              # pack the detector inputs, write the dataset card

# Detectors
.venv/bin/python -I scripts/07_train_detectors.py        # train and select on train/validation, then freeze (10 minutes)
.venv/bin/python -I scripts/08_evaluate_test.py          # the single test evaluation

# Analysis and report
.venv/bin/python -I scripts/10_aggregate.py              # abstention, aggregation and cascade on validation
.venv/bin/python -I scripts/09_report.py                 # outputs/REPORT.md and two figures
.venv/bin/python -I scripts/12_report_figures.py         # figure of VLM accuracy by predicate and variant
```

### What to expect

| Quantity | Saved result |
|---|---|
| Questions (train / validation / test) | 5,568 / 1,216 / 1,216 |
| VLM accuracy on test | 0.707 |
| Test AUROC, 1 - confidence (nothing trained) | 0.680 |
| Test AUROC, MLP on hidden layer 36 | 0.753 |
| Validation accuracy, single question | 0.713 |
| Validation accuracy, four statements aggregated | 0.849 |

The full tables are in `outputs/REPORT.md`. The split, the sample and the detector training
are seeded (seed 42; MLP seeds 42, 43 and 44), so the questions are identical on every run.
VLM outputs can differ in the last decimal places on a different GPU or driver, which can
move the numbers above slightly.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

The tests cover statement generation, the image-level split, feature shapes and the
absence of label leakage, the direction of the risk scores, the abstention threshold, the
bootstrap, model save and reload, and the aggregation and cascade rules. Tests that need
the hidden-state arrays are skipped until stage 6 has been run.

## Files

| Path | Purpose |
|---|---|
| `scripts/common.py` | Paths, predicate mapping, the four statement variants, frozen VLM settings |
| `scripts/vlm.py` | Loads the frozen VLM and scores one statement (Yes/No probability, hidden states) |
| `scripts/01_download_metadata.py` | Downloads the Visual Genome archives into `data/vg/raw/` |
| `scripts/02_parse_filter.py` | Reads the relationships, maps predicates, applies the validity filters |
| `scripts/03_split_sample.py` | Image-level split, per-predicate sampling, statement generation |
| `scripts/04_download_images.py` | Downloads, decodes and hashes the needed images |
| `scripts/05_extract.py` | Runs the VLM on every question; resumable |
| `scripts/06_build_npz.py` | Packs the detector inputs and writes the dataset card |
| `scripts/load_detector_data.py` | Loads detector inputs with checksum checks; the test split is closed by default |
| `scripts/detector_lib.py` | Logistic regression and MLP detectors, metrics, abstention rule, bootstrap |
| `scripts/07_train_detectors.py` | Training and validation-only model selection; freezes the models |
| `scripts/08_evaluate_test.py` | The single test evaluation |
| `scripts/09_report.py` | Builds `outputs/REPORT.md` and two figures from saved results |
| `scripts/10_aggregate.py` | Abstention, aggregation, cascade and detector-informed aggregation (validation only) |
| `scripts/12_report_figures.py` | Figure of VLM accuracy by predicate and statement variant |
| `tests/test_pipeline.py` | Unit tests |

Where the report's results come from:

| Report item | Produced by | Saved in |
|---|---|---|
| Detection table, abstention table, Figure 2 | `08_evaluate_test.py`, `09_report.py` | `outputs/results.json`, `outputs/figure_test_results.png` |
| Figure 1 (VLM accuracy) | `12_report_figures.py` | `outputs/figure_vlm_accuracy.png` |
| Abstention / aggregation / cascade tables, Figure 3 | `10_aggregate.py`, `09_report.py` | `outputs/aggregation.json`, `outputs/figure_cascade.png` |
| Selected hyperparameters | `07_train_detectors.py` | `outputs/detectors/frozen.json`, `lr_candidates.csv`, `mlp_candidates.csv` |

## Detector inputs

`data/vg/detector_ready/{train,val,test}.npz` hold `hidden_L9`, `hidden_L18`, `hidden_L36`
(N x 2048, float32), `p_yes`, `confidence` (float32), and `vlm_pred`, `truth`, `y_error`
(int8). `{split}_rows.csv` lists the matching questions in the same order. `truth`,
`vlm_pred`, identifiers and split are never used as detector inputs.

## Acknowledgements

- Visual Genome (Krishna et al., 2017), CC BY 4.0.
- Qwen2.5-VL-3B-Instruct (Alibaba Cloud), used frozen through Hugging Face Transformers.
- The VLM loading and scoring steps, the metric definitions, the abstention rule, the
  bootstrap and the MLP training loop are shared with the group's VSR experiment, so that
  the two experiments can be set side by side.
- Code written with the help of Claude Code (Anthropic); see the AI Use Statement in the
  report.

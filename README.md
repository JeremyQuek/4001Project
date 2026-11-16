# Spatial-relation errors in a frozen VLM

SC4001 Neural Networks and Deep Learning, Group Project E. This repository contains
our Visual Genome experiment: generate spatial Yes/No questions, score a frozen
Qwen2.5-VL-3B model, then train small classifiers to detect incorrect answers.

Each annotated relationship produces four questions: the original, its converse,
a swapped version and an opposite version. The first two are labelled true and
the last two false. Images are split into train/validation/test before sampling.
Detectors use either `[p_yes, confidence]` or the hidden state at layer 9, 18 or 36.
All fitting uses train; hyperparameters and abstention thresholds use validation.
Test is used only for reporting the selected models.

## Setup

The full extraction uses Python 3.12, an NVIDIA GPU with at least 6 GB VRAM and
CUDA 12.6. Allow about 10 GB for metadata, images, model weights and features.
The saved-result reports do not need a GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell. All commands below use the
activated environment and run from the repository root.

## Rebuild the report from saved results

```bash
python scripts/09_report.py
python scripts/12_report_figures.py
# Optional summary for the team:
python scripts/11_team_summary.py
```

These commands rebuild `outputs/REPORT.md`, `outputs/TEAM_SUMMARY.md` and the
three PNG figures from the committed results, without inference or retraining.
The Word draft is retained as a document; its old JavaScript builder has been
removed. The Markdown report and figures are the reproducible report outputs.
No Node.js dependency is needed.

## Reproduce the experiment

Downloads, VLM weights, per-question outputs and detector NPZ arrays are excluded
from Git because of their size. Scripts 01-06 rebuild them. The committed query
and row tables document the original sample.

Use a separate result folder to keep the submitted results and fitted models:

```bash
export VG_OUTPUT_DIR=outputs/reproduced
```

In PowerShell use `$env:VG_OUTPUT_DIR = "outputs/reproduced"` instead.
This changes the result/model folder; downloaded data and extracted features stay
under `data/vg/` and can be reused.

```bash
# Metadata, questions and images
python scripts/01_download_metadata.py
python scripts/02_parse_filter.py
python scripts/03_split_sample.py
python scripts/04_download_images.py
# If downloads fail, run 03 then 04 again to replace the failed images.

# Frozen VLM outputs and detector inputs
python scripts/05_extract.py --pilot  # optional timing run on 200 training queries
python scripts/05_extract.py
python scripts/06_build_npz.py

# Fit and select on train/validation, then evaluate on test
python scripts/07_train_detectors.py
python scripts/08_evaluate_test.py

# Validation analysis and reports
python scripts/10_aggregate.py
python scripts/09_report.py
python scripts/12_report_figures.py
python scripts/11_team_summary.py
```

Extraction saves one record per question and reuses completed records when the
statement, image checksum and model settings match. Older hash-only records are
recomputed on the next extraction run. The NPZ builder rejects stale records.
Final dataset and model checksums still catch corrupted or mismatched files.

Training protects existing fits from accidental overwriting. To repeat training
in the same result folder, run `07_train_detectors.py --overwrite`, then
`08_evaluate_test.py --rerun` and the analysis/report scripts. Evaluation otherwise
uses cached results and checks that they belong to the current fits. Repeating
an evaluation is allowed; choosing models or thresholds from test results is not
part of this experiment.

To use the submitted results again, run `unset VG_OUTPUT_DIR` (PowerShell:
`Remove-Item Env:VG_OUTPUT_DIR`). Seeds and the model revision are fixed; different
hardware or libraries may still give slightly different results.

## Saved results

| Quantity | Value |
|---|---|
| Questions: train / validation / test | 5,568 / 1,216 / 1,216 |
| VLM test accuracy | 0.707 |
| Test AUROC: 1 - confidence | 0.680 |
| Test AUROC: MLP on hidden layer 36 | 0.753 |
| Validation accuracy: single question | 0.713 |
| Validation accuracy: four statements aggregated | 0.849 |

The aggregation and cascade analyses are exploratory and use validation only.
The group's VSR experiment uses a different dataset, so its scores are not
directly comparable. See `ASSUMPTIONS.md` for the sampling, labels and limitations.

## Files and result sources

| Scripts | Purpose | Outputs |
|---|---|---|
| `01_download_metadata.py` to `04_download_images.py` | Download, filter, split and sample | `data/vg/processed/`, images |
| `05_extract.py`, `vlm.py` | Frozen VLM inference | `data/vg/extraction/`, extraction summaries |
| `06_build_npz.py`, `load_detector_data.py` | Pack and load aligned features | `data/vg/detector_ready/` |
| `07_train_detectors.py`, `detector_lib.py` | LR/MLP fitting and validation selection | `outputs/detectors/` |
| `08_evaluate_test.py` | Test metrics, bootstrap intervals, per-query predictions | `results.json`, `test_metrics.csv`, `test_predictions.csv` |
| `10_aggregate.py` | Validation abstention, aggregation and cascade | `aggregation.json` |
| `09_report.py` | Detection and cascade tables and figures | `REPORT.md`, `figure_test_results.png`, `figure_cascade.png` |
| `12_report_figures.py` | VLM accuracy by predicate and variant | `figure_vlm_accuracy.png` |
| `11_team_summary.py` | Optional prose summary | `TEAM_SUMMARY.md` |

Paths in the last column are relative to `outputs/` unless shown otherwise.
`common.py` holds paths, predicate mappings, statement generation and VLM settings.
The detector arrays contain three `(N, 2048)` hidden-state matrices, `p_yes`,
`confidence`, `vlm_pred`, `truth` and `y_error`. Matching CSV files preserve row
order. Truth labels, predictions, identifiers and split names are never features.

## Tests

```bash
python -m pytest tests/test_pipeline.py -q
```

Tests cover statement generation, splits, feature alignment, risk scores,
thresholds, bootstrap, model save/reload and the synthetic packing-to-evaluation
pipeline. Checks on the full hidden-state arrays skip until script 06 is run.

## Acknowledgements

- Visual Genome (Krishna et al., 2017), CC BY 4.0.
- Qwen2.5-VL-3B-Instruct, Alibaba Cloud, used frozen through Transformers.
- Model scoring, metrics and detector training follow the group's VSR experiment
  so the two experiments use the same definitions.
- Code written with help from Claude Code (Anthropic); see the AI Use Statement
  in the report.

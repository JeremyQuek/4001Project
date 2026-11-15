# VLM spatial-relation error detection on Visual Genome

SC4001 Project E (Hallucination and Reliability in Vision Language Models).

A frozen vision language model (Qwen2.5-VL-3B-Instruct, 4-bit) is asked Yes/No questions
about spatial statements generated from Visual Genome relationships. Its hidden state at
the final prompt token is saved, and small detectors (logistic regression and an MLP) are
trained to predict when its answer is wrong. A confidence baseline built from the Yes/No
logits is trained the same way for comparison.

- Decisions and their reasons: [ASSUMPTIONS.md](ASSUMPTIONS.md)
- Dataset description and checksums: `data/vg/detector_ready/DATASET_CARD.md`
- Results: `outputs/REPORT.md`, `outputs/results.json`, `outputs/figure_test_results.png`

## Requirements

- Linux with an NVIDIA GPU (developed on an RTX 4060 Laptop, 8 GB; peak use about 3.3 GiB)
- Python 3.12 and [uv](https://docs.astral.sh/uv/) (or pip)
- About 10 GB of disk: 7.5 GB model weights, 130 MB Visual Genome metadata, 250 MB images

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

Pinned core versions (same as the teammate's VSR run): torch 2.9.1+cu126,
transformers 4.57.6, accelerate 1.12.0, bitsandbytes 0.49.2.

## Reproducing the results

Run from the repository root, in this order. `-I` runs Python in isolated mode because
these stages read downloaded files.

```bash
# 1. Smoke test: load the VLM and score one synthetic image.
.venv/bin/python scripts/vlm.py

# 2. Download Visual Genome metadata (official links), then parse and filter it.
.venv/bin/python -I scripts/01_download_metadata.py
.venv/bin/python -I scripts/02_parse_filter.py        # prints predicate counts

# 3. Split by image, sample 250 relationships per predicate, write the 8,000 queries,
#    and download only the images that are needed.
.venv/bin/python -I scripts/03_split_sample.py --per-predicate 250
.venv/bin/python -I scripts/04_download_images.py
#    If any image failed, repeat both commands: failed images are excluded and replaced.

# 4. Optional pilot (200 training queries): timing, memory, accuracy.
.venv/bin/python -I scripts/05_extract.py --pilot

# 5. Full extraction (about 50 minutes; safe to interrupt and re-run).
.venv/bin/python -I scripts/05_extract.py

# 6. Build the detector-ready arrays, row tables and dataset card.
.venv/bin/python -I scripts/06_build_npz.py

# 7. Train and select detectors on train/validation; freeze them.
.venv/bin/python -I scripts/07_train_detectors.py

# 8. Evaluate on test ONCE, then write the report and figure.
.venv/bin/python -I scripts/08_evaluate_test.py
.venv/bin/python -I scripts/09_report.py

# 9. Optional exploratory extension (validation only): abstention, aggregation, both, and a cascade.
#    Re-run the report afterwards to include its section.
.venv/bin/python -I scripts/10_aggregate.py
.venv/bin/python -I scripts/09_report.py
.venv/bin/python -I scripts/11_team_summary.py   # plain-language summary for the team

# Unit tests
.venv/bin/python -m pytest tests -q
```

`08_evaluate_test.py` writes `outputs/test_evaluation_started.json` when it opens the test
split. After that, `07_train_detectors.py` refuses to run, and a second run of
`08_evaluate_test.py` only keeps the cached results. `09_report.py` can be re-run at any
time; it reads saved results only.

All sampling uses seeded hashes (seed 42), so stages 2 to 3 give the same queries on every
run. The teammate-overlap filter needs `git fetch origin` first, so that
`origin/isaac:data/vsr/processed/directional_images.csv` is readable; without it the
overlap filter is skipped and the script says so.

## Files

| Path | Purpose |
|---|---|
| `scripts/common.py` | Paths, predicate mapping, query variants, frozen VLM configuration |
| `scripts/01_download_metadata.py` | Download VG archives into `data/vg/raw/` |
| `scripts/02_parse_filter.py` | Stream `relationships.json`, map predicates, apply validity filters |
| `scripts/03_split_sample.py` | Image-level split, per-predicate sampling, statement generation |
| `scripts/04_download_images.py` | Download, decode and hash the needed images |
| `scripts/vlm.py` | Frozen VLM loader and Yes/No scorer (follows the teammate's `run_vlm.py`) |
| `scripts/05_extract.py` | Resumable VLM run: `p_yes` and hidden states per query |
| `scripts/06_build_npz.py` | Shared NPZ contract, row tables, dataset card |
| `scripts/load_detector_data.py` | Loader with checksum and alignment checks; test closed by default |
| `scripts/detector_lib.py` | LR and MLP detectors, metrics, abstention rule, bootstrap |
| `scripts/07_train_detectors.py` | Training and validation-only selection; freezes models |
| `scripts/08_evaluate_test.py` | Single test evaluation |
| `scripts/09_report.py` | Report and figure from saved artifacts |
| `scripts/10_aggregate.py` | Exploratory: abstention, aggregating the four variants, both, and a detect-then-verify cascade (validation only) |
| `scripts/11_team_summary.py` | Short internal summary (`outputs/TEAM_SUMMARY.md`) from saved results |
| `tests/test_pipeline.py` | Split disjointness, leakage, shapes, risk orientation, round trips |

## Output data contract

`data/vg/detector_ready/{train,val,test}.npz` hold `hidden_L9`, `hidden_L18`, `hidden_L36`
(N x 2048 float32), `p_yes`, `confidence` (float32), `vlm_pred`, `truth`, `y_error` (int8).
`{split}_rows.csv` lists the matching rows. `truth`, `vlm_pred`, IDs and split are
metadata and are never detector inputs.

## Acknowledgements

- Visual Genome (Krishna et al., 2017), CC BY 4.0.
- Qwen2.5-VL-3B-Instruct (Alibaba Cloud), used frozen through Hugging Face Transformers.
- The VLM loading and scoring code, the metric definitions, the abstention rule, the
  bootstrap and the MLP training loop follow the teammate's VSR pipeline (branch `isaac`
  of this repository) so that results can be compared.
- Code written with the help of Claude Code (Anthropic); see the AI Use Statement in the
  report.

# SC4001 Project E: Relation Lie Detector

Detecting when a vision-language model answers a spatial-relation question wrong ("left of", "on top of"), from its
hidden states. The plan is in `Proposed Plan.pdf` and `claude/plan.md`.

## Repository structure

```
4001proj/
├── README.md
├── pyproject.toml               dependencies
├── Proposed Plan.pdf
├── project_handout.pdf
├── claude/                      plan files
└── src/
    ├── main.ipynb               entry point: runs every stage in order
    ├── lib/                     shared helpers
    │   ├── paths.py             every file location, in one place
    │   └── device.py            picks cuda, then mps, then cpu
    ├── data/                    library: data shapes, data pipeline, generated data
    │   ├── schemas.py           a pydantic shape for every stage's output (see stage list below)
    │   ├── records.py           reading and writing record files
    │   ├── clean_data.py        stage 1: clean the Visual Genome relationships
    │   ├── generate_left_right.py   stage 2: enrich with LEFT/RIGHT rows from box positions
    │   ├── generate_queries.py  stage 3: yes/no questions with ground truth
    │   ├── visual_genome/       raw download, and images/ with the photos (not in Git)
    │   └── outputs/             generated data (not in Git)
    │       ├── cleaned/         clean_data.json
    │       ├── enriched/        enriched_data.json
    │       ├── queries/         queries_train.json, queries_val.json, queries_test.json
    │       ├── answers/         VLM output: {split}_records.json, {split}_hidden.npy, layers.json
    │       └── validation/      cached re-asked and steered answers
    ├── training/                library: detector training
    │   ├── common.py            loading the answers, scoring, baselines
    │   ├── linear_probe.py      detector 1: logistic regression
    │   ├── mlp.py               detector 2: small neural network
    │   └── results/             detector scores as JSON (committed)
    ├── validation/              library: using the detectors at answer time
    │   ├── common.py            detectors as torch scorers, steering hook, re-asking, caching, metrics
    │   ├── baseline.py          Qwen's plain answers, scored like the methods
    │   ├── linear_feedback.py   linear probe + re-ask when flagged
    │   ├── mlp_feedback.py      MLP + re-ask when flagged
    │   ├── linear_steering.py   linear probe + hidden-state edit in the forward pass
    │   ├── mlp_steering.py      MLP + hidden-state edit in the forward pass
    │   ├── validate.py          stage 7: tune each method on the validation split
    │   ├── test.py              stage 8: baseline vs the four methods on the test split
    │   ├── results/             settings.json, test.json (committed)
    │   └── figures/             stage 9 plots: overview.png, accuracy_by_label.png
    └── model/                   the VLM and every weight file
        ├── vlm.py               Qwen2.5-VL-3B-Instruct wrapper + stage 4: answers every question
        └── models/              weights (not in Git): Qwen2.5-VL-3B-Instruct/, linear_probe.joblib, mlp.pt
```

`src/main.ipynb` is the entry point. Every `.py` file is a library with a `run()` function that the notebook calls,
and each pipeline script can also be run on its own for testing (`python src/data/clean_data.py --help`).

Stage 4, asking Qwen every question and saving its answers and hidden states, is `src/model/vlm.py`. It has been
tested on a tiny randomly initialised Qwen only; the real model hasn't been run yet.

## Setup

Needs Python 3.10 or newer.

```
python3 -m venv .venv
.venv/bin/pip install -e .
```

Dependencies are listed in `pyproject.toml`; there is no `requirements.txt`. The install creates a `*.egg-info/`
folder, which is ignored by Git along with `.venv/`.

## Data

Download these Visual Genome files into `src/data/visual_genome/`:

- `relationships.json.zip`
- `objects.json.zip`
- `relationship_alias.txt`
- `object_alias.txt`
- `image_data.json` (or `.zip`): photo URLs, used by stage 4

Leave the zips as they are; the scripts read them directly. Photos are downloaded into `visual_genome/images/` by
stage 4, only for the images the questions use.

On Colab, the raw data and the outputs can live on Google Drive instead, so they survive a disconnect. Call
`paths.configure(vg_dir=..., output_dir=...)` before running the stages (the setup cell in `main.ipynb` shows how),
or set the `RELATION_VG_DIR` and `RELATION_OUTPUT_DIR` environment variables.

## Qwen

```
.venv/bin/python src/model/vlm.py --download
```

Downloads `Qwen/Qwen2.5-VL-3B-Instruct` (about 7 GB) into `src/model/models/Qwen2.5-VL-3B-Instruct/`. To ask one
question about one image, with the subject boxed in red and the object in blue:

```
.venv/bin/python src/model/vlm.py --image photo.jpg --subj-box 50 60 120 200 --obj-box 300 100 200 150 \
    --question "Is the cup in the red box to the left of the plate in the blue box? Answer yes or no."
```

Each question is one forward pass with no generation: `p_yes` and `p_no` come from the next-token scores of "Yes" and
"No" (normalised over the two), and the hidden states are taken at the last prompt token for every 4th layer plus the
last (`[0, 4, ..., 36]`, 10 layers of 2,048 numbers). Images are resized to at most 448 × 448 pixels' worth of area.
It runs in float16 on CUDA and float32 elsewhere.

## Running

Run `src/main.ipynb` top to bottom. Each stage below can also be run as a script from the repo root.

### 1. Clean

```
.venv/bin/python src/data/clean_data.py
```

Writes `outputs/cleaned/clean_data.json`, one cleaned relation per line, and prints how many rows each filter left
for each predicate. Takes about 30 seconds.

| Option | Effect |
|---|---|
| `--max-per-label N` | Cap on rows per predicate (default 0, which keeps everything) |
| `--keep-odd-names` | Skip the check that names are one to three plain words |
| `--keep-same-name` | Keep relations whose two objects have the same name ("tree behind tree") |
| `--keep-repeated-names` | Keep relations whose names appear more than once in the image |
| `--keep-small-boxes` | Keep boxes smaller than 20 px a side |
| `--keep-overlapping-boxes` | Keep pairs whose boxes are near-identical |
| `--no-bbox-check` | Keep LEFT/RIGHT/UP/DOWN rows whose box positions disagree with the label |
| `--keep-duplicates` | Keep repeated annotations of the same two objects with the same label |

Every stage is on by default. In the notebook the same flags go in `skip`, e.g.
`clean_data.run(skip=["keep_same_name", "keep_repeated_names"])`, which is what `main.ipynb` uses.

### 2. Enrich

Visual Genome has only about 1,300 annotated "left of" and 1,200 "right of" relations. This stage builds more from
`objects.json`: two objects count as side by side when one box ends before the other starts and they share at
least half of the shorter box's height.

```
.venv/bin/python src/data/generate_left_right.py
```

Writes `outputs/enriched/enriched_data.json`: every cleaned row, then the new rows (`"source": "bbox"`). The cleaned
file is not changed. Takes about 30 seconds.

| Option | Effect |
|---|---|
| `--pairs-per-image N` | Max new rows per image (default 2; 0 keeps everything) |
| `--max-per-label N` | Cap on new rows per label (default 0, which keeps everything) |
| `--keep-odd-names` | Skip the check that names are one to three plain words |
| `--keep-small-boxes` | Keep boxes smaller than 20 px a side |
| `--keep-same-name` | Keep pairs whose two objects have the same name |
| `--keep-repeated-names` | Keep objects whose name appears more than once in the image |

Pairs already in the cleaned data are skipped.

### 3. Questions

```
.venv/bin/python src/data/generate_queries.py
```

Each relation becomes four yes/no questions (true, opposite, swapped, swapped + opposite), so half the answers are
yes. The relation's subject is named as being in the red box and its object in the blue box, so the boxes must be
drawn on the image when the question is asked.

Writes `outputs/queries/queries_{train,val,test}.json`. Relations are shared evenly across the 8 labels; a label
with too few rows (DOWN) gives its shortfall to the others. The files cycle through the labels, so the first N
questions are still balanced if a run stops early. Takes about 10 seconds.

| Option | Effect |
|---|---|
| `--train N`, `--val N`, `--test N` | Number of questions per split (default 100000, 20000, 20000) |
| `--input FILE` | Relations to read (default `outputs/enriched/enriched_data.json`) |

### 4. VLM run

```
.venv/bin/python src/model/vlm.py --run --limit 200     # pilot
.venv/bin/python src/model/vlm.py --run                 # every question
```

Needs the Qwen weights (`--download`, see above) and `image_data.json`. For each question it draws the boxes, asks
Qwen, and records the answer. Work is saved in shards of 512 questions under `outputs/answers/{split}_shards/`; a
rerun skips finished shards, so it carries on after a disconnect. `--splits` picks the splits and `--batch-size` the
number of questions per forward pass (default 8, two relations). When a split is done it writes:

| File | Contents |
|---|---|
| `{split}_records.json` | One `TrainRecord` per line (the question, the model's answer, `p_yes`, `p_no`, `wrong`) |
| `{split}_hidden.npy` | float16 array `[N, n_layers, d_model]`; row i belongs to record i |
| `layers.json` (optional) | Which model layer each index of the second axis is, e.g. `[0, 4, 8, ..., 36]` |

### 5–6. Train the detectors

```
.venv/bin/python src/training/linear_probe.py
.venv/bin/python src/training/mlp.py
```

Each tries every saved layer, picks the best on validation AUROC, then scores the test split once. Scores go to
`src/training/results/{linear_probe,mlp}.json` (test AUROC, F1, AUROC per label and per variant, and the
answer-confidence baseline); the trained detector goes to `src/model/models/`.

| Script | Detector | Main options |
|---|---|---|
| `linear_probe.py` | Logistic regression, class-weighted, L2 | `--layers`, `--C` |
| `mlp.py` | d_model → 256 → ReLU → dropout → 1, class-weighted BCE, early stopping | `--layers`, `--seeds`, `--hidden`, `--epochs`, `--device` |

Layer 0 (the embeddings) is skipped by default: at the last prompt token it holds the same token for every question.

`mlp.py` trains the chosen layer with 3 seeds and reports mean ± std. It uses CUDA if available, then Apple's MPS,
then CPU. MPS is skipped on macOS 14 and older, where it crashed in testing.

### 7–8. Use the detectors at answer time

Two ways of using each detector, one script per combination in `src/validation/`:

- **Feedback** (`linear_feedback.py`, `mlp_feedback.py`): if the detector's P(wrong) is above a threshold, Qwen is
  asked again as a second turn ("Your previous answer may be wrong. Look again carefully ...") and the new answer
  replaces the old one.
- **Steering** (`linear_steering.py`, `mlp_steering.py`): a forward hook at the detector's layer edits the hidden
  state at the last prompt token. A flagged state takes Newton steps along the detector's gradient,
  h ← h − (z − z*) / ‖∇z‖² · ∇z, until the detector's logit z is down to its threshold z*, and the total move is
  scaled by α. For the linear probe the gradient is constant (w / σ), so one step lands exactly on z*.

```
.venv/bin/python src/validation/validate.py      # stage 7: choose thresholds and alpha on the validation split
.venv/bin/python src/validation/test.py          # stage 8: final comparison on the test split
```

Both use every question of their split unless `--limit N` is given. `validate.py` writes
`src/validation/results/settings.json`; `test.py` reads it and writes `src/validation/results/test.json` with these
rows:

| Row | What it is |
|---|---|
| `baseline` | Qwen's plain answers from stage 4 (no new run) |
| `{detector}_feedback` | Re-asked when the detector flags the answer |
| `{detector}_feedback_random_flags` | Control: the same number of randomly chosen questions re-asked |
| `perfect_flags_feedback` | Upper bound: exactly the wrong answers re-asked |
| `{detector}_steering` | Hidden state edited along the detector's direction |
| `{detector}_steering_random_direction` | Control: an edit of the same size in a random direction |

Metrics for each row: accuracy, fixed (wrong → right), broken (right → wrong), net gain, share of answers changed,
share flagged and how many flagged answers were really wrong, mean confidence, and accuracy per label.

Qwen runs again in these stages: one re-ask pass per split (shared by both detectors and every threshold), and one
steered pass per detector and alpha on the flagged questions only. Every pass is cached in
`data/outputs/validation/`, so a rerun resumes. Each method script can also run alone, e.g.
`python src/validation/mlp_steering.py --tune --limit 500`.

### 9. Plots

The last cell of `src/main.ipynb` reads `test.json` and draws two figures into `src/validation/results/figures/`
(no extra script):

- `overview.png`: one panel per metric (accuracy change vs baseline, fixed, broken, % changed, % acted on, mean
  confidence). Each method sits next to its control; perfect flags is the upper bound.
- `accuracy_by_label.png`: accuracy per relation for the baseline and each method, as a heatmap.

### Data shapes

Every stage's output is checked against a pydantic model in `src/data/schemas.py`:

| Stage | File | Schema |
|---|---|---|
| 1–2 | `cleaned/clean_data.json`, `enriched/enriched_data.json` | `CleanRelation` |
| 3 | `queries/queries_{split}.json` | `Query` |
| 4 | `answers/{split}_records.json` (+ `{split}_hidden.npy`) | `TrainRecord` |
| 7–8 | `validation/*.json` (cached passes) | `ValidationAnswer` |
| 7 | `results/settings.json` | `ValidationSettings` (a `TuneResult` per method) |
| 8 | `results/test.json` | `TestResults` (a `Metrics` per row) |

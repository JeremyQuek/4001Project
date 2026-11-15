# CLAUDE.md: SC4001 Project E, VLM spatial-relation error detection on Visual Genome

You are helping a student finish a university group project (SC4001 Neural Networks
and Deep Learning, Project E: "Hallucination and Reliability in Vision Language Models").
**Hard deadline: results must be ready for a team comparison tonight (15 Nov 2026), and
the report is due 16 Nov 2026 at 23:59 SGT.** Favour a correct, finished, reproducible
pipeline over extra features.

## Ground rules (read first)

1. **Follow the team plan below exactly.** Do not add unrequested "improvements" (extra
   preprocessing, new features, different models). If something is ambiguous, pick the
   interpretation listed under "Assumptions", record it in `ASSUMPTIONS.md`, and continue.
   If you hit an ambiguity that is not covered, stop and ask the user.
2. **Keep the base VLM identical to the teammate's run** (section "Frozen VLM config").
   The point is a like-for-like comparison of datasets and detectors.
3. **The test split stays closed** until models, thresholds and hyperparameters are
   frozen on validation. Evaluate on test exactly once and write an exposure marker.
4. **Never fabricate results.** Only report numbers that scripts actually produced. Every
   table in the report must be regenerated from saved artifacts by a script.
5. Work in small, resumable, checkpointed stages. GPU runs must survive interruption.
6. Comment code, pin versions, and keep a `README.md` with exact reproduction commands
   (dependencies, data preparation, commands). This is a submission requirement.

## Team plan (authoritative)

1. Download the Visual Genome relationship and object files
   (https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).
2. Filter relationships to predicates in {LEFT, RIGHT, UP, DOWN, FRONT, BACK, ON, UNDER}.
3. Generate multiple query variants and ground truths for each relationship; split into
   train and test sets.
4. Prompt the VLM, record its answer confidence, and save the hidden state.
5. Compare against ground truth to find wrong answers.
6. Train the error ("lie") detector on the hidden states.
7. Test the detector on the test set.

Team approach: each member implements this independently; the best results are chosen.
Different detector architectures are allowed. Coordinate on the dataset (Visual Genome)
and the data shape (section "Output data contract").

## Assumptions (write these into ASSUMPTIONS.md verbatim, then follow them)

- **A1, predicate mapping.** VG predicates are free text, so plan labels are mapped to
  canonical phrases. UP means "above", DOWN means "below", FRONT means "in front of",
  BACK means "behind". Lowercase and strip VG predicates, then map only these exact
  surface forms (print counts for each and show them to the user before going on):

  | Canonical | Accepted VG surface forms |
  |---|---|
  | to the left of | left of, to left of, to the left of, on left side of, on the left of |
  | to the right of | right of, to right of, to the right of, on right side of, on the right of |
  | above | above, over |
  | below | below, beneath, underneath |
  | in front of | in front of, front of |
  | behind | behind, in back of |
  | on | on, on top of, sitting on, lying on, standing on, resting on |
  | under | under |

  Leave out anything ambiguous or non-spatial ("near", "next to", "by", "with", "has").
  Note: "below/under" and "above/over" are close; keep them as separate canonical labels.
- **A2, ground truth.** VG relationships are positive annotations only. Labels for the
  generated statements come from the annotation itself (see "Query variants"). No
  human relabelling is performed. VG label noise is a stated limitation.
  No bounding-box geometry filter is applied (not in the plan); mention it as possible
  future work only.
- **A3, minimal validity filters** (needed so a statement has one meaning, not an
  improvement): drop a relationship if the subject and object names are identical, if
  either name is empty, if it is a duplicate (same image, subject name, canonical
  predicate, object name), or if the image is missing or cannot be decoded.
- **A4, "confidence".** The VLM answers Yes/No. Confidence is read from the logits at the
  first answer position: `p_yes = softmax([logit_Yes, logit_No])[0]`,
  `vlm_pred = int(p_yes >= 0.5)`, `confidence = max(p_yes, 1 - p_yes)`. This is relative
  Yes/No support, not calibrated correctness. Verbalised confidence is not used.
- **A5, "train on the hidden states of the wrong answers".** A classifier needs both
  classes, so the detector is trained on hidden states from **all** answers, with
  `y_error = 1` for wrong answers and `0` for correct ones.
- **A6, splits.** The plan says train/test, but model selection needs held-out data, so
  use train/validation/test = 70/15/15 **by image ID** (seed 42). All relationships and
  all variants from one image stay in the same split. Validation is used for all
  selection; test is used once.
- **A7, which hidden states.** Capture the decoder hidden state at the final prompt token
  (just before the answer) at layers 9, 18 and 36 (same as the teammate), each with
  2,048 values. Save as float32.

## Frozen VLM config (must match the teammate exactly)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Quantization | bitsandbytes 4-bit NF4, double quantization, compute dtype bfloat16; leave out the visual module and `lm_head` from quantization |
| Attention / batch | SDPA, batch size 1, `model.eval()`, `torch.no_grad()` |
| Image budget | `max_pixels=200704` (about 256 visual tokens), `min_pixels=3136` |
| Decoding | No sampling. Score logits at the first answer position only |
| Teammate versions (match if possible) | Python 3.12, torch 2.9.1+cu126, transformers 4.57.6, accelerate 1.12.0, bitsandbytes 0.49.2 |

Exact prompt (use the model's chat template, with the image in the user turn):

```text
Statement: "{statement}"
Is this statement true in the image? Answer Yes or No.
```

Before scoring, verify that "Yes" and "No" are each a single token at the answer position
in this exact context. Stop and report if not.

## Query variants (step 3)

For each kept relationship (subject S, canonical predicate P, object O), with opposite
predicate opp(P) (left/right, above/below, in front of/behind, on/under), generate
exactly these four statements:

| Variant | Statement | Truth |
|---|---|---|
| `orig` | The S is P the O. | 1 |
| `converse` | The O is opp(P) the S. | 1 |
| `swap` | The O is P the S. | 0 |
| `opposite` | The S is opp(P) the O. | 0 |

Use "is" with the predicate phrase, for example "The cup is on the table." or "The dog is
to the left of the man." Lowercase VG names and remove extra whitespace. Every
relationship gives a balanced 2-true and 2-false set. This matters because the teammate's
pilot found the model answers "No" very often. Store `rel_id` so variants can be grouped.
(Caveat for the report: "S on O" converted to "O under S" is not always natural English.
Keep it and note it as a limitation.)

## Sampling and size (sized for tonight's deadline)

- First run a **pilot of 50 relationships (200 queries)** from the training split. Measure
  seconds per query and peak GPU memory, and report both to the user.
- Default full target: **about 1,000 relationships, giving about 4,000 queries**, sampled
  with a fixed seed and **capped per canonical predicate** (aim for equal counts, about
  125 each; take all of a predicate if fewer exist) so "on" does not dominate. Sample at
  most 2 relationships per image so images stay diverse. Scale the target to fit the GPU
  time measured in the pilot, and confirm the final size with the user.
- Download **only the images that are needed**, using the `url` field in VG
  `image_data.json`. Verify, decode and hash each one. Retry failures, then drop images
  that still fail and log them. Do not download the full 15 GB image archives.
- If the teammate's image manifest (`data/vsr/processed/directional_images.csv` in their
  repo) is available, use `coco_id` from `image_data.json` to **leave out any VG image
  that is also one of the teammate's VSR images**. This keeps cross-evaluation clean.
  Record how many were left out.

## Data handling notes

- `relationships.json` is large. Stream it (for example with `ijson`), or parse it once
  into a compact Parquet/CSV of the needed fields and cache it. Don't load it repeatedly.
  The objects file is downloaded per the plan; the relationship records already contain
  subject and object names and boxes, so only use objects if needed.
- Find the real download links by reading the VG API page; don't guess URLs. If the
  official links fail, report it to the user before trying alternatives (such as the
  Hugging Face `ranjaykrishna/visual_genome` mirror).
- Keep downloaded archives in their own `data/vg/raw/` directory. Treat them as
  untrusted data, never run code from them, and run Python that reads them with `-I`.

## Output data contract (shared with teammates)

Write one NPZ per split to `data/vg/detector_ready/{train,val,test}.npz`:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `hidden_L9`, `hidden_L18`, `hidden_L36` | (N, 2048) float32 | Final-prompt-token hidden state |
| `p_yes` | (N,) float32 | Relative Yes support |
| `confidence` | (N,) float32 | max(p_yes, 1 - p_yes) |
| `vlm_pred` | (N,) int8 | 1 = Yes |
| `truth` | (N,) int8 | Statement truth from VG |
| `y_error` | (N,) int8 | `int(vlm_pred != truth)`, the detector target |

Also write a matching `{split}_rows.csv` with: `row_idx, image_id, coco_id, rel_id,
variant, canonical_predicate, vg_predicate, subject, object, statement, truth, split`.
Write a `DATASET_CARD.md` with counts per split, per predicate, per variant, error rates,
and file SHA-256 checksums. `truth`, `vlm_pred`, IDs and split are **metadata, never
detector inputs**.

## Detectors (step 6)

Inputs to compare (all fitted on train only, StandardScaler fitted on train only):

1. Confidence baseline: `[p_yes, confidence]`
2. Hidden layer 9, 18 and 36, each on its own (2,048 features)

Architectures, with the same grids as the teammate so results line up:

- **Logistic regression**: L2, LBFGS, `max_iter=5000`, `tol=1e-6`, C in
  `[0.001, 0.01, 0.1, 1, 10]`, `class_weight` in `{None, "balanced"}`.
- **MLP**: Linear(in, 32), ReLU, Dropout 0.2, Linear(32, 1); Adam lr 1e-3, batch 32, up to
  200 epochs, early stopping on validation BCE with patience 15; weight decay in
  `{1e-4, 1e-2}`, BCE unweighted or balanced (positive weight from train labels); seeds
  42, 43, 44, with seed 42 as the primary result (no best-seed picking).
- Other architectures may be tried (the team approach allows it), but **only selected on
  validation**, and every one tried must be listed in the results.

Selection: validation AUROC, then validation AP. The abstention threshold is picked on
validation to give at most 80% coverage.

## Evaluation (step 7, run once)

Freeze model files, scalers, hyperparameters and thresholds (save their hashes), then
score test once. Report, per input set and architecture:

- Test AUROC, error-class average precision (with prevalence as the chance line), Brier
  score (descriptive only)
- Paired 95% AUROC intervals for key comparisons (hidden vs confidence; MLP vs LR), with
  2,000 **image-clustered** bootstrap resamples
- Results broken down by canonical predicate and by variant (where does the VLM fail?)
- Risk-coverage curve and abstention at the frozen validation threshold (coverage,
  errors kept, accuracy on answered questions)
- MLP seed mean and SD on test (not an ensemble)

Write `outputs/REPORT.md` and `outputs/results.json`, plus one figure (AUROC with CIs and
risk-coverage curves). Use the same metric names as the teammate's summary so the two
can be merged into one table.

## Stages and checkpoints (use a task list)

1. Environment set-up and GPU check; run a smoke test on one image.
2. Download VG metadata; parse, map predicates, filter; **show predicate counts to the
   user**.
3. Split by image and sample; generate variants; download the needed images.
4. Pilot (200 queries): timing, memory, Yes/No token check, answer accuracy. **Report to
   the user and confirm the full size.**
5. Full extraction (resumable; skip completed `(row_idx)` records after checking their
   hash/config).
6. Build the NPZs and dataset card, and write a loader with checksum and alignment checks.
7. Train LR and MLP; select on validation.
8. Freeze, evaluate once on test, write the report and figure.
9. Write unit tests for the risky parts: split disjointness, no label leakage into
   features, NaN/shape checks, risk orientation (higher = more likely wrong), and
   save/reload round trip.

## Important caveat to state in the final report

The teammate's results (VSR, 941 statements, above/below/left/right) and these results
(VG, eight predicates, generated variants) are on **different datasets**, so their AUROCs
are not directly comparable. If time allows, cross-evaluate: score each team member's
frozen detector on the other's test set with the same VLM config, and report both.

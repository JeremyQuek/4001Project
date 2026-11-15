# Assumptions

Assumptions A1 to A7 are copied verbatim from the team instruction set
(`claude/CLAUDE.md`). The section after them records every further decision made
while implementing the plan, with the reason for each.

## Assumptions from the instruction set (verbatim)

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

## Further decisions made during implementation

### Decided by the user

- **A3b, contradictory labels (user decision, 15 Nov 2026).** A relationship is dropped if
  its `swap` or `opposite` statement is itself annotated as true in the same image (for
  example both "man left of woman" and "woman left of man", which happens when an image
  holds several objects with the same name). This removed 2,736 of 539,144 filtered
  relationships, but about 13% of left/right relationships. It only uses VG's own
  annotations; no geometry or relabelling is involved.
- **Dataset size (user decision after the pilot).** 2,000 relationships, 250 per canonical
  predicate, giving 8,000 queries. The pilot was run on the first (1,000-relationship)
  sample; its 200 queries are training rows.

### Data

- **Visual Genome files.** Relationships and objects are version 1.4 and image meta data
  is version 1.2 (the newest of each on the official download page). The objects file is
  downloaded as the plan requires but not read: relationship records already carry the
  subject and object names.
- **Object names.** VG 1.4 stores either `name` or a `names` list; the first listed name
  is used. Names are lowercased and whitespace is collapsed. Nothing else is changed
  (no article, plural or spelling fixes).
- **Duplicates.** The first occurrence in file order is kept.
- **Split.** Each image is assigned by a seeded hash of its image ID (seed 42): the hash
  value in [0, 1) is compared with 0.70 and 0.85. The split therefore does not depend on
  the sample size.
- **Sampling.** Each canonical predicate gets a quota in each split (250 per predicate =
  174 train, 38 validation, 38 test). Relationships are taken in seeded-hash order, at
  most 2 per image, filling the rarest predicate first.
- **Teammate overlap.** The teammate's VSR manifest (branch `origin/isaac`,
  `data/vsr/processed/directional_images.csv`, 855 images) uses COCO image IDs. The 248
  eligible VG images whose `coco_id` is in that manifest are left out (1,657
  relationships).
- **Images.** EXIF orientation is applied and images are converted to RGB, as in the
  teammate's runner.

### VLM

- **Loader and scorer.** `scripts/vlm.py` follows the teammate's `scripts/run_vlm.py`
  step for step: `dtype=bfloat16`, `device_map={'': 0}`, `use_fast=True` processor,
  default chat template (which includes Qwen's default system prompt),
  `use_cache=False`, `logits_to_keep=1`.
- **Layer numbering.** `hidden_states[0]` is the embedding output, so "layer L" is
  `hidden_states[L]`; layer 36 is the last decoder layer before the final norm. These
  are the same indices the teammate saved.
- **Library versions.** Python 3.12.14, torch 2.9.1+cu126, transformers 4.57.6,
  accelerate 1.12.0, bitsandbytes 0.49.2 (all match the teammate). GPU: RTX 4060 Laptop.

### Detectors and evaluation

- **Experiment names** follow the teammate's summary (`all_confidence`, `all_hidden9`,
  `all_hidden18`, `all_hidden36`, prefixed `lr_` or `mlp_`) so the tables can be merged.
- **Heuristic baseline.** `heuristic_confidence` (risk = 1 - confidence, nothing fitted)
  is reported as well because the teammate's summary has it.
- **LR selection.** Validation AUROC, then validation AP, then smaller C, then unweighted
  (the teammate's tie-break order).
- **MLP selection.** Hyperparameters are chosen by the mean validation AUROC over seeds
  42, 43 and 44, then mean validation AP, then larger weight decay, then unweighted (the
  teammate's rule). Seed 42 is the primary result; the other seeds only describe
  training variability. Early stopping uses `min_delta = 1e-5` as in the teammate's
  config.
- **Abstention threshold.** Questions with risk at or below the threshold are answered.
  The threshold is the largest validation risk value whose coverage does not exceed 80%;
  tied scores are kept or dropped together.
- **Bootstrap.** 2,000 resamples of test images (all queries of a drawn image go
  together), one shared set of draws for every model, percentile 95% intervals.
- **Test exposure.** Running the VLM on test images is not treated as opening the test
  split. The test split counts as opened when `y_error` for test rows is first used,
  which happens only in `scripts/08_evaluate_test.py`. The dataset card therefore gives
  error rates for train and validation only.

## Exploratory extension (user request, 15 Nov 2026; not in the team plan)

- **Three-way comparison.** `scripts/10_aggregate.py` compares single-query abstention,
  aggregation of the four variants, and aggregation with abstention. It was added after the
  test evaluation, so it uses the validation split only and never reads test. Nothing is
  fitted. The report labels it exploratory.
- **Aggregation rule.** Support for the VG fact is p_yes for `orig` and `converse` and
  1 - p_yes for `swap` and `opposite`; the mean of the four decides all four answers
  (threshold 0.5). Knowing which variant states the fact and which negates it comes from how
  the statement was rewritten, not from the truth label.
- **Coverage.** Methods are compared at matched validation coverage (lowest-risk queries
  answered, ties together, at most 80% or 60%), not at a frozen threshold.
- **Detector in the hybrid.** The hidden-state MLP with the highest mean validation AUROC
  (seed 42 scores saved in stage 7). It was selected on validation, so its rows are slightly
  optimistic.

- **Cascade.** Each query is treated as one question that costs one VLM call. A router
  (1 - confidence, or the layer-36 LR or MLP risk) picks the riskiest share of questions;
  those get three more calls (the other variants of the same fact) and the aggregated
  answer. The share is swept from 0 to 1 in steps of 0.1 and nothing is tuned. "Random
  router" is the expected accuracy if the same number of questions were escalated at random.
  The optional last step withholds escalated questions with the smallest aggregate margin
  until validation coverage is at most 80%.
- **Detector-informed aggregation.** Three further fixed rules use the layer-36 MLP's error
  probability of each of the four calls: confidence support weighted by 1 - P(wrong);
  detector votes only (each call's Yes/No answer counted as 1 - P(wrong)); and the mean of
  both. The MLP was trained with balanced class weights, so its score is converted back to
  a probability by dividing the odds by (1 - r) / r, with r the training error rate. Nothing
  is fitted. Choosing the best of the four rules from the validation table would be a
  selection on validation.

# GPU pilot results

All observations come from our custom training-image split. No verifier has been trained; these are exploratory measurements of the frozen VLM and individual risk signals. Validation and test images were not evaluated.

## Setup and execution

Model: `Qwen/Qwen2.5-VL-3B-Instruct` at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. 4-bit NF4, bfloat16 computation, vision/output head excluded from quantization, batch size 1, SDPA attention, at most 256 visual tokens.

Completed 419 scoring calls in 274.2 seconds, including model loading. Peak PyTorch allocated GPU memory: 3.281 GiB. All scores were finite. Original Yes/No choices accounted for at least 99.80% of full-vocabulary probability mass.

## Original-answer accuracy

| Cohort | Correct / total | Accuracy | Errors |
|---|---:|---:|---:|
| Full pilot | 82/100 | 82.0% | 18 |
| Provisional consistency subset | 65/73 | 89.0% | 8 |
| Previously flagged for review | 17/27 | 63.0% | 10 |

Of the 18 full-pilot mismatches, 14 were wrong No answers and 4 were wrong Yes answers. These are disagreements with VSR labels; the flagged group contains uncertain spatial interpretations or object grounding.

## Untuned risk signals on the SAME 73 examples

| Signal | AUROC | 95% image-bootstrap interval |
|---|---:|---|
| Low original confidence | 0.808 | 0.669–0.920 |
| Converse disagreement | 0.841 | 0.674–0.976 |
| Exclusive violation | 0.479 | 0.287–0.736 |
| Mirror disagreement | 0.626 | 0.435–0.812 |
| Low image sensitivity | 0.787 | 0.651–0.894 |
| Mean of consistency violations | 0.796 | 0.607–0.953 |

Converse disagreement is promising, but its AUROC improvement over confidence has a paired image-bootstrap 95% interval of -0.159 to 0.225. This pilot does not establish superiority. There are only eight errors in this subset. No parameters or detection thresholds were fitted.

Exclusive violations are weak as a global ranking signal in this pilot. Mirror disagreement is weaker than confidence. The equal-weight mean of consistency violations also does not improve AUROC over confidence; a learned verifier remains to be tested.

## Blind-image check

Blank-image accuracy: 47.0%. The model answered No on 99/100 blank-image prompts. This is a strong default-answer bias. Matching the original and blank answers alone is not proof of image neglect, especially for correct No answers.

On all 100 originals, low image sensitivity had AUROC 0.742 versus 0.720 for low confidence. On the same 73 reviewed examples, it had AUROC 0.788 versus 0.808. Both comparisons are exploratory and do not establish added value beyond confidence.

## Descriptive selective-answering results

Using confidence alone on the 73 reviewed examples:

| Answered | Coverage | Accuracy on answered cases |
|---|---:|---:|
| 73/73 | 100.0% | 89.0% |
| 59/73 | 80.8% | 93.2% |
| 37/73 | 50.7% | 97.3% |

These are rankings on training-pilot data, not a validated deployment threshold. At approximately 81% coverage, both confidence and the equal-weight consistency score retained 59 examples with four errors.

## Concrete example

For “The cat is above the dining table” (audit 100, VSR True), P(Yes) was 0.349. For its proposed equivalent “The dining table is below the cat”, P(Yes) was 0.905. This illustrates a useful potential signal, but other correct answers also show converse disagreement.

## Resolution check

The same 100 originals were scored at a 512-token cap. Accuracy stayed 82/100; 2 errors were fixed and 2 introduced. Retain the 256-token cap for the main experiment. No aggregate gain was demonstrated in this pilot.

## Recommended continuation

1. Freeze model revision, quantization, prompt, pixel budget, scoring and image split seed. Keep official VSR labels and preserve quality flags. Fix transformation eligibility rules using training data, before inspecting validation/test model outputs.
2. Complete preprocessing and image acquisition for the 941 directional candidates (855 unique images). There are 757 additional images beyond the pilot. Review transformations with the same rules; retain a full source-label evaluation plus a documented reviewed-subset sensitivity analysis.
3. Extract features and original hidden states with the frozen setup. Reuse compatible pilot records. All feature-group ablations must use the same examples. Keep examples without eligible consistency transformations in a separate original/blind cohort.
4. Train regularized logistic regression first. Fit scaling on training data only; tune regularization on validation only. Baselines should include original yes-preference/confidence, predicted-answer or relation confounds, then consistency and blind-feature additions. Do not assume equal feature weights.
5. Check error counts in each split. The current split has 659/148/134 candidate statements in train/validation/test. Small error counts imply uncertain metrics; expand with explicitly validated relation rules if needed, before committing the final experiment. Add an MLP or hidden-state probes only after the linear baseline is stable.
6. Select abstention thresholds on validation, then evaluate the untouched test set once using AUROC, average precision, risk–coverage, runtime and paired image-bootstrap intervals. Finally test on compatible R-Bench questions without retraining.

At the measured per-call wall time, five calls across all 941 candidates would take roughly 47 minutes of feature extraction, plus downloads/review and any restarts. This is a planning estimate, not a completed full-dataset measurement.

## Saved artifacts

- `results.jsonl`: full per-question observations and error labels.
- `features.csv`: supervised-learning table; auxiliary feature cells are empty for ineligible cases.
- `hidden/`: three selected decoder-layer states for each of the 100 originals.
- `analysis.json`: metrics and uncertainty intervals.
- `pilot_results.png`: AUROC intervals and selective-answering plot.
- `resolution_comparison.json`: paired resolution results.

# Project E: Detecting Wrong Spatial-Relation Answers in a Vision-Language Model from Hidden States

SC4001 Group Project. Deadline: 16 Nov 2026, 11:59 PM.

> Check every reference, model name and dataset detail in this file against the original source before using it in the report.

> **Plan updated 13 Nov 2026.** The earlier plan (object hallucinations in COCO captions, CHAIR labels, caption
> correction) is replaced by the plan below. Section 2 lists what changed. Items marked **(proposed)** aren't agreed
> by the group yet; Section 10 collects them.

---

## 1. Research question

When a frozen VLM is asked where one object is relative to another (left of, right of, above, below, in front of,
behind, on, under), can a small probe on its hidden states tell when its answer is wrong, better than the model's
own confidence does?

**Why relations and not objects.** Detecting whether an object exists from hidden states turned out to be too easy
to be interesting. Spatial relations are harder for VLMs and harder to read off a single hidden state, since the
model has to bind two objects and a direction.

**The group's plan:**

1. Download the Visual Genome object and relationship files.
2. Filter for predicates in {LEFT, RIGHT, UP, DOWN, FRONT, BACK, ON, UNDER}.
3. Generate several query variants and a ground truth for each relationship, and split into train and test.
4. Ask the model each query. Save its answer, its confidence and its hidden states.
5. Compare against the ground truth to find the wrong answers.
6. Train the lie detector on the hidden states.
7. Test the lie detector on the test set.

**Approach.** Each person implements the pipeline and tries a different detector architecture. The group takes the
best result. Everyone uses the same dataset (Visual Genome), the same model and the same data shape, so the
detectors can be compared directly.

**The VLM is never retrained.** Only the small detector is trained.

```
Photo + "Is the cup to the left of the plate?" ──► VLM (frozen) ──► "Yes" / "No"
                                                       │                 │
                                    hidden states at the answer step     └─ compare with ground truth
                                                       │                             │
                                                       ▼                             ▼
                                                 LIE DETECTOR ◄──── label: answer correct / wrong
```

---

## 2. What changed from the earlier plan

| | Earlier plan | This plan |
|---|---|---|
| Task | Object hallucinations in free captions | Wrong answers to spatial-relation questions |
| Dataset | COCO val2014 | Visual Genome (objects + relationships) |
| Labels | CHAIR matching against COCO object lists | Model answer vs relationship ground truth |
| Model | Qwen2-VL-2B | Qwen 3B (`Qwen/Qwen2.5-VL-3B-Instruct`, exact checkpoint to confirm) |
| Probe position | Object word and 1–3 tokens before it | The answer step of each query |
| Caption correction | Feedback loop and guided generation | Dropped |
| Team split | One part per person | Everyone builds the pipeline; best detector wins |

---

## 3. Part 1: Dataset

### 3.1 Files
From Visual Genome (Krishna et al. 2017), placed in `src/data/`:

| File | Used for |
|---|---|
| `relationships.json` | Subject, predicate, object triples with bounding boxes, per image |
| `objects.json` | All annotated objects per image, with bounding boxes |
| `relationship_alias.txt` | Groups the different wordings of the same predicate |
| `object_alias.txt` | Groups the different names of the same object |
| `image_data.json` | Image width, height and URL |

The raw files are large. Keep `src/data/` out of Git (GitHub rejects files over 100 MB) and download only the
images that survive cleaning, using the URLs in `image_data.json`.

### 3.2 Cleaning
1. Normalise predicate text (lowercase, strip) and collapse wordings with `relationship_alias.txt`.
2. Map the canonical predicates to the 8 labels and drop everything else. The alias file gives names like
   "above" and "left of", not "UP" and "LEFT", so the group needs one shared map **(proposed)**:

| Label | Predicates | Opposite |
|---|---|---|
| LEFT | left of, to the left of | RIGHT |
| RIGHT | right of, to the right of | LEFT |
| UP | above, over | DOWN |
| DOWN | below | UP |
| FRONT | in front of | BACK |
| BACK | behind | FRONT |
| ON | on, on top of | UNDER |
| UNDER | under, beneath, underneath | ON |

   Confirm each wording against the alias file once it is in the repo.
3. Normalise object names with `object_alias.txt`.
4. Drop relationships that would make an ambiguous question **(proposed)**:
   - the subject's or the object's name appears more than once in that image ("the man" could be either man)
   - the subject and the object have the same name
   - either box is tiny, or the two boxes are near-identical
5. Check the label against the boxes where that is possible **(proposed)**. For LEFT, RIGHT, UP and DOWN, compare
   the box centres and drop the relationships that disagree with the annotation. FRONT, BACK, ON and UNDER can't
   be checked from 2D boxes, so they rely on the Visual Genome label.
6. Remove duplicate triples within an image.
7. "on" is far more frequent than the others. Cap the number of relationships per label so the set isn't mostly ON.

### 3.3 Query variants and ground truth
For each cleaned relationship (subject, label, object), generate **(proposed)**:

| Variant | Example | Ground truth |
|---|---|---|
| True statement | "Is the cup to the left of the plate?" | yes |
| Opposite predicate | "Is the cup to the right of the plate?" | no |
| Swapped objects | "Is the plate to the left of the cup?" | no |
| Paraphrase of each | "In this image, is the cup located to the left of the plate?" | same as the original |

The false versions give a balanced yes/no set, which matters because VLMs tend to answer "yes". Every query ends
with the same instruction (`Answer yes or no.`) so the answer is one token.

### 3.4 Split
Split **by image**, with a fixed seed, before generating queries. All queries about one image, including every
variant of the same relationship, stay in the same split.

| Split | Share | Used for |
|---|---|---|
| Train | 70% | Training the detector |
| Validation | 15% | Choosing the layer, thresholds and hyperparameters |
| Test | 15% | Final results only, used once |

### 3.5 Data shape (to coordinate on)
One record per query, as JSON lines **(proposed)**:

| Field | Meaning |
|---|---|
| `query_id` | Unique ID; also the key for the hidden-state file |
| `image_id`, `split` | Visual Genome image ID; train / val / test |
| `subj`, `obj` | Normalised names |
| `subj_bbox`, `obj_bbox` | `[x, y, w, h]` |
| `label` | One of the 8 labels (the true relation) |
| `variant` | true / opposite / swapped, plus the paraphrase index |
| `query_text` | The question as asked |
| `gt_answer` | yes / no |
| `model_answer` | yes / no (filled in Part 2) |
| `correct` | Whether `model_answer` equals `gt_answer` (filled in Part 2) |
| `p_yes`, `p_no` | Probability of each answer token (filled in Part 2) |

Hidden states: one array per query, shape `[n_layers + 1, d_model]`, taken at the last prompt token.

**Output:** `data/clean/relationships.jsonl`, `data/queries/{split}.jsonl`

---

## 4. Part 2: Running the VLM

- **Model:** Qwen 3B, taken to be `Qwen/Qwen2.5-VL-3B-Instruct`. Everyone uses the same Hugging Face ID.
- **Precision:** the same for everyone, since a 4-bit run gives different hidden states from an fp16 one.
  Qwen recommends bf16, which a T4 lacks, and fp16 may produce NaNs. Inspect the pilot outputs; if they are broken,
  use fp32 if it fits.
- **Image resolution:** Qwen-VL resizes dynamically, so fix `min_pixels` and `max_pixels` in the processor.
- **Prompt:** the same chat template and wording for everyone.
- **One forward pass per query**, no sampling. From it we save:
  - the answer: whichever of "Yes" / "No" has the higher probability as the first answer token
  - the confidence: `p_yes` and `p_no` from the logits at that step. We don't ask the model to state its
    confidence, since stated confidence is unreliable.
  - hidden states at the **last prompt token** (the state that produces the answer), for every layer or a subset.
    `hidden_states[0]` is the embedding layer. From memory the checkpoint has 36 layers with hidden size 2048, so
    `[37, 2048]` per query; confirm against `config.json`.
  - stored as fp16 `.npy` shards keyed by `query_id`
- **Pilot first:** ~200 queries. Check that the outputs are sane, measure the error rate for each label and the
  time per query, then fix the dataset size. If the model is nearly always right on a label, there are too few
  wrong answers to train on, and that label needs more queries or harder variants.

**Output:** `data/answers/{split}.jsonl`, `data/hidden/{split}/*.npy`

---

## 5. Part 3: Lie detector

### 5.1 Labels
`1` = the model's answer is wrong, `0` = it is right. The detector is trained on the hidden states of **both**
right and wrong answers; with only wrong answers it would have nothing to tell them apart from. Wrong answers will
be the minority, so report the base rate and use class-weighted losses.

### 5.2 Detectors
Each person tries a different architecture on the same saved hidden states. Starting points:

| Variant | Model |
|---|---|
| Linear | Logistic regression (scikit-learn), class-weighted, L2 regularised |
| MLP | 2-layer MLP (hidden 256, ReLU, dropout 0.1), PyTorch, class-weighted BCE |
| Multi-layer | Concatenate or attention-pool the hidden states of several layers |
| Hidden state + confidence | Any of the above with `p_yes`, `p_no` added as features |

- Input standardised using the train-split mean and standard deviation.
- Layer and hyperparameters chosen on **validation** AUROC.
- 3 seeds for anything with random initialisation; report mean ± standard deviation.

### 5.3 Baselines
| Baseline | Signal |
|---|---|
| Answer confidence | Low `max(p_yes, p_no)` → wrong |
| Answer entropy | High entropy over yes / no → wrong |
| **Label only** (confound) | The training-set error rate of that label (e.g. LEFT is wrong more often than ON) |
| **Variant only** (confound) | The training-set error rate of that query variant |
| Label + variant + object names | Together. The detector must beat this to show it reads more than the question |
| Blank-image gap (optional) | Change in `p_yes` when the image is replaced by a blank one; a small change suggests the model isn't using the image |

### 5.4 Ablations
- Layer: early vs middle vs late
- Linear vs non-linear
- Training-set size: 25% / 50% / 100% of train images
- **Leave-one-label-out:** train on 7 labels, test on the 8th. Shows whether the detector learned a general
  "I'm wrong" signal or one pattern per relation.

**Output:** `checkpoints/detector/`, `results/detection/*.csv`

---

## 6. Evaluation

| Metric | Notes |
|---|---|
| VLM accuracy per label and per variant | Reported first; shows how often and where the model is wrong |
| AUROC | Main, threshold-independent |
| Precision / recall / F1 | At the validation-tuned threshold |
| AUROC per label | Which relations the detector handles |
| AUROC vs layer | Where in the model the signal appears |
| Leave-one-label-out AUROC | Generalisation to an unseen relation |

**Failure analysis (for the report):** 5–10 hand-picked examples per failure type, with the image, the question,
the model's answer and the detector's score:
- wrong answers the detector missed
- right answers it flagged
- queries where the Visual Genome label itself is wrong or ambiguous (label noise). Two team members each check
  ~100 queries against the images and report the error rate.

**Fair-comparison rules:**
- Detector, layer and thresholds are chosen on **validation**. Test images are used once, at the end.
- Every person's detector uses the same queries, the same hidden states and the same test split.
- Fixed seeds throughout.

---

## 7. Compute plan (free Colab T4)

| Task | Notes |
|---|---|
| Cleaning and query generation | CPU, minutes |
| Pilot (~200 queries) | Gives the time per query and the error rate per label |
| Full VLM run | One short forward pass per query. Size the dataset from the pilot timing; split across accounts |
| Detector training | Seconds to minutes each |

- Save answers and hidden states to Drive every few hundred queries, so a disconnect doesn't lose the run.
- All the variants of one image share the image, so process queries grouped by image.

**Cut order if short of time:**
1. the blank-image baseline
2. the training-size ablation
3. extra seeds (keep 1)
4. paraphrase variants (keep the true / opposite / swapped set)

---

## 8. Timeline

| Date | Work |
|---|---|
| 13 Nov | Visual Genome files in `src/data/`; data cleaner; query generator; agree on the data shape |
| 14 Nov | Pilot, then the full VLM run with saved hidden states; mark wrong answers |
| 15 Nov | Detectors (one architecture per person), baselines, evaluation, figures |
| 15–16 Nov | Report (≤10 pages, Arial 10pt), README with reproduction commands, AI Use Statement |

Everyone must be able to explain every part.

---

## 9. Planned figures and tables
1. Pipeline diagram (photo + question → VLM → answer and hidden state → lie detector)
2. Table: VLM accuracy per label and per variant
3. Line chart: detector AUROC at each layer
4. Bar chart: each detector architecture vs the baselines (AUROC, F1)
5. Table: AUROC per label, and leave-one-label-out
6. Example questions with the model's answer and the detector's score

---

## 10. Open decisions

| Decision | Proposal |
|---|---|
| Exact checkpoint for "Qwen 3B" | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Label-to-predicate map (Section 3.2) | As tabled, confirmed against the alias file |
| Ground truth for LEFT / RIGHT / UP / DOWN | Keep only relationships whose boxes agree with the annotation |
| Query variants (Section 3.3) | True, opposite, swapped, plus one paraphrase of each |
| Record fields (Section 3.5) | As tabled |
| Dataset size | Fixed after the pilot timing |
| Precision, image resolution, prompt wording | One setting for everyone, fixed after the pilot |

---

## 11. Planned repository layout

```
4001proj/                          # repo root, branch jeremy_new
├── Proposed Plan.pdf
├── claude/                        # plan files and the conversation transcript
├── src/
│   ├── data/                      # Visual Genome files (not committed)
│   ├── clean_data.py              # alias mapping, 8-label filter, ambiguity and box checks
│   ├── generate_queries.py        # query variants, ground truth, split by image
│   ├── run_vlm.py                 # ask each query; save answer, confidence, hidden states
│   ├── train_detector.py          # one detector architecture per person
│   └── evaluate.py                # AUROC, per-label results, baselines, plots
└── README.md                      # setup, data download, commands to reproduce every result
```

File names under `src/` are placeholders until the code is written.

---

## 12. References (verify before citing)
- Krishna et al. 2017. *Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations.*
- Bai et al. 2025. *Qwen2.5-VL Technical Report.* arXiv 2502.13923.
- Azaria & Mitchell 2023. *The Internal State of an LLM Knows When It's Lying.*
- Liu et al. 2023. *Visual Spatial Reasoning* (VSR). Spatial-relation benchmark for VLMs.
- Kamath et al. 2023. *What's "up" with vision-language models? Investigating their struggle with spatial reasoning.* Includes spatial subsets built from Visual Genome.
- Yuksekgonul et al. 2023. *When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It?* (ARO; includes a Visual Genome Relation test).
- Li et al. 2023. *Evaluating Object Hallucination in Large Vision-Language Models* (POPE). Source of the yes/no probing format.
- CH-Wang et al. 2024. *Do Androids Know They're Only Dreaming of Electric Sheep?* Probes for hallucination in text LLMs.
- *TruthPrInt*, arXiv 2503.10602 (ICCV 2025). Probes internal states of VLMs. Check the authors.

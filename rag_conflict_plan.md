# Project F: Conflict Detection in Retrieval-Augmented Generation

SC4001 Group Project. Deadline: 16 Nov 2026, 11:59 PM.

> Check every reference in this file against the original paper before citing it in the report.

---

## 1. Research question

When a RAG system retrieves passages that contradict each other, can a small, question-aware conflict detector,
fine-tuned on synthetically generated conflicts, catch those conflicts more reliably than off-the-shelf NLI or
answer disagreement? And does placing it before the language model make the final answers more correct and
more faithful?

**Our three contributions:**

1. **Conflict dataset generation:** a controlled, labelled set of question + passage-pair examples, each tagged with its conflict type.
2. **Conflict detection model:** an NLI model fine-tuned to flag conflicting evidence for a given question.
3. **Evaluation framework:** measures detection quality and its downstream effect on RAG answers, using established metrics.

**Supporting pipeline:** existing tools, not a contribution. We credit them in the report.

```
Question → BM25 / dense retrieval → reranker (pretrained) → CONFLICT DETECTOR → LLM (Qwen2.5-3B-Instruct)
                                                              ├─ no conflict → answer normally
                                                              └─ conflict    → resolution strategy
```

---

## 2. Part 1: Conflict dataset generation

### 2.1 Source data
- **HotpotQA, distractor setting** (Yang et al. 2018). Each question has 2 gold paragraphs, 8 distractors,
  and annotated *supporting-fact* sentences.
- Split **by question ID** into train / validation / test **before** generating any pairs, so that no question
  or passage appears in more than one split.

### 2.2 Pair types

Every example is `(question, passage A, passage B, label, conflict_type)`.

| Pair type | How it is generated | Label | `conflict_type` |
|---|---|---|---|
| Entity swap | Copy gold passage; replace the answer entity with another entity of the same NER type (PERSON, ORG, GPE, …) | Conflict | `entity` |
| Date swap | Replace a DATE or year in the answer span or supporting sentence | Conflict | `date` |
| Number swap | Replace a CARDINAL, QUANTITY or PERCENT with a different value (more than ±10%, so it is clearly different) | Conflict | `number` |
| Paraphrased swap *(if time allows)* | Entity swap, then reword it with an LLM so the surface text differs | Conflict | `paraphrase_conflict` |
| Same fact, reworded *(if time allows)* | LLM paraphrase of the gold passage that keeps the answer | No conflict | `paraphrase_consistent` |
| Complementary | The two gold paragraphs of a HotpotQA question (different facts, both true) | No conflict | `complementary` |
| Unrelated | Gold passage + a distractor paragraph | No conflict | `unrelated` |
| Time-based, not a conflict *(hard negative)* | Same entity with different values at clearly different times (e.g. "in 2019" vs "in 2023") | No conflict | `temporal` |

### 2.3 Generation rules
- Use **spaCy** (`en_core_web_trf` or `en_core_web_sm`) to find entities and their types.
- The pool of replacement entities comes **only from the same split**, so test entities never appear in training.
- The swapped value must differ from the original after normalisation: lowercase, punctuation stripped.
- Include **both orders**, (A, B) and (B, A), so the model cannot learn a position shortcut.
- **Balance** the labels at about 50% conflict / 50% no conflict, and stratify by `conflict_type`.
- Save the generation seed and metadata (original span, replaced span, character offsets) for each example,
  for reproducibility and failure analysis.

### 2.4 Target sizes
| Split | Pairs |
|---|---|
| Train | ~20,000 |
| Validation | ~2,000 |
| Test (synthetic) | ~2,000 |
| Test (real conflicts) | ~100–200 |

### 2.5 Real-conflict test set
- First choice: **WikiContradict** (Hou et al. 2024), converted to `(question, passage A, passage B, label)` form.
- Fallback: hand-label 100–200 real pairs. Each one is checked by two team members.
- Used **only for testing**, never for training or tuning.

**Output:** `data/conflicts/{train,val,test_synth,test_real}.jsonl`

---

## 3. Part 2: Conflict detection model

### 3.1 Starting point
- A pretrained NLI checkpoint, e.g. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  (or `microsoft/deberta-v3-base` with an NLI head). Check the exact model name on Hugging Face.
- Why: NLI already models *entailment / contradiction / neutral*, which is the closest existing task
  (SNLI, Bowman et al. 2015; MultiNLI, Williams et al. 2018). Using synthetic perturbations as training data
  follows FactCC (Kryściński et al. 2020).

### 3.2 Input format (what makes it question-aware)
```
premise    = "Question: {question} Passage: {passage_A}"
hypothesis = "Passage: {passage_B}"
```
- Max length is 512 tokens. If the pair is too long, keep the supporting-fact or swapped sentences and the
  sentences around them, and truncate the rest.
- **Output:** a binary conflict / no-conflict head, trained on top of the NLI model's existing classifier.
  Record which option we use in the report.

### 3.3 Training setup
| Setting | Value |
|---|---|
| Loss | Cross-entropy |
| Epochs | 2–3, early stopping on validation F1 |
| Learning rate | 2e-5, linear decay, 10% warm-up |
| Batch size | 16 (with gradient accumulation if memory is tight) |
| Precision | fp16 |
| Seeds | 3 runs; report the mean ± standard deviation |
| Hardware | Colab T4, ~30–60 min per run |

Save checkpoints to Google Drive after each epoch.

### 3.4 Variants (ablations)
| Variant | What it tests |
|---|---|
| **Off-the-shelf NLI** (no fine-tuning; "contradiction" probability, threshold tuned on validation) | Is fine-tuning needed at all? |
| Fine-tuned **without** the question in the input | Does question-awareness matter? |
| Fine-tuned **without** the hard negatives (`temporal`, `complementary`) | Do hard negatives reduce false alarms? |
| **Full model** | Main result |

### 3.5 Non-NLI baselines
- **Answer disagreement:** the LLM answers from each passage separately, and different normalised answers mean a conflict.
- **LLM self-check:** prompt Qwen2.5-3B: "Do these two passages give conflicting answers to the question? Yes/No."

**Output:** `checkpoints/conflict_detector/`, plus predictions in `results/detection/*.csv`

---

## 4. Part 3: Evaluation framework

We use existing, published metrics and don't invent our own scoring.

### 4.1 Detection quality (does the detector find conflicts?)
| Metric | Notes |
|---|---|
| Precision, recall, F1 | Main detection metrics |
| AUROC | Threshold-independent comparison |
| F1 per `conflict_type` | Shows which kinds of conflict are hard |
| False-alarm rate on `temporal` and `complementary` | Shows whether the detector over-flags |
| Synthetic test → real test gap | **Key question:** does synthetic training transfer to real conflicts? |

### 4.2 Effect on final answers (does detection help RAG answers?)

**Evidence conditions**, built from the test split:
| Condition | Passages given to the LLM |
|---|---|
| Clean | Gold passages + distractors |
| Irrelevant | Distractors only |
| Contradictory | Gold passages + a conflicting copy (from Part 1) |

**Systems compared**
1. No detector (plain RAG)
2. Off-the-shelf NLI detector
3. Our fine-tuned detector

For 2 and 3, when a conflict is flagged, apply each **resolution strategy**:
- **S1: Memory arbitration.** The LLM picks the answer it believes is correct.
- **S2: Report both.** "Sources disagree: X or Y."
- **S3: Abstain.** "The evidence is conflicting / I don't know."

**Metrics**
| Metric | Tool / source | What it measures |
|---|---|---|
| Exact match, token-F1 | Standard SQuAD / HotpotQA scripts | Is the core answer correct? |
| Faithfulness | **MiniCheck** (Tang et al. 2024); **AlignScore** (Zha et al. 2023) as an alternative | Is the answer supported by the passages? |
| Abstention precision / recall | Our own counts | Does it abstain when it should, and only then? |
| Conflict acknowledgement rate | String / NLI match for both answers (S2 only) | Does it report both sides when there's a real conflict? |
| Unnecessary hedging rate | Hedging or abstaining on Clean examples | What do false alarms cost? |

### 4.3 Analysis splits
- **Known vs unknown:** ask every test question closed-book first. Split results by whether the LLM already
  knew the answer. This is needed to read S1 correctly, because synthetic swaps usually leave the true answer
  in the model's memory.
- **Detector right vs wrong:** shows how detector errors carry through to the final answer.
- **Per `conflict_type`:** entity / date / number / paraphrase.

### 4.4 Failure analysis (for the report)
- 5–10 hand-picked examples per failure type:
  - missed conflicts
  - false alarms on `temporal` and `complementary` pairs
  - number conflicts with close values
  - cases where S1 picked the wrong side from memory
- An explanation of *why* each fails, e.g. the question doesn't mention a time, so time-based pairs look contradictory.

### 4.5 Fair-comparison rules
- Tune every threshold and hyperparameter on **validation** only. The test sets are used once, at the end.
- All systems use the same retrieved passages, the same LLM and the same prompts, except for the part being tested.
- Fixed seeds. LLM generation is greedy (temperature 0).

**Output:** `results/end_to_end/*.csv`, figures in `figures/`

---

## 5. Compute plan (free Colab T4)

| Task | Estimated time |
|---|---|
| Dataset generation (spaCy, CPU) | ~30–60 min |
| Detector fine-tuning (×3 seeds, ×4 variants) | ~3–4 h total, split across team accounts |
| LLM answering (~500 questions × 3 conditions × systems/strategies) | ~2–4 h, batched |
| MiniCheck scoring | ~30 min |

- Save checkpoints and generated answers to Drive every ~100 questions.
- Backup GPUs: Kaggle Notebooks or the course's compute resources.

---

## 6. Timeline

| Date | Work |
|---|---|
| 11–12 Nov | Part 1: dataset generation script + real-conflict test set; run the off-the-shelf NLI baseline |
| 13 Nov | Part 2: train the detector and its variants; detection results (Section 4.1) |
| 14 Nov | Part 3: end-to-end RAG runs + MiniCheck scoring; failure analysis |
| 15–16 Nov | Report (≤10 pages, Arial 10pt), README with reproduction commands, AI Use Statement |

---

## 7. Planned repository layout

```
rag_conflict/
├── data_gen/make_conflicts.py      # Part 1
├── data_gen/build_real_test.py     # Part 1 (WikiContradict / hand-labelled)
├── detector/train.py               # Part 2
├── detector/predict.py
├── baselines/answer_disagreement.py
├── baselines/llm_selfcheck.py
├── pipeline/rag.py                 # retrieval + reranker + detector + LLM
├── eval/detection_metrics.py       # Part 3 (4.1)
├── eval/end_to_end.py              # Part 3 (4.2–4.3)
├── notebooks/run_all.ipynb         # Colab entry point
└── README.md                       # dependencies, data prep, commands
```

---

## 8. References (verify before citing)
- Yang et al. 2018. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.*
- Bowman et al. 2015. *A large annotated corpus for learning natural language inference* (SNLI).
- Williams et al. 2018. *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference* (MultiNLI).
- Kryściński et al. 2020. *Evaluating the Factual Consistency of Abstractive Text Summarization* (FactCC).
- Zha et al. 2023. *AlignScore: Evaluating Factual Consistency with a Unified Alignment Function.*
- Tang et al. 2024. *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents.*
- Hou et al. 2024. *WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia.*
- Xie et al. 2023. *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.*
- Shi et al. 2023. *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding* (related work only).

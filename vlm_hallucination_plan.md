# Project E: Detecting Hallucinations in Vision-Language Models from Hidden States

SC4001 Group Project. Deadline: 16 Nov 2026, 11:59 PM.

> Check every reference, model name and dataset detail in this file against the original source before using it in the report.

---

## 1. Research question

Can a small probe trained on a frozen VLM's hidden states detect which objects in a generated caption are
hallucinated, better than confidence-based signals? How early before the word is written can it tell? And does using
the probe to correct captions, either after generation or during it, reduce hallucinations without removing real
objects?

**Our contributions:**

1. **Hallucination dataset with hidden states:** captions from a real VLM, with object-level hallucination labels
   and the hidden states that produced each object word.
2. **Hallucination detector (probe):** a lightweight classifier on the frozen VLM's hidden states. It's tested at the
   object word itself and 1–3 tokens *before* it (early warning).
3. **Correction with the detector:** (a) after generation, as a feedback loop; (b) during generation, banning the word
   and resampling. Both are compared against random-flag and perfect-flag controls.
4. **Evaluation:** existing metrics (CHAIR, AUROC/F1, POPE-style transfer), plus failure analysis.

**The VLM is never retrained.** Only the small probe is trained.

```
Photo ──► VLM (frozen) ──► caption, word by word
              │
              └─ hidden states at each step ──► PROBE ──► "real" / "hallucinated"
                                                            │
                                    ┌───────────────────────┴───────────────────────┐
                              after generation                               during generation
                         (feedback → VLM rewrites)                    (ban the word → VLM re-picks)
```

---

## 2. Part 1: Dataset and labels

### 2.1 Images and ground truth
- **COCO val2014** (Lin et al. 2014): images, `instances_val2014.json` (object labels from 80 categories),
  `captions_val2014.json` (5 human captions per image).
- Sample **~4,000 images** with a fixed seed. Split **by image** **before** anything else:

| Split | Images | Used for |
|---|---|---|
| Train | ~2,800 (70%) | Training the probe |
| Validation | ~600 (15%) | Choosing the layer, thresholds and hyperparameters |
| Test | ~600 (15%) | Final results only, used once |

### 2.2 Caption generation
- **VLM:** `Qwen/Qwen2-VL-2B-Instruct` in fp16 on a T4. Use `attn_implementation="eager"` if we need attention
  weights for a baseline; flash-attention isn't supported on a T4.
  - Backup: `llava-hf/llava-1.5-7b-hf` in 4-bit (slower; known to hallucinate more, which gives more examples).
- **Prompt:** `"Describe this image in detail."` A long caption gives more object mentions, and so more hallucinations.
- **Decoding:** greedy, `max_new_tokens=256`, fixed seed.
- **Saved in the same pass** (`output_hidden_states=True`, `return_dict_in_generate=True`):
  - the generated token IDs, plus each token's probability and the entropy of the next-token distribution
  - hidden states for a **subset of layers** (e.g. every 4th layer plus the last one; check the number of
    layers in the model config)
  - only at the positions we need (Section 2.4), stored as fp16 `.npy` shards on Google Drive

### 2.3 Hallucination labels (CHAIR-style)
Following **CHAIR** (Rohrbach et al. 2018), using its released synonym list:
1. Find object words in each generated caption and map them to the 80 COCO categories ("puppy" → dog,
   "man" → person). Words outside the 80 categories are **ignored** (not labelled).
2. Ground truth for the image = categories in the **instance annotations** ∪ categories mentioned in **any of the
   5 human captions**.
3. Object mention in ground truth → **grounded (0)**. Not in ground truth → **hallucinated (1)**.

**Label-noise check:** two team members each check ~100 "hallucinated" labels against the actual image.
Report the share of false labels (real objects COCO doesn't list) and the agreement between the two of them.

**Class imbalance:** expect hallucinations to be the minority. Report the base rate and use class-weighted losses.

### 2.4 Which hidden states to save
For each object mention, let `t` = the position of its **first token**.

| Probe position | Hidden state at | Meaning |
|---|---|---|
| `post` | position `t` (the object token after it is fed back in) | Detection after the word is written |
| `pre-1` | position `t-1` (the state whose output chose the object token) | Early warning, just in time |
| `pre-2` | position `t-2` | Early warning, 1 token earlier |
| `pre-3` | position `t-3` | Early warning, 2 tokens earlier |

Estimated storage: ~4,000 images × ~10 object mentions × 4 positions × ~5–8 layers at fp16 comes to a few GB.
Reduce the number of layers if Drive space is short.

**Output:** `data/captions/*.jsonl`, `data/labels/*.jsonl`, `data/hidden/{split}/layer{L}_{pos}.npy`

---

## 3. Part 2: Hallucination detector

### 3.1 Probe
| Variant | Model |
|---|---|
| Linear | Logistic regression (scikit-learn), class-weighted, L2 regularised |
| MLP | 2-layer MLP (hidden 256, ReLU, dropout 0.1), PyTorch, class-weighted BCE |

- Input: one hidden-state vector (standardised using train-split mean and standard deviation).
- Train one probe per **(layer, position)**. Pick the best layer on **validation** AUROC.
- 3 seeds for the MLP; report mean ± standard deviation.
- Training takes seconds to minutes on CPU or GPU.

### 3.2 Baselines (no training, or off-the-shelf)
| Baseline | Signal |
|---|---|
| Token probability | Low probability of the object's first token → hallucinated |
| Token entropy | High entropy at the step that chose it → hallucinated |
| Attention to the image | Low share of attention on image tokens at that step → hallucinated (needs eager attention) |
| External detector | OWL-ViT (Minderer et al. 2022, `google/owlvit-base-patch32`) looks for the object name in the image; low score → hallucinated |
| VLM self-check | Ask the same VLM "Is there a {object} in the image? Answer yes or no." |

Thresholds for all baselines are tuned on validation only.

### 3.3 Ablations
- Layer: early vs middle vs late
- Position: `post` vs `pre-1` vs `pre-2` vs `pre-3` (the **early-warning curve**)
- Linear vs MLP
- Training-set size: 25% / 50% / 100% of train images

**Output:** `checkpoints/probe/`, `results/detection/*.csv`

---

## 4. Part 3: Using the detector to correct captions

### 4.1 After generation: feedback loop
1. The VLM writes caption 1.
2. The probe (`post` position, best layer) flags hallucinated objects.
3. Re-prompt with the same image:
   `"Your previous description mentioned: {flagged objects}. These are likely not in the image. Rewrite the description without them, and do not add objects you cannot see."`
4. Run the probe on the new caption. Repeat for at most 2 rounds.

### 4.2 During generation: early warning
- A custom generation loop, one token at a time.
- When the chosen token starts an object word (from the CHAIR synonym list), run the probe on the `pre-1` hidden state,
  which is already computed.
- If P(hallucinated) > τ, **ban that token** (set its logit to −∞) and re-pick at that step.
- Limit: at most 3 bans per object slot and 10 per caption, so it can't loop forever.
- τ is tuned on validation.

### 4.3 Controls (same pipeline, different source of flags)
| Condition | Where the flags come from | Purpose |
|---|---|---|
| No correction | none | Baseline |
| Simple deletion | probe; delete the flagged words from the text, no rewrite | Is the rewrite needed? |
| Random flags | the same *number* of flags as the probe, chosen at random | Is it the probe that helps, or just asking for a rewrite? |
| **Probe flags** | our detector | Main result |
| Best baseline's flags | e.g. token probability or OWL-ViT | Is the probe worth it? |
| Perfect flags | ground-truth labels | Upper bound on improvement |

Run all of these for both 4.1 (after) and 4.2 (during), where they apply. Simple deletion only applies to 4.1.

**Output:** `results/correction/*.jsonl`, `results/correction/metrics.csv`

---

## 5. Evaluation

### 5.1 Detection (Part 2)
| Metric | Notes |
|---|---|
| AUROC | Main, threshold-independent |
| Precision / recall / F1 | At the validation-tuned threshold |
| AUROC vs position | The early-warning curve (`post`, `pre-1`, `pre-2`, `pre-3`) |
| AUROC vs layer | Where in the model the signal appears |

### 5.2 Caption quality after correction (Part 3)
| Metric | Source | What it measures |
|---|---|---|
| CHAIR-i | Rohrbach et al. 2018 | Share of object mentions that are hallucinated (lower is better) |
| CHAIR-s | Rohrbach et al. 2018 | Share of captions with at least one hallucination (lower is better) |
| Object recall | Ours | Share of real objects still mentioned (shows over-deletion) |
| New hallucinations | Ours | Objects in the corrected caption that weren't in the original and are hallucinated |
| Caption length | Ours | Checks that hallucinations don't drop just because captions got shorter |
| Latency | Ours | Seconds per caption, after vs during |

Report bootstrap 95% confidence intervals for the CHAIR differences between conditions.

### 5.3 Generalisation (if time allows)
- **Task transfer:** apply the captioning-trained probe to **POPE** (Li et al. 2023) yes/no questions, on the hidden
  state at the "Yes" answer token. Does it predict when "Yes" is wrong? Test the random, popular and adversarial splits.
- **Dataset transfer:** run on **AMBER** (Wang et al. 2023) images with AMBER's own annotations.

### 5.4 Failure analysis (for the report)
- 5–10 hand-picked examples per failure type:
  - objects that often appear together (fork / dining table, car / traffic light)
  - small or hidden real objects that get flagged
  - real objects COCO doesn't list (label noise)
  - the VLM ignoring the feedback, or adding new hallucinations in the rewrite
  - an awkward sentence after a ban during generation
- Show the image, the caption with flagged words highlighted, and the probe scores.

### 5.5 Fair-comparison rules
- Probe, layer and thresholds are chosen on **validation**. Test images are used once, at the end.
- All conditions use the same VLM, prompt, decoding settings and test images.
- Fixed seeds throughout.

---

## 6. Compute plan (free Colab T4)

| Task | Estimated time |
|---|---|
| Captions + hidden states for ~4,000 images (batched) | ~1–3 h |
| Labelling (CPU) | minutes |
| Probe training (all layers × positions × seeds) | < 1 h |
| OWL-ViT + VLM self-check baselines | ~30–60 min |
| Feedback loop (test split, ≤2 rounds × conditions) | ~1–2 h |
| Guided generation (test split × conditions) | ~1–2 h |

- Save captions and hidden states to Drive every ~200 images, so a disconnect doesn't lose the run.
- Split GPU work across team members' accounts. Backups: Kaggle Notebooks or the course's compute resources.

---

## 7. Timeline

| Date | Work |
|---|---|
| 12 Nov | Part 1: sample and split images, generate captions with saved hidden states, CHAIR labels |
| 13 Nov | Part 2: label-noise check, probes for every layer and position, baselines, detection results |
| 14 Nov | Part 3: feedback loop + guided generation, all controls on the test split |
| 15 Nov | CHAIR and bootstrap results, failure analysis, figures; generalisation if time allows |
| 15–16 Nov | Report (≤10 pages, Arial 10pt), README with reproduction commands, AI Use Statement |

**Suggested split of work:**
- **A:** Part 1 (captions, hidden states, labels, label-noise check)
- **B:** Part 2 (probes, baselines, early-warning curve)
- **C:** Part 3 (feedback loop, guided generation, CHAIR evaluation, figures)

Everyone must be able to explain every part.

---

## 8. Planned figures and tables
1. Pipeline diagram (photo → VLM → probe → after / during correction)
2. Line chart: probe AUROC at each layer (one line per position)
3. Bar chart: probe vs baselines (AUROC, F1)
4. Early-warning curve: AUROC against how many tokens early
5. Table: CHAIR-i, CHAIR-s, object recall, new hallucinations, latency for each condition (after and during)
6. Example captions before and after, with flagged words highlighted

---

## 9. Planned repository layout

```
vlm_halluc/
├── data/prepare_coco.py          # sample + split images, load annotations
├── data/chair_labels.py          # CHAIR-style object matching and labels
├── generate/caption_and_dump.py  # VLM captions + hidden states + token probs/entropy
├── probe/train_probe.py          # linear / MLP probes per layer & position
├── probe/baselines.py            # token prob, entropy, attention, OWL-ViT, self-check
├── correct/feedback_loop.py      # after-generation correction
├── correct/guided_generate.py    # during-generation ban & re-pick
├── eval/detection_metrics.py
├── eval/chair_eval.py            # CHAIR-i/s, recall, new hallucinations, bootstrap CIs
├── eval/pope_transfer.py         # optional
├── notebooks/run_all.ipynb       # Colab entry point
└── README.md                     # dependencies, data download, commands
```

---

## 10. References (verify before citing)
- Lin et al. 2014. *Microsoft COCO: Common Objects in Context.*
- Rohrbach et al. 2018. *Object Hallucination in Image Captioning* (CHAIR).
- Li et al. 2023. *Evaluating Object Hallucination in Large Vision-Language Models* (POPE).
- Wang et al. 2023. *AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation.*
- Wang et al. 2024. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.*
- Minderer et al. 2022. *Simple Open-Vocabulary Object Detection with Vision Transformers* (OWL-ViT).
- Azaria & Mitchell 2023. *The Internal State of an LLM Knows When It's Lying.*
- Fieback et al. 2024. *MetaToken: Detecting Hallucination in Image Descriptions by Meta Classification.*
- Yin et al. 2023. *Woodpecker: Hallucination Correction for Multimodal Large Language Models.*
- Zhou et al. 2023. *Analyzing and Mitigating Object Hallucination in Large Vision-Language Models* (LURE).
- Leng et al. 2024. *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding* (VCD).
- Huang et al. 2024. *OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation.*

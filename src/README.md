# Detecting VLM hallucinations from hidden states (SC4001 Project E)

A small probe reads a **frozen** Qwen2-VL-2B's hidden-layer activations and predicts which objects in
its image captions are hallucinated. We also test how early it can tell (1–3 tokens before the word),
and whether its flags reduce hallucinations when used to correct captions, either after generation
(feedback loop) or during it (candidate check / skip). Full plan: `../claude/vlm_hallucination_plan.md`.

## Layout

```
src/
├── notebooks/colab_runner.ipynb   entry point: runs the stage scripts in order on Colab
├── scripts/                       one thin script per stage (01 … 09, label_audit)
├── lib/                           library code
│   ├── config.py                  YAML config + --set overrides
│   ├── common.py                  shared paths/loaders
│   ├── data/coco.py               COCO download, POPE exclusion, split by image
│   ├── data/chair.py              CHAIR matching with character offsets
│   ├── vlm/model.py               load Qwen2-VL, build prompts
│   ├── vlm/generate.py            captions (pass 1) + teacher-forced activations (pass 2)
│   ├── vlm/tokens.py              token ↔ character alignment
│   ├── vlm/activations.py         per-mention features at post/pre1/pre2/pre3
│   ├── vlm/guided.py              token-by-token loop with probe check (during generation)
│   ├── probe/                     dataset loading, logistic-regression/MLP probes, training
│   ├── baselines/                 token prob, entropy, attention, confounds, OWL-ViT, self-check
│   ├── correct/feedback.py        after-generation feedback loop + controls
│   └── eval/                      detection metrics, CHAIR metrics, paired bootstrap, plots
├── configs/default.yaml           every setting in one place
└── tests/                         CPU unit tests (CHAIR, alignment, probes, metrics)
```

## Dependencies

- Python ≥ 3.10, a CUDA GPU (a free Colab T4 is enough)
- `pip install -r requirements.txt && pip install -e .`
- NLTK data: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`

Downloaded automatically by stage 01 / Hugging Face:

| What | Source |
|---|---|
| COCO 2014 annotations | http://images.cocodataset.org/annotations/annotations_trainval2014.zip |
| COCO val2014 images (sampled ones only) | each image's `coco_url` |
| CHAIR synonyms | https://github.com/LisaAnne/Hallucination (`data/synonyms.txt`) |
| POPE COCO question files (to exclude their images) | https://github.com/AoiDragon/POPE |
| VLM | `Qwen/Qwen2-VL-2B-Instruct` |
| External detector baseline | `google/owlvit-base-patch32` |

Disk: ~5 GB for the model cache, ~0.7 GB of images, a few GB of saved activations.

## Reproducing the results

**On Colab:** open `notebooks/colab_runner.ipynb`, select a T4 runtime, and run the cells top to bottom.

**From a shell** (same commands the notebook runs):

```bash
cd src
python scripts/01_prepare_data.py             # data, splits, image download
python scripts/02_generate_captions.py --pilot # check hallucination rate / fp16 first
python scripts/02_generate_captions.py         # captions for all images
python scripts/03_label.py                     # CHAIR labels
python scripts/04_extract_activations.py       # hidden states at object mentions
python scripts/05_train_probes.py              # probes per layer × position
python scripts/06_baselines.py --owlvit --selfcheck
python scripts/07_guided_generation.py         # Part 3 base captions (add --guided for during-generation)
python scripts/08_feedback_loop.py             # after-generation correction
python scripts/09_evaluate.py                  # tables/ and figures/
```

Any setting can be overridden: `--set model.dtype=float32 --set part3.n_images=200`.
Outputs go to `<paths.out_root>/<run_name>/`. The exact config used is saved there as `config_used.yaml`.

### What each stage writes

| Stage | Output (under the run folder) |
|---|---|
| 01 | `splits.json`, `images.json` (split + CHAIR ground truth per image) |
| 02 | `captions.jsonl` (caption + exact token ids); `pilot/` for the pilot |
| 03 | `labels.jsonl` (one row per object mention, hallucinated 0/1) |
| 04 | `acts/<split>/shard_*.npz`, `stats_*.jsonl` |
| 05 | `results/probe_metrics.csv`, `probe_train_size.csv`, `probe_scores.csv`, `probes/best_<pos>.pkl` |
| 06 | `results/baseline_metrics.csv`, `baseline_scores.csv`, `baseline_thresholds.json` |
| 07 | `correction/base.jsonl`, `correction/guided_<policy>_<source>.jsonl` |
| 08 | `correction/feedback_<source>.jsonl`, `feedback_delete.jsonl` |
| 09 | `tables/*.csv`, `figures/*.png` |

### Experimental hygiene
- Images are split into train/val/test **by image** before anything else; POPE images are excluded.
- Layers, thresholds and hyperparameters are chosen on **validation** only; test is used for reporting.
- Every Part 3 condition decodes through the same token-by-token loop, on the same test images.
- Correction CIs are paired bootstraps over images.

## Tests

```bash
cd src && pytest -q tests        # CPU only; the MLP test is skipped without torch
```

## Acknowledgements
CHAIR metric and synonym list: Rohrbach et al. 2018. POPE: Li et al. 2023. Qwen2-VL: Wang et al. 2024.
OWL-ViT: Minderer et al. 2022. COCO: Lin et al. 2014. Hugging Face `transformers`, scikit-learn, NLTK.

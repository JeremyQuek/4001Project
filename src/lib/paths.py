"""Every file location in the project, in one place.

    src/data/visual_genome/   raw Visual Genome download; images/ holds the photos, fetched as needed
    src/data/outputs/         every generated data file, one folder per stage:
        cleaned/clean_data.json
        enriched/enriched_data.json
        queries/queries_{split}.json
        answers/{split}_records.json, {split}_hidden.npy, layers.json
        validation/  cached re-asked and steered answers, so validation runs resume
    src/model/models/         weights: Qwen2.5-VL-3B-Instruct/, linear_probe.joblib, mlp.pt
    src/training/results/     detector scores
    src/validation/results/   chosen settings and the final baseline-vs-methods comparison

The raw-data and output folders can be moved, e.g. to Google Drive on Colab, with configure() or the
RELATION_VG_DIR / RELATION_OUTPUT_DIR environment variables. Paths are looked up when a stage runs, so
configure() must be called before the stages, not before the imports.
"""

import os
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
MODEL_DIR = SRC / "model" / "models"
QWEN_DIR = MODEL_DIR / "Qwen2.5-VL-3B-Instruct"
RESULTS_DIR = SRC / "training" / "results"
VALIDATION_RESULTS_DIR = SRC / "validation" / "results"

_vg_dir = Path(os.environ.get("RELATION_VG_DIR", SRC / "data" / "visual_genome"))
_output_dir = Path(os.environ.get("RELATION_OUTPUT_DIR", SRC / "data" / "outputs"))


def configure(vg_dir: str | Path | None = None, output_dir: str | Path | None = None) -> None:
    global _vg_dir, _output_dir
    if vg_dir is not None:
        _vg_dir = Path(vg_dir)
    if output_dir is not None:
        _output_dir = Path(output_dir)


def vg_dir() -> Path:
    return _vg_dir


def images_dir() -> Path:
    return _vg_dir / "images"


def output_dir() -> Path:
    return _output_dir


def cleaned_file() -> Path:
    return _output_dir / "cleaned" / "clean_data.json"


def enriched_file() -> Path:
    return _output_dir / "enriched" / "enriched_data.json"


def queries_file(split: str) -> Path:
    return _output_dir / "queries" / f"queries_{split}.json"


def answers_dir() -> Path:
    return _output_dir / "answers"


def validation_dir() -> Path:
    return _output_dir / "validation"

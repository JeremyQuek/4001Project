"""Shared loading and scoring for the detectors.

Each split is read from two files in the answers folder (outputs/answers/), written by the VLM run:

    {split}_records.json   a JSON array of TrainRecord, one per line
    {split}_hidden.npy     float16 array [N, n_layers, d_model]; row i belongs to record i

An optional layers.json lists which model layer each index of the second axis is (e.g. [0, 4, 8, ..., 36]).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from data.records import read_records
from data.schemas import TrainRecord


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[TrainRecord]]:
    """Hidden states (memory-mapped), targets (1 = the model's answer is wrong) and records of one split."""
    records = read_records(data_dir / f"{split}_records.json", TrainRecord)
    hidden = np.load(data_dir / f"{split}_hidden.npy", mmap_mode="r")
    if len(hidden) != len(records):
        raise ValueError(f"{split}: {len(hidden)} hidden-state rows but {len(records)} records")
    y = np.array([r.wrong for r in records], dtype=np.int64)
    return hidden, y, records


def layer_names(data_dir: Path, n_layers: int) -> list[int]:
    path = data_dir / "layers.json"
    if path.exists():
        return json.loads(path.read_text())
    return list(range(n_layers))


def default_layers(names: list[int]) -> list[int]:
    """Every saved layer except 0, the embeddings: at the last prompt token they hold the same token for every
    question, so they carry no information, and steering can't edit them."""
    return [layer for layer in names if layer != 0] or names


def layer_features(hidden: np.ndarray, layer_idx: int) -> np.ndarray:
    return np.asarray(hidden[:, layer_idx, :], dtype=np.float32)


def fit_standardiser(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std from the train split only."""
    return x_train.mean(axis=0), x_train.std(axis=0) + 1e-6


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold with the highest F1, chosen on validation."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    return float(max(candidates, key=lambda t: f1_score(y, scores >= t, zero_division=0)))


def report(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": threshold,
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def confidence_scores(records: list[TrainRecord]) -> np.ndarray:
    """Baseline: the less sure the model is of its answer, the more likely it is wrong."""
    return np.array([1 - max(r.p_yes, r.p_no) for r in records])


def auroc_by(records: list[TrainRecord], y: np.ndarray, scores: np.ndarray, key) -> dict:
    """AUROC within each group (e.g. each label); None where a group has only one class."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[key(r)].append(i)
    out = {}
    for name, idx in sorted(groups.items()):
        out[name] = float(roc_auc_score(y[idx], scores[idx])) if len(set(y[idx])) == 2 else None
    return out


def save_results(name: str, results: dict) -> Path:
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = paths.RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(results, indent=2))
    return path

"""Load saved activation shards and join them with hallucination labels."""
from pathlib import Path

import numpy as np

from ..utils.io import read_jsonl


def acts_dir(run_dir, split):
    return Path(run_dir) / "acts" / split


def load_split(run_dir, split, layer, pos, exclude_overlap=False):
    """Returns (X float32 [n, D], y int [n], uids [n]) for one (layer, position)."""
    labels = {r["uid"]: r["hallucinated"] for r in read_jsonl(Path(run_dir) / "labels.jsonl")}
    xs, uids = [], []
    for shard in sorted(acts_dir(run_dir, split).glob("shard_*.npz")):
        z = np.load(shard)
        if len(z["uid"]) == 0:
            continue
        keep = z[f"valid_{pos}"].copy()
        if exclude_overlap:
            keep &= ~z[f"overlap_{pos}"]
        xs.append(z[f"L{layer}_{pos}"][keep])
        uids.extend(z["uid"][keep].tolist())
    if not xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, int), []
    X = np.concatenate(xs).astype(np.float32)
    y = np.array([labels[u] for u in uids], dtype=int)
    return X, y, uids


def saved_layers(run_dir, split="train"):
    shards = sorted(acts_dir(run_dir, split).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No activation shards in {acts_dir(run_dir, split)}; run 04_extract_activations.py")
    return np.load(shards[0])["layers"].tolist()

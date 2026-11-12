"""Paths and loaders shared by the stage scripts."""
from collections import defaultdict
from pathlib import Path

from .data.chair import Chair
from .utils.io import read_json, read_jsonl


def images_meta(cfg):
    """{image_id: {"file_name", "coco_url", "split", "gt": [...], "coco_categories": [...]}} from stage 01."""
    path = cfg.run_dir / "images.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/01_prepare_data.py first")
    return {int(k): v for k, v in read_json(path).items()}


def image_file(cfg, meta):
    from .data.coco import image_path
    return image_path(cfg.data_dir, meta["file_name"])


def get_chair(cfg):
    return Chair(Path(cfg.data_dir) / "chair_synonyms.txt")


def captions_path(cfg, pilot=False):
    return cfg.run_dir / ("pilot" if pilot else "") / "captions.jsonl"


def labels_by_image(cfg):
    """{image_id: [label rows ordered by mention index]}"""
    out = defaultdict(list)
    for r in read_jsonl(cfg.run_dir / "labels.jsonl"):
        out[r["image_id"]].append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["k"])
    return out


def mention_stats(cfg, split):
    """Per-mention token statistics written by stage 04, in shard order."""
    rows = []
    for p in sorted((cfg.run_dir / "acts" / split).glob("stats_*.jsonl")):
        rows.extend(read_jsonl(p))
    return rows


def num_layers(model):
    n = getattr(model.config, "num_hidden_layers", None)
    return n if n is not None else model.config.text_config.num_hidden_layers


def test_subset(cfg, images):
    """The fixed test images used for all Part 3 (correction) experiments."""
    test = [i for i, m in images.items() if m["split"] == "test"]
    order = read_json(cfg.run_dir / "splits.json")["test"]
    rank = {i: k for k, i in enumerate(order)}
    return sorted(test, key=rank.get)[: cfg.part3.n_images]

"""Enrich the cleaned data with LEFT/RIGHT rows built from object box positions.

Visual Genome has only about 1,300 annotated "left of" and 1,200 "right of" relations. Every object in
objects.json has a box, so for two objects side by side we can work out which one is on the left.

Reads   outputs/cleaned/clean_data.json (from clean_data.py), visual_genome/objects.json.zip, object_alias.txt
Writes  outputs/enriched/enriched_data.json: every cleaned row, then the new bbox rows.
        The cleaned file is not changed.

Library: run(pairs_per_image=2, max_per_label=0, skip=["keep_same_name", ...]); skip takes the flag names in SKIP_FLAGS.
Script:  python src/data/generate_left_right.py [--pairs-per-image N] [--max-per-label N] [--keep-same-name] ...
         (--help lists one flag per stage)
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from data.clean_data import (
    MIN_BOX_SIDE,
    NAME_RE,
    SAME_OBJECT_IOU,
    SEED,
    iou,
    load_object_alias,
    load_zipped_json,
    norm_name,
    split_of,
    xywh,
)
from data.records import read_lines, write_lines
from data.schemas import BBox, CleanRelation, Predicate, Source, Split

MIN_Y_OVERLAP = 0.5  # share of the shorter box's height the two boxes must share, so the pair is side by side

STAGES = [
    "candidate pairs",
    "valid names",
    "boxes large enough",
    "different names",
    "names unique in image",
    "side by side",
    "not already in clean data",
    "after per-image limit",
    "after cap",
]

# command-line flag -> the stage it switches off
SKIP_FLAGS = {
    "keep_odd_names": "valid names",
    "keep_small_boxes": "boxes large enough",
    "keep_same_name": "different names",
    "keep_repeated_names": "names unique in image",
}


def image_objects(image: dict, alias: dict[str, str]) -> list[dict]:
    """Objects of one image with a normalised name, dropping same-name boxes that are one object annotated twice."""
    kept = []
    for obj in image["objects"]:
        if not obj.get("names") or obj["w"] <= 0 or obj["h"] <= 0:
            continue
        name = norm_name(obj["names"][0], alias)
        box = xywh(obj)
        if any(k["name"] == name and iou(box, k["box"]) >= SAME_OBJECT_IOU for k in kept):
            continue
        kept.append({"id": obj["object_id"], "name": name, "box": box})
    return kept


def side_by_side(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    """left ends before right starts, and the two share most of their height."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    if lx + lw > rx:
        return False
    y_overlap = min(ly + lh, ry + rh) - max(ly, ry)
    return y_overlap >= MIN_Y_OVERLAP * min(lh, rh)


def generate(pairs_per_image: int, max_per_label: int, skip: set[str]) -> tuple[list[CleanRelation], dict[str, Counter]]:
    with open(paths.cleaned_file(), encoding="utf-8") as f:
        existing = {(r["image_id"], r["subj_id"], r["obj_id"]) for r in json.load(f)}
    alias = load_object_alias()
    rng = random.Random(SEED)
    stats = {stage: Counter() for stage in STAGES}
    by_label = defaultdict(list)

    for image in load_zipped_json("objects.json"):
        image_id = image["image_id"]
        objects = image_objects(image, alias)
        if "valid names" not in skip:
            objects = [o for o in objects if NAME_RE.fullmatch(o["name"])]
        if "boxes large enough" not in skip:
            objects = [o for o in objects if min(o["box"][2:]) >= MIN_BOX_SIDE]
        name_counts = Counter(o["name"] for o in objects)

        found = []
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                stats["candidate pairs"]["pairs"] += 1
                if "different names" not in skip and a["name"] == b["name"]:
                    continue
                if "names unique in image" not in skip and (name_counts[a["name"]] > 1 or name_counts[b["name"]] > 1):
                    continue
                if side_by_side(a["box"], b["box"]):
                    left, right = a, b
                elif side_by_side(b["box"], a["box"]):
                    left, right = b, a
                else:
                    continue
                stats["side by side"]["pairs"] += 1
                # One row per pair. Which object is the subject is random, which balances LEFT and RIGHT;
                # the opposite direction is asked later by the swapped query.
                if rng.random() < 0.5:
                    found.append((left, Predicate.LEFT, right))
                else:
                    found.append((right, Predicate.RIGHT, left))

        found = [f for f in found if (image_id, f[0]["id"], f[2]["id"]) not in existing
                 and (image_id, f[2]["id"], f[0]["id"]) not in existing]
        for subj, label, obj in found:
            stats["not already in clean data"][label] += 1
        if pairs_per_image and len(found) > pairs_per_image:
            found = rng.sample(found, pairs_per_image)

        for subj, label, obj in found:
            stats["after per-image limit"][label] += 1
            by_label[label].append(
                CleanRelation(
                    rel_id=f"{image_id}_{subj['id']}_{obj['id']}",
                    image_id=image_id,
                    subj_id=subj["id"],
                    subj=subj["name"],
                    subj_bbox=BBox(x=subj["box"][0], y=subj["box"][1], w=subj["box"][2], h=subj["box"][3]),
                    obj_id=obj["id"],
                    obj=obj["name"],
                    obj_bbox=BBox(x=obj["box"][0], y=obj["box"][1], w=obj["box"][2], h=obj["box"][3]),
                    predicate_raw=None,
                    label=label,
                    source=Source.BBOX,
                    split=split_of(image_id),
                )
            )

    rows = []
    for label in (Predicate.LEFT, Predicate.RIGHT):
        kept = by_label[label]
        if max_per_label and len(kept) > max_per_label:
            kept = rng.sample(kept, max_per_label)
        stats["after cap"][label] = len(kept)
        rows.extend(kept)
    rows.sort(key=lambda r: (r.image_id, r.rel_id))
    return rows, stats


def print_stats(rows: list[CleanRelation], stats: dict[str, Counter]) -> None:
    width = max(len(s) for s in STAGES)
    print(f"candidate object pairs: {stats['candidate pairs']['pairs']}, side by side: {stats['side by side']['pairs']}\n")
    print(f"{'rows left after':<{width}}  {'LEFT':>9}{'RIGHT':>9}")
    for stage in STAGES[-3:]:
        print(f"{stage:<{width}}  {stats[stage][Predicate.LEFT]:>9}{stats[stage][Predicate.RIGHT]:>9}")
    splits = Counter(r.split for r in rows)
    print(f"\n{len(rows)} new rows from {len({r.image_id for r in rows})} images: "
          + ", ".join(f"{s.value} {splits[s]}" for s in Split))


def run(pairs_per_image: int = 2, max_per_label: int = 0, skip: Iterable[str] = ()) -> Path:
    """Write the cleaned rows plus the new LEFT/RIGHT rows to outputs/enriched/. skip takes flag names from SKIP_FLAGS."""
    unknown = set(skip) - SKIP_FLAGS.keys()
    if unknown:
        raise ValueError(f"unknown skip flags {sorted(unknown)}; choose from {list(SKIP_FLAGS)}")
    rows, stats = generate(pairs_per_image, max_per_label, {SKIP_FLAGS[flag] for flag in skip})
    old_lines = read_lines(paths.cleaned_file())
    path = write_lines(paths.enriched_file(), old_lines + [r.model_dump_json() for r in rows])
    print_stats(rows, stats)
    print(f"kept {len(old_lines)} cleaned rows; wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs-per-image", type=int, default=2,
                        help="max new rows per image, so busy images don't dominate; 0 keeps everything")
    parser.add_argument("--max-per-label", type=int, default=0, help="cap on new rows per label; 0 keeps everything")
    for flag, stage in SKIP_FLAGS.items():
        parser.add_argument("--" + flag.replace("_", "-"), action="store_true", help=f'skip the "{stage}" stage')
    args = parser.parse_args()
    run(args.pairs_per_image, args.max_per_label, [flag for flag in SKIP_FLAGS if getattr(args, flag)])


if __name__ == "__main__":
    main()

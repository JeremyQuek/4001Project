"""Clean the Visual Genome relationships into CleanRelation rows.

Reads   visual_genome/relationships.json.zip, objects.json.zip, object_alias.txt
Writes  outputs/cleaned/clean_data.json (a JSON array, one CleanRelation per line)

Library: run(max_per_label=0, skip=["keep_same_name", ...]); skip takes the flag names in SKIP_FLAGS.
Script:  python src/data/clean_data.py [--max-per-label N] [--keep-same-name] [--keep-repeated-names] ...
         (--help lists one flag per cleaning stage)





┌──────────────────────────────────────┬───────┬───────┬────────┬───────┬────────┬────────┬─────────┬────────┐
│           Rows left after            │ LEFT  │ RIGHT │   UP   │ DOWN  │ FRONT  │  BACK  │   ON    │ UNDER  │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Spatial predicate matched            │ 1,304 │ 1,205 │ 23,586 │ 3,472 │ 14,256 │ 41,373 │ 740,904 │ 22,103 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Valid names and boxes                │ 1,297 │ 1,197 │ 23,398 │ 3,453 │ 14,173 │ 41,219 │ 732,724 │ 21,964 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Subject and object named differently │ 1,180 │ 1,079 │ 23,077 │ 3,362 │ 13,946 │ 40,409 │ 727,102 │ 21,718 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Both names unique in the image       │ 538   │ 476   │ 13,937 │ 2,146 │ 6,999  │ 22,146 │ 362,734 │ 12,534 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Boxes at least 20 px a side          │ 469   │ 405   │ 12,101 │ 1,912 │ 6,289  │ 20,456 │ 295,363 │ 11,224 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Boxes not near-identical             │ 469   │ 405   │ 12,024 │ 1,906 │ 6,269  │ 20,365 │ 290,299 │ 11,092 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Label agrees with box positions      │ 411   │ 337   │ 11,055 │ 1,743 │ 6,269  │ 20,365 │ 290,299 │ 11,092 │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Duplicates removed                   │ 398   │ 328   │ 9,763  │ 1,586 │ 5,812  │ 17,782 │ 216,750 │ 9,631  │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────┼─────────┼────────┤
│ Final (ON capped at 20,000)          │ 398   │ 328   │ 9,763  │ 1,586 │ 5,812  │ 17,782 │ 20,000  │ 9,631  │

"""

import argparse
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from data.records import write_records
from data.schemas import RAW_TO_PREDICATE, BBox, CleanRelation, Predicate, Source, Split

SEED = 4001
FILLER = {"is", "are", "a", "an", "the"}
NAME_RE = re.compile(r"[a-z]+( [a-z]+){0,2}")  # one to three plain words
MIN_BOX_SIDE = 20  # pixels
MAX_IOU = 0.9
SAME_OBJECT_IOU = 0.5  # same-name boxes overlapping this much are one object annotated twice

STAGES = [
    "spatial predicate",
    "valid names and boxes",
    "different names",
    "names unique in image",
    "boxes large enough",
    "boxes not near-identical",
    "label agrees with boxes",
    "no duplicates",
    "after cap",
]

# command-line flag -> the cleaning stage it switches off
SKIP_FLAGS = {
    "keep_odd_names": "valid names and boxes",
    "keep_same_name": "different names",
    "keep_repeated_names": "names unique in image",
    "keep_small_boxes": "boxes large enough",
    "keep_overlapping_boxes": "boxes not near-identical",
    "no_bbox_check": "label agrees with boxes",
    "keep_duplicates": "no duplicates",
}


def load_zipped_json(name: str):
    with zipfile.ZipFile(paths.vg_dir() / f"{name}.zip") as z:
        with z.open(name) as f:
            return json.load(f)


def load_object_alias() -> dict[str, str]:
    """Each line is "canonical,variant,variant,..."; map every variant to its canonical name."""
    alias = {}
    with open(paths.vg_dir() / "object_alias.txt", encoding="utf-8") as f:
        for line in f:
            names = [n.strip().lower() for n in line.split(",") if n.strip()]
            for name in names[1:]:
                alias.setdefault(name, names[0])
    return alias


def norm_predicate(raw: str) -> str:
    return " ".join(w for w in raw.lower().split() if w not in FILLER)


def norm_name(raw: str, alias: dict[str, str]) -> str:
    name = " ".join(raw.lower().split())
    return alias.get(name, name)


def first_name(obj: dict) -> str:
    # Visual Genome stores either "name" or a "names" list
    if "name" in obj:
        return obj["name"]
    return obj["names"][0] if obj.get("names") else ""


def split_of(image_id: int) -> Split:
    """Split by image, from a hash, so it stays the same whatever the filters keep."""
    bucket = int(hashlib.md5(f"{SEED}:{image_id}".encode()).hexdigest(), 16) % 100
    if bucket < 70:
        return Split.TRAIN
    return Split.VAL if bucket < 85 else Split.TEST


def xywh(obj: dict) -> tuple[int, int, int, int]:
    return obj["x"], obj["y"], obj["w"], obj["h"]


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def count_distinct(boxes: list[tuple[int, int, int, int]]) -> int:
    """Number of objects among same-name boxes, treating heavily overlapping boxes as one object."""
    kept = []
    for box in boxes:
        if all(iou(box, k) < SAME_OBJECT_IOU for k in kept):
            kept.append(box)
    return len(kept)


def boxes_agree(label: Predicate, subj: BBox, obj: BBox) -> bool:
    """Compare box centres. Only LEFT/RIGHT/UP/DOWN can be checked from 2D boxes."""
    if label is Predicate.LEFT:
        return subj.cx < obj.cx
    if label is Predicate.RIGHT:
        return subj.cx > obj.cx
    if label is Predicate.UP:
        return subj.cy < obj.cy
    if label is Predicate.DOWN:
        return subj.cy > obj.cy
    return True


def name_counts_per_image(alias: dict[str, str]) -> dict[int, Counter]:
    """How many distinct objects in each image carry each normalised name."""
    counts = {}
    for image in load_zipped_json("objects.json"):
        boxes = defaultdict(list)
        for obj in image["objects"]:
            for name in {norm_name(n, alias) for n in obj.get("names", [])}:
                boxes[name].append(xywh(obj))
        counts[image["image_id"]] = Counter({name: count_distinct(b) for name, b in boxes.items()})
    return counts


def clean(max_per_label: int, skip: set[str]) -> tuple[list[CleanRelation], dict[str, Counter]]:
    """skip holds the names of the stages to switch off; their rows pass straight through."""
    alias = load_object_alias()
    check_unique = "names unique in image" not in skip
    name_counts = name_counts_per_image(alias) if check_unique else {}
    stats = {stage: Counter() for stage in STAGES}
    seen = set()
    by_label = defaultdict(list)

    for image in load_zipped_json("relationships.json"):
        image_id = image["image_id"]
        counts = name_counts.get(image_id, Counter())
        for rel in image["relationships"]:
            label = RAW_TO_PREDICATE.get(norm_predicate(rel["predicate"]))
            if label is None:
                continue
            stats["spatial predicate"][label] += 1

            s, o = rel["subject"], rel["object"]
            subj, obj = norm_name(first_name(s), alias), norm_name(first_name(o), alias)
            # boxes with no area are always dropped: BBox can't hold them
            if min(s["w"], s["h"], o["w"], o["h"]) <= 0:
                continue
            if "valid names and boxes" not in skip and not (NAME_RE.fullmatch(subj) and NAME_RE.fullmatch(obj)):
                continue
            stats["valid names and boxes"][label] += 1

            if "different names" not in skip and subj == obj:
                continue
            stats["different names"][label] += 1

            if check_unique and (counts[subj] > 1 or counts[obj] > 1):
                continue
            stats["names unique in image"][label] += 1

            if "boxes large enough" not in skip and min(s["w"], s["h"], o["w"], o["h"]) < MIN_BOX_SIDE:
                continue
            stats["boxes large enough"][label] += 1

            if "boxes not near-identical" not in skip and iou(xywh(s), xywh(o)) > MAX_IOU:
                continue
            stats["boxes not near-identical"][label] += 1

            subj_bbox = BBox(x=s["x"], y=s["y"], w=s["w"], h=s["h"])
            obj_bbox = BBox(x=o["x"], y=o["y"], w=o["w"], h=o["h"])
            if "label agrees with boxes" not in skip and not boxes_agree(label, subj_bbox, obj_bbox):
                continue
            stats["label agrees with boxes"][label] += 1

            # same two objects with the same label, matched by object ID so two different trees stay apart
            key = (image_id, s["object_id"], o["object_id"], label)
            if "no duplicates" not in skip:
                if key in seen:
                    continue
                seen.add(key)
            stats["no duplicates"][label] += 1

            by_label[label].append(
                CleanRelation(
                    rel_id=str(rel["relationship_id"]),
                    image_id=image_id,
                    subj_id=s["object_id"],
                    subj=subj,
                    subj_bbox=subj_bbox,
                    obj_id=o["object_id"],
                    obj=obj,
                    obj_bbox=obj_bbox,
                    predicate_raw=rel["predicate"],
                    label=label,
                    source=Source.ANNOTATION,
                    split=split_of(image_id),
                )
            )

    rng = random.Random(SEED)
    rows = []
    for label in Predicate:
        kept = by_label[label]
        if max_per_label and len(kept) > max_per_label:
            kept = rng.sample(kept, max_per_label)
        stats["after cap"][label] = len(kept)
        rows.extend(kept)
    rows.sort(key=lambda r: (r.image_id, int(r.rel_id)))
    return rows, stats


def print_stats(rows: list[CleanRelation], stats: dict[str, Counter]) -> None:
    width = max(len(s) for s in STAGES)
    print(f"{'rows left after':<{width}}  " + "".join(f"{p.value:>9}" for p in Predicate))
    for stage in STAGES:
        print(f"{stage:<{width}}  " + "".join(f"{stats[stage][p]:>9}" for p in Predicate))
    splits = Counter(r.split for r in rows)
    print(f"\n{len(rows)} rows from {len({r.image_id for r in rows})} images: "
          + ", ".join(f"{s.value} {splits[s]}" for s in Split))


def run(max_per_label: int = 0, skip: Iterable[str] = ()) -> Path:
    """Clean the relationships and write them to outputs/cleaned/. skip takes flag names from SKIP_FLAGS."""
    unknown = set(skip) - SKIP_FLAGS.keys()
    if unknown:
        raise ValueError(f"unknown skip flags {sorted(unknown)}; choose from {list(SKIP_FLAGS)}")
    rows, stats = clean(max_per_label, {SKIP_FLAGS[flag] for flag in skip})
    path = write_records(paths.cleaned_file(), rows)
    print_stats(rows, stats)
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-per-label", type=int, default=0, help="cap on rows per label; 0 keeps everything")
    for flag, stage in SKIP_FLAGS.items():
        parser.add_argument("--" + flag.replace("_", "-"), action="store_true", help=f'skip the "{stage}" stage')
    args = parser.parse_args()
    run(args.max_per_label, [flag for flag in SKIP_FLAGS if getattr(args, flag)])


if __name__ == "__main__":
    main()

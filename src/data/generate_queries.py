"""Turn cleaned relations into yes/no questions with ground truth.

Each relation gives four questions, one per Variant, so half the answers are yes and half are no.
The relation's subject is boxed in red and its object in blue, and the question names them by colour.

Reads   outputs/enriched/enriched_data.json (from generate_left_right.py)
Writes  outputs/queries/queries_{train,val,test}.json (a JSON array, one Query per line)

Library: run(train=100_000, val=20_000, test=20_000)
Script:  python src/data/generate_queries.py [--train N] [--val N] [--test N] [--input FILE]
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from data.records import read_records, write_records
from data.schemas import CleanRelation, Predicate, Query, Split, Variant

SEED = 4001

# One template per relation, picked at random; all four variants of a relation use the same one
TEMPLATES = [
    "Is the {a} {phrase} the {b}? Answer yes or no.",
    "Looking at the image, is the {a} {phrase} the {b}? Answer yes or no.",
    "Is it true that the {a} is {phrase} the {b}? Answer yes or no.",
]


def questions_for(rel: CleanRelation, template_idx: int) -> list[Query]:
    red = f"{rel.subj} in the red box"
    blue = f"{rel.obj} in the blue box"
    asked = {
        Variant.TRUE: (red, rel.label, blue),
        Variant.OPPOSITE: (red, rel.label.opposite, blue),
        Variant.SWAPPED: (blue, rel.label, red),
        Variant.SWAPPED_OPPOSITE: (blue, rel.label.opposite, red),
    }
    queries = []
    for variant, (a, predicate, b) in asked.items():
        queries.append(
            Query(
                query_id=f"{rel.rel_id}_{variant.value}_{template_idx}",
                rel_id=rel.rel_id,
                image_id=rel.image_id,
                split=rel.split,
                subj=rel.subj,
                subj_bbox=rel.subj_bbox,
                obj=rel.obj,
                obj_bbox=rel.obj_bbox,
                label=rel.label,
                variant=variant,
                paraphrase=template_idx,
                query_text=TEMPLATES[template_idx].format(a=a, phrase=predicate.phrase, b=b),
                gt_answer=variant.answer,
            )
        )
    return queries


def share_out(total: int, available: dict[Predicate, int]) -> dict[Predicate, int]:
    """Split total evenly across labels; a label with too few rows gives its shortfall to the others."""
    quota = {}
    remaining = total
    by_size = sorted(available, key=lambda p: available[p])
    for i, label in enumerate(by_size):
        quota[label] = min(available[label], remaining // (len(by_size) - i))
        remaining -= quota[label]
    return quota


def interleave(groups: list[list]) -> list:
    """Round-robin across groups, so any prefix of the result covers every label evenly."""
    out = []
    for i in range(max(len(g) for g in groups)):
        out.extend(g[i] for g in groups if i < len(g))
    return out


def build_split(rows: list[CleanRelation], n_questions: int, rng: random.Random) -> tuple[list[Query], dict]:
    by_label = {label: [r for r in rows if r.label is label] for label in Predicate}
    quota = share_out(n_questions // len(Variant), {p: len(r) for p, r in by_label.items()})
    picked = [rng.sample(by_label[label], quota[label]) for label in Predicate]
    queries = []
    for rel in interleave(picked):
        queries.extend(questions_for(rel, rng.randrange(len(TEMPLATES))))
    return queries, quota


def run(train: int = 100_000, val: int = 20_000, test: int = 20_000, input_path: Path | None = None) -> dict[Split, Path]:
    """Write one question file per split to outputs/queries/; input_path defaults to the enriched data."""
    rows = read_records(input_path or paths.enriched_file(), CleanRelation)
    rng = random.Random(SEED)
    sizes = {Split.TRAIN: train, Split.VAL: val, Split.TEST: test}

    written = {}
    print(f"{'relations per label':<20}" + "".join(f"{p.value:>8}" for p in Predicate) + f"{'questions':>11}")
    for split, n in sizes.items():
        queries, quota = build_split([r for r in rows if r.split is split], n, rng)
        written[split] = write_records(paths.queries_file(split.value), queries)
        print(f"{split.value:<20}" + "".join(f"{quota[p]:>8}" for p in Predicate) + f"{len(queries):>11}")

    answers = Counter(q.gt_answer.value for q in queries)
    print(f"\nanswers in {split.value}: {dict(answers)}; files written to {paths.queries_file('x').parent}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, help="cleaned relations to read (default: the enriched data)")
    parser.add_argument("--train", type=int, default=100_000, help="number of train questions")
    parser.add_argument("--val", type=int, default=20_000, help="number of validation questions")
    parser.add_argument("--test", type=int, default=20_000, help="number of test questions")
    args = parser.parse_args()
    run(args.train, args.val, args.test, args.input)


if __name__ == "__main__":
    main()

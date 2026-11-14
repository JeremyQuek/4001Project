"""Baseline: Qwen's plain answers from stage 4, with no detector, scored with the same metrics as the methods.

No new Qwen run is needed; the answers are read from outputs/answers/{split}_records.json.

Library: run(split="test", limit=None) -> {"baseline": metrics}
Script:  python src/validation/baseline.py [--split test] [--limit N]
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.common import load_answers, metrics, original_p_yes, print_table


def run(split: str = "test", limit: int | None = None) -> dict[str, dict]:
    records, _, _ = load_answers(split, limit)
    return {"baseline": metrics(records, original_p_yes(records))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N questions of the split (default: all)")
    args = parser.parse_args()
    print_table(run(args.split, args.limit))


if __name__ == "__main__":
    main()

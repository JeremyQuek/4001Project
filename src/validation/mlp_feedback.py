"""MLP detector + feedback loop.

When the MLP gives P(wrong) above the threshold, Qwen is asked the question again as a second turn
("Your previous answer may be wrong. Look again ...") and the new answer replaces the old one.
Control: re-asking the same number of randomly chosen questions.

Library: tune(vlm, limit=None) -> chosen threshold on the validation split
         run(vlm, threshold, split="test", limit=None) -> metrics for the method and its control
Script:  python src/validation/mlp_feedback.py [--tune] [--split test] [--limit N] [--threshold X]
         (without --threshold, uses the value in validation/results/settings.json from validate.py)
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.schemas import ValidationSettings
from lib import paths
from model.vlm import VLM
from validation.common import feedback_tune, feedback_run, print_table

DETECTOR = "mlp"
NAME = "mlp_feedback"


def tune(vlm: VLM, limit: int | None = None, batch_size: int = 8) -> dict:
    return feedback_tune(vlm, DETECTOR, limit, batch_size)


def run(vlm: VLM, threshold: float, split: str = "test", limit: int | None = None, batch_size: int = 8) -> dict[str, dict]:
    return feedback_run(vlm, DETECTOR, threshold, split, limit, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true", help="choose the threshold on the validation split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N questions of the split (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, help="default: from validation/results/settings.json")
    args = parser.parse_args()

    vlm = VLM()
    if args.tune:
        result = tune(vlm, args.limit, args.batch_size)
        print(f"best threshold {result['threshold']} (val accuracy {result['val']['accuracy']:.4f})")
        return
    value = args.threshold
    if value is None:
        settings = ValidationSettings.model_validate_json((paths.VALIDATION_RESULTS_DIR / "settings.json").read_text())
        value = getattr(settings, NAME).threshold
    print_table(run(vlm, value, args.split, args.limit, args.batch_size))


if __name__ == "__main__":
    main()

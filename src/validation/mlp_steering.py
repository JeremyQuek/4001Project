"""MLP detector + steering inside the forward pass.

A hook at the MLP's layer edits the hidden state at the last prompt token. The MLP has no single direction,
so a flagged state takes Newton steps h <- h - (z - z*) / ||grad z||^2 grad z until its logit is at the
threshold z*; the total move is scaled by alpha. Control: a random direction with the same size of move.

Library: tune(vlm, limit=None) -> chosen alpha on the validation split
         run(vlm, alpha, split="test", limit=None) -> metrics for the method and its control
Script:  python src/validation/mlp_steering.py [--tune] [--split test] [--limit N] [--alpha X]
         (without --alpha, uses the value in validation/results/settings.json from validate.py)
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
from validation.common import steering_tune, steering_run, print_table

DETECTOR = "mlp"
NAME = "mlp_steering"


def tune(vlm: VLM, limit: int | None = None, batch_size: int = 8, alphas: tuple[float, ...] = (0.5, 1.0, 2.0)) -> dict:
    return steering_tune(vlm, DETECTOR, alphas, limit, batch_size)


def run(vlm: VLM, alpha: float, split: str = "test", limit: int | None = None, batch_size: int = 8) -> dict[str, dict]:
    return steering_run(vlm, DETECTOR, alpha, split, limit, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true", help="choose the alpha on the validation split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N questions of the split (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, help="default: from validation/results/settings.json")
    args = parser.parse_args()

    vlm = VLM()
    if args.tune:
        result = tune(vlm, args.limit, args.batch_size)
        print(f"best alpha {result['alpha']} (val accuracy {result['val']['accuracy']:.4f})")
        return
    value = args.alpha
    if value is None:
        settings = ValidationSettings.model_validate_json((paths.VALIDATION_RESULTS_DIR / "settings.json").read_text())
        value = getattr(settings, NAME).alpha
    print_table(run(vlm, value, args.split, args.limit, args.batch_size))


if __name__ == "__main__":
    main()

"""Tune the four detector + method combinations on the validation split.

feedback   the P(wrong) threshold above which Qwen is asked again
steering   the strength alpha of the hidden-state edit

Reads   outputs/answers/val_*, the trained detectors in src/model/models/
Writes  src/validation/results/settings.json (read by test.py), cached passes in outputs/validation/

Library: run(limit=None, alphas=(0.5, 1.0, 2.0), vlm=None) -> ValidationSettings
Script:  python src/validation/validate.py [--limit N] [--alphas 0.5 1 2] [--batch-size 8]
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.schemas import ValidationSettings
from lib import paths
from model.vlm import VLM
from validation import linear_feedback, linear_steering, mlp_feedback, mlp_steering

FEEDBACK = (linear_feedback, mlp_feedback)
STEERING = (linear_steering, mlp_steering)


def run(limit: int | None = None, alphas: Sequence[float] = (0.5, 1.0, 2.0), batch_size: int = 8,
        vlm: VLM | None = None) -> ValidationSettings:
    """Uses every validation question unless limit is set; returns and saves the chosen settings."""
    vlm = vlm or VLM()
    settings = {}
    for module in FEEDBACK:
        print(f"tuning {module.NAME} (threshold)")
        settings[module.NAME] = module.tune(vlm, limit, batch_size)
    for module in STEERING:
        print(f"tuning {module.NAME} (alpha over {list(alphas)})")
        settings[module.NAME] = module.tune(vlm, limit, batch_size, tuple(alphas))

    settings = ValidationSettings.model_validate(settings)
    paths.VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = paths.VALIDATION_RESULTS_DIR / "settings.json"
    path.write_text(settings.model_dump_json(indent=2))

    print(f"\n{'method':<20}{'setting':>18}{'val accuracy':>14}{'baseline':>10}")
    for name, s in settings:
        setting = f"threshold {s.threshold:g}" if s.threshold is not None else f"alpha {s.alpha:g}"
        print(f"{name:<20}{setting:>18}{s.val.accuracy:>14.4f}{s.val.baseline_accuracy:>10.4f}")
    print(f"wrote {path}")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="first N validation questions (default: all)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args.limit, args.alphas, args.batch_size)


if __name__ == "__main__":
    main()

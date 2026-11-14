"""Final comparison on the test split: baseline Qwen vs the four detector + method combinations, with controls.

Rows: baseline; each feedback method and its random-flags control; feedback with perfect flags (upper bound);
each steering method and its random-direction control. Settings come from validate.py; the test split is used once.

Reads   outputs/answers/test_*, src/validation/results/settings.json, the trained detectors
Writes  src/validation/results/test.json, cached passes in outputs/validation/

Library: run(limit=None, vlm=None) -> TestResults
Script:  python src/validation/test.py [--limit N] [--batch-size 8]
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.schemas import TestResults, ValidationSettings
from lib import paths
from model.vlm import VLM
from validation import baseline, linear_feedback, linear_steering, mlp_feedback, mlp_steering
from validation.common import perfect_feedback_run, print_table


def run(limit: int | None = None, batch_size: int = 8, vlm: VLM | None = None) -> TestResults:
    """Uses every test question unless limit is set; returns and saves every row's metrics."""
    settings_path = paths.VALIDATION_RESULTS_DIR / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"{settings_path} not found; run validate.py first")
    settings = ValidationSettings.model_validate_json(settings_path.read_text())
    vlm = vlm or VLM()

    rows = baseline.run("test", limit)
    for module in (linear_feedback, mlp_feedback):
        rows |= module.run(vlm, getattr(settings, module.NAME).threshold, "test", limit, batch_size)
    rows |= perfect_feedback_run(vlm, "test", limit, batch_size)
    for module in (linear_steering, mlp_steering):
        rows |= module.run(vlm, getattr(settings, module.NAME).alpha, "test", limit, batch_size)

    paths.VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = TestResults(settings=settings, results=rows)
    path = paths.VALIDATION_RESULTS_DIR / "test.json"
    path.write_text(results.model_dump_json(indent=2))
    print_table(rows)
    print(f"wrote {path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="first N test questions (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(args.limit, args.batch_size)


if __name__ == "__main__":
    main()

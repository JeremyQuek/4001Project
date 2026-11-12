"""Label-noise check for the CHAIR labels.

  export: write a CSV of N hallucinated + N grounded mentions (train split) with image links.
          Two team members each fill in their own column independently:
          1 = the object is NOT in the image, 0 = it IS in the image.
  score:  read the filled CSV; report each annotator's agreement with CHAIR per label and
          Cohen's kappa between the two annotators.
"""
import random

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from lib.common import captions_path, images_meta
from lib.config import base_parser, config_from_args
from lib.utils.io import read_jsonl


def export(cfg, n, out):
    images = images_meta(cfg)
    caps = {c["image_id"]: c["caption"] for c in read_jsonl(captions_path(cfg))}
    labels = [r for r in read_jsonl(cfg.run_dir / "labels.jsonl") if r["split"] == "train"]
    rng = random.Random(cfg.seed)
    pick = []
    for flag in (1, 0):
        pool = [r for r in labels if r["hallucinated"] == flag]
        pick += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(pick)  # annotators should not see the CHAIR label order
    rows = [{
        "uid": r["uid"], "image_url": images[r["image_id"]]["coco_url"], "word": r["word"],
        "category": r["category"], "caption": caps[r["image_id"]],
        "chair_label": r["hallucinated"], "annotator_1": "", "annotator_2": "",
    } for r in pick]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; hide the chair_label column from annotators")


def score(path):
    df = pd.read_csv(path).dropna(subset=["annotator_1", "annotator_2"])
    for a in ("annotator_1", "annotator_2"):
        for flag, name in ((1, "hallucinated"), (0, "grounded")):
            sub = df[df.chair_label == flag]
            err = (sub[a].astype(int) != flag).mean()
            print(f"{a}: CHAIR '{name}' labels judged wrong: {err:.1%} (n={len(sub)})")
    k = cohen_kappa_score(df.annotator_1.astype(int), df.annotator_2.astype(int))
    print(f"Cohen's kappa between annotators: {k:.3f} (n={len(df)})")


def main():
    p = base_parser(__doc__)
    p.add_argument("action", choices=["export", "score"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    path = args.csv or str(cfg.run_dir / "label_audit.csv")
    export(cfg, args.n, path) if args.action == "export" else score(path)


if __name__ == "__main__":
    main()

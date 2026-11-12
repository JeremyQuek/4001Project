"""Stage 03: CHAIR-label every object mention in the generated captions.

Writes <run>/labels.jsonl: one row per mention with its character span and hallucinated 0/1.
"""
from collections import Counter

from lib.common import captions_path, get_chair, images_meta
from lib.config import base_parser, config_from_args
from lib.data.chair import chair_metrics
from lib.utils.io import read_jsonl, write_jsonl


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    images = images_meta(cfg)
    chair = get_chair(cfg)

    rows, per_split = [], {}
    for c in read_jsonl(captions_path(cfg)):
        meta = images[c["image_id"]]
        mentions = chair.label(c["caption"], set(meta["gt"]))
        per_split.setdefault(c["split"], []).append(mentions)
        for k, m in enumerate(mentions):
            rows.append({"uid": f"{c['image_id']}_{k}", "image_id": c["image_id"], "split": c["split"], "k": k,
                         **m.to_dict()})
    write_jsonl(cfg.run_dir / "labels.jsonl", rows)

    for split, labelled in per_split.items():
        m = chair_metrics(labelled)
        print(f"{split:5s} captions={m['n_captions']} mentions={m['n_mentions']} "
              f"hallucinated={m['n_hallucinated']} CHAIR-i={m['chair_i']:.3f} CHAIR-s={m['chair_s']:.3f}")
    top = Counter(r["category"] for r in rows if r["hallucinated"]).most_common(10)
    print("most hallucinated categories:", top)


if __name__ == "__main__":
    main()

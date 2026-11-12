"""Stage 02: greedy captions from the frozen VLM (pass 1 of 2). Saves the exact token ids so
stage 04 can teacher-force them. Resumable: images already in captions.jsonl are skipped.

--pilot runs the first `generation.pilot_n` train images into <run>/pilot/ and prints the
hallucination rate and a precision sanity check, so you can decide before the full run.
"""
from tqdm import tqdm

from lib.common import captions_path, get_chair, image_file, images_meta
from lib.config import base_parser, config_from_args
from lib.data.chair import chair_metrics
from lib.utils.io import append_jsonl, done_keys, read_jsonl
from lib.utils.seed import set_seed
from lib.vlm.generate import generate_captions
from lib.vlm.model import load_vlm


def pilot_report(cfg, rows, images):
    chair = get_chair(cfg)
    labelled = [chair.label(r["caption"], set(images[r["image_id"]]["gt"])) for r in rows]
    m = chair_metrics(labelled)
    empty = sum(1 for r in rows if len(r["caption"].split()) < 3)
    print("\n=== PILOT ===")
    print(f"captions: {len(rows)}  object mentions: {m['n_mentions']}  hallucinated: {m['n_hallucinated']}")
    print(f"CHAIR-i: {m['chair_i']:.3f}  CHAIR-s: {m['chair_s']:.3f}")
    print(f"near-empty captions (<3 words): {empty}  -> if many, fp16 may be broken: use --set model.dtype=float32")
    est = m["n_hallucinated"] / max(len(rows), 1) * cfg.data.n_images * cfg.data.split[0]
    print(f"estimated hallucinated mentions in the train split: ~{est:.0f} (aim for >= 500)")
    for r in rows[:3]:
        print("-" * 60, "\n", r["caption"][:500])


def main():
    p = base_parser(__doc__)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)

    images = images_meta(cfg)
    out = captions_path(cfg, args.pilot)
    todo = [i for i, m in images.items() if m["split"] in args.splits]
    if args.pilot:
        todo = [i for i, m in images.items() if m["split"] == "train"][: cfg.generation.pilot_n]
    done = done_keys(out, "image_id")
    todo = [i for i in todo if i not in done]

    model, processor = load_vlm(cfg)
    g = cfg.generation
    for b in tqdm(range(0, len(todo), g.batch_size), desc="captions"):
        ids = todo[b:b + g.batch_size]
        paths = [image_file(cfg, images[i]) for i in ids]
        results = generate_captions(model, processor, paths, g.prompt, g.max_new_tokens, cfg)
        append_jsonl(out, [
            {"image_id": i, "split": images[i]["split"], "caption": cap, "ids": tok_ids}
            for i, (tok_ids, cap) in zip(ids, results)
        ])

    if args.pilot:
        pilot_report(cfg, read_jsonl(out), images)


if __name__ == "__main__":
    main()

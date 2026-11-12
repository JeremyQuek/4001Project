"""Stage 08: after-generation correction. Starts from the stage-07 base captions, flags objects with
each source (probe / baseline / random / perfect), asks the VLM to rewrite, for up to
`part3.feedback_rounds` rounds. Also runs the simple-deletion control (probe flags, no rewrite).

Writes <run>/correction/feedback_<source>.jsonl and feedback_delete.jsonl. Resumable per image.
"""
import random

from tqdm import tqdm

from lib.common import get_chair, image_file, images_meta
from lib.config import base_parser, config_from_args
from lib.correct.feedback import (baseline_flags, delete_sentences, feedback_prompt, perfect_flags,
                                     post_features, probe_flags, random_flags)
from lib.probe.models import SavedProbe
from lib.utils.io import append_jsonl, done_keys, read_json, read_jsonl
from lib.utils.seed import set_seed
from lib.vlm.generate import teacher_forced
from lib.vlm.tokens import mention_token_spans, token_char_ends
from lib.vlm.guided import guided_generate
from lib.vlm.model import load_vlm, stop_token_ids


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    run, p3 = cfg.run_dir, cfg.part3
    base_path = run / "correction" / "base.jsonl"
    base = {r["image_id"]: r for r in read_jsonl(base_path)}
    if not base:
        raise FileNotFoundError(f"{base_path} is empty; run 07_guided_generation.py first")

    images = images_meta(cfg)
    chair = get_chair(cfg)
    probe = SavedProbe.load(run / "probes" / "best_post.pkl")
    base_thr = read_json(run / "results" / "baseline_thresholds.json")[p3.baseline_score] \
        if "baseline" in p3.feedback_sources else None
    model, processor = load_vlm(cfg)
    tok = processor.tokenizer
    stops = stop_token_ids(model, processor)

    def flags_for(source, img, caption, ids, mentions, rng):
        gt = set(images[img]["gt"])
        if source == "perfect":
            return perfect_flags(mentions, gt)
        firsts = [a for a, _ in mention_token_spans(token_char_ends(tok, ids), [m.to_dict() for m in mentions])]
        tf = teacher_forced(model, processor, image_file(cfg, images[img]), cfg.generation.prompt,
                            ids, [probe.layer], cfg)
        pf = probe_flags(probe, post_features(tf, firsts, 0))
        if source in ("probe", "delete"):
            return pf
        if source == "random":
            return random_flags(len(mentions), len(pf), rng)
        if source == "baseline":
            if p3.baseline_score != "token_prob":
                raise ValueError("feedback baseline source supports token_prob only")
            return baseline_flags([1.0 - float(tf["prob"][t]) for t in firsts], base_thr)
        raise ValueError(source)

    for source in list(p3.feedback_sources) + ["delete"]:
        path = run / "correction" / f"feedback_{source}.jsonl"
        done = done_keys(path, "image_id")
        for img in tqdm([i for i in base if i not in done], desc=f"feedback {source}"):
            rng = random.Random(cfg.seed + img)
            caption, ids = base[img]["caption"], base[img]["ids"]
            # Latency includes the original caption, so conditions compare end-to-end time per image.
            history, seconds = [caption], base[img]["seconds"]
            rounds = 1 if source == "delete" else p3.feedback_rounds
            for _ in range(rounds):
                mentions = chair.find_mentions(caption)
                if not mentions:
                    break
                flagged = flags_for(source, img, caption, ids, mentions, rng)
                if not flagged:
                    break
                if source == "delete":
                    caption = delete_sentences(caption, mentions, flagged)
                else:
                    prompt = feedback_prompt(caption, [mentions[k].word for k in flagged])
                    res = guided_generate(model, processor, chair, image_file(cfg, images[img]), prompt, cfg,
                                          p3.max_new_tokens, policy="none", stops=stops)
                    caption, ids, seconds = res["caption"], res["ids"], seconds + res["seconds"]
                history.append(caption)
            append_jsonl(path, [{"image_id": img, "caption": caption, "history": history,
                                 "rounds_used": len(history) - 1, "seconds": seconds}])


if __name__ == "__main__":
    main()

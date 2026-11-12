"""Stage 07: Part 3 captions through the token-by-token loop.

Always produces the "no correction" reference (policy none) -> <run>/correction/base.jsonl,
which stage 08 builds on. With --guided, also runs the during-generation policies
(candidate / skip) for each flag source -> correction/guided_<policy>_<source>.jsonl.
Resumable per image.
"""
from tqdm import tqdm

from lib.common import get_chair, image_file, images_meta, test_subset
from lib.config import base_parser, config_from_args
from lib.probe.models import SavedProbe
from lib.utils.io import append_jsonl, done_keys, read_json
from lib.utils.seed import set_seed
from lib.vlm.guided import Flagger, guided_generate
from lib.vlm.model import load_vlm, stop_token_ids


def main():
    p = base_parser(__doc__)
    p.add_argument("--guided", action="store_true", help="also run the during-generation policies")
    args = p.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    p3 = cfg.part3
    run = cfg.run_dir
    out_dir = run / "correction"

    images = images_meta(cfg)
    ids = test_subset(cfg, images)
    chair = get_chair(cfg)
    model, processor = load_vlm(cfg)
    stops = stop_token_ids(model, processor)

    jobs = [("none", None)]
    if args.guided:
        jobs += [(pol, src) for pol in p3.guided_policies for src in p3.guided_sources]

    for policy, source in jobs:
        name = "base" if policy == "none" else f"guided_{policy}_{source}"
        path = out_dir / f"{name}.jsonl"
        probe, meta = None, {}
        if source in ("probe", "random"):
            pos = "post" if policy == "candidate" else "pre1"
            probe = SavedProbe.load(run / "probes" / f"best_{pos}.pkl")
            meta = read_json(run / "probes" / f"best_{pos}.json")
        done = done_keys(path, "image_id")
        for i in tqdm([i for i in ids if i not in done], desc=name):
            flagger = None
            if source is not None:
                flagger = Flagger(source, probe=probe, gt=set(images[i]["gt"]),
                                  rate=meta.get("val_flag_rate", 0.0), seed=cfg.seed + i)
            res = guided_generate(
                model, processor, chair, image_file(cfg, images[i]), cfg.generation.prompt, cfg,
                p3.max_new_tokens, policy=policy, flagger=flagger,
                probe_layer=probe.layer if probe is not None else None,
                max_repicks_per_slot=p3.max_repicks_per_slot,
                max_repicks_per_caption=p3.max_repicks_per_caption, stops=stops)
            append_jsonl(path, [{"image_id": i, **res}])


if __name__ == "__main__":
    main()

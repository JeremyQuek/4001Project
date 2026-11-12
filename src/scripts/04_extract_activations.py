"""Stage 04: teacher-forced pass (pass 2 of 2). For every labelled object mention, save the hidden
states at the probe positions (post, pre1, pre2, pre3) for the selected layers, plus token
probability, entropy and (optionally) attention-to-image.

Writes <run>/acts/<split>/shard_XXXX.npz and stats_XXXX.jsonl. A shard is skipped if its .npz
exists, so the stage can be resumed after a disconnect.
"""
import numpy as np
from tqdm import tqdm

from lib.common import captions_path, image_file, images_meta, labels_by_image, num_layers
from lib.config import base_parser, config_from_args
from lib.utils.io import read_jsonl, write_jsonl
from lib.vlm.activations import mention_features, pack_shard, select_layers
from lib.vlm.generate import teacher_forced
from lib.vlm.model import load_vlm


def main():
    p = base_parser(__doc__)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()
    cfg = config_from_args(args)
    a = cfg.activations

    images = images_meta(cfg)
    labels = labels_by_image(cfg)
    captions = {c["image_id"]: c for c in read_jsonl(captions_path(cfg))}

    model, processor = load_vlm(cfg, attn_implementation="eager" if a.attention else None)
    layers = select_layers(cfg, num_layers(model))
    print("saving layers", layers, "positions", a.positions)

    for split in args.splits:
        ids = [i for i in captions if captions[i]["split"] == split and labels.get(i)]
        out_dir = cfg.run_dir / "acts" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in tqdm(range(0, len(ids), a.shard_size), desc=f"{split} shards"):
            shard = s // a.shard_size
            npz = out_dir / f"shard_{shard:04d}.npz"
            if npz.exists():
                continue
            uids, stats = [], []
            feats = {pos: [] for pos in a.positions}
            valid = {pos: [] for pos in a.positions}
            overlap = {pos: [] for pos in a.positions}
            for i in ids[s:s + a.shard_size]:
                c, ments = captions[i], labels[i]
                tf = teacher_forced(model, processor, image_file(cfg, images[i]), cfg.generation.prompt,
                                    c["ids"], layers, cfg, want_attention=a.attention)
                f, v, o, st = mention_features(processor.tokenizer, c["ids"], ments, tf, a.positions)
                for pos in a.positions:
                    feats[pos] += f[pos]
                    valid[pos] += v[pos]
                    overlap[pos] += o[pos]
                for m, row in zip(ments, st):
                    uids.append(m["uid"])
                    stats.append({"uid": m["uid"], "image_id": i, "split": split, "category": m["category"],
                                  "word": m["word"], "hallucinated": m["hallucinated"], **row})
            # Stats first, npz last: an existing npz means the whole shard finished.
            write_jsonl(out_dir / f"stats_{shard:04d}.jsonl", stats)
            np.savez(npz, **pack_shard(layers, a.positions, uids, feats, valid, overlap))


if __name__ == "__main__":
    main()

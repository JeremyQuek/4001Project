"""Turn one teacher-forced pass into per-mention probe features and token statistics."""
import numpy as np

from .tokens import MAX_OFFSET, POSITION_OFFSETS, mention_token_spans, token_char_ends


def select_layers(cfg, n_layers):
    if cfg.activations.get("layers"):
        return sorted(cfg.activations.layers)
    stride = cfg.activations.layer_stride
    return sorted(set(range(stride, n_layers + 1, stride)) | {n_layers})


def mention_features(tokenizer, gen_ids, mentions, tf, positions):
    """For every mention: hidden state at each probe position plus token-level stats.

    Position rule: pre-k reads the state k tokens before the mention's first token T.
    pre1 at T-1 is always valid (for the first caption token it is the last prompt token, which
    is the state that chose it). pre2/pre3 are dropped if they fall inside the prompt.
    A position is tagged `overlap` if it lies on another mention's tokens.
    """
    ends = token_char_ends(tokenizer, gen_ids)
    spans = mention_token_spans(ends, mentions)
    covered = {}
    for k, (a, b) in enumerate(spans):
        for i in range(a, b + 1):
            covered[i] = k

    feats = {pos: [] for pos in positions}
    valid = {pos: [] for pos in positions}
    overlap = {pos: [] for pos in positions}
    stats = []
    for k, (first, last) in enumerate(spans):
        for pos in positions:
            off = POSITION_OFFSETS[pos]
            i = first - off                      # caption-relative index (may be negative)
            ok = i >= (-1 if off == 1 else 0)    # pre1 may sit on the last prompt token
            feats[pos].append(np.asarray(tf["hidden"][:, MAX_OFFSET + i]) if ok else
                              np.zeros(tf["hidden"].shape[::2], dtype=np.float16))
            valid[pos].append(ok)
            overlap[pos].append(bool(off > 0 and covered.get(i, k) != k))
        row = {
            "token_idx": first,
            "n_tokens": last - first + 1,
            "caption_len": len(gen_ids),
            "token_prob": float(tf["prob"][first]),
            "token_entropy": float(tf["entropy"][first]),
        }
        if "att_image" in tf:
            row["att_image"] = tf["att_image"][:, first].tolist()
        stats.append(row)
    return feats, valid, overlap, stats


def pack_shard(layers, positions, uids, feats, valid, overlap):
    """Arrays for one npz shard: key L{layer}_{pos} -> [n, D] float16."""
    out = {"uid": np.array(uids), "layers": np.array(layers)}
    for pos in positions:
        stacked = np.stack(feats[pos]) if feats[pos] else np.zeros((0, len(layers), 0), np.float16)
        for j, l in enumerate(layers):
            out[f"L{l}_{pos}"] = stacked[:, j, :].astype(np.float16) if len(stacked) else stacked
        out[f"valid_{pos}"] = np.array(valid[pos], dtype=bool)
        out[f"overlap_{pos}"] = np.array(overlap[pos], dtype=bool)
    return out

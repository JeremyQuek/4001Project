"""After-generation correction: flag objects in a finished caption, then ask the VLM to rewrite it.

Flag sources share one interface: given a caption (and its token ids), return the indices of the
mentions to flag.
  probe     our probe on the `post` state of each mention (teacher-forced under the ORIGINAL
            describe prompt, i.e. the same setting the probe was trained in)
  baseline  a confidence baseline (e.g. 1 - token probability) with its validation threshold
  random    the same NUMBER of mentions the probe flags, chosen at random
  perfect   ground-truth CHAIR labels (upper bound)
"""
import random
import re

import numpy as np

FEEDBACK_TEMPLATE = (
    'Here is a description of this image:\n"{caption}"\n'
    "These objects are likely not in the image: {objects}. "
    "Rewrite the description without them, and do not add objects you cannot see."
)


def feedback_prompt(caption, flagged_words):
    objects = ", ".join(sorted(set(flagged_words)))
    return FEEDBACK_TEMPLATE.format(caption=caption.strip(), objects=objects)


_SENT_RE = re.compile(r"[^.!?]*[.!?]?")


def delete_sentences(caption, mentions, flagged_idx):
    """Simple-deletion control: drop every sentence that contains a flagged mention."""
    spans = [(m.start, m.end) for k, m in enumerate(mentions) if k in flagged_idx]
    kept = []
    for s in _SENT_RE.finditer(caption):
        if not s.group().strip():
            continue
        if any(s.start() <= a < s.end() for a, _ in spans):
            continue
        kept.append(s.group().strip())
    return " ".join(kept)


def probe_flags(probe, feats_post):
    """feats_post: [n_mentions, D] post states at the probe's layer."""
    if len(feats_post) == 0:
        return set()
    s = probe.score(feats_post)
    return {k for k, v in enumerate(s) if v >= probe.threshold}


def baseline_flags(scores, threshold):
    return {k for k, v in enumerate(scores) if v >= threshold}


def random_flags(n_mentions, k, rng: random.Random):
    k = min(k, n_mentions)
    return set(rng.sample(range(n_mentions), k)) if k else set()


def perfect_flags(mentions, gt):
    return {k for k, m in enumerate(mentions) if m.category not in gt}


def post_features(tf, mention_first_tokens, layer_index):
    """Post-state features for each mention from a teacher_forced() result."""
    from ..vlm.tokens import MAX_OFFSET
    if not mention_first_tokens:
        return np.zeros((0, tf["hidden"].shape[-1]), np.float32)
    return np.stack([tf["hidden"][layer_index, MAX_OFFSET + t].float().numpy() for t in mention_first_tokens])

"""Caption-level metrics for the correction experiments, with paired bootstrap CIs.

Every condition is compared on the same images. For one image we keep:
  mentions   CHAIR-labelled mentions of the corrected caption
  base       CHAIR-labelled mentions of the uncorrected caption (policy "none")
"""
import numpy as np


def per_image_counts(mentions, base_mentions, n_words, seconds):
    cats = {m["category"] for m in mentions}
    real_base = {m["category"] for m in base_mentions if not m["hallucinated"]}
    halluc_base = {m["category"] for m in base_mentions if m["hallucinated"]}
    halluc_now = {m["category"] for m in mentions if m["hallucinated"]}
    return {
        "n_mentions": len(mentions),
        "n_halluc": sum(m["hallucinated"] for m in mentions),
        "has_halluc": int(any(m["hallucinated"] for m in mentions)),
        "real_base": len(real_base),
        "real_kept": len(real_base & cats),
        "new_halluc": len(halluc_now - halluc_base),
        "n_words": n_words,
        "seconds": seconds,
    }


def aggregate(rows):
    tot = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    n = len(rows)
    return {
        "chair_i": tot["n_halluc"] / max(tot["n_mentions"], 1),
        "chair_s": tot["has_halluc"] / max(n, 1),
        "object_recall": tot["real_kept"] / max(tot["real_base"], 1),
        "new_halluc_per_caption": tot["new_halluc"] / max(n, 1),
        "mean_words": tot["n_words"] / max(n, 1),
        "mean_seconds": tot["seconds"] / max(n, 1),
        "n_images": n,
    }


def paired_bootstrap(rows_a, rows_b, metric, n_boot=1000, seed=0):
    """95% CI for metric(a) - metric(b), resampling images (rows are aligned by image)."""
    assert len(rows_a) == len(rows_b)
    rng = np.random.default_rng(seed)
    n = len(rows_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = aggregate([rows_a[i] for i in idx])[metric]
        b = aggregate([rows_b[i] for i in idx])[metric]
        diffs.append(a - b)
    point = aggregate(rows_a)[metric] - aggregate(rows_b)[metric]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi)}

"""Training-free confidence signals. Every score is oriented so that higher = more likely hallucinated."""
import numpy as np


def token_prob_score(stats):
    return np.array([1.0 - r["token_prob"] for r in stats])


def entropy_score(stats):
    return np.array([r["token_entropy"] for r in stats])


def attention_scores(stats, layers):
    """{layer: 1 - share of attention on image tokens} when attention was recorded."""
    if not stats or "att_image" not in stats[0]:
        return {}
    att = np.array([r["att_image"] for r in stats])  # [n, n_layers]
    return {l: 1.0 - att[:, j] for j, l in enumerate(layers)}

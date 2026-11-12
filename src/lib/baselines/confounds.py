"""Confound baselines: can "late in the caption" or "risky object class" alone predict hallucination?
The probe has to beat these to show it learned more than position or object identity."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def position_features(stats):
    return np.array([[r["token_idx"], r["token_idx"] / max(r["caption_len"], 1)] for r in stats], dtype=float)


def class_rates(train_stats, train_y, alpha=1.0):
    """Smoothed training-set hallucination rate per category."""
    prior = float(np.mean(train_y)) if len(train_y) else 0.0
    counts, hits = {}, {}
    for r, y in zip(train_stats, train_y):
        c = r["category"]
        counts[c] = counts.get(c, 0) + 1
        hits[c] = hits.get(c, 0) + int(y)
    rates = {c: (hits[c] + alpha * prior) / (counts[c] + alpha) for c in counts}
    return rates, prior


def class_feature(stats, rates, prior):
    p = np.array([rates.get(r["category"], prior) for r in stats])
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))[:, None]


def fit_confounds(train_stats, train_y):
    """Returns a function stats -> {"position": s, "class": s, "position_class": s}."""
    rates, prior = class_rates(train_stats, train_y)
    Xp = position_features(train_stats)
    Xc = class_feature(train_stats, rates, prior)
    pos_clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xp, train_y)
    both_clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(np.hstack([Xp, Xc]), train_y)

    def score(stats):
        if not stats:
            return {"position": np.zeros(0), "class": np.zeros(0), "position_class": np.zeros(0)}
        xp = position_features(stats)
        xc = class_feature(stats, rates, prior)
        return {
            "position": pos_clf.predict_proba(xp)[:, 1],
            "class": xc[:, 0],
            "position_class": both_clf.predict_proba(np.hstack([xp, xc]))[:, 1],
        }
    return score

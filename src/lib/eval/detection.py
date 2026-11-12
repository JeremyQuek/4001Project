"""Detection metrics for any score where higher = more likely hallucinated."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from ..probe.train import best_f1_threshold


def detection_metrics(y, s, threshold):
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= threshold).astype(int)
    two = len(set(y.tolist())) == 2
    return {
        "auroc": roc_auc_score(y, s) if two else float("nan"),
        "ap": average_precision_score(y, s) if two else float("nan"),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": float(threshold),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "n": int(len(y)),
    }


def evaluate_with_val_threshold(y_val, s_val, y_test, s_test):
    """Pick the F1-optimal threshold on validation, then report test metrics at it."""
    thr = best_f1_threshold(y_val, s_val)
    return detection_metrics(y_test, s_test, thr)

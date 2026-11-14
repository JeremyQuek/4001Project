"""Linear probe: logistic regression on one layer's hidden state, predicting whether the VLM's answer is wrong.

Tries every saved layer and each regularisation strength, picks the best on validation AUROC, then scores the
test split once.

Reads   outputs/answers/{train,val,test}_records.json and _hidden.npy (see common.py)
Writes  src/model/models/linear_probe.joblib, src/training/results/linear_probe.json

Library: run(layers=None, C=(0.01, 0.1, 1.0), data_dir=None)
Script:  python src/training/linear_probe.py [--data-dir DIR] [--layers 8 16 24] [--C 0.01 0.1 1]
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from training.common import (
    auroc_by,
    default_layers,
    best_threshold,
    confidence_scores,
    fit_standardiser,
    layer_features,
    layer_names,
    load_split,
    report,
    save_results,
)

SEED = 4001


def run(layers: Sequence[int] | None = None, C: Sequence[float] = (0.01, 0.1, 1.0),
        data_dir: Path | None = None) -> dict:
    """Train, pick layer and C on validation, score test once; returns the results dict."""
    data_dir = data_dir or paths.answers_dir()
    h_train, y_train, _ = load_split(data_dir, "train")
    h_val, y_val, _ = load_split(data_dir, "val")
    names = layer_names(data_dir, h_train.shape[1])
    layers = list(layers) if layers else default_layers(names)

    sweep = []
    best = None
    for layer in layers:
        idx = names.index(layer)
        x_train = layer_features(h_train, idx)
        mean, std = fit_standardiser(x_train)
        x_train = (x_train - mean) / std
        x_val = (layer_features(h_val, idx) - mean) / std
        for c in C:
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)
            clf.fit(x_train, y_train)
            val_auroc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
            sweep.append({"layer": layer, "C": c, "val_auroc": float(val_auroc)})
            print(f"layer {layer:>3}  C {c:<6}  val AUROC {val_auroc:.4f}")
            if best is None or val_auroc > best["val_auroc"]:
                best = {"layer": layer, "idx": idx, "C": c, "val_auroc": val_auroc, "clf": clf, "mean": mean, "std": std}

    val_scores = best["clf"].predict_proba((layer_features(h_val, best["idx"]) - best["mean"]) / best["std"])[:, 1]
    threshold = best_threshold(y_val, val_scores)

    # the test split is used once, with everything chosen on validation
    h_test, y_test, test_records = load_split(data_dir, "test")
    x_test = (layer_features(h_test, best["idx"]) - best["mean"]) / best["std"]
    test_scores = best["clf"].predict_proba(x_test)[:, 1]
    conf = confidence_scores(test_records)

    results = {
        "detector": "linear_probe",
        "layer": best["layer"],
        "C": best["C"],
        "sweep": sweep,
        "test": report(y_test, test_scores, threshold),
        "test_confidence_baseline_auroc": float(roc_auc_score(y_test, conf)),
        "test_auroc_by_label": auroc_by(test_records, y_test, test_scores, lambda r: r.label.value),
        "test_auroc_by_variant": auroc_by(test_records, y_test, test_scores, lambda r: r.variant.value),
    }
    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: best[k] for k in ("clf", "mean", "std", "layer", "C")} | {"threshold": threshold},
                paths.MODEL_DIR / "linear_probe.joblib")
    path = save_results("linear_probe", results)

    t = results["test"]
    print(f"\nbest: layer {best['layer']}, C {best['C']} (val AUROC {best['val_auroc']:.4f})")
    print(f"test AUROC {t['auroc']:.4f}  F1 {t['f1']:.4f}  vs confidence baseline AUROC "
          f"{results['test_confidence_baseline_auroc']:.4f}")
    print(f"wrote {path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, help="answers folder (default: outputs/answers)")
    parser.add_argument("--layers", type=int, nargs="*", help="model layers to try (default: every saved layer except 0)")
    parser.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0],
                        help="inverse L2 strengths to try")
    args = parser.parse_args()
    run(args.layers, args.C, args.data_dir)


if __name__ == "__main__":
    main()

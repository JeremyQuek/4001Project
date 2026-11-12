"""Stage 05: train probes for every (layer, position), select on validation, report test.

Writes:
  <run>/results/probe_metrics.csv    one row per (model, position, layer, exclude_overlap, split)
  <run>/results/probe_train_size.csv training-set-size ablation for the best post-position probe
  <run>/results/probe_scores.csv     per-mention val/test scores of the selected probe at each position
  <run>/probes/best_{position}.pkl   saved probe (+ .json metadata) used by stages 07/08
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.config import base_parser, config_from_args
from lib.eval.detection import detection_metrics
from lib.probe.dataset import load_split, saved_layers
from lib.probe.models import SavedProbe
from lib.probe.train import best_f1_threshold, score, train_logreg, train_mlp
from lib.utils.io import write_json
from lib.utils.seed import set_seed


def fit(kind, Xtr, ytr, Xva, yva, cfg, seed):
    if kind == "logreg":
        return train_logreg(Xtr, ytr, Xva, yva, cfg.probe.logreg_C, seed)
    return train_mlp(Xtr, ytr, Xva, yva, cfg.probe.mlp, seed)


def main():
    p = base_parser(__doc__)
    p.add_argument("--skip-mlp", action="store_true", help="logistic regression only (fast)")
    args = p.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    run = cfg.run_dir
    (run / "results").mkdir(exist_ok=True)
    (run / "probes").mkdir(exist_ok=True)

    layers = saved_layers(run)
    kinds = [k for k in cfg.probe.models if not (args.skip_mlp and k == "mlp")]
    rows = []
    best = {}  # position -> (val_auroc, kind, layer, fitted, threshold)

    for pos in cfg.activations.positions:
        # Early-warning positions are also run without samples that sit on another object's tokens.
        overlap_variants = [False, True] if pos != "post" else [False]
        for excl in overlap_variants:
            for layer in tqdm(layers, desc=f"{pos} excl={excl}"):
                tr = load_split(run, "train", layer, pos, excl)
                va = load_split(run, "val", layer, pos, excl)
                te = load_split(run, "test", layer, pos, excl)
                if len(tr[1]) == 0 or len(set(tr[1])) < 2:
                    continue
                for kind in kinds:
                    seeds = cfg.probe.seeds if kind == "mlp" else [cfg.seed]
                    per_seed = []
                    for seed in seeds:
                        fitted = fit(kind, tr[0], tr[1], va[0], va[1], cfg, seed)
                        s_va, s_te = score(kind, fitted, va[0]), score(kind, fitted, te[0])
                        thr = best_f1_threshold(va[1], s_va)
                        m_va = detection_metrics(va[1], s_va, thr)
                        m_te = detection_metrics(te[1], s_te, thr)
                        per_seed.append((m_va, m_te))
                        if not excl and (pos not in best or m_va["auroc"] > best[pos][0]):
                            best[pos] = (m_va["auroc"], kind, layer, fitted, thr, s_va, s_te, va[2], te[2])
                    for split, k in (("val", 0), ("test", 1)):
                        ms = [x[k] for x in per_seed]
                        row = {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}
                        row["auroc_std"] = float(np.std([m["auroc"] for m in ms]))
                        rows.append({"model": kind, "position": pos, "layer": layer, "exclude_overlap": excl,
                                     "split": split, "n_seeds": len(ms), **row})
    pd.DataFrame(rows).to_csv(run / "results" / "probe_metrics.csv", index=False)

    # Save the validation-selected probe for each position, plus its per-mention scores.
    score_rows = []
    for pos, (auc, kind, layer, fitted, thr, s_va, s_te, u_va, u_te) in best.items():
        probe = SavedProbe(kind, layer, pos, fitted["scaler"], fitted["model"], thr)
        probe.save(run / "probes" / f"best_{pos}.pkl")
        write_json(run / "probes" / f"best_{pos}.json", {
            "kind": kind, "layer": layer, "position": pos, "val_auroc": auc, "threshold": thr,
            "val_flag_rate": float(np.mean(s_va >= thr)),
        })
        score_rows += [{"uid": u, "split": "val", "position": pos, "score": float(s)} for u, s in zip(u_va, s_va)]
        score_rows += [{"uid": u, "split": "test", "position": pos, "score": float(s)} for u, s in zip(u_te, s_te)]
        print(f"best {pos}: {kind} layer {layer} val AUROC {auc:.3f}")
    pd.DataFrame(score_rows).to_csv(run / "results" / "probe_scores.csv", index=False)

    # Training-set size ablation (logistic regression, best post layer).
    if "post" in best:
        layer = best["post"][2]
        tr = load_split(run, "train", layer, "post")
        va = load_split(run, "val", layer, "post")
        te = load_split(run, "test", layer, "post")
        rng = np.random.default_rng(cfg.seed)
        size_rows = []
        for frac in cfg.probe.train_fractions:
            idx = rng.permutation(len(tr[1]))[: int(frac * len(tr[1]))]
            fitted = train_logreg(tr[0][idx], tr[1][idx], va[0], va[1], cfg.probe.logreg_C, cfg.seed)
            s_va, s_te = score("logreg", fitted, va[0]), score("logreg", fitted, te[0])
            m = detection_metrics(te[1], s_te, best_f1_threshold(va[1], s_va))
            size_rows.append({"train_fraction": frac, "n_train": len(idx), "layer": layer, **m})
        pd.DataFrame(size_rows).to_csv(run / "results" / "probe_train_size.csv", index=False)


if __name__ == "__main__":
    main()

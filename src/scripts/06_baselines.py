"""Stage 06: baselines on the same mentions as the probe.

Always: token probability, entropy, attention-to-image (if recorded), position-only,
object-class-only, position+class. Optional (GPU): --owlvit, --selfcheck.
Thresholds are chosen on validation; metrics are reported on test.

Writes <run>/results/baseline_metrics.csv, baseline_scores.csv, baseline_thresholds.json
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.baselines.confidence import attention_scores, entropy_score, token_prob_score
from lib.baselines.confounds import fit_confounds
from lib.common import image_file, images_meta, mention_stats
from lib.config import base_parser, config_from_args
from lib.eval.detection import detection_metrics
from lib.probe.dataset import saved_layers
from lib.probe.train import best_f1_threshold
from lib.utils.io import write_json


def per_image_lookup(stats, fn):
    """Call fn(image_id, [categories]) once per image and map the result back to each mention."""
    by_img = {}
    for r in stats:
        by_img.setdefault(r["image_id"], set()).add(r["category"])
    table = {}
    for i, cats in tqdm(by_img.items(), desc=fn.__name__ if hasattr(fn, "__name__") else "lookup"):
        res = fn(i, sorted(cats))
        for c, v in res.items():
            table[(i, c)] = v
    return np.array([table[(r["image_id"], r["category"])] for r in stats])


def main():
    p = base_parser(__doc__)
    p.add_argument("--owlvit", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    run = cfg.run_dir

    stats = {s: mention_stats(cfg, s) for s in ("train", "val", "test")}
    y = {s: np.array([r["hallucinated"] for r in stats[s]]) for s in stats}
    scores = {s: {} for s in ("val", "test")}

    for s in ("val", "test"):
        scores[s]["token_prob"] = token_prob_score(stats[s])
        scores[s]["entropy"] = entropy_score(stats[s])
        for layer, v in attention_scores(stats[s], saved_layers(run)).items():
            scores[s][f"attention_L{layer}"] = v

    confounds = fit_confounds(stats["train"], y["train"])
    for s in ("val", "test"):
        scores[s].update(confounds(stats[s]))

    images = images_meta(cfg)
    if args.owlvit:
        from lib.baselines.owlvit import OwlVitChecker
        owl = OwlVitChecker()

        def owlvit(i, cats):
            return owl.presence(image_file(cfg, images[i]), cats)
        for s in ("val", "test"):
            scores[s]["owlvit"] = 1.0 - per_image_lookup(stats[s], owlvit)

    if args.selfcheck:
        from lib.baselines.selfcheck import p_yes
        from lib.vlm.model import load_vlm
        model, processor = load_vlm(cfg)

        def selfcheck(i, cats):
            return p_yes(model, processor, image_file(cfg, images[i]), cats, cfg)
        for s in ("val", "test"):
            scores[s]["selfcheck"] = 1.0 - per_image_lookup(stats[s], selfcheck)

    rows, thresholds, score_rows = [], {}, []
    for name in scores["val"]:
        thr = best_f1_threshold(y["val"], scores["val"][name])
        thresholds[name] = thr
        rows.append({"baseline": name, "split": "val", **detection_metrics(y["val"], scores["val"][name], thr)})
        rows.append({"baseline": name, "split": "test", **detection_metrics(y["test"], scores["test"][name], thr)})
    for s in ("val", "test"):
        for k, r in enumerate(stats[s]):
            score_rows.append({"uid": r["uid"], "split": s, **{n: float(v[k]) for n, v in scores[s].items()}})

    (run / "results").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "results" / "baseline_metrics.csv", index=False)
    pd.DataFrame(score_rows).to_csv(run / "results" / "baseline_scores.csv", index=False)
    write_json(run / "results" / "baseline_thresholds.json", thresholds)
    print(df[df.split == "test"][["baseline", "auroc", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()

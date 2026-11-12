"""Stage 09: report tables and figures from everything above.

Detection: validation-selected layer per (model, position) -> test metrics; baselines on test;
early-warning curve; layer curves.
Correction: CHAIR-i/s, object recall, new hallucinations, length, latency for every condition,
with paired bootstrap CIs against the no-correction captions.

Writes <run>/tables/*.csv and <run>/figures/*.png
"""
import pandas as pd

from lib.common import get_chair, images_meta
from lib.config import base_parser, config_from_args
from lib.eval import plots
from lib.eval.chair_metrics import aggregate, paired_bootstrap, per_image_counts
from lib.utils.io import read_jsonl


def detection_tables(run, tables, figs):
    pm_path = run / "results" / "probe_metrics.csv"
    if not pm_path.exists():
        print("no probe_metrics.csv; skipping detection tables")
        return
    pm = pd.read_csv(pm_path)
    val = pm[pm.split == "val"]
    test = pm[pm.split == "test"]
    keys = ["model", "position", "exclude_overlap"]
    best_layer = val.loc[val.groupby(keys).auroc.idxmax(), keys + ["layer"]]
    best = best_layer.merge(test, on=keys + ["layer"])
    best.to_csv(tables / "probe_best_layer_test.csv", index=False)
    print("\nProbe (validation-selected layer), test:")
    print(best[keys + ["layer", "auroc", "auroc_std", "f1"]].to_string(index=False))

    plots.layer_curve(pm[pm.exclude_overlap == False], figs / "layer_curve_logreg.png", "logreg")  # noqa: E712
    if (pm.model == "mlp").any():
        plots.layer_curve(pm[pm.exclude_overlap == False], figs / "layer_curve_mlp.png", "mlp")  # noqa: E712
    plots.early_warning_curve(best, figs / "early_warning.png")

    bm_path = run / "results" / "baseline_metrics.csv"
    if bm_path.exists():
        bm = pd.read_csv(bm_path)
        bt = bm[bm.split == "test"][["baseline", "auroc", "ap", "f1"]]
        post = best[(best.position == "post") & (best.exclude_overlap == False)]  # noqa: E712
        probe_rows = pd.DataFrame({"baseline": "probe_" + post.model, "auroc": post.auroc, "ap": post.ap, "f1": post.f1})
        comp = pd.concat([probe_rows, bt]).sort_values("auroc", ascending=False)
        comp.to_csv(tables / "detection_vs_baselines.csv", index=False)
        plots.bar(comp, "baseline", "auroc", figs / "detection_vs_baselines.png", "test AUROC")
        print("\nProbe vs baselines (test):")
        print(comp.to_string(index=False))


def correction_tables(cfg, tables, figs):
    run = cfg.run_dir
    base_rows = {r["image_id"]: r for r in read_jsonl(run / "correction" / "base.jsonl")}
    if not base_rows:
        print("no correction/base.jsonl; skipping correction tables")
        return
    images = images_meta(cfg)
    chair = get_chair(cfg)

    def labelled(img, caption):
        return [m.to_dict() for m in chair.label(caption, set(images[img]["gt"]))]

    order = sorted(base_rows)
    base_lab = {i: labelled(i, base_rows[i]["caption"]) for i in order}

    def counts(rows):
        out = []
        for i in order:
            r = rows[i]
            out.append(per_image_counts(labelled(i, r["caption"]), base_lab[i],
                                        len(r["caption"].split()), r.get("seconds", 0.0)))
        return out

    ref = counts(base_rows)
    summary = [{"condition": "none", **aggregate(ref)}]
    for path in sorted((run / "correction").glob("*.jsonl")):
        name = path.stem
        if name == "base":
            continue
        rows = {r["image_id"]: r for r in read_jsonl(path)}
        if any(i not in rows for i in order):
            print(f"{name}: incomplete ({len(rows)}/{len(order)} images); skipped")
            continue
        c = counts(rows)
        row = {"condition": name, **aggregate(c)}
        for metric in ("chair_i", "chair_s", "object_recall"):
            ci = paired_bootstrap(c, ref, metric, cfg.eval.n_bootstrap, cfg.seed)
            row[f"{metric}_diff"] = ci["diff"]
            row[f"{metric}_ci"] = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        summary.append(row)
    df = pd.DataFrame(summary)
    df.to_csv(tables / "correction.csv", index=False)
    plots.bar(df, "condition", "chair_i", figs / "correction_chair_i.png", "CHAIR-i (lower is better)")
    print("\nCorrection (test subset):")
    cols = ["condition", "chair_i", "chair_s", "object_recall", "new_halluc_per_caption", "mean_words", "mean_seconds"]
    print(df[cols].to_string(index=False))


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    tables, figs = cfg.run_dir / "tables", cfg.run_dir / "figures"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    detection_tables(cfg.run_dir, tables, figs)
    correction_tables(cfg, tables, figs)


if __name__ == "__main__":
    main()

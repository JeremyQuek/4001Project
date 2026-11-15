"""MLP detector: a small neural network on one layer's hidden state, predicting whether the VLM's answer is wrong.

Architecture: d_model -> hidden (256) -> ReLU -> dropout -> 1, trained with class-weighted BCE.
Picks the layer on validation AUROC with one seed, then trains the chosen layer with every seed and scores the
test split once; reports mean and std over seeds.

Reads   outputs/answers/{train,val,test}_records.json and _hidden.npy (see common.py)
Writes  src/model/models/mlp.pt, src/training/results/mlp.json

Library: run(layers=None, seeds=(0, 1, 2), config=TrainConfig(), device="auto", data_dir=None)
Script:  python src/training/mlp.py [--data-dir DIR] [--layers 8 16 24] [--seeds 0 1 2] [--epochs 30]
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm.auto import tqdm

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import paths
from lib.device import pick_device
from training.common import (
    auroc_by,
    default_layers,
    best_threshold,
    confidence_metrics,
    detector_metrics,
    fit_standardiser,
    layer_features,
    layer_names,
    load_split,
    report,
    save_results,
    save_scores,
)


@dataclass
class TrainConfig:
    hidden: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256
    epochs: int = 30
    patience: int = 5  # epochs without val improvement before stopping


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model: MLP, x: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(x[i:i + batch]).to(device))).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def train(x_train, y_train, x_val, y_val, args: TrainConfig, seed: int, device: torch.device,
          desc: str = "epochs") -> tuple[MLP, float]:
    """Train with early stopping on validation AUROC; returns the best epoch's model and its AUROC."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(x_train.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # float32: Apple's MPS backend has no float64
    pos_weight = torch.tensor((y_train == 0).sum() / max((y_train == 1).sum(), 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train.astype(np.float32))

    best_auroc, best_state, stale = -1.0, None, 0
    bar = tqdm(range(args.epochs), desc=desc, unit="epoch", leave=False)
    for epoch in bar:
        model.train()
        for idx in np.array_split(rng.permutation(len(x_t)), max(len(x_t) // args.batch, 1)):
            opt.zero_grad()
            loss = loss_fn(model(x_t[idx].to(device)), y_t[idx].to(device))
            loss.backward()
            opt.step()
        val_auroc = roc_auc_score(y_val, predict(model, x_val, device))
        bar.set_postfix(val_auroc=f"{val_auroc:.4f}")
        if val_auroc > best_auroc:
            best_auroc, stale = val_auroc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    bar.close()
    model.load_state_dict(best_state)
    return model, float(best_auroc)


def run(layers: Sequence[int] | None = None, seeds: Sequence[int] = (0, 1, 2), config: TrainConfig | None = None,
        device: str = "auto", data_dir: Path | None = None) -> dict:
    """Pick the layer on validation, train it with every seed, score test once; returns the results dict."""
    args = config or TrainConfig()
    seeds = list(seeds)
    data_dir = data_dir or paths.answers_dir()
    device = pick_device(device)

    h_train, y_train, _ = load_split(data_dir, "train")
    h_val, y_val, val_records = load_split(data_dir, "val")
    names = layer_names(data_dir, h_train.shape[1])
    layers = list(layers) if layers else default_layers(names)

    def features(layer: int):
        idx = names.index(layer)
        x_train = layer_features(h_train, idx)
        mean, std = fit_standardiser(x_train)
        return (x_train - mean) / std, (layer_features(h_val, idx) - mean) / std, mean, std

    # layer sweep with the first seed
    sweep = []
    for layer in tqdm(layers, desc="MLP layer sweep", unit="layer"):
        x_train, x_val, _, _ = features(layer)
        _, val_auroc = train(x_train, y_train, x_val, y_val, args, seeds[0], device, desc=f"layer {layer}")
        sweep.append({"layer": layer, "val_auroc": val_auroc})
        tqdm.write(f"layer {layer:>3}  val AUROC {val_auroc:.4f}")
    best_layer = max(sweep, key=lambda s: s["val_auroc"])["layer"]

    # the chosen layer with every seed; the test split is used once per seed
    x_train, x_val, mean, std = features(best_layer)
    h_test, y_test, test_records = load_split(data_dir, "test")
    x_test = (layer_features(h_test, names.index(best_layer)) - mean) / std
    per_seed, best_run = [], None
    for seed in tqdm(seeds, desc=f"MLP seeds (layer {best_layer})", unit="seed"):
        model, val_auroc = train(x_train, y_train, x_val, y_val, args, seed, device, desc=f"seed {seed}")
        val_scores = predict(model, x_val, device)
        threshold = best_threshold(y_val, val_scores)
        test_scores = predict(model, x_test, device)
        seed_run = {"seed": seed, "val_auroc": val_auroc, "test": report(y_test, test_scores, threshold)}
        per_seed.append(seed_run)
        tqdm.write(f"seed {seed}  val AUROC {val_auroc:.4f}  test AUROC {seed_run['test']['auroc']:.4f}")
        if best_run is None or val_auroc > best_run[0]["val_auroc"]:
            best_run = (seed_run, model, threshold, val_scores, test_scores)

    test_aurocs = [r["test"]["auroc"] for r in per_seed]
    test_f1s = [r["test"]["f1"] for r in per_seed]
    seed_run, model, threshold, val_scores, test_scores = best_run
    conf = confidence_metrics(val_records, y_val, test_records, y_test)
    results = {
        "detector": "mlp",
        "layer": best_layer,
        "config": asdict(args) | {"seeds": seeds},
        "sweep": sweep,
        "per_seed": per_seed,
        "test_auroc_mean": float(np.mean(test_aurocs)),
        "test_auroc_std": float(np.std(test_aurocs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        # the saved model: the seed with the best validation AUROC
        "test": detector_metrics(y_val, val_scores, y_test, test_scores, threshold),
        "test_confidence_baseline_auroc": conf["auroc"],
        "test_confidence_baseline": conf,
        "test_auroc_by_label": auroc_by(test_records, y_test, test_scores, lambda r: r.label.value),
        "test_auroc_by_variant": auroc_by(test_records, y_test, test_scores, lambda r: r.variant.value),
    }
    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_in": x_train.shape[1], "hidden": args.hidden,
                "dropout": args.dropout, "layer": best_layer, "seed": seed_run["seed"], "threshold": threshold,
                "mean": mean, "std": std}, paths.MODEL_DIR / "mlp.pt")
    save_scores("mlp", test_scores)
    path = save_results("mlp", results)

    print(f"\nbest layer {best_layer}: test AUROC {results['test_auroc_mean']:.4f} ± {results['test_auroc_std']:.4f}, "
          f"F1 {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}; confidence baseline AUROC "
          f"{results['test_confidence_baseline_auroc']:.4f}")
    print(f"wrote {path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, help="answers folder (default: outputs/answers)")
    parser.add_argument("--layers", type=int, nargs="*", help="model layers to try (default: every saved layer except 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        parser.add_argument("--" + f.name.replace("_", "-"), type=f.type, default=getattr(defaults, f.name))
    parser.add_argument("--device", default="auto", help="auto (cuda, then mps, then cpu), or cuda, mps or cpu")
    args = parser.parse_args()
    config = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    run(args.layers, args.seeds, config, args.device, args.data_dir)


if __name__ == "__main__":
    main()

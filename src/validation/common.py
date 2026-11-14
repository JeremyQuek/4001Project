"""Shared code for using the detectors at answer time, for the four detector + method scripts.

Two ways of using a detector:
    feedback   if the detector flags an answer, Qwen is asked again in a second turn
    steering   a forward hook at the detector's layer moves the hidden state at the last prompt token until the
               detector's logit is down to its threshold, then the forward pass carries on with the edited state

The baseline is Qwen's plain answers from stage 4 (outputs/answers/{split}_records.json), so it needs no new run.
Every new pass (re-asked or steered answers) is cached in outputs/validation/, chunk by chunk, so runs resume.
"""

import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.records import read_records, write_records
from data.schemas import Answer, Predicate, TrainRecord, ValidationAnswer
from lib import paths
from model.vlm import VLM, draw_boxes, load_image
from training.common import layer_names, load_split
from training.mlp import MLP

SEED = 4001
DETECTORS = ("linear_probe", "mlp")
CHUNK = 512  # questions per cache write
MAX_STEPS = 10  # gradient steps when steering; the linear probe needs exactly one
FOLLOW_UP = "Your previous answer may be wrong. Look again carefully at the red box and the blue box. {question}"


# ---------- detectors as torch scorers ----------

@dataclass(frozen=True)
class Scorer:
    """A trained detector as a differentiable function of one raw hidden state: logit of P(answer is wrong)."""

    name: str
    layer: int
    threshold: float  # on P(wrong), chosen on validation when the detector was trained
    mean: torch.Tensor
    std: torch.Tensor
    net: Callable[[torch.Tensor], torch.Tensor]  # standardised hidden state [B, D] -> logit [B]

    @property
    def z_target(self) -> float:
        return math.log(self.threshold / (1 - self.threshold))

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        return self.net((h - self.mean) / self.std)

    def to(self, device: torch.device) -> "Scorer":
        net = _Linear(self.net.w.to(device), self.net.b) if isinstance(self.net, _Linear) else self.net.to(device)
        return replace(self, mean=self.mean.to(device), std=self.std.to(device), net=net)


@dataclass(frozen=True)
class _Linear:
    w: torch.Tensor
    b: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b


def load_detector(name: str) -> Scorer:
    """A detector saved by src/training/ (in src/model/models/) as a Scorer on the CPU."""
    if name == "linear_probe":
        saved = joblib.load(paths.MODEL_DIR / "linear_probe.joblib")
        clf = saved["clf"]
        net = _Linear(torch.tensor(clf.coef_[0], dtype=torch.float32), float(clf.intercept_[0]))
    elif name == "mlp":
        saved = torch.load(paths.MODEL_DIR / "mlp.pt", weights_only=False)
        net = MLP(saved["d_in"], saved["hidden"], saved["dropout"])
        net.load_state_dict(saved["state_dict"])
        net.eval().requires_grad_(False)
    else:
        raise ValueError(f"unknown detector {name}; choose from {DETECTORS}")
    return Scorer(name=name, layer=saved["layer"], threshold=float(saved["threshold"]),
                  mean=torch.as_tensor(saved["mean"], dtype=torch.float32),
                  std=torch.as_tensor(saved["std"], dtype=torch.float32), net=net)


def load_answers(split: str, limit: int | None = None) -> tuple[list[TrainRecord], np.ndarray, list[int]]:
    """Stage 4's answers for a split: records, hidden states [N, n_layers, d_model] and their layer numbers."""
    hidden, _, records = load_split(paths.answers_dir(), split)
    layers = layer_names(paths.answers_dir(), hidden.shape[1])
    return records[:limit], hidden[:limit], layers


@torch.no_grad()
def saved_logits(scorer: Scorer, hidden: np.ndarray, layers: list[int]) -> np.ndarray:
    """The detector's logit for every saved hidden state, as at stage 4 (before any edit)."""
    idx = layers.index(scorer.layer)
    out = []
    for i in range(0, len(hidden), 4096):
        x = torch.from_numpy(np.asarray(hidden[i:i + 4096, idx, :], dtype=np.float32))
        out.append(scorer.logit(x).numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ---------- steering ----------

def steer_vectors(scorer: Scorer, h: torch.Tensor, alpha: float, control: str | None,
                  generator: torch.Generator) -> torch.Tensor:
    """Edited hidden states [B, D].

    Each row whose logit z is above the target z* is moved by Newton steps along the detector's gradient,
        x <- x - (z(x) - z*) / ||grad z(x)||^2 * grad z(x),
    until z = z*. For the linear probe the gradient is constant (w / std), so one step lands exactly on z*; the MLP
    takes a few. The total move is then scaled by alpha (1 = stop at the threshold, 2 = go twice as far).
    control="random" keeps each row's move size but points it in a random direction.
    """
    h0 = h.detach().float()
    x = h0.clone()
    for _ in range(MAX_STEPS):
        with torch.enable_grad():
            x.requires_grad_(True)
            z = scorer.logit(x)
            (grad,) = torch.autograd.grad(z.sum(), x)
        over = (z.detach() - scorer.z_target).clamp(min=0)
        if not (over > 1e-4).any():
            break
        x = x.detach() - (over / grad.pow(2).sum(-1).clamp(min=1e-12)).unsqueeze(-1) * grad
    delta = alpha * (x.detach() - h0)
    if control == "random":
        r = torch.randn(delta.shape, generator=generator).to(delta.device)
        delta = r / r.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
    return h0 + delta


@contextmanager
def steering(vlm: VLM, scorer: Scorer, alpha: float, control: str | None = None) -> Iterator[None]:
    """While active, every forward pass of vlm has its hidden state at scorer.layer edited at the last token."""
    scorer = scorer.to(vlm.device)
    generator = torch.Generator().manual_seed(SEED)

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs.clone()
        hs[:, -1, :] = steer_vectors(scorer, hs[:, -1, :], alpha, control, generator).to(hs.dtype)
        return (hs, *output[1:]) if isinstance(output, tuple) else hs

    handle = vlm.hook_module(scorer.layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------- running Qwen on records, with a cache ----------

def ask_records(vlm: VLM, records: list[TrainRecord], batch_size: int, reask: bool = False) -> list[ValidationAnswer]:
    """Ask Qwen the records' questions again: plainly, or (reask=True) as a second turn after its first answer."""
    out = []
    boxed = {}
    for b in range(0, len(records), batch_size):
        batch = records[b:b + batch_size]
        for r in batch:
            if r.rel_id not in boxed:
                if len(boxed) > 64:
                    boxed.clear()
                boxed[r.rel_id] = draw_boxes(load_image(r.image_id), r.subj_bbox, r.obj_bbox)
        images = [boxed[r.rel_id] for r in batch]
        questions = [r.query_text for r in batch]
        if reask:
            previous = ["Yes." if r.model_answer is Answer.YES else "No." for r in batch]
            follow_ups = [FOLLOW_UP.format(question=r.query_text) for r in batch]
            answers = vlm.ask(images, questions, previous, follow_ups)
        else:
            answers = vlm.ask(images, questions)
        out.extend(ValidationAnswer(query_id=r.query_id, p_yes=a.p_yes, p_no=a.p_no) for r, a in zip(batch, answers))
    return out


def cached(path: Path, records: list[TrainRecord],
           compute: Callable[[list[TrainRecord]], list[ValidationAnswer]]) -> dict[str, ValidationAnswer]:
    """Answers for records, computing only those not already in the cache file; saved after every chunk."""
    done = {a.query_id: a for a in read_records(path, ValidationAnswer)} if path.exists() else {}
    missing = [r for r in records if r.query_id not in done]
    for i in range(0, len(missing), CHUNK):
        done.update((a.query_id, a) for a in compute(missing[i:i + CHUNK]))
        write_records(path, done.values())
        print(f"  {path.stem}: {min(i + CHUNK, len(missing))}/{len(missing)} new answers")
    return done


def reasked(vlm: VLM, split: str, records: list[TrainRecord], batch_size: int) -> np.ndarray:
    """p_yes after asking every question a second time; shared by both detectors and every threshold."""
    path = paths.validation_dir() / f"{split}_reask.json"
    answers = cached(path, records, lambda rs: ask_records(vlm, rs, batch_size, reask=True))
    return np.array([answers[r.query_id].p_yes for r in records])


def steered(vlm: VLM, split: str, records: list[TrainRecord], scorer: Scorer, alpha: float,
            control: str | None, batch_size: int) -> dict[str, ValidationAnswer]:
    suffix = "_random" if control == "random" else ""
    path = paths.validation_dir() / f"{split}_{scorer.name}_steer_a{alpha:g}{suffix}.json"

    def compute(rs: list[TrainRecord]) -> list[ValidationAnswer]:
        with steering(vlm, scorer, alpha, control):
            return ask_records(vlm, rs, batch_size)

    return cached(path, records, compute)


# ---------- metrics ----------

def original_p_yes(records: list[TrainRecord]) -> np.ndarray:
    return np.array([r.p_yes for r in records])


def metrics(records: list[TrainRecord], p_yes: np.ndarray, flagged: np.ndarray | None = None) -> dict:
    """How the final answers (p_yes >= 0.5 means yes) compare with the ground truth and with the baseline."""
    gt_yes = np.array([r.gt_answer is Answer.YES for r in records])
    before_yes = np.array([r.model_answer is Answer.YES for r in records])
    after_yes = p_yes >= 0.5
    before, after = before_yes == gt_yes, after_yes == gt_yes
    labels = np.array([r.label.value for r in records])
    out = {
        "n": len(records),
        "accuracy": float(after.mean()),
        "baseline_accuracy": float(before.mean()),
        "fixed": int((~before & after).sum()),
        "broken": int((before & ~after).sum()),
        "net_gain": int((~before & after).sum() - (before & ~after).sum()),
        "changed_rate": float((after_yes != before_yes).mean()),
        "mean_confidence": float(np.maximum(p_yes, 1 - p_yes).mean()),
        "accuracy_by_label": {p.value: float(after[labels == p.value].mean()) for p in Predicate
                              if (labels == p.value).any()},
    }
    if flagged is not None:
        out["flagged_rate"] = float(flagged.mean())
        out["flag_precision"] = float((~before[flagged]).mean()) if flagged.any() else 0.0
    return out


def random_flags(n: int, k: int) -> np.ndarray:
    flags = np.zeros(n, dtype=bool)
    flags[np.random.default_rng(SEED).choice(n, size=k, replace=False)] = True
    return flags


def print_table(rows: dict[str, dict]) -> None:
    print(f"\n{'method':<36}{'accuracy':>9}{'fixed':>8}{'broken':>8}{'net':>7}{'changed':>9}{'flagged':>9}{'conf':>7}")
    for name, m in rows.items():
        flagged = f"{m['flagged_rate']:.1%}" if "flagged_rate" in m else "-"
        print(f"{name:<36}{m['accuracy']:>9.4f}{m['fixed']:>8}{m['broken']:>8}{m['net_gain']:>7}"
              f"{m['changed_rate']:>9.1%}{flagged:>9}{m['mean_confidence']:>7.3f}")


# ---------- the two methods, shared by the four scripts ----------

def feedback_tune(vlm: VLM, detector: str, limit: int | None, batch_size: int) -> dict:
    """Pick the P(wrong) threshold for re-asking that gives the best validation accuracy.

    Every validation question is re-asked once (cached), so trying a threshold costs nothing.
    """
    records, hidden, layers = load_answers("val", limit)
    scorer = load_detector(detector)
    p_wrong = 1 / (1 + np.exp(-saved_logits(scorer, hidden, layers)))
    new_p, old_p = reasked(vlm, "val", records, batch_size), original_p_yes(records)
    grid = sorted({round(scorer.threshold, 4), *np.round(np.linspace(0.05, 0.95, 19), 2).tolist()})
    tried = []
    for t in grid:
        flagged = p_wrong >= t
        tried.append((t, metrics(records, np.where(flagged, new_p, old_p), flagged)))
    # best accuracy; on a tie, the higher threshold (fewer re-asks)
    t, best = max(tried, key=lambda tm: (tm[1]["accuracy"], tm[0]))
    return {"threshold": t, "val": best, "grid": [{"threshold": t, "accuracy": m["accuracy"]} for t, m in tried]}


def feedback_run(vlm: VLM, detector: str, threshold: float, split: str, limit: int | None,
                 batch_size: int) -> dict[str, dict]:
    """The detector's re-asks, and a control re-asking the same number of random questions."""
    records, hidden, layers = load_answers(split, limit)
    scorer = load_detector(detector)
    flagged = 1 / (1 + np.exp(-saved_logits(scorer, hidden, layers))) >= threshold
    rand = random_flags(len(records), int(flagged.sum()))
    new_p, old_p = reasked(vlm, split, records, batch_size), original_p_yes(records)
    return {
        f"{detector}_feedback": metrics(records, np.where(flagged, new_p, old_p), flagged),
        f"{detector}_feedback_random_flags": metrics(records, np.where(rand, new_p, old_p), rand),
    }


def perfect_feedback_run(vlm: VLM, split: str, limit: int | None, batch_size: int) -> dict[str, dict]:
    """Upper bound for feedback: re-ask exactly the questions the model got wrong."""
    records, _, _ = load_answers(split, limit)
    flagged = np.array([bool(r.wrong) for r in records])
    new_p, old_p = reasked(vlm, split, records, batch_size), original_p_yes(records)
    return {"perfect_flags_feedback": metrics(records, np.where(flagged, new_p, old_p), flagged)}


def _steered_p_yes(vlm: VLM, split: str, records: list[TrainRecord], flagged: np.ndarray, scorer: Scorer,
                   alpha: float, control: str | None, batch_size: int) -> np.ndarray:
    """Final p_yes: steered for flagged questions, unchanged otherwise (the hook would leave them alone anyway)."""
    p = original_p_yes(records)
    subset = [r for r, f in zip(records, flagged) if f]
    answers = steered(vlm, split, subset, scorer, alpha, control, batch_size) if subset else {}
    for i, r in enumerate(records):
        if flagged[i]:
            p[i] = answers[r.query_id].p_yes
    return p


def steering_tune(vlm: VLM, detector: str, alphas: tuple[float, ...], limit: int | None, batch_size: int) -> dict:
    """Pick the steering strength alpha that gives the best validation accuracy.

    Only questions the detector flags are run again, since the hook doesn't change the others.
    """
    records, hidden, layers = load_answers("val", limit)
    scorer = load_detector(detector)
    flagged = saved_logits(scorer, hidden, layers) > scorer.z_target
    tried = []
    for alpha in alphas:
        p = _steered_p_yes(vlm, "val", records, flagged, scorer, alpha, None, batch_size)
        tried.append((alpha, metrics(records, p, flagged)))
    # best accuracy; on a tie, the smaller alpha (smaller edit)
    alpha, best = max(tried, key=lambda am: (am[1]["accuracy"], -am[0]))
    return {"alpha": alpha, "val": best, "grid": [{"alpha": a, "accuracy": m["accuracy"]} for a, m in tried]}


def steering_run(vlm: VLM, detector: str, alpha: float, split: str, limit: int | None,
                 batch_size: int) -> dict[str, dict]:
    """Steering with the detector's direction, and a control with random directions of the same size."""
    records, hidden, layers = load_answers(split, limit)
    scorer = load_detector(detector)
    flagged = saved_logits(scorer, hidden, layers) > scorer.z_target
    return {
        f"{detector}_steering": metrics(
            records, _steered_p_yes(vlm, split, records, flagged, scorer, alpha, None, batch_size), flagged),
        f"{detector}_steering_random_direction": metrics(
            records, _steered_p_yes(vlm, split, records, flagged, scorer, alpha, "random", batch_size), flagged),
    }

import numpy as np
import pytest

from lib.baselines.confounds import fit_confounds
from lib.correct.feedback import delete_sentences, feedback_prompt, random_flags
from lib.eval.chair_metrics import aggregate, paired_bootstrap, per_image_counts
from lib.eval.detection import evaluate_with_val_threshold
from lib.probe.models import SavedProbe
from lib.probe.train import train_logreg, score


def synthetic(n, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += 2.0 * y  # one informative direction
    return X, y


def test_logreg_probe_learns_and_round_trips(tmp_path):
    Xtr, ytr = synthetic(2000)
    Xva, yva = synthetic(500, seed=1)
    fitted = train_logreg(Xtr, ytr, Xva, yva, [0.01, 0.1, 1.0])
    assert fitted["val_auroc"] > 0.8
    probe = SavedProbe("logreg", 12, "post", fitted["scaler"], fitted["model"], threshold=0.5)
    probe.save(tmp_path / "p.pkl")
    loaded = SavedProbe.load(tmp_path / "p.pkl")
    np.testing.assert_allclose(loaded.score(Xva), score("logreg", fitted, Xva))
    assert loaded.layer == 12 and loaded.position == "post"


def test_mlp_probe():
    pytest.importorskip("torch")
    from lib.config import Config
    from lib.probe.train import train_mlp
    Xtr, ytr = synthetic(2000)
    Xva, yva = synthetic(500, seed=1)
    mcfg = Config.wrap({"hidden": 32, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
                        "batch_size": 128, "max_epochs": 20, "patience": 3})
    assert train_mlp(Xtr, ytr, Xva, yva, mcfg)["val_auroc"] > 0.8


def test_val_threshold_then_test():
    Xva, yva = synthetic(500, seed=1)
    Xte, yte = synthetic(500, seed=2)
    m = evaluate_with_val_threshold(yva, Xva[:, 0], yte, Xte[:, 0])
    assert m["auroc"] > 0.8 and 0 < m["f1"] <= 1


def test_confounds_detect_position_effect():
    rng = np.random.default_rng(0)
    stats, y = [], []
    for k in range(2000):
        idx = int(rng.integers(0, 200))
        stats.append({"token_idx": idx, "caption_len": 200, "category": "dog" if k % 2 else "chair"})
        y.append(int(rng.random() < idx / 200))  # later tokens hallucinate more
    s = fit_confounds(stats, np.array(y))(stats)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, s["position"]) > 0.7
    assert roc_auc_score(y, s["class"]) == pytest.approx(0.5, abs=0.05)


def test_delete_sentences(chair):
    cap = "A man sits. A cat sleeps on the chair. The sky is blue."
    ms = chair.find_mentions(cap)
    out = delete_sentences(cap, ms, {1})  # flag "cat"
    assert out == "A man sits. The sky is blue."


def test_feedback_prompt_and_random_flags():
    import random
    p = feedback_prompt(' A dog. ', ["dog", "cat", "dog"])
    assert "cat, dog" in p and '"A dog."' in p
    assert len(random_flags(5, 2, random.Random(0))) == 2
    assert random_flags(1, 3, random.Random(0)) == {0}


def test_correction_metrics_and_paired_bootstrap():
    real = {"category": "person", "hallucinated": 0}
    fake = {"category": "dog", "hallucinated": 1}
    base = [per_image_counts([real, fake], [real, fake], 10, 1.0) for _ in range(20)]
    fixed = [per_image_counts([real], [real, fake], 8, 2.0) for _ in range(20)]
    a = aggregate(fixed)
    assert a["chair_i"] == 0 and a["object_recall"] == 1 and a["new_halluc_per_caption"] == 0
    ci = paired_bootstrap(fixed, base, "chair_i", n_boot=200)
    assert ci["diff"] == pytest.approx(-0.5) and ci["ci_high"] < 0

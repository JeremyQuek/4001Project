"""Training both probe types with model selection on the validation split only."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .models import Standardizer, fit_logreg, make_mlp


def safe_auroc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")


def train_logreg(Xtr, ytr, Xva, yva, Cs, seed=0):
    """Fit one logistic regression per C, keep the best by validation AUROC."""
    scaler = Standardizer().fit(Xtr)
    Ztr, Zva = scaler(Xtr), scaler(Xva)
    best = None
    for C in Cs:
        clf = fit_logreg(Ztr, ytr, C, seed)
        auc = safe_auroc(yva, clf.predict_proba(Zva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    auc, C, clf = best
    return {"scaler": scaler, "model": clf, "val_auroc": auc, "C": C}


def train_mlp(Xtr, ytr, Xva, yva, mcfg, seed=0):
    """2-layer MLP with class-weighted BCE and early stopping on validation AUROC."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = Standardizer().fit(Xtr)
    Ztr = torch.from_numpy(scaler(Xtr)).float()
    Zva = torch.from_numpy(scaler(Xva)).float()
    ytr_t = torch.from_numpy(ytr).float()

    model = make_mlp(Ztr.shape[1], mcfg.hidden, mcfg.dropout)
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg.lr, weight_decay=mcfg.weight_decay)

    best_auc, best_state, bad = -1.0, None, 0
    n = len(Ztr)
    for _ in range(mcfg.max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, mcfg.batch_size):
            idx = perm[i:i + mcfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Ztr[idx]).squeeze(-1), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = safe_auroc(yva, model(Zva).squeeze(-1).numpy())
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= mcfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"scaler": scaler, "model": model, "val_auroc": best_auc}


def score(kind, fitted, X):
    Z = fitted["scaler"](X)
    if kind == "logreg":
        return fitted["model"].predict_proba(Z)[:, 1]
    import torch
    with torch.no_grad():
        return torch.sigmoid(fitted["model"](torch.from_numpy(Z).float()).squeeze(-1)).numpy()


def best_f1_threshold(y, s):
    """Threshold maximising F1 on (validation) scores."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    k = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[k]) if len(t) else 0.5

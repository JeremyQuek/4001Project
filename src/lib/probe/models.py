"""The two probe types and a single saved-probe format used by the correction stages."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class Standardizer:
    def fit(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mean) / self.std


def fit_logreg(X, y, C, seed=0):
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
    clf.fit(X, y)
    return clf


def make_mlp(dim, hidden, dropout):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class SavedProbe:
    """A trained probe plus everything needed to score raw hidden states.

    kind: "logreg" or "mlp"; layer/position say which hidden state it reads."""

    def __init__(self, kind, layer, position, scaler, model, threshold=0.5):
        self.kind, self.layer, self.position = kind, layer, position
        self.scaler, self.model, self.threshold = scaler, model, threshold

    def score(self, X):
        """P(hallucinated) for raw hidden states X [n, D] (numpy)."""
        Z = self.scaler(np.asarray(X, dtype=np.float32))
        if self.kind == "logreg":
            return self.model.predict_proba(Z)[:, 1]
        import torch
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Z).float()).squeeze(-1)
        return torch.sigmoid(logits).numpy()

    def save(self, path):
        import pickle
        state = {"kind": self.kind, "layer": self.layer, "position": self.position,
                 "mean": self.scaler.mean, "std": self.scaler.std, "threshold": self.threshold}
        if self.kind == "logreg":
            state["model"] = self.model
        else:
            state["mlp_state"] = {k: v.cpu() for k, v in self.model.state_dict().items()}
            state["mlp_shape"] = (self.model[0].in_features, self.model[0].out_features)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path, dropout=0.0):
        import pickle
        with open(path, "rb") as f:
            s = pickle.load(f)
        scaler = Standardizer()
        scaler.mean, scaler.std = s["mean"], s["std"]
        if s["kind"] == "logreg":
            model = s["model"]
        else:
            model = make_mlp(*s["mlp_shape"], dropout)
            model.load_state_dict(s["mlp_state"])
            model.eval()
        return cls(s["kind"], s["layer"], s["position"], scaler, model, s["threshold"])

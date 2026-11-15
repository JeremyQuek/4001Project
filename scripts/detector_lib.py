"""Detector models, metrics and evaluation helpers.

The metric definitions, abstention rule, bootstrap and MLP training loop are
adapted from the teammate's scripts on branch origin/isaac
(train_logistic_baselines.py, train_mlp_detectors.py,
evaluate_detector_comparison.py, by isaacwvh) so that both sets of results use
the same definitions and metric names.

Risk orientation everywhere: a HIGHER score means the VLM answer is MORE likely wrong.
"""
import copy
import math
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Input sets compared; names match the teammate's summary so tables can be merged.
EXPERIMENTS = [
    {'name': 'all_confidence', 'feature_set': 'confidence', 'layer': None},
    {'name': 'all_hidden9', 'feature_set': 'hidden', 'layer': 9},
    {'name': 'all_hidden18', 'feature_set': 'hidden', 'layer': 18},
    {'name': 'all_hidden36', 'feature_set': 'hidden', 'layer': 36},
]
LR_CONFIG = {'seed': 42, 'C_grid': [0.001, 0.01, 0.1, 1.0, 10.0], 'class_weight_grid': [None, 'balanced'],
             'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-6}
MLP_CONFIG = {'hidden_units': 32, 'dropout': 0.2, 'learning_rate': 1e-3, 'batch_size': 32,
              'max_epochs': 200, 'patience': 15, 'min_delta': 1e-5,
              'weight_decay_grid': [1e-4, 1e-2], 'class_weight_grid': [None, 'balanced'],
              'seeds': [42, 43, 44], 'primary_seed': 42}
TARGET_COVERAGE = 0.8
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261115


def metrics(y, risk):
    """AUROC, error-class average precision and Brier score of a risk score in [0, 1]."""
    risk = np.asarray(risk)
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError('Invalid risk scores.')
    return {'auroc': float(roc_auc_score(y, risk)),
            'average_precision': float(average_precision_score(y, risk)),
            'brier_score': float(brier_score_loss(y, risk))}


def coverage_threshold(y, risk, target=TARGET_COVERAGE):
    """Largest risk threshold whose validation coverage is at most `target`.

    Queries with risk <= threshold are answered. Tied scores stay together.
    """
    y, risk = np.asarray(y), np.asarray(risk)
    target_n = math.floor(target * len(risk))
    values, counts = np.unique(risk, return_counts=True)
    eligible = np.flatnonzero(np.cumsum(counts) <= target_n)
    threshold = float(values[eligible[-1]]) if len(eligible) else float(np.nextafter(values[0], -np.inf))
    keep = risk <= threshold
    answered = int(keep.sum())
    return {'threshold': threshold, 'target_coverage': target, 'answered': answered,
            'coverage': answered / len(y), 'retained_errors': int(y[keep].sum()),
            'retained_vlm_accuracy': float(1 - y[keep].mean()) if answered else None}


def operating_point(y, risk, threshold):
    """Abstention results when a frozen threshold is applied to new data."""
    y = np.asarray(y)
    keep = np.asarray(risk) <= threshold
    answered = int(keep.sum())
    errors = int(y[keep].sum())
    rejected = len(y) - answered
    caught = int(y[~keep].sum())
    return {'validation_threshold': float(threshold), 'answered': answered,
            'coverage': answered / len(y), 'retained_errors': errors,
            'retained_vlm_accuracy': float(1 - errors / answered) if answered else None,
            'abstained': rejected, 'errors_caught': caught,
            'error_recall': float(caught / y.sum()) if y.sum() else None,
            'abstention_precision': float(caught / rejected) if rejected else None}


def risk_coverage(y, risk):
    """Coverage and error rate among answered queries as the threshold rises (ties kept whole)."""
    risk = np.asarray(risk)
    order = np.argsort(risk, kind='stable')
    sorted_risk = risk[order]
    end = np.r_[np.flatnonzero(np.diff(sorted_risk) != 0), len(order) - 1]
    answered = end + 1
    errors = np.cumsum(np.asarray(y)[order])[end]
    return answered / len(risk), errors / answered


def paired_bootstrap(y, predictions, image_ids, pairs, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """Image-clustered bootstrap; every model is scored on the same resamples.

    Returns 95% AUROC intervals per model and 95% intervals of AUROC differences
    for each (model_a, model_b) pair.
    """
    y = np.asarray(y)
    groups = {}
    for index, image_id in enumerate(image_ids):
        groups.setdefault(str(image_id), []).append(index)
    groups = list(groups.values())
    generator = np.random.default_rng(seed)
    samples = {name: [] for name in predictions}
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in generator.integers(0, len(groups), len(groups))])
        if len(np.unique(y[indices])) != 2:
            continue
        for name, risk in predictions.items():
            samples[name].append(float(roc_auc_score(y[indices], risk[indices])))
    intervals = {name: np.quantile(values, [.025, .975]).tolist() for name, values in samples.items()}
    comparisons = {}
    for label, (a, b) in pairs.items():
        deltas = np.asarray(samples[a]) - np.asarray(samples[b])
        comparisons[label] = {
            'model_a': a, 'model_b': b,
            'auroc_delta': float(roc_auc_score(y, predictions[a]) - roc_auc_score(y, predictions[b])),
            'paired_image_bootstrap95': np.quantile(deltas, [.025, .975]).tolist()}
    return {'auroc_intervals': intervals, 'comparisons': comparisons,
            'valid_resamples': len(next(iter(samples.values()))), 'resampling_unit': 'image'}


def fit_logistic(X, y, C, class_weight, config=LR_CONFIG):
    """StandardScaler + L2 logistic regression, both fitted on the given (training) rows only."""
    if not np.isfinite(X).all() or len(np.unique(y)) != 2:
        raise ValueError('Training requires finite inputs and both target classes.')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=C, class_weight=class_weight, solver=config['solver'],
                                          max_iter=config['max_iter'], tol=config['tol'],
                                          random_state=config['seed']))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if pipeline['classifier'].classes_.tolist() != [0, 1]:
        raise ValueError('Risk output must correspond to error class 1.')
    return pipeline, converged


class ErrorMLP(torch.nn.Module):
    """Linear(in, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1); output is an error logit."""

    def __init__(self, input_dim, hidden_units=32, dropout=.2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_units), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_units, 1))

    def forward(self, X):
        return self.network(X).squeeze(-1)


def risk_scores(model, X):
    """Sigmoid error probability for already-scaled inputs."""
    model.eval()
    with torch.inference_mode():
        values = torch.sigmoid(model(torch.as_tensor(np.asarray(X, dtype=np.float32)))).numpy()
    if not np.isfinite(values).all():
        raise ValueError('Invalid MLP risk scores.')
    return values


def fit_mlp(X, y, V, v, seed, weight_decay, class_weight, config=MLP_CONFIG):
    """Train one MLP on scaled train rows; keep the epoch with the lowest validation BCE."""
    torch.manual_seed(seed)
    model = ErrorMLP(X.shape[1], config['hidden_units'], config['dropout'])
    # "balanced" weights the error class by (#correct / #wrong) from the training labels.
    positive_weight = float((y == 0).sum() / (y == 1).sum()) if class_weight == 'balanced' else 1.
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=weight_decay)
    train_X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    train_y = torch.as_tensor(y, dtype=torch.float32)
    val_X = torch.as_tensor(np.asarray(V, dtype=np.float32))
    val_y = torch.as_tensor(v, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, stale, history = float('inf'), None, 0, 0, []
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator)
        total = 0.
        for start in range(0, len(order), config['batch_size']):
            indices = order[start:start + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_X[indices]), train_y[indices])
            if not torch.isfinite(loss).item():
                raise ValueError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(val_X), val_y))
        history.append({'epoch': epoch, 'train_bce': total / len(train_y), 'validation_bce': val_loss})
        if val_loss < best_loss - config['min_delta']:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    val_risk = risk_scores(model, V)
    return {'model': model, 'seed': seed, 'weight_decay': weight_decay, 'class_weight': class_weight,
            'positive_weight': positive_weight, 'best_epoch': best_epoch, 'epochs_run': len(history),
            'best_validation_bce': best_loss, 'history': history,
            'training_metrics': metrics(y, risk_scores(model, X)),
            'validation_metrics': metrics(v, val_risk), 'validation_risk': val_risk}


def save_mlp(path, run, input_dim, config=MLP_CONFIG):
    torch.save({'state_dict': run['model'].state_dict(), 'input_dim': input_dim,
                'hidden_units': config['hidden_units'], 'dropout': config['dropout'],
                'seed': run['seed'], 'best_epoch': run['best_epoch'],
                'weight_decay': run['weight_decay'], 'class_weight': run['class_weight']}, path)


def load_mlp(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ErrorMLP(saved['input_dim'], saved['hidden_units'], saved['dropout'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model

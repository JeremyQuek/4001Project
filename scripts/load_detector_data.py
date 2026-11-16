"""Loader for the detector-ready arrays, with checksum and alignment checks.

    X, y, info = load_detector_data('train', 'hidden', layer=18)
    X, y, info = load_detector_data('val', 'confidence')

Detector inputs are `p_yes`, `confidence` and the hidden states.
Training and model selection use train and validation; test is for evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import DETECTOR_READY, read_json, sha256_file

SPLITS = ('train', 'val', 'test')
LAYERS = (9, 18, 36)


def load_detector_data(split, feature_set, layer=None, folder=DETECTOR_READY):
    """Return (X float32, y_error int, info) for one split and feature set."""
    folder = Path(folder)
    if split not in SPLITS:
        raise ValueError(f'unknown split {split}')
    metadata = read_json(folder / 'metadata.json')
    for name in (f'{split}.npz', f'{split}_rows.csv'):
        if sha256_file(folder / name) != metadata['checksums'][name]:
            raise ValueError(f'checksum mismatch for {name}')
    with np.load(folder / f'{split}.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    rows = pd.read_csv(folder / f'{split}_rows.csv')

    # Alignment between the arrays and the row table.
    n = len(rows)
    if not np.array_equal(rows['row_idx'], np.arange(n)) or set(rows['split']) != {split}:
        raise ValueError('row table order != npz')
    if any(len(value) != n for value in arrays.values()):
        raise ValueError('length mismatch')
    if not np.array_equal(arrays['truth'], rows['truth']):
        raise ValueError('truth mismatch')
    # outcome columns should agree with each other
    if not np.array_equal(arrays['vlm_pred'], (arrays['p_yes'] >= 0.5).astype(np.int8)):
        raise ValueError('vlm_pred != (p_yes >= 0.5)')
    if not np.array_equal(arrays['y_error'], (arrays['vlm_pred'] != arrays['truth']).astype(np.int8)):
        raise ValueError('y_error != (vlm_pred != truth)')

    if feature_set == 'confidence':
        X = np.column_stack([arrays['p_yes'], arrays['confidence']])
    elif feature_set == 'hidden':
        if layer not in LAYERS:
            raise ValueError('layer must be 9, 18 or 36')
        X = arrays[f'hidden_L{layer}']
    else:
        raise ValueError(f'unknown feature set {feature_set}')
    X = np.asarray(X, dtype=np.float32)
    expected_width = 2 if feature_set == 'confidence' else 2048
    if X.shape != (n, expected_width) or not np.isfinite(X).all():
        raise ValueError('bad feature shape or NaN/inf')
    y = arrays['y_error'].astype(int)
    info = {'rows': rows, 'image_ids': rows['image_id'].to_numpy(),
            'confidence': arrays['confidence'], 'feature_set': feature_set, 'layer': layer}
    return X, y, info

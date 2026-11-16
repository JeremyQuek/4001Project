"""Evaluate the selected detectors on the held-out test split.

No fitting or threshold selection happens here. Existing results are kept
unless --rerun is passed.
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from common import DETECTOR_READY, OUTPUTS, ROOT, read_json, sha256_file, write_csv, write_json
from detector_lib import (BOOTSTRAP_REPETITIONS, load_mlp, metrics, operating_point, paired_bootstrap,
                          risk_scores)
from load_detector_data import load_detector_data

FROZEN = OUTPUTS / 'detectors/frozen.json'
RESULTS = OUTPUTS / 'results.json'


def check_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f'Frozen artifact changed: {path}')


def group_breakdown(frame, column, model_names):
    """VLM behaviour and detector AUROC inside each predicate or variant."""
    result = {}
    for value, part in frame.groupby(column, sort=False):
        y = part['y_error'].to_numpy()
        entry = {'rows': len(part), 'errors': int(y.sum()), 'vlm_error_rate': float(y.mean()),
                 'vlm_accuracy': float(1 - y.mean()), 'vlm_yes_rate': float(part['vlm_pred'].mean()),
                 'auroc': {}}
        for name in model_names:
            # AUROC is undefined when a group has only one outcome class.
            entry['auroc'][name] = float(roc_auc_score(y, part[name])) if 0 < y.sum() < len(y) else None
        result[str(value)] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun', action='store_true', help='Recompute test results from saved models.')
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    frozen = read_json(FROZEN)
    if RESULTS.exists() and not args.rerun:
        if read_json(RESULTS)['protocol']['frozen_sha256'] != sha256_file(FROZEN):
            parser.error('Saved results belong to different fits. Use --rerun to evaluate the current models.')
        print(f'Using saved results: {RESULTS}. Pass --rerun to recompute.')
        return
    models = frozen['models']
    # Verify the saved inputs and models; source-code edits do not invalidate a run.
    for name, checksum in frozen['data_checksums'].items():
        check_hash(DETECTOR_READY / name, checksum)
    for entry in models.values():
        if entry['kind'] == 'lr':
            check_hash(ROOT / entry['model_file'], entry['model_sha256'])
        elif entry['kind'] == 'mlp':
            check_hash(ROOT / entry['scaler_file'], entry['scaler_sha256'])
            for run in entry['seeds'].values():
                check_hash(ROOT / run['checkpoint'], run['checkpoint_sha256'])

    protocol = {'frozen_sha256': sha256_file(FROZEN), 'evaluator_sha256': sha256_file(__file__),
                'primary_seed': frozen['mlp_config']['primary_seed'],
                'bootstrap_repetitions': BOOTSTRAP_REPETITIONS,
                'test_thresholds': 'Unchanged validation thresholds; test coverage can differ from 80%.',
                'test_use': 'Final evaluation only; no model, seed or threshold selection.',
                'other_seeds': 'Separate stability runs, not an ensemble or seed search.'}
    _, y, info = load_detector_data('test', 'confidence')
    rows = info['rows']
    for split in ('train', 'val'):
        other = pd.read_csv(DETECTOR_READY / f'{split}_rows.csv')
        if set(rows['image_id']) & set(other['image_id']):
            raise ValueError('Test image leakage across splits.')

    predictions = {'heuristic_confidence': 1 - info['confidence']}
    thresholds = {'heuristic_confidence': models['heuristic_confidence']['coverage80']['threshold']}
    seed_stability, pairs = {}, {}
    primary = str(frozen['mlp_config']['primary_seed'])
    for experiment in frozen['experiments']:
        name = experiment['name']
        X, targets, _ = load_detector_data('test', experiment['feature_set'], experiment['layer'])
        if not np.array_equal(targets, y):
            raise ValueError('Test rows are misaligned between feature sets.')
        lr_entry = models[f'lr_{name}']
        pipeline = joblib.load(ROOT / lr_entry['model_file'])
        if pipeline['classifier'].classes_.tolist() != [0, 1]:
            raise ValueError('LR output class order changed.')
        predictions[f'lr_{name}'] = pipeline.predict_proba(X)[:, 1]
        thresholds[f'lr_{name}'] = lr_entry['coverage80']['threshold']

        mlp_entry = models[f'mlp_{name}']
        scaled = joblib.load(ROOT / mlp_entry['scaler_file']).transform(X).astype(np.float32)
        per_seed = []
        for seed, run in mlp_entry['seeds'].items():
            risk = risk_scores(load_mlp(ROOT / run['checkpoint']), scaled)
            per_seed.append({'seed': int(seed), **metrics(y, risk),
                             'operating_point': operating_point(y, risk, run['coverage80']['threshold'])})
            if seed == primary:  # seed 42 is the reported MLP; no best-seed picking
                predictions[f'mlp_{name}'] = risk
                thresholds[f'mlp_{name}'] = run['coverage80']['threshold']
        aurocs = [r['auroc'] for r in per_seed]
        seed_stability[f'mlp_{name}'] = {'seeds': per_seed, 'test_auroc_mean': float(np.mean(aurocs)),
                                         'test_auroc_std': float(np.std(aurocs, ddof=1))}
        # Key paired comparisons: MLP vs LR, each vs the unfitted heuristic,
        # and hidden states vs the fitted confidence baseline.
        pairs[f'{name}_mlp_minus_lr'] = (f'mlp_{name}', f'lr_{name}')
        pairs[f'{name}_lr_minus_heuristic'] = (f'lr_{name}', 'heuristic_confidence')
        pairs[f'{name}_mlp_minus_heuristic'] = (f'mlp_{name}', 'heuristic_confidence')
        if experiment['feature_set'] == 'hidden':
            pairs[f'lr_{name}_minus_lr_all_confidence'] = (f'lr_{name}', 'lr_all_confidence')
            pairs[f'mlp_{name}_minus_mlp_all_confidence'] = (f'mlp_{name}', 'mlp_all_confidence')

    bootstrap = paired_bootstrap(y, predictions, info['image_ids'], pairs)
    group = {'rows': len(y), 'images': int(len(set(info['image_ids']))), 'errors': int(y.sum()),
             'error_prevalence': float(y.mean()), 'unfiltered_vlm_accuracy': float(1 - y.mean()),
             'models': {}, 'seed_stability': seed_stability, 'bootstrap': bootstrap}
    metric_rows = []
    for name, risk in predictions.items():
        result = {**metrics(y, risk), 'auroc_image_bootstrap95': bootstrap['auroc_intervals'][name],
                  'operating_point': operating_point(y, risk, thresholds[name])}
        group['models'][name] = result
        metric_rows.append({'cohort': 'all', 'model': name, 'rows': len(y), 'errors': int(y.sum()),
                            **metrics(y, risk), 'auroc_ci_low': result['auroc_image_bootstrap95'][0],
                            'auroc_ci_high': result['auroc_image_bootstrap95'][1], **result['operating_point']})

    # One line per test query with every model's risk (source for curves and breakdowns).
    with np.load(DETECTOR_READY / 'test.npz') as saved:
        vlm_pred = saved['vlm_pred']
    frame = rows[['row_idx', 'image_id', 'rel_id', 'variant', 'canonical_predicate', 'truth']].copy()
    frame['vlm_pred'] = vlm_pred
    frame['y_error'] = y
    for name, risk in predictions.items():
        frame[name] = risk
    frame.to_csv(OUTPUTS / 'test_predictions.csv', index=False)
    write_csv(OUTPUTS / 'test_metrics.csv', metric_rows)

    summary = {
        'dataset': 'Visual Genome spatial relations (generated variants)', 'protocol': protocol,
        'test_opened': True, 'cohorts': {'all': group},
        'breakdowns': {'canonical_predicate': group_breakdown(frame, 'canonical_predicate', list(predictions)),
                       'variant': group_breakdown(frame, 'variant', list(predictions))},
        'test_predictions_sha256': sha256_file(OUTPUTS / 'test_predictions.csv'),
        'evaluation_wall_seconds': round(time.perf_counter() - started, 1),
    }
    write_json(RESULTS, summary)
    print(f'Test evaluated: {len(y)} rows, {int(y.sum())} VLM errors. Results in {RESULTS}.', flush=True)


if __name__ == '__main__':
    main()

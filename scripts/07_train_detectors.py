"""Train logistic-regression and MLP error detectors.

Fits use the training split only (scalers included); hyperparameters and the
abstention thresholds are chosen on validation only. The script then freezes
every model file, threshold and the test-file checksum in
outputs/detectors/frozen.json. The test split is never opened here.
"""
import argparse
import itertools
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from common import DETECTOR_READY, OUTPUTS, ROOT, read_json, sha256_file, write_csv, write_json
from detector_lib import (EXPERIMENTS, LR_CONFIG, MLP_CONFIG, TARGET_COVERAGE, coverage_threshold, fit_logistic,
                          fit_mlp, metrics, save_mlp)
from load_detector_data import load_detector_data

OUT = OUTPUTS / 'detectors'


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Replace saved detector fits.')
    args = parser.parse_args()
    if (OUT / 'frozen.json').exists() and not args.overwrite:
        parser.error('Detectors already exist. Use --overwrite or set VG_OUTPUT_DIR to a new folder.')
    torch.set_num_threads(1)
    started = time.perf_counter()
    (OUT / 'models').mkdir(parents=True, exist_ok=True)
    (OUT / 'histories').mkdir(exist_ok=True)

    models = {}        # frozen entry per reported model name
    lr_rows, mlp_rows = [], []  # every candidate tried, for the report
    validation_predictions = {}

    for experiment in EXPERIMENTS:
        name = experiment['name']
        kwargs = {'feature_set': experiment['feature_set'], 'layer': experiment['layer']}
        X, y, info = load_detector_data('train', **kwargs)
        V, v, val_info = load_detector_data('val', **kwargs)
        if set(info['image_ids']) & set(val_info['image_ids']):
            raise ValueError('Training/validation image overlap.')

        if name == 'all_confidence':
            # Unfitted baseline: risk = 1 - confidence. Only its threshold uses validation.
            risk = 1 - val_info['confidence']
            models['heuristic_confidence'] = {
                'kind': 'heuristic', 'validation_metrics': metrics(v, risk),
                'coverage80': coverage_threshold(v, risk)}
            validation_predictions['heuristic_confidence'] = risk

        # ---- Logistic regression: validation AUROC, then AP, then smaller C, then unweighted.
        best = None
        for C, weight in itertools.product(LR_CONFIG['C_grid'], LR_CONFIG['class_weight_grid']):
            pipeline, converged = fit_logistic(X, y, C, weight)
            risk = pipeline.predict_proba(V)[:, 1]
            scores = metrics(v, risk)
            lr_rows.append({'experiment': name, 'C': C, 'class_weight': str(weight), 'converged': converged,
                            'iterations': int(pipeline['classifier'].n_iter_[0]),
                            **{f'validation_{k}': val for k, val in scores.items()},
                            'training_auroc': metrics(y, pipeline.predict_proba(X)[:, 1])['auroc']})
            key = (scores['auroc'], scores['average_precision'], -C, int(weight is None))
            if best is None or key > best['key']:
                best = {'key': key, 'pipeline': pipeline, 'C': C, 'class_weight': weight,
                        'risk': risk, 'scores': scores, 'converged': converged}
        path = OUT / 'models' / f'lr_{name}.joblib'
        joblib.dump(best['pipeline'], path)
        models[f'lr_{name}'] = {
            'kind': 'lr', **experiment, 'train_rows': len(y), 'validation_rows': len(v),
            'selected_C': best['C'], 'selected_class_weight': best['class_weight'],
            'converged': best['converged'], 'validation_metrics': best['scores'],
            'coverage80': coverage_threshold(v, best['risk']),
            'model_file': relative(path), 'model_sha256': sha256_file(path)}
        validation_predictions[f'lr_{name}'] = best['risk']
        print(f'lr_{name}: validation AUROC={best["scores"]["auroc"]:.3f} '
              f'(C={best["C"]}, weights={best["class_weight"]})', flush=True)

        # ---- MLP: scaler fitted on train only; selection by mean validation AUROC over seeds.
        scaler = StandardScaler().fit(X)
        train_scaled = scaler.transform(X).astype(np.float32)
        val_scaled = scaler.transform(V).astype(np.float32)
        best = None
        for decay, weight in itertools.product(MLP_CONFIG['weight_decay_grid'], MLP_CONFIG['class_weight_grid']):
            runs = [fit_mlp(train_scaled, y, val_scaled, v, seed, decay, weight) for seed in MLP_CONFIG['seeds']]
            aurocs = [run['validation_metrics']['auroc'] for run in runs]
            mean_auroc = float(np.mean(aurocs))
            mean_ap = float(np.mean([run['validation_metrics']['average_precision'] for run in runs]))
            row = {'experiment': name, 'weight_decay': decay, 'class_weight': str(weight),
                   'mean_validation_auroc': mean_auroc, 'std_validation_auroc': float(np.std(aurocs, ddof=1)),
                   'mean_validation_average_precision': mean_ap}
            for run in runs:
                row[f'seed{run["seed"]}_auroc'] = run['validation_metrics']['auroc']
                row[f'seed{run["seed"]}_best_epoch'] = run['best_epoch']
            mlp_rows.append(row)
            key = (mean_auroc, mean_ap, decay, int(weight is None))
            if best is None or key > best['key']:
                best = {'key': key, 'runs': runs, 'row': row, 'weight_decay': decay, 'class_weight': weight}
        scaler_path = OUT / 'models' / f'mlp_{name}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        entry = {'kind': 'mlp', **experiment, 'train_rows': len(y), 'validation_rows': len(v),
                 'selected_weight_decay': best['weight_decay'], 'selected_class_weight': best['class_weight'],
                 'selection_metrics': best['row'], 'scaler_file': relative(scaler_path),
                 'scaler_sha256': sha256_file(scaler_path), 'seeds': {}}
        for run in best['runs']:
            seed = run['seed']
            path = OUT / 'models' / f'mlp_{name}_seed{seed}.pt'
            save_mlp(path, run, X.shape[1])
            write_json(OUT / 'histories' / f'mlp_{name}_seed{seed}.json', run['history'])
            entry['seeds'][str(seed)] = {
                'checkpoint': relative(path), 'checkpoint_sha256': sha256_file(path),
                'best_epoch': run['best_epoch'], 'epochs_run': run['epochs_run'],
                'training_metrics': run['training_metrics'], 'validation_metrics': run['validation_metrics'],
                'coverage80': coverage_threshold(v, run['validation_risk'])}
            if seed == MLP_CONFIG['primary_seed']:
                validation_predictions[f'mlp_{name}'] = run['validation_risk']
        models[f'mlp_{name}'] = entry
        print(f'mlp_{name}: mean validation AUROC={best["row"]["mean_validation_auroc"]:.3f} '
              f'+/- {best["row"]["std_validation_auroc"]:.3f} '
              f'(decay={best["weight_decay"]}, weights={best["class_weight"]})', flush=True)

    write_csv(OUT / 'lr_candidates.csv', lr_rows)
    write_csv(OUT / 'mlp_candidates.csv', mlp_rows)
    np.savez(OUT / 'validation_predictions.npz', **validation_predictions)

    data = read_json(DETECTOR_READY / 'metadata.json')
    frozen = {
        'models': models, 'experiments': EXPERIMENTS, 'lr_config': LR_CONFIG, 'mlp_config': MLP_CONFIG,
        'target_validation_coverage': TARGET_COVERAGE,
        'selection': {'lr': 'validation AUROC, then average precision, then smaller C, then unweighted',
                      'mlp': 'mean validation AUROC over seeds 42-44, then mean average precision, '
                             'then larger weight decay, then unweighted'},
        'fit_split': 'train', 'selection_split': 'val', 'test_opened': False,
        'data_checksums': data['checksums'],
        'versions': {p: version(p) for p in ('scikit-learn', 'numpy', 'torch', 'joblib')},
        'candidates_tried': {'lr': len(lr_rows), 'mlp_configurations': len(mlp_rows),
                             'mlp_seed_fits': len(mlp_rows) * len(MLP_CONFIG['seeds'])},
        'training_wall_seconds': round(time.perf_counter() - started, 1),
    }
    write_json(OUT / 'frozen.json', frozen)
    print(f'Frozen {len(models)} models before test; {frozen["training_wall_seconds"]} s.', flush=True)


if __name__ == '__main__':
    main()

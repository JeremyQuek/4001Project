"""Unit tests for the risky parts of the pipeline.

Tests on the real arrays are skipped until stage 6 has produced them. They read
only row metadata for the test split and never its outcomes.
"""
import importlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from common import (CANONICAL, DETECTOR_READY, OPPOSITE, SURFACE_TO_CANONICAL,
                    extraction_settings, make_statements, read_json, write_json)
from detector_lib import (coverage_threshold, fit_logistic, fit_mlp, load_mlp, metrics, operating_point,
                          paired_bootstrap, risk_coverage, risk_scores, save_mlp)
from load_detector_data import load_detector_data

needs_data = pytest.mark.skipif(not all((DETECTOR_READY / f'{split}.npz').exists() for split in ('train', 'val', 'test')),
                                reason='hidden-state arrays not built yet (run stages 1 to 6)')


def synthetic(n=400, seed=0):
    """Feature 0 is higher for errors; feature 1 is noise."""
    generator = np.random.default_rng(seed)
    y = (generator.random(n) < 0.3).astype(int)
    X = np.column_stack([y + generator.normal(0, 0.7, n), generator.normal(0, 1, n)]).astype(np.float32)
    return X, y


# ---- Query generation ----

def test_variants_are_balanced_and_correct():
    statements = make_statements('cup', 'on', 'table')
    assert statements == [('orig', 'The cup is on the table.', 1),
                          ('converse', 'The table is under the cup.', 1),
                          ('swap', 'The table is on the cup.', 0),
                          ('opposite', 'The cup is under the table.', 0)]
    assert sum(truth for _, _, truth in statements) == 2


def test_opposites_are_symmetric_and_mapping_is_exact():
    assert set(OPPOSITE) == set(CANONICAL)
    assert all(OPPOSITE[OPPOSITE[p]] == p and OPPOSITE[p] != p for p in CANONICAL)
    assert SURFACE_TO_CANONICAL['over'] == 'above' and SURFACE_TO_CANONICAL['under'] == 'under'
    for excluded in ('near', 'next to', 'by', 'with', 'has'):
        assert excluded not in SURFACE_TO_CANONICAL


def test_split_is_deterministic():
    sampler = importlib.import_module('03_split_sample')
    assert [sampler.split_of(i) for i in range(50)] == [sampler.split_of(i) for i in range(50)]
    shares = pd.Series([sampler.split_of(i) for i in range(20000)]).value_counts(normalize=True)
    assert abs(shares['train'] - 0.70) < 0.02 and abs(shares['val'] - 0.15) < 0.02


# ---- Real data: split disjointness, leakage, NaN/shape ----

@needs_data
def test_splits_share_no_image_or_relationship():
    rows = {s: pd.read_csv(DETECTOR_READY / f'{s}_rows.csv') for s in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not set(rows[a]['image_id']) & set(rows[b]['image_id'])
        assert not set(rows[a]['rel_id']) & set(rows[b]['rel_id'])
    for frame in rows.values():  # every relationship has its four variants, 2 true and 2 false
        grouped = frame.groupby('rel_id')['truth'].agg(['size', 'sum'])
        assert (grouped['size'] == 4).all() and (grouped['sum'] == 2).all()


@needs_data
@pytest.mark.parametrize('split', ['train', 'val'])
def test_features_have_right_shape_and_no_label_leakage(split):
    with np.load(DETECTOR_READY / f'{split}.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    assert set(arrays) == {'hidden_L9', 'hidden_L18', 'hidden_L36', 'p_yes', 'confidence',
                           'vlm_pred', 'truth', 'y_error'}
    n = len(arrays['truth'])
    X, y, info = load_detector_data(split, 'confidence')
    # Confidence inputs are exactly [p_yes, confidence]: no truth, prediction or id column.
    assert X.shape == (n, 2) and X.dtype == np.float32
    assert np.array_equal(X[:, 0], arrays['p_yes']) and np.array_equal(X[:, 1], arrays['confidence'])
    assert np.allclose(X[:, 1], np.maximum(X[:, 0], 1 - X[:, 0]))
    assert np.array_equal(y, (arrays['vlm_pred'] != arrays['truth']).astype(int))
    assert len(info['rows']) == n
    for layer in (9, 18, 36):
        H, target, _ = load_detector_data(split, 'hidden', layer=layer)
        assert H.shape == (n, 2048) and H.dtype == np.float32 and np.isfinite(H).all()
        assert np.array_equal(target, y)
        # No single hidden unit is a copy of a label column.
        for label in (arrays['truth'], arrays['vlm_pred'], arrays['y_error']):
            assert not (H == label[:, None].astype(np.float32)).all(axis=0).any()


@needs_data
def test_loader_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        load_detector_data('train', 'truth')
    with pytest.raises(ValueError):
        load_detector_data('train', 'hidden', layer=12)


# ---- Risk orientation, thresholds, bootstrap ----

def test_risk_orientation_higher_means_more_likely_wrong():
    X, y = synthetic()
    pipeline, _ = fit_logistic(X, y, 1.0, None)
    risk = pipeline.predict_proba(X)[:, 1]
    assert risk[y == 1].mean() > risk[y == 0].mean()
    assert metrics(y, risk)['auroc'] > 0.8
    scaled = pipeline['scaler'].transform(X).astype(np.float32)
    run = fit_mlp(scaled, y, scaled, y, seed=42, weight_decay=1e-4, class_weight='balanced')
    mlp_risk = risk_scores(run['model'], scaled)
    assert mlp_risk[y == 1].mean() > mlp_risk[y == 0].mean()
    assert run['positive_weight'] == pytest.approx((y == 0).sum() / (y == 1).sum())
    # Heuristic: low confidence must give high risk.
    confidence = np.array([0.99, 0.55])
    assert (1 - confidence)[1] > (1 - confidence)[0]


def test_metrics_reject_invalid_risk():
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([0.2, np.nan]))
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([0.2, 1.5]))


def test_coverage_threshold_never_exceeds_target_and_keeps_ties_together():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    risk = np.array([.1, .2, .3, .4, .5, .6, .7, .7, .7, .9])
    chosen = coverage_threshold(y, risk, 0.8)
    assert chosen['coverage'] <= 0.8 and chosen['threshold'] == 0.6  # the three tied 0.7s stay out
    point = operating_point(y, risk, chosen['threshold'])
    assert point['answered'] == 6 and point['retained_errors'] == 2 and point['errors_caught'] == 3
    assert point['answered'] + point['abstained'] == len(y)
    coverage, error_rate = risk_coverage(y, risk)
    assert coverage[-1] == 1 and error_rate[-1] == pytest.approx(y.mean())


def test_bootstrap_is_image_clustered_and_paired():
    X, y = synthetic(300, seed=1)
    images = np.repeat(np.arange(75), 4)
    good, bad = X[:, 0], X[:, 1]
    scale = lambda v: (v - v.min()) / (v.max() - v.min())
    result = paired_bootstrap(y, {'good': scale(good), 'bad': scale(bad)}, images,
                              {'good_minus_bad': ('good', 'bad')}, repetitions=200, seed=3)
    low, high = result['comparisons']['good_minus_bad']['paired_image_bootstrap95']
    assert low > 0 and high > low and result['resampling_unit'] == 'image'
    again = paired_bootstrap(y, {'good': scale(good), 'bad': scale(bad)}, images,
                             {'good_minus_bad': ('good', 'bad')}, repetitions=200, seed=3)
    assert again == result


# ---- Save / reload round trip ----

def test_save_reload_round_trip(tmp_path):
    X, y = synthetic()
    pipeline, _ = fit_logistic(X, y, 0.1, 'balanced')
    joblib.dump(pipeline, tmp_path / 'lr.joblib')
    assert np.array_equal(joblib.load(tmp_path / 'lr.joblib').predict_proba(X), pipeline.predict_proba(X))
    scaled = pipeline['scaler'].transform(X).astype(np.float32)
    run = fit_mlp(scaled, y, scaled, y, seed=43, weight_decay=1e-2, class_weight=None)
    save_mlp(tmp_path / 'mlp.pt', run, X.shape[1])
    assert np.array_equal(risk_scores(load_mlp(tmp_path / 'mlp.pt'), scaled), run['validation_risk'])


def test_mlp_training_is_reproducible():
    X, y = synthetic(200, seed=2)
    first = fit_mlp(X, y, X, y, seed=42, weight_decay=1e-4, class_weight=None)
    second = fit_mlp(X, y, X, y, seed=42, weight_decay=1e-4, class_weight=None)
    assert np.array_equal(first['validation_risk'], second['validation_risk'])


def test_extraction_reuses_only_matching_records(tmp_path, monkeypatch):
    stage = importlib.import_module('05_extract')
    monkeypatch.setattr(stage, 'RECORDS', tmp_path)
    metadata = {'torch': 'test', 'packages': {'transformers': 'test'}}
    settings = extraction_settings(metadata)
    np.savez(tmp_path / 'q1.npz', hidden=np.zeros((3, 2048), dtype=np.float32),
             p_yes=np.float32(.8), statement='The cup is on the table.',
             image_sha='image1', settings=settings)
    args = ('q1', 'The cup is on the table.', 'image1', settings)
    assert stage.load_record(*args) is not None
    assert stage.load_record('q1', 'The table is on the cup.', 'image1', settings) is None
    assert stage.load_record('q1', args[1], 'changed-image', settings) is None
    assert stage.load_record('q1', args[1], 'image1', 'changed-settings') is None
    (tmp_path / 'q1.npz').write_bytes(b'incomplete output')
    assert stage.load_record(*args) is None


def test_pack_train_and_evaluate_can_be_reproduced(tmp_path, monkeypatch):
    """Exercise the real pipeline with synthetic VLM outputs, without downloading a model."""
    pack = importlib.import_module('06_build_npz')
    train = importlib.import_module('07_train_detectors')
    evaluate = importlib.import_module('08_evaluate_test')
    processed, extraction = tmp_path / 'processed', tmp_path / 'extraction'
    ready, outputs = tmp_path / 'ready', tmp_path / 'outputs'
    for folder in (processed, extraction / 'records', outputs):
        folder.mkdir(parents=True)
    metadata = {'torch': 'test', 'packages': {'transformers': 'test'}}
    settings = extraction_settings(metadata)
    queries, images = [], []
    generator = np.random.default_rng(42)
    for image_id, split in enumerate(('train', 'val', 'test')):
        images.append({'image_id': image_id, 'sha256': f'image{image_id}'})
        for index, (variant, statement, truth) in enumerate(make_statements('cup', 'on', 'table')):
            query_id = f'{split}_{variant}'
            queries.append({'query_id': query_id, 'image_id': image_id, 'coco_id': -1,
                            'rel_id': image_id, 'variant': variant, 'canonical_predicate': 'on',
                            'vg_predicate': 'on', 'subject': 'cup', 'object': 'table',
                            'statement': statement, 'truth': truth, 'split': split})
            np.savez(extraction / 'records' / f'{query_id}.npz',
                     hidden=generator.normal(size=(3, 2048)).astype(np.float32),
                     p_yes=np.float32([.8, .3, .7, .2][index]), statement=statement,
                     image_sha=f'image{image_id}', settings=settings)
    pd.DataFrame(queries).to_csv(processed / 'queries.csv', index=False)
    pd.DataFrame(images).to_csv(processed / 'image_manifest.csv', index=False)
    write_json(processed / 'sample_summary.json', {})
    write_json(outputs / 'extraction_summary.json', {'vlm': metadata})
    for name, value in {'PROCESSED': processed, 'EXTRACTION': extraction,
                        'DETECTOR_READY': ready, 'OUTPUTS': outputs}.items():
        monkeypatch.setattr(pack, name, value)
    pack.main()
    for split in ('train', 'val', 'test'):
        X, y, info = load_detector_data(split, 'confidence', folder=ready)
        assert X.shape == (4, 2) and y.tolist() == [0, 1, 1, 0]
        assert set(info['image_ids']) == {('train', 'val', 'test').index(split)}

    out = outputs / 'detectors'
    for name, value in {'ROOT': tmp_path, 'OUT': out, 'DETECTOR_READY': ready}.items():
        monkeypatch.setattr(train, name, value)
    small_lr = {**train.LR_CONFIG, 'C_grid': [.1], 'class_weight_grid': [None]}
    small_mlp = {**train.MLP_CONFIG, 'max_epochs': 3, 'weight_decay_grid': [1e-4],
                 'class_weight_grid': [None]}
    monkeypatch.setattr(train, 'LR_CONFIG', small_lr)
    monkeypatch.setattr(train, 'MLP_CONFIG', small_mlp)
    monkeypatch.setattr(train, 'fit_mlp', lambda *a: fit_mlp(*a, config=small_mlp))
    loader = lambda *a, **kw: load_detector_data(*a, folder=ready, **kw)
    monkeypatch.setattr(train, 'load_detector_data', loader)
    monkeypatch.setattr(sys, 'argv', ['07_train_detectors.py'])
    train.main()
    # This historical marker no longer prevents reproduction.
    write_json(outputs / 'test_evaluation_started.json', {'old': 'protocol'})
    with pytest.raises(SystemExit):
        train.main()  # existing fits require an explicit overwrite
    monkeypatch.setattr(sys, 'argv', ['07_train_detectors.py', '--overwrite'])
    train.main()

    for name, value in {'ROOT': tmp_path, 'OUTPUTS': outputs, 'DETECTOR_READY': ready,
                        'FROZEN': out / 'frozen.json', 'RESULTS': outputs / 'results.json'}.items():
        monkeypatch.setattr(evaluate, name, value)
    monkeypatch.setattr(evaluate, 'load_detector_data', loader)
    monkeypatch.setattr(evaluate, 'BOOTSTRAP_REPETITIONS', 20)
    monkeypatch.setattr(evaluate, 'paired_bootstrap',
                        lambda *a: paired_bootstrap(*a, repetitions=20))
    monkeypatch.setattr(sys, 'argv', ['08_evaluate_test.py'])
    evaluate.main()
    first = read_json(evaluate.RESULTS)
    assert first['cohorts']['all']['rows'] == 4
    assert len(first['cohorts']['all']['models']) == 9
    saved = evaluate.RESULTS.read_bytes()
    evaluate.main()  # cached result, no rewrite
    assert evaluate.RESULTS.read_bytes() == saved
    monkeypatch.setattr(sys, 'argv', ['08_evaluate_test.py', '--rerun'])
    evaluate.main()
    assert read_json(evaluate.RESULTS)['cohorts'] == first['cohorts']


# ---- Exploratory aggregation (stage 10) ----

def test_aggregation_answers_four_variants_together():
    stage = importlib.import_module('10_aggregate')
    variant = np.array(['orig', 'converse', 'swap', 'opposite'] * 2)
    rel_id = np.repeat([7, 9], 4)
    # Relationship 7: one wrong single answer (converse) is outvoted. Relationship 9: support below 0.5.
    p_yes = np.array([0.9, 0.4, 0.1, 0.2, 0.3, 0.2, 0.6, 0.9])
    answer, support, margin = stage.aggregate(p_yes, variant, rel_id)
    assert answer.tolist() == [1, 1, 0, 0, 0, 0, 1, 1]
    assert support[:4] == pytest.approx(0.75) and support[4:] == pytest.approx(0.25)
    assert margin == pytest.approx(0.25)
    with pytest.raises(ValueError):
        stage.aggregate(p_yes[:3], variant[:3], rel_id[:3])
    mask = stage.answered_mask(np.array([.1, .2, .2, .2, .9]), 0.6)
    assert mask.tolist() == [True, False, False, False, False]  # the tied 0.2s stay together


def test_cascade_escalates_only_the_riskiest_queries():
    stage = importlib.import_module('10_aggregate')
    single, aggregated = np.array([1, 0, 1, 0]), np.array([1, 1, 0, 1])
    margin, risk = np.array([.4, .3, .2, .1]), np.array([.1, .9, .2, .8])
    correct, escalated, answered = stage.cascade(single, aggregated, margin, risk, 0.5)
    assert escalated.tolist() == [False, True, False, True] and answered.all()
    assert correct.tolist() == [1, 1, 1, 1]  # the two single-query errors were both verified
    correct, escalated, answered = stage.cascade(single, aggregated, margin, risk, 0.0)
    assert not escalated.any() and correct.tolist() == single.tolist()
    # With abstention, only escalated queries can be withheld, smallest margin first.
    _, _, answered = stage.cascade(single, aggregated, margin, risk, 0.5, coverage=0.75)
    assert answered.tolist() == [True, True, True, False]


def test_detector_informed_aggregation_rules():
    stage = importlib.import_module('10_aggregate')
    variant, rel_id = np.array(['orig', 'converse', 'swap', 'opposite']), np.array([5, 5, 5, 5])
    # Three calls weakly deny the fact, one strongly affirms it; the detector distrusts the three.
    p_yes, error = np.array([0.95, 0.45, 0.6, 0.6]), np.array([0.05, 0.9, 0.9, 0.9])
    assert stage.aggregate(p_yes, variant, rel_id)[0].tolist() == [1, 1, 0, 0]
    rules = stage.aggregate_with_detector(p_yes, error, variant, rel_id)
    assert set(rules) == {'trust_weighted', 'detector_votes', 'combined'}
    assert rules['detector_votes'][1] == pytest.approx((0.95 + 0.9 * 3) / 4)
    assert rules['trust_weighted'][1] == pytest.approx((.95 * .95 + .1 * (.45 + .4 + .4)) / (.95 + .3))
    # A detector that trusts every call equally leaves the confidence rule unchanged.
    same = stage.aggregate_with_detector(p_yes, np.full(4, 0.2), variant, rel_id)['trust_weighted']
    assert same[1] == pytest.approx(stage.aggregate(p_yes, variant, rel_id)[1])
    # Undoing balanced weighting maps 0.5 back to the training error rate and keeps the order.
    fixed = stage.undo_balanced_weighting(np.array([0.2, 0.5, 0.8]), 0.3)
    assert fixed[1] == pytest.approx(0.3) and (np.diff(fixed) > 0).all()

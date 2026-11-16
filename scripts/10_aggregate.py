"""Stage 10 (exploratory, validation only): three ways to use the signals.

Compares, on the validation split only:
  1. abstention on a single query (risk = 1 - confidence, or a frozen detector's risk);
  2. aggregation of the four query variants of one relationship;
  3. aggregation followed by abstention.

It also evaluates a cascade ("detect, then verify"): every question gets one VLM
call; only the questions a router marks as risky get the three extra calls and
the aggregated answer. Optionally the least decided escalated questions are then
withheld.

Nothing is fitted or selected here and the test split is never read. The VLM
outputs and the frozen detectors' validation risks are reused as saved.

Aggregation: every variant is evidence about one underlying fact F (the VG
relationship). `orig` and `converse` state F; `swap` and `opposite` state its
negation. Support for F is p_yes for the first two and 1 - p_yes for the others.
The mean support over the four queries decides all four answers together: Yes to
`orig`/`converse` and No to `swap`/`opposite` when the mean is at least 0.5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import CANONICAL, DETECTOR_READY, OUTPUTS, read_json, write_json
from detector_lib import BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED
from load_detector_data import load_detector_data

FROZEN = OUTPUTS / 'detectors/frozen.json'
COVERAGES = [1.0, 0.8, 0.6]
ESCALATION_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CASCADE_ABSTAIN_COVERAGE = 0.8
STATES_FACT = {'orig': True, 'converse': True, 'swap': False, 'opposite': False}


def decide(support, states_fact, rel_id, weights=None):
    """Answer the four variants of each relationship from the (weighted) mean support for its fact."""
    weights = np.ones(len(support)) if weights is None else weights
    _, inverse, counts = np.unique(rel_id, return_inverse=True, return_counts=True)
    if not (counts == 4).all():
        raise ValueError('Every relationship needs its four variants')
    mean_support = (np.bincount(inverse, weights=weights * support) / np.bincount(inverse, weights=weights))[inverse]
    fact_true = mean_support >= 0.5
    answer = np.where(states_fact, fact_true, ~fact_true).astype(int)
    return answer, mean_support, np.abs(mean_support - 0.5)


def aggregate(p_yes, variant, rel_id):
    """Per-query (aggregated answer, mean support for the fact, margin from 0.5)."""
    states_fact = np.array([STATES_FACT[v] for v in variant])
    return decide(np.where(states_fact, p_yes, 1 - p_yes).astype(float), states_fact, rel_id)


def undo_balanced_weighting(risk, error_rate):
    """Turn a class-balanced detector score back into an error probability.

    Balanced training multiplies the odds of "wrong" by (1 - rate) / rate, where rate is
    the training error rate; dividing the odds by the same factor removes it. Not a fit.
    """
    risk = np.clip(risk, 1e-6, 1 - 1e-6)
    odds = risk / (1 - risk) * error_rate / (1 - error_rate)
    return odds / (1 + odds)


def aggregate_with_detector(p_yes, error_probability, variant, rel_id):
    """Aggregation rules that also use the detector's error probability of each of the four calls.

    trust_weighted: the confidence support of each call, weighted by 1 - P(wrong).
    detector_votes: each call votes with its Yes/No answer; the vote counts as 1 - P(wrong)
        for the side it chose and P(wrong) for the other side. p_yes is used only for the answer.
    combined: the mean of the confidence support and the detector vote of each call.
    """
    states_fact = np.array([STATES_FACT[v] for v in variant])
    support = np.where(states_fact, p_yes, 1 - p_yes).astype(float)
    votes_for_fact = (p_yes >= 0.5) == states_fact
    belief = np.where(votes_for_fact, 1 - error_probability, error_probability)
    return {'trust_weighted': decide(support, states_fact, rel_id, weights=1 - error_probability),
            'detector_votes': decide(belief, states_fact, rel_id),
            'combined': decide((support + belief) / 2, states_fact, rel_id)}


def group_mean(values, rel_id):
    _, inverse = np.unique(rel_id, return_inverse=True)
    return (np.bincount(inverse, weights=values) / np.bincount(inverse))[inverse]


def answered_mask(risk, target):
    """Answer the lowest-risk queries; tied risks stay together; coverage never exceeds target."""
    if target >= 1:
        return np.ones(len(risk), dtype=bool)
    best = np.zeros(len(risk), dtype=bool)
    for value in np.unique(risk):
        mask = risk <= value
        if mask.mean() > target:
            break
        best = mask
    return best


def cascade(single_correct, aggregate_correct, margin, risk, fraction, coverage=1.0):
    """One cascade setting: escalate the riskiest queries, optionally withhold the least decided.

    Escalated queries (highest router risk, ties together, at most `fraction` of all)
    cost four VLM calls and take the aggregated answer; the rest keep the single answer.
    With coverage < 1, escalated queries with the smallest aggregate margin are withheld.
    """
    escalated = answered_mask(-risk, fraction) if fraction > 0 else np.zeros(len(risk), dtype=bool)
    correct = np.where(escalated, aggregate_correct, single_correct)
    answered = np.ones(len(risk), dtype=bool)
    if coverage < 1:
        # Non-escalated queries are always answered; escalated ones in order of decreasing margin.
        answered = answered_mask(np.where(escalated, -margin, -np.inf), coverage)
    return correct, escalated, answered


def bootstrap_gain(gain, image_ids, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """95% interval of a mean per-query gain, resampling images."""
    generator = np.random.default_rng(seed)
    images, inverse = np.unique(image_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=gain)
    sizes = np.bincount(inverse)
    draws = generator.integers(0, len(images), size=(repetitions, len(images)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    frozen = read_json(FROZEN)
    hidden_mlps = {name: entry for name, entry in frozen['models'].items()
                   if entry['kind'] == 'mlp' and entry['feature_set'] == 'hidden'}
    # The hidden-state MLP with the highest mean validation AUROC (already chosen in stage 7).
    detector = max(hidden_mlps, key=lambda name: hidden_mlps[name]['selection_metrics']['mean_validation_auroc'])
    with np.load(OUTPUTS / 'detectors/validation_predictions.npz') as saved:
        detector_risk = saved[detector].astype(float)

    X, y_error, info = load_detector_data('val', 'confidence')
    rows = info['rows']
    p_yes, confidence = X[:, 0].astype(float), X[:, 1].astype(float)
    truth = rows['truth'].to_numpy()
    rel_id, image_id = rows['rel_id'].to_numpy(), rows['image_id'].to_numpy()
    single_correct = (y_error == 0).astype(int)
    answer, _, margin = aggregate(p_yes, rows['variant'].to_numpy(), rel_id)
    aggregate_correct = (answer == truth).astype(int)

    # name: (label, correctness of the answers given, risk used to decide what to withhold)
    methods = {
        'single_no_abstention': ('Single query, no abstention', single_correct, None),
        'abstain_confidence': ('1. Abstain on 1 - confidence', single_correct, 1 - confidence),
        'abstain_detector': (f'1. Abstain on `{detector}` risk', single_correct, detector_risk),
        'aggregate': ('2. Aggregate four variants', aggregate_correct, None),
        'aggregate_abstain_margin': ('3. Aggregate, abstain on small margin', aggregate_correct, -margin),
        'aggregate_abstain_detector': (f'3. Aggregate, abstain on mean `{detector}` risk', aggregate_correct,
                                       group_mean(detector_risk, rel_id)),
    }
    results = {}
    for name, (label, correct, risk) in methods.items():
        points = []
        for target in COVERAGES if risk is not None else [1.0]:
            mask = answered_mask(risk, target) if risk is not None else np.ones(len(correct), dtype=bool)
            points.append({'target_coverage': target, 'answered': int(mask.sum()),
                           'coverage': float(mask.mean()), 'accuracy_on_answered': float(correct[mask].mean())})
        results[name] = {'label': label, 'vlm_calls_per_question': 4 if 'aggregate' in name else 1, 'points': points}

    # Cascade: one call for every query, three more only where the router sees risk.
    with np.load(OUTPUTS / 'detectors/validation_predictions.npz') as saved:
        lr_detector = detector.replace('mlp_', 'lr_')
        routers = {'confidence': ('1 - confidence', 1 - confidence),
                   lr_detector: (f'`{lr_detector}` risk', saved[lr_detector].astype(float)),
                   detector: (f'`{detector}` risk', detector_risk)}
    single_accuracy, aggregate_accuracy = float(single_correct.mean()), float(aggregate_correct.mean())
    cascade_results = {}
    for name, (label, risk) in routers.items():
        points = []
        for fraction in ESCALATION_FRACTIONS:
            correct, escalated, _ = cascade(single_correct, aggregate_correct, margin, risk, fraction)
            share = float(escalated.mean())
            point = {'target_escalation': fraction, 'escalated': share, 'mean_vlm_calls': 1 + 3 * share,
                     'accuracy': float(correct.mean()),
                     # What routing at random would give for the same number of calls.
                     'random_router_accuracy': single_accuracy + share * (aggregate_accuracy - single_accuracy),
                     'single_errors_escalated': float(escalated[single_correct == 0].mean())}
            point['gain_over_random_image_bootstrap95'] = bootstrap_gain(
                correct - (single_correct + share * (aggregate_correct - single_correct)), image_id)
            if 1 - share <= CASCADE_ABSTAIN_COVERAGE:
                correct, _, answered = cascade(single_correct, aggregate_correct, margin, risk, fraction,
                                               CASCADE_ABSTAIN_COVERAGE)
                point['with_abstention'] = {'coverage': float(answered.mean()),
                                            'accuracy_on_answered': float(correct[answered].mean())}
            points.append(point)
        cascade_results[name] = {'label': label, 'points': points}

    # Hallucination in the strict sense: "Yes" to a false statement (swap, opposite).
    false_statement, vlm_yes = truth == 0, (p_yes >= 0.5)
    error_types = {
        'false_statements': int(false_statement.sum()), 'true_statements': int((~false_statement).sum()),
        'single_false_yes_rate': float(vlm_yes[false_statement].mean()),
        'single_false_no_rate': float((~vlm_yes)[~false_statement].mean()),
        'aggregate_false_yes_rate': float((answer == 1)[false_statement].mean()),
        'aggregate_false_no_rate': float((answer == 0)[~false_statement].mean()),
        'note': 'Aggregation answers the four variants together, so its two rates are equal by construction.',
    }

    # Aggregation rules that also use the detector's error probability of every call.
    balanced = hidden_mlps[detector]['selected_class_weight'] == 'balanced'
    train_error_rate = read_json(DETECTOR_READY / 'metadata.json')['counts']['train']['error_rate']
    error_probability = undo_balanced_weighting(detector_risk, train_error_rate) if balanced else detector_risk
    rules = {'confidence': ('Confidence only (mean support)', answer, margin)}
    labels = {'trust_weighted': 'Confidence, weighted by detector trust',
              'detector_votes': 'Detector votes only', 'combined': 'Confidence and detector votes'}
    for name, (rule_answer, _, rule_margin) in aggregate_with_detector(
            p_yes, error_probability, rows['variant'].to_numpy(), rel_id).items():
        rules[name] = (labels[name], rule_answer, rule_margin)
    rule_results = {}
    for name, (label, rule_answer, rule_margin) in rules.items():
        correct = (rule_answer == truth).astype(int)
        entry = {'label': label, 'accuracy': float(correct.mean()),
                 'minus_confidence_rule': float((correct - aggregate_correct).mean()),
                 'minus_confidence_rule_image_bootstrap95': bootstrap_gain(correct - aggregate_correct, image_id),
                 'false_yes_rate': float((rule_answer == 1)[false_statement].mean()),
                 'abstention': [], 'cascade': []}
        for target in COVERAGES[1:]:
            mask = answered_mask(-rule_margin, target)
            entry['abstention'].append({'target_coverage': target, 'coverage': float(mask.mean()),
                                        'accuracy_on_answered': float(correct[mask].mean())})
        for fraction in ESCALATION_FRACTIONS:  # routed by the detector, as in the cascade above
            mixed, escalated, _ = cascade(single_correct, correct, rule_margin, detector_risk, fraction)
            entry['cascade'].append({'target_escalation': fraction, 'mean_vlm_calls': 1 + 3 * float(escalated.mean()),
                                     'accuracy': float(mixed.mean())})
        rule_results[name] = entry

    gain = aggregate_correct - single_correct
    by_predicate = {}
    for predicate in CANONICAL:
        mask = (rows['canonical_predicate'] == predicate).to_numpy()
        by_predicate[predicate] = {'rows': int(mask.sum()), 'single_accuracy': float(single_correct[mask].mean()),
                                   'aggregate_accuracy': float(aggregate_correct[mask].mean())}
    votes = np.bincount(np.unique(rel_id, return_inverse=True)[1], weights=single_correct)
    output = {
        'status': 'Exploratory extension added after the test evaluation. Validation split only; test never read.',
        'split': 'val', 'rows': int(len(truth)), 'relationships': int(len(votes)),
        'images': int(len(np.unique(image_id))),
        'detector_used': detector,
        'detector_note': 'This detector was selected on validation, so its validation rows are slightly optimistic.',
        'coverage_rule': 'Lowest-risk queries answered, ties together, coverage at most the target; '
                         'matched on validation itself, not a frozen threshold.',
        'methods': results,
        'cascade': {'rule': 'Queries with the highest router risk (ties together, at most the target share) '
                            'get three more VLM calls and the aggregated answer; the rest keep the single answer.',
                    'abstain_coverage': CASCADE_ABSTAIN_COVERAGE, 'routers': cascade_results},
        'error_types': error_types,
        'aggregation_rules': {
            'detector': detector,
            'error_probability': ('Detector risk with class-balanced weighting undone using the training error '
                                  f'rate ({train_error_rate:.3f}).') if balanced else 'Detector risk as saved.',
            'note': 'Four fixed rules, none tuned. Picking the best of them from this table is a selection on '
                    'validation, so its number is optimistic.',
            'rules': rule_results},
        'aggregate_minus_single': {'accuracy_gain': float(gain.mean()),
                                   'image_bootstrap95': bootstrap_gain(gain, image_id),
                                   'bootstrap_repetitions': BOOTSTRAP_REPETITIONS},
        'consistency': {'all_four_correct': float((votes == 4).mean()), 'three_correct': float((votes == 3).mean()),
                        'two_correct': float((votes == 2).mean()), 'one_correct': float((votes == 1).mean()),
                        'none_correct': float((votes == 0).mean())},
        'by_predicate': by_predicate,
    }
    write_json(OUTPUTS / 'aggregation.json', output)
    for name, entry in results.items():
        print(entry['label'] + ': ' + '; '.join(f"{p['coverage']:.3f} -> {p['accuracy_on_answered']:.3f}"
                                                for p in entry['points']))
    for name, entry in cascade_results.items():
        print(f"cascade via {name}: " + '; '.join(f"{p['mean_vlm_calls']:.1f} calls -> {p['accuracy']:.3f}"
                                                  for p in entry['points']))
    for name, entry in rule_results.items():
        print(f"rule {name}: acc {entry['accuracy']:.3f} ({entry['minus_confidence_rule']:+.3f} "
              f"{[round(v, 3) for v in entry['minus_confidence_rule_image_bootstrap95']]}); abstain "
              + '; '.join(f"{p['coverage']:.3f} -> {p['accuracy_on_answered']:.3f}" for p in entry['abstention'])
              + ' | cascade ' + ', '.join(f"{p['accuracy']:.3f}" for p in entry['cascade']))
    print(error_types)
    print(f"Aggregate minus single: {gain.mean():+.3f} {output['aggregate_minus_single']['image_bootstrap95']}")


if __name__ == '__main__':
    main()

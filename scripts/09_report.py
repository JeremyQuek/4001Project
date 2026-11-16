"""Stage 9: regenerate outputs/REPORT.md and the figure from saved artifacts.

Reads only files written by earlier stages (results.json, frozen.json,
test_predictions.csv, candidate tables); it never refits or rescores anything,
so it can be re-run freely after the single test evaluation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import CANONICAL, DETECTOR_READY, OUTPUTS, VARIANTS, VLM, read_json
from detector_lib import risk_coverage

INPUT_LABELS = {'all_confidence': 'Confidence [p_yes, confidence]', 'all_hidden9': 'Hidden layer 9',
                'all_hidden18': 'Hidden layer 18', 'all_hidden36': 'Hidden layer 36'}
# Colour = architecture throughout the figure; text stays in ink colours.
BLUE, ORANGE, GREY, INK, MUTED, SURFACE = '#2a78d6', '#eb6834', '#8a8983', '#0b0b0b', '#52514e', '#fcfcfb'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(lines)


def interval(pair):
    return f'[{pair[0]:.3f}, {pair[1]:.3f}]'


def signed_interval(pair):
    return f'[{pair[0]:+.3f}, {pair[1]:+.3f}]'


def best_hidden(models, kind):
    """Hidden-state model with the highest VALIDATION AUROC (never chosen on test)."""
    def validation_auroc(name):
        entry = models[name]
        return (entry['validation_metrics']['auroc'] if kind == 'lr'
                else entry['selection_metrics']['mean_validation_auroc'])
    return max((f'{kind}_all_hidden{layer}' for layer in VLM['hidden_layers']), key=validation_auroc)


def make_figure(results, frozen, predictions, path):
    group = results['cohorts']['all']
    models = group['models']
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)

    # Panel 1: test AUROC with image-bootstrap 95% intervals, LR and MLP per input set.
    names = list(INPUT_LABELS)
    for offset, kind, colour, label in ((0.14, 'lr', BLUE, 'Logistic regression'),
                                        (-0.14, 'mlp', ORANGE, 'MLP (seed 42)')):
        for position, name in enumerate(names):
            entry = models[f'{kind}_{name}']
            low, high = entry['auroc_image_bootstrap95']
            y = len(names) - 1 - position + offset
            left.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle='round')
            left.plot(entry['auroc'], y, 'o', color=colour, markersize=7, markeredgecolor=SURFACE,
                      markeredgewidth=1.5, label=label if position == 0 else None)
            left.annotate(f'{entry["auroc"]:.3f}', (high, y), xytext=(5, 0), textcoords='offset points',
                          va='center', fontsize=8, color=MUTED)
    heuristic = models['heuristic_confidence']
    left.axvline(heuristic['auroc'], color=GREY, linestyle=':', linewidth=1.5,
                 label=f'1 - confidence, nothing fitted ({heuristic["auroc"]:.3f})')
    left.axvline(0.5, color=MUTED, linewidth=0.8)
    left.annotate('chance', (0.5, len(names) - 0.45), xytext=(3, 0), textcoords='offset points',
                  fontsize=8, color=MUTED)
    left.set_yticks(range(len(names)), [INPUT_LABELS[name] for name in reversed(names)])
    left.set_ylim(-0.6, len(names) - 0.3)
    left.set_xlabel('Test AUROC for detecting a wrong VLM answer (95% image-bootstrap interval)')
    left.set_title('Detector AUROC by input', loc='left', fontsize=10, fontweight='bold')
    left.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=8)

    # Panel 2: risk-coverage curves; hidden-layer models are the validation-selected layers.
    y = predictions['y_error'].to_numpy()
    lr_hidden, mlp_hidden = best_hidden(frozen['models'], 'lr'), best_hidden(frozen['models'], 'mlp')
    curves = [(lr_hidden, BLUE, '-', f'LR, hidden layer {frozen["models"][lr_hidden]["layer"]}'),
              (mlp_hidden, ORANGE, '-', f'MLP, hidden layer {frozen["models"][mlp_hidden]["layer"]}'),
              ('lr_all_confidence', BLUE, '--', 'LR, confidence'),
              ('heuristic_confidence', GREY, ':', '1 - confidence, nothing fitted')]
    for name, colour, style, label in curves:
        coverage, error_rate = risk_coverage(y, predictions[name].to_numpy())
        keep = coverage >= 0.05  # the first few answered queries are too noisy to read
        right.plot(coverage[keep], error_rate[keep], color=colour, linestyle=style, linewidth=2, label=label)
        point = models[name]['operating_point']
        right.plot(point['coverage'], 1 - point['retained_vlm_accuracy'], 'o', color=colour, markersize=7,
                   markeredgecolor=SURFACE, markeredgewidth=1.5)
    right.axhline(group['error_prevalence'], color=MUTED, linewidth=0.8)
    right.annotate(f'no abstention: {group["error_prevalence"]:.3f}', (0.05, group['error_prevalence']),
                   xytext=(0, 4), textcoords='offset points', fontsize=8, color=MUTED)
    right.set_xlim(0, 1.01)
    right.set_ylim(bottom=0)
    right.set_xlabel('Coverage (share of test questions answered)')
    right.set_ylabel('VLM error rate on answered questions')
    right.set_title('Risk-coverage (dots: frozen threshold)', loc='left', fontsize=10,
                    fontweight='bold')
    right.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8)

    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x' if axis is left else 'y', color='#e4e3df', linewidth=0.6)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return lr_hidden, mlp_hidden


def make_cascade_figure(data, path):
    """Accuracy against mean VLM calls for each cascade router (validation, exploratory)."""
    routers = data['cascade']['routers']
    figure, axis = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    styles = {'confidence': (GREY, ':')}
    for name, entry in routers.items():
        colour, style = styles.get(name, (BLUE if name.startswith('lr_') else ORANGE, '-'))
        points = entry['points']
        axis.plot([p['mean_vlm_calls'] for p in points], [p['accuracy'] for p in points], color=colour,
                  linestyle=style, linewidth=2, marker='o', markersize=4, label='Router: ' + entry['label'].replace('`', ''))
    ends = next(iter(routers.values()))['points']
    axis.plot([1, 4], [ends[0]['accuracy'], ends[-1]['accuracy']], color=MUTED, linestyle='--', linewidth=1,
              label='Router: random')
    axis.annotate(f"single query: {ends[0]['accuracy']:.3f}", (1, ends[0]['accuracy']), xytext=(8, -4),
                  textcoords='offset points', fontsize=8, color=MUTED, va='top')
    axis.annotate(f"always four: {ends[-1]['accuracy']:.3f}", (4, ends[-1]['accuracy']), xytext=(0, 8),
                  textcoords='offset points', fontsize=8, color=MUTED, ha='right')
    axis.set_xlabel('Mean VLM calls per question')
    axis.set_ylabel('Accuracy (all validation questions answered)')
    axis.set_title('Cascade: verify only the risky answers', loc='left', fontsize=10, fontweight='bold')
    axis.set_xlim(0.9, 4.1)
    axis.set_facecolor(SURFACE)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', color='#e4e3df', linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def cascade_section(data):
    routers = data['cascade']['routers']
    names = list(routers)
    make_cascade_figure(data, OUTPUTS / 'figure_cascade.png')
    rows, withheld = [], []
    for index, reference in enumerate(routers[names[-1]]['points']):
        points = [routers[name]['points'][index] for name in names]
        rows.append([f"{reference['target_escalation']:.1f}", f"{reference['mean_vlm_calls']:.2f}",
                     f"{reference['random_router_accuracy']:.3f}"] + [f"{p['accuracy']:.3f}" for p in points]
                    + [signed_interval(reference['gain_over_random_image_bootstrap95']),
                       f"{reference['single_errors_escalated']:.3f}"])
        if 'with_abstention' in reference:
            withheld.append([f"{reference['target_escalation']:.1f}", f"{reference['mean_vlm_calls']:.2f}"]
                            + [f"{p['with_abstention']['accuracy_on_answered']:.3f} at "
                               f"{p['with_abstention']['coverage']:.3f}" if 'with_abstention' in p else 'n/a'
                               for p in points])
    labels = [routers[name]['label'] for name in names]
    errors = data['error_types']
    return f"""
### Cascade: one call first, three more only when the router sees risk

{data['cascade']['rule']} Mean calls and the random-router column are for
the `{names[-1]}` router's escalated share; shares differ slightly between routers because tied
risks are kept together. The last two columns are for the `{names[-1]}` router.

{table(['Target share escalated', 'Mean VLM calls', 'Random router'] + ['Router: ' + label for label in labels]
       + ['Gain over random, 95% interval', 'Single-query errors escalated'], rows)}

![Cascade accuracy against VLM calls](figure_cascade.png)

Adding a last step that withholds the least decided escalated questions (smallest aggregate
margin) until coverage is at most {data['cascade']['abstain_coverage']:.0%} gives (accuracy on answered at coverage):

{table(['Target share escalated', 'Mean VLM calls'] + ['Router: ' + label for label in labels], withheld)}

### Error types (validation)

A hallucination in the strict sense is a "Yes" to a false statement.

{table(['', 'Yes to a false statement', 'No to a true statement'],
       [['Single query', f"{errors['single_false_yes_rate']:.3f}", f"{errors['single_false_no_rate']:.3f}"],
        ['Aggregated', f"{errors['aggregate_false_yes_rate']:.3f}", f"{errors['aggregate_false_no_rate']:.3f}"]])}

{errors['note']}
"""


def rules_section(data):
    """Aggregation rules that also use the detector's error probability of each call."""
    block = data['aggregation_rules']
    rows = []
    for entry in block['rules'].values():
        by_calls = {round(p['target_escalation'], 1): p for p in entry['cascade']}
        rows.append([entry['label'], f"{entry['accuracy']:.3f}",
                     f"{entry['minus_confidence_rule']:+.3f} {signed_interval(entry['minus_confidence_rule_image_bootstrap95'])}",
                     f"{entry['false_yes_rate']:.3f}"]
                    + [f"{p['accuracy_on_answered']:.3f} at {p['coverage']:.3f}" for p in entry['abstention']]
                    + [f"{by_calls[share]['accuracy']:.3f}" for share in (0.3, 0.5, 0.7)])
    return f"""
### Aggregation rules that also use the detector on every call

Detector: `{block['detector']}`. {block['error_probability']}
Trust weighting scales each call's confidence support by 1 - P(wrong). Detector votes
replace the confidence with 1 - P(wrong) for the side the call chose. The last three
columns are the cascade (routed by the same detector) at 30%, 50% and 70% escalated.

{table(['Aggregation rule', 'Accuracy', 'Change against confidence only (95% interval)', 'Yes to a false statement',
        'Abstain to 80%', 'Abstain to 60%', 'Cascade 1.9 calls', 'Cascade 2.5 calls', 'Cascade 3.1 calls'], rows)}

{block['note']}
"""


def aggregation_section():
    """Exploratory section from outputs/aggregation.json (stage 10); empty if that stage was not run."""
    path = OUTPUTS / 'aggregation.json'
    if not path.exists():
        return ''
    data = read_json(path)
    rows = [[entry['label'], entry['vlm_calls_per_question'], f"{point['coverage']:.3f}", point['answered'],
             f"{point['accuracy_on_answered']:.3f}"]
            for entry in data['methods'].values() for point in entry['points']]
    predicates = [[name, entry['rows'], f"{entry['single_accuracy']:.3f}", f"{entry['aggregate_accuracy']:.3f}",
                   f"{entry['aggregate_accuracy'] - entry['single_accuracy']:+.3f}"]
                  for name, entry in data['by_predicate'].items()]
    gain, consistency = data['aggregate_minus_single'], data['consistency']
    return f"""
## Exploratory extension: abstention, aggregation, or both (validation only)

**Added after the test evaluation; not a test result.** Computed by `scripts/10_aggregate.py` on the
validation split ({data['rows']} queries, {data['relationships']} relationships, {data['images']} images).
Nothing was fitted; the test split was not read.

Aggregation averages the VLM's support for one underlying fact over its four query variants
(p_yes for `orig` and `converse`, 1 - p_yes for `swap` and `opposite`) and answers all four
from that average. It needs four VLM calls per question.

{table(['Method', 'VLM calls', 'Coverage', 'Answered', 'Accuracy on answered'], rows)}

- Coverage rule: {data['coverage_rule']}
- Aggregation changes accuracy by {gain['accuracy_gain']:+.3f} at full coverage
  (95% image-bootstrap interval {signed_interval(gain['image_bootstrap95'])}).
- Consistency of the single-query VLM: all four variants correct for
  {consistency['all_four_correct']:.3f} of relationships, three for {consistency['three_correct']:.3f},
  two for {consistency['two_correct']:.3f}, one for {consistency['one_correct']:.3f},
  none for {consistency['none_correct']:.3f}.
- {data['detector_note']}
- The variants were generated from an annotated true relationship, so each group of four is
  balanced by construction. Accuracy on free-form questions was not measured.

{table(['Predicate', 'Rows', 'Single-query accuracy', 'Aggregated accuracy', 'Change'], predicates)}
{cascade_section(data) if 'cascade' in data else ''}{rules_section(data) if 'aggregation_rules' in data else ''}"""


def main():
    results = read_json(OUTPUTS / 'results.json')
    frozen = read_json(OUTPUTS / 'detectors/frozen.json')
    data = read_json(DETECTOR_READY / 'metadata.json')
    predictions = pd.read_csv(OUTPUTS / 'test_predictions.csv')
    group = results['cohorts']['all']
    models, trained = group['models'], frozen['models']
    lr_hidden, mlp_hidden = make_figure(results, frozen, predictions, OUTPUTS / 'figure_test_results.png')

    def validation_auroc(name):
        entry = trained[name]
        if entry['kind'] == 'mlp':
            return entry['seeds'][str(frozen['mlp_config']['primary_seed'])]['validation_metrics']['auroc']
        return entry['validation_metrics']['auroc']

    def chosen(name):
        entry = trained[name]
        if entry['kind'] == 'lr':
            return f'C={entry["selected_C"]}, weights={entry["selected_class_weight"]}'
        if entry['kind'] == 'mlp':
            return f'decay={entry["selected_weight_decay"]}, weights={entry["selected_class_weight"]}'
        return 'nothing fitted'

    main_rows, abstain_rows = [], []
    for name, entry in models.items():
        point = entry['operating_point']
        main_rows.append([f'`{name}`', chosen(name), f'{validation_auroc(name):.3f}', f'{entry["auroc"]:.3f}',
                          interval(entry['auroc_image_bootstrap95']), f'{entry["average_precision"]:.3f}',
                          f'{entry["brier_score"]:.3f}'])
        abstain_rows.append([f'`{name}`', f'{point["validation_threshold"]:.4f}', f'{point["coverage"]:.3f}',
                             point['answered'], point['retained_errors'],
                             f'{point["retained_vlm_accuracy"]:.3f}', point['errors_caught'],
                             f'{point["error_recall"]:.3f}', f'{point["abstention_precision"]:.3f}'])

    comparison_rows = []
    for label, entry in group['bootstrap']['comparisons'].items():
        low, high = entry['paired_image_bootstrap95']
        verdict = 'interval excludes 0' if low > 0 or high < 0 else 'interval includes 0'
        comparison_rows.append([f'`{entry["model_a"]}` minus `{entry["model_b"]}`',
                                f'{entry["auroc_delta"]:+.3f}', signed_interval([low, high]), verdict])

    seed_rows = []
    for name, entry in group['seed_stability'].items():
        by_seed = {run['seed']: run['auroc'] for run in entry['seeds']}
        seed_rows.append([f'`{name}`'] + [f'{by_seed[seed]:.3f}' for seed in frozen['mlp_config']['seeds']]
                         + [f'{entry["test_auroc_mean"]:.3f}', f'{entry["test_auroc_std"]:.3f}'])

    shown = ['heuristic_confidence', 'lr_all_confidence', lr_hidden, mlp_hidden]

    def breakdown_rows(key, order):
        rows = []
        for value in order:
            entry = results['breakdowns'][key][value]
            rows.append([value, entry['rows'], entry['errors'], f'{entry["vlm_accuracy"]:.3f}',
                         f'{entry["vlm_yes_rate"]:.3f}']
                        + ['n/a' if entry['auroc'][m] is None else f'{entry["auroc"][m]:.3f}' for m in shown])
        return rows
    breakdown_header = ['Group', 'Rows', 'VLM errors', 'VLM accuracy', 'VLM Yes rate'] + [f'AUROC `{m}`' for m in shown]

    counts = data['counts']
    tried = frozen['candidates_tried']
    # Say so when validation picked the smallest C in the planned grid (the grid was not extended).
    smallest_c = min(frozen['lr_config']['C_grid'])
    at_edge = [f'`{name}`' for name, entry in trained.items()
               if entry['kind'] == 'lr' and entry['selected_C'] == smallest_c]
    edge_note = (f"- Validation chose the smallest planned C ({smallest_c}) for {', '.join(at_edge)}. The grid was\n"
                 "  kept as planned, so stronger regularisation was not explored.\n") if at_edge else ''
    report = f"""# Results: VLM spatial-relation error detection on Visual Genome

Generated by `scripts/09_report.py` from saved artifacts (`outputs/results.json`,
`outputs/detectors/frozen.json`, `outputs/test_predictions.csv`). Do not edit by hand.

## Set-up

- VLM: `{VLM['model_id']}` (revision `{VLM['model_revision'][:12]}`), 4-bit NF4, frozen;
  same configuration as the group's VSR experiment.
- Data: Visual Genome relationships with eight canonical spatial predicates; four generated
  statements per relationship (2 true, 2 false). Split by image.
  Train {counts['train']['rows']} queries ({counts['train']['images']} images),
  validation {counts['val']['rows']} ({counts['val']['images']}),
  test {counts['test']['rows']} ({counts['test']['images']}).
- Target: `y_error = 1` when the VLM's Yes/No answer disagrees with the VG-derived truth.
- Train error rate {counts['train']['error_rate']:.3f}, validation {counts['val']['error_rate']:.3f}.
- All fitting used train only; hyperparameters and abstention thresholds used validation only.
  Test was scored once ({results['protocol']['bootstrap_repetitions']} image-clustered bootstrap resamples).

## Test set

{group['rows']} queries from {group['images']} images; {group['errors']} VLM errors.
VLM accuracy {group['unfiltered_vlm_accuracy']:.3f}; error prevalence {group['error_prevalence']:.3f}
(this is the chance line for error-class average precision).

## Detection results

Higher risk means the answer is more likely wrong. MLP rows are seed 42 (primary).
Brier score is descriptive only.

{table(['Model', 'Selected on validation', 'Validation AUROC', 'Test AUROC', '95% interval', 'Test AP', 'Brier'], main_rows)}

![Test AUROC and risk-coverage curves](figure_test_results.png)

## Paired AUROC comparisons (same bootstrap resamples)

Comparisons are exploratory and not adjusted for multiplicity. Intervals condition on the
frozen fits.

{table(['Comparison', 'AUROC difference', '95% interval', ''], comparison_rows)}

## Abstention at the frozen validation threshold

Thresholds were chosen on validation for at most 80% coverage; test coverage can differ.
Without abstention the VLM accuracy is {group['unfiltered_vlm_accuracy']:.3f}.

{table(['Model', 'Threshold', 'Coverage', 'Answered', 'Errors kept', 'Accuracy on answered', 'Errors caught', 'Error recall', 'Abstention precision'], abstain_rows)}

## Where the VLM fails: by canonical predicate

Detector AUROC columns show the unfitted heuristic, the fitted confidence baseline, and the
hidden-state LR and MLP whose layers had the highest validation AUROC.

{table(breakdown_header, breakdown_rows('canonical_predicate', CANONICAL))}

## Where the VLM fails: by variant

`orig` and `converse` are true statements; `swap` and `opposite` are false.

{table(breakdown_header, breakdown_rows('variant', VARIANTS))}

## MLP seed variability on test (separate runs, not an ensemble)

{table(['Model'] + [f'Seed {s}' for s in frozen['mlp_config']['seeds']] + ['Mean', 'SD'], seed_rows)}

## Everything that was tried

- Logistic regression: {tried['lr']} fits (4 input sets x 5 values of C x 2 class weightings);
  see `outputs/detectors/lr_candidates.csv`.
- MLP: {tried['mlp_configurations']} configurations x 3 seeds = {tried['mlp_seed_fits']} fits
  (4 input sets x 2 weight decays x 2 class weightings); see `outputs/detectors/mlp_candidates.csv`.
- `heuristic_confidence` (risk = 1 - confidence) has no fitted parameters.
- No other architectures or input sets were tried.
{aggregation_section()}
## Caveats

- **Not directly comparable with the group's VSR results.** Those detectors were evaluated on
  VSR (941 statements; above/below/left/right); these are on Visual Genome with eight
  predicates and generated variants. The AUROCs come from different datasets.
- Truth labels come from VG annotations with no relabelling and no bounding-box geometry
  check, so some "errors" are label noise. A geometry filter is possible future work.
- Converse statements such as "O is under S" from "S is on O" are not always natural English.
- An image may contain several objects with the same name, which can make a statement
  ambiguous even after contradictory annotations were dropped.
- Confidence is relative Yes/No support from the logits, not a calibrated probability.
- Intervals reflect test-image sampling for the frozen detectors, not retraining variability.
{edge_note}"""
    (OUTPUTS / 'REPORT.md').write_text(report, encoding='utf-8')
    print('Wrote outputs/REPORT.md and outputs/figure_test_results.png')


if __name__ == '__main__':
    main()

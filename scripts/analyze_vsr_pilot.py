"""Describe training-pilot accuracy and untuned error-detection signals."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'confidence_uncertainty': 'Low answer confidence',
    'converse_disagreement': 'Converse disagreement',
    'exclusive_violation': 'Exclusive violation',
    'mirror_disagreement': 'Mirror disagreement',
    'consistency_mean': 'Mean consistency violation',
    'visual_independence': 'Low image sensitivity',
    'blind_support': 'Blind support for answer',
}


def basic(rows):
    n = len(rows)
    correct = sum(not r['error_label'] for r in rows)
    p = correct / n
    z = 1.95996398454
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {
        'n': n, 'correct': correct, 'errors': n-correct, 'accuracy': p,
        'accuracy_wilson95': [center-half, center+half],
        'false_yes': sum(r['truth_label']==0 and r['prediction']==1 for r in rows),
        'false_no': sum(r['truth_label']==1 and r['prediction']==0 for r in rows),
        'true_statements': sum(r['truth_label']==1 for r in rows),
        'false_statements': sum(r['truth_label']==0 for r in rows),
        'predicted_yes': sum(r['prediction']==1 for r in rows),
    }


def risks(rows, eligible):
    values = {
        'confidence_uncertainty': np.array([1-r['features']['confidence'] for r in rows]),
        'visual_independence': np.array([1-r['features']['image_sensitivity'] for r in rows]),
        'blind_support': np.array([r['features']['blind_support_for_original_answer'] for r in rows]),
    }
    if eligible:
        for name in ('converse_disagreement', 'exclusive_violation', 'mirror_disagreement'):
            values[name] = np.array([r['features'][name] for r in rows])
        values['consistency_mean'] = sum(values[n] for n in ('converse_disagreement','exclusive_violation','mirror_disagreement')) / 3
    return values


def detection(rows, eligible, repetitions):
    y = np.array([r['error_label'] for r in rows])
    values = risks(rows, eligible)
    if len(np.unique(y)) != 2:
        return {'n': len(rows), 'errors': int(y.sum()), 'signals': {}, 'note': 'AUROC undefined with one class.'}
    metrics = {name: {'auroc': float(roc_auc_score(y, score)),
        'average_precision': float(average_precision_score(y, score))} for name, score in values.items()}
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['image_id'], []).append(i)
    groups = list(groups.values())
    generator = np.random.default_rng(42)
    samples = {name: [] for name in values}
    deltas = {name: [] for name in values}
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in generator.integers(0, len(groups), len(groups))])
        if len(np.unique(y[indices])) < 2:
            continue
        baseline = roc_auc_score(y[indices], values['confidence_uncertainty'][indices])
        for name, score in values.items():
            auc = float(roc_auc_score(y[indices], score[indices]))
            samples[name].append(auc)
            deltas[name].append(auc-baseline)
    for name, item in metrics.items():
        item['auroc_image_bootstrap95'] = np.quantile(samples[name], [.025,.975]).tolist()
        item['delta_vs_confidence'] = item['auroc']-metrics['confidence_uncertainty']['auroc']
        item['delta_vs_confidence_image_bootstrap95'] = np.quantile(deltas[name], [.025,.975]).tolist()
        item['bootstrap_samples'] = len(samples[name])
    return {'n': len(rows), 'errors': int(y.sum()), 'error_prevalence': float(y.mean()), 'signals': metrics}


def coverage_curve(rows, score):
    # Fixed, label-independent tie break. Coverage values are descriptive rankings,
    # not deployment thresholds selected on this pilot.
    order = sorted(range(len(rows)), key=lambda i: (score[i], rows[i]['example_id']))
    errors = np.array([rows[i]['error_label'] for i in order])
    retained = np.arange(1, len(rows)+1)
    return retained/len(rows), np.cumsum(errors)/retained, np.cumsum(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/vsr_pilot_256')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir/'results.jsonl').read_text().splitlines()]
    run = json.loads((args.run_dir/'run.json').read_text())
    if len(rows) != run['examples'] or any(r['split'] != 'train' for r in rows):
        raise ValueError('Incomplete or non-training pilot results.')
    for row in rows:
        for result in row['scores'].values():
            if not math.isfinite(result['p_yes']) or abs(result['p_yes']+result['p_no']-1)>1e-5:
                raise ValueError('Invalid binary probability.')
        if row['error_label'] != int(row['prediction'] != row['truth_label']):
            raise ValueError('Incorrect error-label construction.')
    eligible = [r for r in rows if r['consistency_eligible']]
    flagged = [r for r in rows if not r['consistency_eligible']]
    calls = [s for r in rows for s in r['scores'].values()]
    mass = [r['scores']['original']['yes_no_full_vocabulary_mass'] for r in rows]
    summary = {
        'scope': 'Training-only exploratory pilot; no verifier trained and no held-out performance claim.',
        'all': basic(rows), 'eligible': basic(eligible), 'flagged': basic(flagged),
        'by_relation': {relation: basic([r for r in rows if r['relation']==relation]) for relation in sorted({r['relation'] for r in rows})},
        'eligible_by_relation': {relation: basic([r for r in eligible if r['relation']==relation]) for relation in sorted({r['relation'] for r in eligible})},
        'blind_accuracy_all': mean(int((r['scores']['blind']['p_yes']>=.5)==r['truth_label']) for r in rows),
        'blind_yes_count': sum(r['scores']['blind']['p_yes']>=.5 for r in rows),
        'original_blind_answer_agreement': mean(int((r['scores']['blind']['p_yes']>=.5)==bool(r['prediction'])) for r in rows),
        'high_confidence': {str(threshold): {
            'n': sum(r['features']['confidence']>=threshold for r in rows),
            'errors': sum(r['features']['confidence']>=threshold and r['error_label']==1 for r in rows)
        } for threshold in (.9,.95,.99)},
        'scoring_checks': {'all_scores_finite': True, 'yes_no_mass_min': min(mass),
            'yes_no_mass_median': median(mass), 'originals_mass_below_half': sum(x<.5 for x in mass)},
        'runtime': {'scored_variants':len(calls), 'wall_seconds':run['total_wall_seconds'],
            'load_seconds':run['model_load_seconds'], 'mean_forward_seconds':mean(s['forward_seconds'] for s in calls),
            'median_forward_seconds':median(s['forward_seconds'] for s in calls),
            'mean_example_wall_seconds':mean(r['example_wall_seconds'] for r in rows),
            'peak_allocated_vram_gib':run['max_peak_allocated_vram_gib']},
    }
    print('Computing untuned signal metrics and image-cluster bootstrap intervals...', flush=True)
    summary['detection_all'] = detection(rows, False, args.bootstrap)
    summary['detection_eligible'] = detection(eligible, True, args.bootstrap)
    summary['diagnostics'] = {}
    for name in ('converse', 'mirrored_equivalent', 'exclusive'):
        if name=='exclusive':
            marked = [r for r in eligible if r['prediction']==1 and r['scores'][name]['p_yes']>=.5]
        else:
            marked = [r for r in eligible if r['prediction'] != int(r['scores'][name]['p_yes']>=.5)]
        summary['diagnostics'][name] = {'flagged':len(marked),'original_errors_among_flagged':sum(r['error_label'] for r in marked),
            'original_correct_among_flagged':sum(not r['error_label'] for r in marked)}
    summary['coverage_eligible'] = {}
    risk_values = risks(eligible, True)
    for name in ('confidence_uncertainty','consistency_mean','visual_independence'):
        coverage, risk, cumulative = coverage_curve(eligible,risk_values[name])
        summary['coverage_eligible'][name] = {}
        for fraction in (1.,.9,.8,.5):
            n = math.ceil(fraction*len(eligible))
            summary['coverage_eligible'][name][str(fraction)] = {'answered':n,'coverage':float(coverage[n-1]),
                'errors':int(cumulative[n-1]),'accuracy':float(1-risk[n-1])}
    summary['examples'] = {
        'confident_errors': [{'audit_index':r['audit_index'],'statement':r['statement'],'truth_label':r['truth_label'],
            'prediction':r['prediction'],'confidence':r['features']['confidence'], 'consistency_eligible':r['consistency_eligible'],
            'features':r['features']} for r in sorted([r for r in rows if r['error_label']], key=lambda r:-r['features']['confidence'])[:10]],
        'correct_with_large_consistency_disagreement': [{'audit_index':r['audit_index'],'statement':r['statement'],'features':r['features']}
            for r in sorted([r for r in eligible if not r['error_label']], key=lambda r:-(r['features']['converse_disagreement']+r['features']['mirror_disagreement']))[:5]]
    }
    (args.run_dir/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    signals = summary['detection_eligible']['signals']
    names = list(LABELS)
    aurocs = [signals[name]['auroc'] for name in names]
    lo = [signals[name]['auroc_image_bootstrap95'][0] for name in names]
    hi = [signals[name]['auroc_image_bootstrap95'][1] for name in names]
    ypos = np.arange(len(names))
    axes[0].hlines(ypos, lo, hi, color='#54738c', linewidth=2)
    axes[0].scatter(aurocs,ypos,color='#156384',zorder=3)
    axes[0].set_yticks(ypos,[LABELS[n] for n in names])
    axes[0].invert_yaxis()
    axes[0].axvline(.5,color='#777777',linestyle='--',linewidth=1)
    axes[0].set(xlim=(0,1),xlabel='Error-detection AUROC (higher is better)',title=f'{len(eligible)} reviewed training examples; {sum(r["error_label"] for r in eligible)} errors')
    for name,color in [('confidence_uncertainty','#156384'),('consistency_mean','#bb6735'),('visual_independence','#567a40')]:
        coverage,risk,_ = coverage_curve(eligible,risk_values[name])
        axes[1].plot(coverage,1-risk,label=LABELS[name],color=color,linewidth=2)
    axes[1].axhline(summary['eligible']['accuracy'],color='#777777',linestyle='--',label='Answer everything')
    axes[1].set(xlim=(.2,1),ylim=(0,1.02),xlabel='Coverage: fraction answered',ylabel='Accuracy on answered examples',title='Pilot risk–coverage comparison')
    axes[1].legend(fontsize=8,loc='lower left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Exploratory training pilot — no trained verifier or test-set evaluation',fontsize=12)
    fig.savefig(args.run_dir/'pilot_results.png',dpi=160)
    plt.close(fig)
    print(json.dumps({key:value for key,value in summary.items() if key not in ('examples',)},indent=2),flush=True)


if __name__ == '__main__':
    main()

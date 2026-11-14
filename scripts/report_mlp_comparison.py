"""Build the comparison report and figures from cached evaluation only."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_detector_comparison import risk_coverage, check_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mlp_comparison'


def main():
    summary=json.loads((OUT/'test_summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((OUT/'frozen_comparison.json').read_text(encoding='utf-8'))
    selection=frozen['selection']
    lr=json.loads((ROOT/'outputs/logistic_baselines/summary.json').read_text(encoding='utf-8'))
    check_hash(OUT/'test_predictions.csv',summary['test_predictions_sha256'])
    with (OUT/'test_predictions.csv').open(encoding='utf-8',newline='') as handle:
        records=list(csv.DictReader(handle))
    lines=['# Frozen MLP versus logistic-regression comparison','',
        'Final held-out test evaluation. Hyperparameters, checkpoints, primary seed 42, and abstention thresholds were frozen before test inference. No test tuning or seed selection was performed.','',
        'The primary comparisons do not establish an MLP advantage. On all relations, the core MLP scored 0.746 AUROC versus 0.755 for logistic regression (paired difference −0.010, 95% interval [−0.034, +0.012]). On the vertical consistency subset, the behavior MLP scored 0.810 versus 0.802 (difference +0.008, interval [−0.026, +0.043]). Both intervals include zero. Keep the practical logistic core baseline; retain the MLP as a comparison rather than claiming a demonstrated improvement.','',
        'The more promising signal is consistency information: on the same vertical subset, MLP core/behavior AUROC was 0.774/0.810 and LR core/behavior AUROC was 0.761/0.802. The MLP increment remains uncertain (paired interval [−0.025, +0.107]); a larger independent evaluation is needed. Hidden-state MLPs did not beat the practical core baseline on the primary seed; early/middle probes had large train/test gaps. These results support testing behavioral features further, rather than adding model capacity alone.','',
        '## Method','',
        'Two learned layers: standardized inputs → Linear(32) → ReLU → Dropout(0.2) → Linear(1). Sigmoid converts the output logit to an error-risk score. The VLM and its extracted features were unchanged.','',
        'Scalers and weights used training only. Adam, learning rate 0.001, batch size 32, up to 200 epochs; early stopping after 15 epochs without validation-BCE improvement, restoring the best checkpoint. A fixed search compared weight decay 0.0001/0.01 and unweighted/training-balanced BCE. Settings were selected by mean validation AUROC across seeds 42/43/44, then mean AP. Seed 42 was predeclared for the primary test comparison; the three runs are not an ensemble.','',
        f"Eight matched feature/cohort experiments, 32 candidate configurations, 96 seed fits; training took {selection['training_wall_seconds']:.1f} seconds on CPU. No new VLM calls were needed.",
        '', 'Target: `1` means the original VLM answer disagrees with the accepted VSR label; `0` means it agrees. This measures answer error, not intent to deceive. Inputs exclude labels, image IDs, and dataset split.','',
        'AUROC measures error ranking (0.5 random). AP is average precision for the error class; its chance reference is the cohort error prevalence. Higher AUROC/AP is better. Brier scores are descriptive and do not establish probability calibration.','']
    labels={'confidence':'Confidence (2)','core':'Core (4)','hidden9':'Hidden layer 9 (2048)',
        'hidden18':'Hidden layer 18 (2048)','hidden36':'Hidden layer 36 (2048)','behavior':'Behavior (7)'}
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,cohort in enumerate(('all','consistency')):
        group=summary['cohorts'][cohort]
        models=group['models']
        experiments=[e for e in selection['experiments'] if e['cohort']==cohort]
        title='All relations' if cohort=='all' else 'Vertical consistency subset'
        lines.extend([f'## {title}','',
            f"Test: {group['rows']} pairs across {group['images']} images, {group['errors']} VLM errors ({group['error_prevalence']:.1%}). Unfiltered VLM accuracy: {group['unfiltered_vlm_accuracy']:.1%}.",'',
            '| Inputs | LR validation AUROC | MLP validation mean ± SD | LR test AUROC | MLP test AUROC (seed 42) | LR / MLP test AP | MLP test mean ± SD (3 seeds) |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        heuristic=models['heuristic_confidence']
        lines.append(f"| Raw confidence risk | {lr['cohorts'][cohort]['models']['heuristic_confidence']['auroc']:.3f} | — | {heuristic['auroc']:.3f} | — | {heuristic['average_precision']:.3f} / — | — |")
        x=np.arange(len(experiments))
        for i,e in enumerate(experiments):
            name=e['name']
            suffix=name.removeprefix(cohort+'_')
            label=labels[suffix]
            a=models['lr_'+name]
            b=models['mlp_'+name]
            v=selection['models'][name]['selection_metrics']
            s=group['seed_stability'][name]
            lines.append(f"| {label} | {lr['cohorts'][cohort]['models'][name]['auroc']:.3f} | {v['mean_validation_auroc']:.3f} ± {v['std_validation_auroc']:.3f} | {a['auroc']:.3f} | {b['auroc']:.3f} | {a['average_precision']:.3f} / {b['average_precision']:.3f} | {s['test_auroc_mean']:.3f} ± {s['test_auroc_std']:.3f} |")
            for offset,result,color,legend in [(-.12,a,'#3973ac','Logistic regression'),(.12,b,'#d87529','MLP, seed 42')]:
                low,high=result['auroc_image_bootstrap95']
                axes[0,col].errorbar(i+offset,result['auroc'],yerr=[[result['auroc']-low],[high-result['auroc']]],
                    fmt='o',color=color,capsize=3,label=legend if i==0 else None)
        ax=axes[0,col]
        ax.set_xticks(x,[labels[e['name'].removeprefix(cohort+'_')].replace('Hidden layer ','Layer ') for e in experiments],rotation=18)
        ax.axhline(.5,color='gray',linestyle=':',linewidth=1)
        ax.set_ylim(.3,1)
        ax.set_ylabel('Test AUROC (95% image-bootstrap interval)')
        ax.set_title(f'{title}: n={group["rows"]}, errors={group["errors"]}')
        ax.legend(fontsize=9)
        ax.grid(axis='y',alpha=.2)
        lines.extend(['','### Frozen validation-threshold abstention policy','',
            'Each threshold targeted at most 80% coverage on validation, keeping whole ties. These unchanged thresholds can yield different coverage on test; operating-point accuracies are therefore not matched-coverage comparisons.','',
            '| Model | Test answered | Test coverage | Errors retained | Accuracy on answered | Errors caught / total |',
            '|---|---:|---:|---:|---:|---:|'])
        main_name='all_core' if cohort=='all' else 'consistency_behavior'
        curve_keys=['heuristic_confidence','lr_'+main_name,'mlp_'+main_name]
        curve_labels=['Raw confidence','Logistic regression','MLP, seed 42']
        for key,label,color in zip(curve_keys,curve_labels,['#777777','#3973ac','#d87529']):
            p=models[key]['operating_point']
            accuracy=f"{p['retained_vlm_accuracy']:.1%}" if p['retained_vlm_accuracy'] is not None else 'n/a'
            lines.append(f"| {label} | {p['answered']}/{group['rows']} | {p['coverage']:.1%} | {p['retained_errors']} | {accuracy} | {p['errors_caught']}/{group['errors']} |")
            cache_key=key+'_seed42' if key.startswith('mlp_') else key
            rs=[r for r in records if r['cohort']==cohort and r['model']==cache_key]
            if len(rs)!=group['rows']:
                raise ValueError('Cached prediction count does not match summary.')
            y=np.array([int(r['y_error']) for r in rs])
            risk=np.array([float(r['risk']) for r in rs])
            coverage,error=risk_coverage(y,risk)
            axes[1,col].step(coverage,error,where='post',color=color,label=label)
            if p['answered']:
                axes[1,col].scatter([p['coverage']],[1-p['retained_vlm_accuracy']],color=color,s=25)
        ax=axes[1,col]
        ax.axhline(group['error_prevalence'],color='black',linestyle=':',linewidth=1,label='Unfiltered VLM error rate')
        ax.set(xlim=(.2,1),ylim=(0,.4),xlabel='Coverage: fraction answered',ylabel='Risk: error rate among answered',
            title='Descriptive test ranking; dots use frozen thresholds')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        lines.extend(['','### Paired AUROC differences','',
            'Image-cluster bootstrap, 2,000 resamples. A 95% interval spanning zero does not establish a reliable improvement. Intervals condition on the frozen fits; seed variability is separate. The core full-cohort and behavior subset comparisons were predeclared primary; other feature comparisons are exploratory, without multiplicity adjustment.','',
            '| MLP minus matched LR | Difference | 95% paired interval |','|---|---:|---:|'])
        for e in experiments:
            name=e['name']
            comparison=group['bootstrap']['comparisons'][name+'_mlp_minus_lr']
            low,high=comparison['paired_image_bootstrap95']
            lines.append(f"| {labels[name.removeprefix(cohort+'_')]} | {comparison['auroc_delta']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
        if cohort=='consistency':
            comparison=group['bootstrap']['comparisons']['mlp_behavior_minus_core']
            low,high=comparison['paired_image_bootstrap95']
            lines.extend(['',f"Adding the three consistency features to the MLP on this same subset changes test AUROC by {comparison['auroc_delta']:+.3f}, paired 95% interval [{low:+.3f}, {high:+.3f}]."])
        lines.extend(['','### MLP generalization and selected settings','',
            '| Inputs | Train AUROC (seed 42) | Validation AUROC (seed 42) | Test AUROC (seed 42) | Selected decay | Class weights | Best epoch |',
            '|---|---:|---:|---:|---:|---|---:|'])
        for e in experiments:
            entry=selection['models'][e['name']]
            run=entry['seeds']['42']
            lines.append(f"| {labels[e['name'].removeprefix(cohort+'_')]} | {run['training_metrics']['auroc']:.3f} | {run['validation_metrics']['auroc']:.3f} | {models['mlp_'+e['name']]['auroc']:.3f} | {entry['selected_weight_decay']} | {entry['selected_class_weight']} | {run['best_epoch']} |")
        lines.append('')
    fig.suptitle('Frozen VSR error detectors: held-out comparison',fontsize=15)
    fig.savefig(OUT/'test_comparison.png',dpi=180)
    fig.savefig(OUT/'test_comparison.pdf')
    plt.close(fig)
    lines.extend(['## Interpretation limits and next steps','',
        'The primary comparison is seed 42, fixed before test. Three-seed mean ± sample SD describes initialization/training variability and is not a confidence interval or an ensemble. Do not choose a seed or layer using the test table. Small test cohorts yield substantial sampling uncertainty.','',
        'Consistency features are evaluated only on the vertical subset (above/below). Its metrics cannot be compared directly with the all-relations cohort. Input dimensionality is large relative to the number of training errors for hidden-state probes; train/validation gaps and seed sensitivity matter. Probe performance alone does not show that a model knew the correct answer internally.','',
        'Risk scores rank likely answer errors; they are not established calibrated probabilities. Abstention improves reliability by declining questions, rather than correcting answers. Images were split without overlap, but this is one VSR-derived sample and one fixed VLM/configuration. External validity is still untested.','',
        'This held-out split has now been used. Future architectural or feature tuning needs a fresh holdout or an external benchmark such as a separately prepared R-Bench evaluation. Retain the frozen logistic baseline, quantify gains on that new evaluation, and calibrate/choose any deployment policy on validation data.','',
        '## Saved artifacts and usage','',
        '`models/{experiment}_seed42.pt` is the primary MLP; `models/{experiment}_scaler.joblib` is its training-fitted scaler. Seeds 43 and 44 are separately saved. See `validation_selection.json` for checkpoint metadata and unchanged validation thresholds. Never fit another scaler at inference.','',
        'For `all_core`, input order is `[p_original, confidence, p_blind, image_sensitivity]`, with `confidence=max(p_original,1-p_original)` and `image_sensitivity=abs(p_original-p_blind)`. The behavior input appends converse disagreement, exclusive violation, and mirror disagreement, on eligible vertical examples only. Hidden models take the original-prompt 2,048-value vector at their named layer.','',
        '```python',
        'import sys, joblib, numpy as np',
        "sys.path.insert(0, 'scripts')",
        'from train_mlp_detectors import load_checkpoint, risk_scores',
        "scaler = joblib.load('outputs/mlp_comparison/models/all_core_scaler.joblib')",
        "model, _ = load_checkpoint('outputs/mlp_comparison/models/all_core_seed42.pt')",
        'X = np.asarray([[p_original, confidence, p_blind, image_sensitivity]], dtype=np.float32)',
        'risk = risk_scores(model, scaler.transform(X).astype(np.float32))',
        '```','',
        'Reproduction entry points (training deliberately refuses to run after test exposure):','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts\\train_mlp_detectors.py',
        '.\\.venv\\Scripts\\python.exe scripts\\evaluate_detector_comparison.py',
        '.\\.venv\\Scripts\\python.exe scripts\\report_mlp_comparison.py',
        '```','',
        'The evaluator checks frozen file hashes and returns cached results after successful completion. The report reads cached predictions only. Preserve `frozen_comparison.json`, `test_evaluation_started.json`, `test_summary.json`, `test_predictions.csv` and model files together. The source LR report remains a historical validation-only report.',''])
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved REPORT.md, test_comparison.png and test_comparison.pdf from cached evaluation.')


if __name__=='__main__':
    main()

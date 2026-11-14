"""Report saved baseline models; uses training and validation only."""
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from train_logistic_baselines import load_training_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/logistic_baselines'


def paired_delta(y,left,right,images,repetitions=1000):
    groups={}
    for i,key in enumerate(images):
        groups.setdefault(key,[]).append(i)
    groups=list(groups.values())
    rng=np.random.default_rng(42)
    values=[]
    for _ in range(repetitions):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[indices]))==2:
            values.append(roc_auc_score(y[indices],left[indices])-roc_auc_score(y[indices],right[indices]))
    return {'delta':float(roc_auc_score(y,left)-roc_auc_score(y,right)),
        'image_bootstrap95':np.quantile(values,[.025,.975]).tolist(),
        'note':'Conditional on validation-selected models; not an independent test estimate.'}


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    if summary['test_opened']:
        raise ValueError('Expected a validation-only training stage.')
    gaps=[]
    risks={}
    for cohort,group in summary['cohorts'].items():
        for name,model_info in group['models'].items():
            if not model_info['trained']:
                X,y,info=load_training_split('validation',feature_set='confidence',cohort=cohort)
                risks[(cohort,name)]=(1-X[:,1],y,info)
                continue
            path=ROOT/model_info['model_file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=model_info['model_sha256']:
                raise ValueError('Saved model checksum mismatch.')
            model=joblib.load(path)
            kwargs={k:model_info[k] for k in ('feature_set','cohort','layer')}
            X,y,_=load_training_split('train',**kwargs)
            V,v,info=load_training_split('validation',**kwargs)
            train_risk=model.predict_proba(X)[:,1]
            val_risk=model.predict_proba(V)[:,1]
            auc=float(roc_auc_score(v,val_risk))
            if not np.isclose(auc,model_info['auroc'],rtol=0,atol=1e-12):
                raise ValueError('Validation metrics changed after reloading model.')
            risks[(cohort,name)]=(val_risk,v,info)
            gaps.append({'model':name,'cohort':cohort,'train_auroc':float(roc_auc_score(y,train_risk)),
                'validation_auroc':auc,'train_average_precision':float(average_precision_score(y,train_risk)),
                'validation_average_precision':model_info['average_precision']})
    with (OUT/'generalization_gaps.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)
    comparisons={}
    for cohort,left,right in [('all','all_core','all_confidence'),
                             ('consistency','consistency_behavior','consistency_core')]:
        l,y,info=risks[(cohort,left)]
        r,other_y,other_info=risks[(cohort,right)]
        if not np.array_equal(y,other_y) or not np.array_equal(info['example_ids'],other_info['example_ids']):
            raise ValueError('Paired comparison uses different examples.')
        comparisons[left+'_minus_'+right]=paired_delta(y,l,r,info['image_ids'])
    (OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    display_names={'heuristic_confidence':'Raw VLM confidence','all_confidence':'LR: confidence features',
        'all_core':'LR: 4 core features','all_hidden9':'LR: layer 9 hidden state',
        'all_hidden18':'LR: layer 18 hidden state','all_hidden36':'LR: layer 36 hidden state',
        'consistency_confidence':'LR: confidence features','consistency_core':'LR: 4 core features',
        'consistency_behavior':'LR: 7 behavioral features'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,(cohort,group) in zip(axes,summary['cohorts'].items()):
        models=list(group['models'].items())
        positions=np.arange(len(models))
        for position,(name,model) in zip(positions,models):
            lo,hi=model['auroc_image_bootstrap95']
            color='#6b7680' if not model['trained'] else '#206d70'
            ax.hlines(position,lo,hi,color=color,linewidth=2)
            ax.scatter(model['auroc'],position,color=color,zorder=3)
        ax.set_yticks(positions,[display_names[name] for name,_ in models])
        ax.invert_yaxis()
        ax.axvline(.5,color='#999999',linestyle='--',linewidth=1)
        ax.set(xlim=(0,1),xlabel='Validation error-detection AUROC',
            title=f'{group["validation_rows"]} validation pairs; {group["validation_errors"]} VLM errors')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Logistic-regression baselines — validation only; test reserved',fontsize=13)
    fig.savefig(OUT/'validation_results.png',dpi=160)
    plt.close(fig)
    lines=['# Logistic-regression detector baselines', '',
        'Eight fitted detectors are saved. Each StandardScaler and classifier was fitted only on the training split. '
        'Validation selected C, class weighting and provisional abstention thresholds. The test split was not opened by this training/reporting stage.', '',
        'Target: 1 = the original frozen VLM answer disagrees with VSR truth; 0 = it agrees. '
        'Outputs are error-risk rankings, not assumed calibrated correctness probabilities.', '',
        '## Protocol', '',
        'C grid: 0.001, 0.01, 0.1, 1, 10. Class weighting: none or balanced. Eight experiments, 80 candidate fits. '
        'L2-regularized logistic regression, LBFGS, at most 5,000 iterations, seed 42, one numerical thread. '
        'Selection: validation AUROC, then average precision, then stronger regularization and unweighted fits for exact ties. '
        'All selected fits converged and reproduced predictions after save/reload.', '',
        'The VLM and feature-extraction configuration are unchanged. Baselines in each cohort use identical training/validation examples. '
        'Do not compare the 148-row and 95-row cohorts as if they were the same evaluation set.', '']
    for cohort,group in summary['cohorts'].items():
        lines += [f'## {"All examples" if cohort=="all" else "Consistency-eligible vertical subset"}', '',
            f'Training: {group["training_rows"]} pairs. Validation: {group["validation_rows"]} pairs, '
            f'{group["validation_errors"]} errors. Average-precision reference prevalence: {group["validation_error_prevalence"]:.3f}.', '',
            '| Detector | Validation AUROC | Average precision | 95% AUROC interval |', '|---|---:|---:|---|']
        for name,model in group['models'].items():
            lo,hi=model['auroc_image_bootstrap95']
            lines.append(f'| {display_names[name]} | {model["auroc"]:.3f} | {model["average_precision"]:.3f} | {lo:.3f}–{hi:.3f} |')
        lines.append('')
    lines += ['## Interpretation and limits', '',
        'The core detector is the practical starting baseline on all examples. Its validation AUROC is only slightly above the learned confidence-only baseline. '
        'The seven-feature detector increases average precision on the consistency subset but does not increase AUROC over its four-feature counterpart. '
        'These results do not establish that the extra diagnostic calls improve detection.', '',
        'The final-layer linear hidden-state probe performs better than the early/middle probes here. '
        'The probes have 2,048 input dimensions and only 659 training examples; consult generalization_gaps.csv for train–validation gaps. '
        'This is not evidence that a particular layer contains or loses the correct spatial answer.', '',
        'All reported intervals resample images and condition on the selected models. They do not account for hyperparameter-selection optimism on the same validation set. '
        'No final generalization claim should be made before the later untouched-test comparison.', '',
        '## Provisional abstention policy', '',
        'Thresholds target at most 80% validation coverage, accepting whole tied-score groups. '
        'They measure retained accuracy of the original VLM answers, not classification accuracy of the error detector. '
        'For the full-cohort core baseline:', '']
    g=summary['cohorts']['all']
    point=g['models']['all_core']['coverage80']
    lines += [f'- Original VLM validation accuracy: {100*g["unfiltered_vlm_accuracy"]:.1f}%.',
        f'- Retained: {point["answered"]}/{g["validation_rows"]} ({100*point["coverage"]:.1f}% coverage).',
        f'- Retained VLM errors: {point["retained_errors"]}; retained accuracy: {100*point["retained_vlm_accuracy"]:.1f}%.',
        f'- Selected risk threshold: {point["threshold"]:.6f}.', '',
        'Class-weighted models especially should not be presented as calibrated error probabilities. '
        'A later calibrated-probability claim requires separate validation/calibration design.', '',
        '## Files and next step', '',
        '- models/*.joblib: fitted scaler + classifier pipelines; JSON sidecars document inputs and settings.',
        '- protocol.json: frozen search and data fingerprints.',
        '- hyperparameter_search.csv: all 80 candidate validation results.',
        '- validation_metrics.csv and validation_predictions.csv: selected-model validation outputs.',
        '- generalization_gaps.csv: training versus validation performance.',
        '- paired_comparisons.json: matched incremental AUROC comparisons.',
        '- summary.json: metrics, intervals and coverage thresholds.', '',
        'Next: train a small MLP with the same feature sets and cohorts, using training-only preprocessing and validation-based selection. '
        'Freeze the comparison and then evaluate the selected LR/MLP models once on test. Keep these LR artifacts unchanged for that comparison.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'generalization_gaps':gaps,'paired_comparisons':comparisons},indent=2))


if __name__=='__main__':
    main()

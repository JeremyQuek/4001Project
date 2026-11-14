"""One frozen, held-out evaluation; never fits models or selects test thresholds."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from load_detector_data import load_detector_data
from train_logistic_baselines import metrics, write_csv
from train_mlp_detectors import load_checkpoint, risk_scores

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mlp_comparison'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hash(path,expected):
    if sha(path)!=expected:
        raise ValueError(f'Frozen artifact changed: {path}')


def operating_point(y,risk,threshold):
    """Apply the saved validation threshold, including entire score ties."""
    y=np.asarray(y)
    keep=np.asarray(risk)<=threshold
    answered=int(keep.sum())
    errors=int(y[keep].sum())
    rejected=len(y)-answered
    caught=int(y[~keep].sum())
    return {'validation_threshold':float(threshold),'answered':answered,
        'coverage':answered/len(y),'retained_errors':errors,
        'retained_vlm_accuracy':float(1-errors/answered) if answered else None,
        'abstained':rejected,'errors_caught':caught,
        'error_recall':float(caught/y.sum()) if y.sum() else None,
        'abstention_precision':float(caught/rejected) if rejected else None}


def risk_coverage(y,risk):
    """Descriptive test curve, retaining whole ties; not a deployed threshold."""
    order=np.argsort(risk,kind='stable')
    sorted_risk=np.asarray(risk)[order]
    end=np.r_[np.flatnonzero(np.diff(sorted_risk)!=0),len(order)-1]
    n=end+1
    errors=np.cumsum(np.asarray(y)[order])[end]
    return n/len(y),errors/n


def paired_bootstrap(y,predictions,image_ids,pairs,repetitions,seed=20261114):
    """Resample images and reuse each draw for all frozen model comparisons."""
    groups={}
    for i,image_id in enumerate(image_ids):
        groups.setdefault(str(image_id),[]).append(i)
    groups=list(groups.values())
    generator=np.random.default_rng(seed)
    samples={name:[] for name in predictions}
    for _ in range(repetitions):
        indices=np.concatenate([groups[i] for i in generator.integers(0,len(groups),len(groups))])
        if len(np.unique(y[indices]))!=2:
            continue
        for name,risk in predictions.items():
            samples[name].append(float(roc_auc_score(y[indices],risk[indices])))
    intervals={name:np.quantile(values,[.025,.975]).tolist() for name,values in samples.items()}
    comparisons={}
    for label,(a,b) in pairs.items():
        deltas=np.asarray(samples[a])-np.asarray(samples[b])
        comparisons[label]={'model_a':a,'model_b':b,
            'auroc_delta':float(roc_auc_score(y,predictions[a])-roc_auc_score(y,predictions[b])),
            'paired_image_bootstrap95':np.quantile(deltas,[.025,.975]).tolist()}
    return {'auroc_intervals':intervals,'comparisons':comparisons,
        'valid_resamples':len(next(iter(samples.values()))),'resampling_unit':'image',
        'note':'Intervals condition on frozen fits; seed variability is reported separately. Exploratory comparisons are not multiplicity adjusted.'}


def main():
    started=time.perf_counter()
    torch.set_num_threads(1)
    freeze_path=OUT/'frozen_comparison.json'
    frozen=json.loads(freeze_path.read_text(encoding='utf-8'))
    check_hash(ROOT/'configs/mlp_comparison.json',frozen['config_sha256'])
    check_hash(OUT/'training_protocol.json',frozen['training_protocol_sha256'])
    check_hash(ROOT/'outputs/logistic_baselines/summary.json',frozen['lr_summary_sha256'])
    check_hash(ROOT/'data/vsr/detector_ready/test.npz',frozen['test_npz_sha256'])
    selection=frozen['selection']
    lr=json.loads((ROOT/'outputs/logistic_baselines/summary.json').read_text(encoding='utf-8'))
    # Check all model bytes before opening any test arrays.
    for name,entry in selection['models'].items():
        check_hash(ROOT/entry['scaler_file'],entry['scaler_sha256'])
        for seed,run in entry['seeds'].items():
            check_hash(ROOT/run['checkpoint'],run['checkpoint_sha256'])
        baseline=lr['cohorts'][entry['cohort']]['models'][name]
        check_hash(ROOT/baseline['model_file'],baseline['model_sha256'])
    eval_protocol={'frozen_comparison_sha256':sha(freeze_path),'evaluator_sha256':sha(__file__),
        'primary_seed':frozen['primary_seed'],'primary_experiments':frozen['primary_experiments'],
        'test_thresholds':'Apply unchanged validation thresholds; actual test coverage can differ from 80%.',
        'bootstrap_repetitions':selection['config']['bootstrap_repetitions'],
        'test_use':'Final evaluation only; no model/seed/threshold selection.',
        'other_seeds':'Separate stability runs, not an ensemble or seed search.'}
    marker_path=OUT/'test_evaluation_started.json'
    if marker_path.exists():
        old=json.loads(marker_path.read_text(encoding='utf-8'))
        if old['protocol']!=eval_protocol:
            raise ValueError('Test was already exposed under a different protocol. Do not tune/re-evaluate.')
        if (OUT/'test_summary.json').exists():
            print('Evaluation already complete. Read cached test_summary.json; no test data reopened.')
            return
    else:
        marker_path.write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':eval_protocol},indent=2),encoding='utf-8')
    seed0=str(frozen['primary_seed'])
    summary={'protocol':eval_protocol,'cohorts':{},'test_opened':True}
    rows=[]
    prediction_rows=[]
    with threadpool_limits(limits=1):
        for cohort in ('all','consistency'):
            X,y,info=load_detector_data('test',feature_set='confidence',cohort=cohort)
            for split in ('train','validation'):
                _,_,other=load_detector_data(split,feature_set='confidence',cohort=cohort)
                if set(info['image_ids'])&set(other['image_ids']):
                    raise ValueError('Test image leakage across splits.')
            group={'rows':len(y),'images':len(set(info['image_ids'])),'errors':int(y.sum()),
                'error_prevalence':float(y.mean()),'unfiltered_vlm_accuracy':float(1-y.mean()),
                'models':{},'seed_stability':{}}
            predictions={'heuristic_confidence':1-X[:,1]}
            thresholds={'heuristic_confidence':lr['cohorts'][cohort]['models']['heuristic_confidence']['coverage80']['threshold']}
            pairs={}
            for experiment in selection['experiments']:
                if experiment['cohort']!=cohort:
                    continue
                name=experiment['name']
                entry=selection['models'][name]
                kwargs={k:experiment[k] for k in ('feature_set','cohort','layer')}
                inputs,targets,meta=load_detector_data('test',**kwargs)
                if not np.array_equal(targets,y) or not np.array_equal(meta['example_ids'],info['example_ids']):
                    raise ValueError('Test feature/target rows are misaligned.')
                baseline=lr['cohorts'][cohort]['models'][name]
                lr_model=joblib.load(ROOT/baseline['model_file'])
                if lr_model['classifier'].classes_.tolist()!=[0,1]:
                    raise ValueError('LR output class order changed.')
                lr_key='lr_'+name
                predictions[lr_key]=lr_model.predict_proba(inputs)[:,1]
                thresholds[lr_key]=baseline['coverage80']['threshold']
                scaler=joblib.load(ROOT/entry['scaler_file'])
                if scaler.n_samples_seen_!=entry['train_rows']:
                    raise ValueError('MLP scaler training rows changed.')
                scaled=scaler.transform(inputs).astype(np.float32)
                seed_metrics=[]
                for seed,run in entry['seeds'].items():
                    model,_=load_checkpoint(ROOT/run['checkpoint'])
                    risk=risk_scores(model,scaled)
                    key='mlp_'+name+'_seed'+seed
                    seed_metrics.append({'seed':int(seed),**metrics(y,risk),
                        'operating_point':operating_point(y,risk,run['coverage80']['threshold'])})
                    for i,score in enumerate(risk):
                        prediction_rows.append({'cohort':cohort,'model':key,'example_id':str(info['example_ids'][i]),
                            'image_id':str(info['image_ids'][i]),'y_error':int(y[i]),'risk':float(score)})
                    if seed==seed0:
                        predictions['mlp_'+name]=risk
                        thresholds['mlp_'+name]=run['coverage80']['threshold']
                group['seed_stability'][name]={'seeds':seed_metrics,
                    'test_auroc_mean':float(np.mean([r['auroc'] for r in seed_metrics])),
                    'test_auroc_std':float(np.std([r['auroc'] for r in seed_metrics],ddof=1))}
                pairs[name+'_mlp_minus_lr']=('mlp_'+name,lr_key)
                pairs[name+'_mlp_minus_heuristic']=('mlp_'+name,'heuristic_confidence')
            if cohort=='consistency':
                pairs['mlp_behavior_minus_core']=('mlp_consistency_behavior','mlp_consistency_core')
                pairs['lr_behavior_minus_core']=('lr_consistency_behavior','lr_consistency_core')
            print(f'{cohort}: {len(y)} test rows, {int(y.sum())} VLM errors; frozen predictions complete.',flush=True)
            bootstrap=paired_bootstrap(y,predictions,info['image_ids'],pairs,
                selection['config']['bootstrap_repetitions'])
            group['bootstrap']=bootstrap
            for name,risk in predictions.items():
                result={**metrics(y,risk),'auroc_image_bootstrap95':bootstrap['auroc_intervals'][name],
                    'operating_point':operating_point(y,risk,thresholds[name])}
                group['models'][name]=result
                rows.append({'cohort':cohort,'model':name,'rows':len(y),'errors':int(y.sum()),
                    **metrics(y,risk),'auroc_ci_low':result['auroc_image_bootstrap95'][0],
                    'auroc_ci_high':result['auroc_image_bootstrap95'][1],**result['operating_point']})
                if not name.startswith('mlp_'):
                    for i,score in enumerate(risk):
                        prediction_rows.append({'cohort':cohort,'model':name,'example_id':str(info['example_ids'][i]),
                            'image_id':str(info['image_ids'][i]),'y_error':int(y[i]),'risk':float(score)})
            summary['cohorts'][cohort]=group
            print(f'{cohort}: paired image-bootstrap evaluation complete.',flush=True)
    write_csv(OUT/'test_metrics.csv',rows)
    write_csv(OUT/'test_predictions.csv',prediction_rows)
    summary['evaluation_wall_seconds']=time.perf_counter()-started
    summary['test_predictions_sha256']=sha(OUT/'test_predictions.csv')
    (OUT/'test_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Held-out evaluation saved; model choices and validation thresholds were unchanged.',flush=True)


if __name__=='__main__':
    main()

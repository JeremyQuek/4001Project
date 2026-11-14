"""Fit frozen-feature error detectors using train/validation only. Test stays closed."""
import csv
import hashlib
import json
import math
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from load_detector_data import load_detector_data

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/logistic_baselines'


def load_training_split(split,**kwargs):
    if split not in ('train','validation'):
        raise ValueError('This training stage may not open the test split.')
    return load_detector_data(split,**kwargs)


def fit_candidate(X,y,C,class_weight,config):
    if not np.isfinite(X).all() or len(np.unique(y))!=2:
        raise ValueError('Training requires finite inputs and both target classes.')
    pipeline=Pipeline([
        ('scaler',StandardScaler()),
        ('classifier',LogisticRegression(C=C,class_weight=class_weight,solver=config['solver'],
            max_iter=config['max_iter'],tol=config['tol'],random_state=config['seed']))])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipeline.fit(X,y)
    if pipeline['scaler'].n_samples_seen_!=len(X) or not np.allclose(pipeline['scaler'].mean_,X.mean(axis=0,dtype=np.float64),rtol=1e-6,atol=1e-7):
        raise ValueError('Scaler does not match training-only statistics.')
    if pipeline['classifier'].classes_.tolist()!=[0,1]:
        raise ValueError('Risk output must correspond to error class 1.')
    return pipeline


def metrics(y,risk):
    if not np.isfinite(risk).all() or np.any((risk<0)|(risk>1)):
        raise ValueError('Invalid risk scores.')
    return {'auroc':float(roc_auc_score(y,risk)),
        'average_precision':float(average_precision_score(y,risk)),
        'brier_score':float(brier_score_loss(y,risk))}


def coverage_threshold(y,risk,target=.8):
    """Retain a whole tie group, choosing greatest attainable coverage <= target."""
    target_n=math.floor(target*len(risk))
    values,counts=np.unique(risk,return_counts=True)
    totals=np.cumsum(counts)
    eligible=np.flatnonzero(totals<=target_n)
    if not len(eligible):
        threshold=float(np.nextafter(values[0],-np.inf))
    else:
        threshold=float(values[eligible[-1]])
    mask=risk<=threshold
    n=int(mask.sum())
    return {'threshold':threshold,'target_coverage':target,'answered':n,
        'coverage':n/len(y),'retained_errors':int(y[mask].sum()),
        'retained_vlm_accuracy':float(1-y[mask].mean()) if n else None,
        'tie_policy':'Whole risk-score tie groups retained; coverage cannot exceed target.'}


def bootstrap_intervals(y,predictions,image_ids,repetitions,seed):
    """Fixed selected models; conditional validation intervals, not nested selection CIs."""
    grouped={}
    for i,image in enumerate(image_ids):
        grouped.setdefault(image,[]).append(i)
    groups=list(grouped.values())
    generator=np.random.default_rng(seed)
    samples={name:[] for name in predictions}
    for _ in range(repetitions):
        indices=np.concatenate([groups[i] for i in generator.integers(0,len(groups),len(groups))])
        if len(np.unique(y[indices]))<2:
            continue
        for name,risk in predictions.items():
            samples[name].append(float(roc_auc_score(y[indices],risk[indices])))
    baseline=np.array(samples['heuristic_confidence'])
    return {name:{'auroc_image_bootstrap95':np.quantile(values,[.025,.975]).tolist(),
        'paired_delta_vs_heuristic95':np.quantile(np.array(values)-baseline,[.025,.975]).tolist(),
        'bootstrap_samples':len(values)} for name,values in samples.items()}


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    started=time.perf_counter()
    config_path=ROOT/'configs/logistic_baselines.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    data_meta=json.loads((ROOT/'data/vsr/detector_ready/metadata.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True,exist_ok=True)
    model_folder=OUT/'models'
    model_folder.mkdir(exist_ok=True)
    protocol={'config':config,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'data_checksums':{key:data_meta['checksums'][key] for key in ('train.npz','validation.npz')},
        'target':'1 = original VLM answer incorrect; 0 = correct',
        'training_splits_opened':['train','validation'],'test_opened':False,
        'scaler_fit':'train only','model_fit':'train only','hyperparameter_selection':'validation only',
        'threshold_selection':'validation only; target 80% coverage',
        'test_policy':'Reserved for final comparison after MLP/model selection is frozen.',
        'versions':{p:version(p) for p in ('scikit-learn','numpy','scipy','joblib')}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    groups={}
    for cohort in ('all','consistency'):
        trainX,trainY,train_info=load_training_split('train',feature_set='confidence',cohort=cohort)
        valX,valY,val_info=load_training_split('validation',feature_set='confidence',cohort=cohort)
        if set(train_info['image_ids'])&set(val_info['image_ids']):
            raise ValueError('Image overlap between training and validation.')
        heuristic=1-valX[:,1]
        groups[cohort]={'train_y':trainY,'train_ids':train_info['example_ids'],
            'val_y':valY,'val_info':val_info,'predictions':{'heuristic_confidence':heuristic},
            'models':{'heuristic_confidence':{'name':'heuristic_confidence','trained':False,
                **metrics(valY,heuristic),'coverage80':coverage_threshold(valY,heuristic)}}}
    grid_rows=[]
    with threadpool_limits(limits=1):
        for experiment in config['experiments']:
            name=experiment['name']
            kwargs={k:experiment[k] for k in ('feature_set','cohort','layer')}
            X,y,info=load_training_split('train',**kwargs)
            V,v,vi=load_training_split('validation',**kwargs)
            group=groups[experiment['cohort']]
            if not (np.array_equal(info['example_ids'],group['train_ids']) and np.array_equal(y,group['train_y']) and np.array_equal(vi['example_ids'],group['val_info']['example_ids']) and np.array_equal(v,group['val_y'])):
                raise ValueError('Feature-group rows or labels are misaligned.')
            best=None
            for C in config['C_grid']:
                for weight in config['class_weight_grid']:
                    fitted=fit_candidate(X,y,C,weight,config)
                    risk=fitted.predict_proba(V)[:,1]
                    scores=metrics(v,risk)
                    grid_rows.append({'experiment':name,'C':C,'class_weight':str(weight),**scores})
                    key=(scores['auroc'],scores['average_precision'],-C,int(weight is None))
                    if best is None or key>best['key']:
                        best={'key':key,'model':fitted,'risk':risk,'C':C,'class_weight':weight,'metrics':scores}
            path=model_folder/f'{name}.joblib'
            joblib.dump(best['model'],path)
            restored=joblib.load(path)
            if not np.allclose(restored.predict_proba(V)[:,1],best['risk'],rtol=0,atol=1e-12):
                raise ValueError('Saved model predictions changed on reload.')
            selected={'name':name,'trained':True,**kwargs,'input_dimensions':X.shape[1],
                'input_feature_names':({'confidence':['p_original','confidence'],
                    'core':data_meta['core_feature_names'],'behavior':data_meta['behavior_feature_names']}.get(experiment['feature_set'])),
                'uses_hidden_state':experiment['feature_set']=='hidden',
                'training_rows':len(y),'training_errors':int(y.sum()),'validation_rows':len(v),
                'C':best['C'],'class_weight':best['class_weight'],**best['metrics'],
                'coverage80':coverage_threshold(v,best['risk'],config['target_validation_coverage']),
                'model_file':str(path.relative_to(ROOT)),
                'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'converged':True,'scaler_training_rows':int(restored['scaler'].n_samples_seen_),
                'iterations':restored['classifier'].n_iter_.tolist(),
                'risk_note':'Ranking score; especially with class weighting, not assumed calibrated probability.'}
            if X.shape[1]<=7:
                selected['standardized_coefficients']=restored['classifier'].coef_[0].tolist()
                selected['intercept']=restored['classifier'].intercept_.tolist()
            group['models'][name]=selected
            group['predictions'][name]=best['risk']
            (model_folder/f'{name}.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
            print(f'{name}: train={len(y)}, validation={len(v)}, validation AUROC={selected["auroc"]:.3f}, AP={selected["average_precision"]:.3f}, C={best["C"]}, weights={best["class_weight"]}',flush=True)
    write_csv(OUT/'hyperparameter_search.csv',grid_rows)
    summary={'stage':'Training and validation only; test reserved.',
        'models_trained':len(config['experiments']),'candidate_fits':len(grid_rows),'test_opened':False,
        'interval_note':'Image-bootstrap intervals condition on the selected models. Validation also selected hyperparameters, so intervals do not remove selection optimism.',
        'cohorts':{}}
    prediction_rows=[]
    metric_rows=[]
    for cohort,group in groups.items():
        intervals=bootstrap_intervals(group['val_y'],group['predictions'],group['val_info']['image_ids'],config['bootstrap_repetitions'],config['seed'])
        for name,item in group['models'].items():
            item.update(intervals[name])
            metric_rows.append({'cohort':cohort,'model':name,'AUROC':item['auroc'],'average_precision':item['average_precision'],
                'AUROC_low95':item['auroc_image_bootstrap95'][0],'AUROC_high95':item['auroc_image_bootstrap95'][1],
                'C':item.get('C',''),'class_weight':str(item.get('class_weight','')),
                'validation_coverage':item['coverage80']['coverage'],'retained_vlm_accuracy':item['coverage80']['retained_vlm_accuracy']})
            for i,risk in enumerate(group['predictions'][name]):
                prediction_rows.append({'cohort':cohort,'model':name,'example_id':group['val_info']['example_ids'][i],
                    'image_id':group['val_info']['image_ids'][i],'y_error':int(group['val_y'][i]),'risk':float(risk)})
        summary['cohorts'][cohort]={'training_rows':len(group['train_y']),'validation_rows':len(group['val_y']),
            'validation_errors':int(group['val_y'].sum()),'validation_error_prevalence':float(group['val_y'].mean()),
            'unfiltered_vlm_accuracy':float(1-group['val_y'].mean()),'models':group['models']}
    summary['wall_seconds']=round(time.perf_counter()-started,3)
    write_csv(OUT/'validation_metrics.csv',metric_rows)
    write_csv(OUT/'validation_predictions.csv',prediction_rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'models_trained':summary['models_trained'],'candidate_fits':summary['candidate_fits'],
        'test_opened':False,'wall_seconds':summary['wall_seconds']},indent=2),flush=True)


if __name__=='__main__':
    main()

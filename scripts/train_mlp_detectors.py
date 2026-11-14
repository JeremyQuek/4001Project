"""Train MLP error detectors with train-only scaling and validation-only selection."""
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from train_logistic_baselines import load_training_split,metrics,coverage_threshold,write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mlp_comparison'


class ErrorMLP(torch.nn.Module):
    def __init__(self,input_dim,hidden_units=32,dropout=.2):
        super().__init__()
        self.network=torch.nn.Sequential(torch.nn.Linear(input_dim,hidden_units),torch.nn.ReLU(),
            torch.nn.Dropout(dropout),torch.nn.Linear(hidden_units,1))

    def forward(self,X):
        return self.network(X).squeeze(-1)


def risk_scores(model,X):
    model.eval()
    with torch.inference_mode():
        values=torch.sigmoid(model(torch.as_tensor(np.asarray(X,dtype=np.float32)))).numpy()
    if not np.isfinite(values).all():
        raise ValueError('Invalid MLP risk scores.')
    return values


def fit_seed(X,y,V,v,config,seed,weight_decay,class_weight):
    torch.manual_seed(seed)
    model=ErrorMLP(X.shape[1],config['hidden_units'],config['dropout'])
    positive_weight=float((y==0).sum()/(y==1).sum()) if class_weight=='balanced' else 1.
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=weight_decay)
    trainX=torch.as_tensor(np.asarray(X,dtype=np.float32))
    trainY=torch.as_tensor(y,dtype=torch.float32)
    valX=torch.as_tensor(np.asarray(V,dtype=np.float32))
    valY=torch.as_tensor(v,dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed)
    best_loss=float('inf')
    best_state=None
    best_epoch=0
    stale=0
    history=[]
    for epoch in range(1,config['max_epochs']+1):
        model.train()
        order=torch.randperm(len(trainY),generator=generator)
        total=0.
        for start in range(0,len(order),config['batch_size']):
            indices=order[start:start+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(trainX[indices]),trainY[indices])
            if not torch.isfinite(loss).item():
                raise ValueError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(indices)
        model.eval()
        with torch.inference_mode():
            val_loss=float(loss_fn(model(valX),valY))
        if not np.isfinite(val_loss):
            raise ValueError('Non-finite validation loss.')
        history.append({'epoch':epoch,'train_bce':total/len(trainY),'validation_bce':val_loss})
        if val_loss<best_loss-config['min_delta']:
            best_loss=val_loss
            best_state=copy.deepcopy(model.state_dict())
            best_epoch=epoch
            stale=0
        else:
            stale+=1
        if stale>=config['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    train_risk=risk_scores(model,X)
    val_risk=risk_scores(model,V)
    return {'model':model,'seed':seed,'weight_decay':weight_decay,'class_weight':class_weight,
        'positive_weight':positive_weight,'best_epoch':best_epoch,'epochs_run':len(history),
        'best_validation_bce':best_loss,'history':history,
        'training_metrics':metrics(y,train_risk),'validation_metrics':metrics(v,val_risk),
        'validation_risk':val_risk}


def load_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=ErrorMLP(saved['input_dim'],saved['hidden_units'],saved['dropout'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model,saved


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    cfgpath=ROOT/'configs/mlp_comparison.json'
    config=json.loads(cfgpath.read_text(encoding='utf-8'))
    lr_summary=json.loads((ROOT/'outputs/logistic_baselines/summary.json').read_text(encoding='utf-8'))
    lr_protocol=json.loads((ROOT/'outputs/logistic_baselines/protocol.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/'data/vsr/detector_ready/metadata.json').read_text(encoding='utf-8'))
    if any(data['checksums'][name]!=value for name,value in lr_protocol['data_checksums'].items()):
        raise ValueError('Training/validation arrays differ from LR baseline inputs.')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'test_evaluation_started.json').exists():
        raise ValueError('Test evaluation has started. Do not retrain/tune this comparison against the exposed test set.')
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'histories').mkdir(exist_ok=True)
    protocol={'config':config,'config_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
        'experiments':lr_protocol['config']['experiments'],'data_checksums':lr_protocol['data_checksums'],
        'fit_splits':['train'],'selection_split':'validation','test_opened':False,
        'device':'CPU','torch_version':torch.__version__,'numerical_threads':1,'ensemble':False}
    (OUT/'training_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    selected={}
    candidates=[]
    for experiment in protocol['experiments']:
        name=experiment['name']
        kwargs={k:experiment[k] for k in ('feature_set','cohort','layer')}
        X,y,info=load_training_split('train',**kwargs)
        V,v,vi=load_training_split('validation',**kwargs)
        if set(info['image_ids'])&set(vi['image_ids']):
            raise ValueError('Training/validation image overlap.')
        baseline=lr_summary['cohorts'][experiment['cohort']]['models'][name]
        if baseline['training_rows']!=len(y) or baseline['validation_rows']!=len(v):
            raise ValueError('MLP and LR cohorts do not match.')
        scaler=StandardScaler().fit(X)
        if scaler.n_samples_seen_!=len(X) or not np.allclose(scaler.mean_,X.mean(axis=0,dtype=np.float64)):
            raise ValueError('Incorrect scaler fitting boundary.')
        train_scaled=scaler.transform(X).astype(np.float32)
        val_scaled=scaler.transform(V).astype(np.float32)
        best=None
        for decay,weight in itertools.product(config['weight_decay_grid'],config['class_weight_grid']):
            runs=[fit_seed(train_scaled,y,val_scaled,v,config,seed,decay,weight) for seed in config['seeds']]
            auc=float(np.mean([r['validation_metrics']['auroc'] for r in runs]))
            ap=float(np.mean([r['validation_metrics']['average_precision'] for r in runs]))
            row={'experiment':name,'weight_decay':decay,'class_weight':str(weight),'mean_validation_auroc':auc,
                'std_validation_auroc':float(np.std([r['validation_metrics']['auroc'] for r in runs],ddof=1)),
                'mean_validation_average_precision':ap}
            for run in runs:
                row[f'seed{run["seed"]}_auroc']=run['validation_metrics']['auroc']
                row[f'seed{run["seed"]}_best_epoch']=run['best_epoch']
            candidates.append(row)
            key=(auc,ap,decay,int(weight is None))
            if best is None or key>best['key']:
                best={'key':key,'runs':runs,'config_metrics':row,'weight_decay':decay,'class_weight':weight}
        scaler_path=OUT/'models'/f'{name}_scaler.joblib'
        joblib.dump(scaler,scaler_path)
        entry={**experiment,'input_dim':X.shape[1],'train_rows':len(y),'validation_rows':len(v),
            'selected_weight_decay':best['weight_decay'],'selected_class_weight':best['class_weight'],
            'selection_metrics':best['config_metrics'],'scaler_file':str(scaler_path.relative_to(ROOT)),
            'scaler_sha256':hashlib.sha256(scaler_path.read_bytes()).hexdigest(),'seeds':{}}
        for run in best['runs']:
            seed=run['seed']
            path=OUT/'models'/f'{name}_seed{seed}.pt'
            checkpoint={'state_dict':run['model'].state_dict(),'input_dim':X.shape[1],
                'hidden_units':config['hidden_units'],'dropout':config['dropout'],'seed':seed,
                'best_epoch':run['best_epoch'],'weight_decay':best['weight_decay'],'class_weight':best['class_weight']}
            torch.save(checkpoint,path)
            restored,_=load_checkpoint(path)
            if not np.array_equal(risk_scores(restored,val_scaled),run['validation_risk']):
                raise ValueError('MLP predictions changed after save/reload.')
            history_path=OUT/'histories'/f'{name}_seed{seed}.json'
            history_path.write_text(json.dumps(run['history']),encoding='utf-8')
            entry['seeds'][str(seed)]={'checkpoint':str(path.relative_to(ROOT)),
                'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'best_epoch':run['best_epoch'],'epochs_run':run['epochs_run'],
                'training_metrics':run['training_metrics'],'validation_metrics':run['validation_metrics'],
                'coverage80':coverage_threshold(v,run['validation_risk'],config['target_validation_coverage'])}
        selected[name]=entry
        print(f'{name}: mean validation AUROC={entry["selection_metrics"]["mean_validation_auroc"]:.3f} '
              f'+/- {entry["selection_metrics"]["std_validation_auroc"]:.3f}; decay={best["weight_decay"]}; weights={best["class_weight"]}',flush=True)
    write_csv(OUT/'candidate_validation_results.csv',candidates)
    result={'models':selected,'config':config,'experiments':protocol['experiments'],
        'candidate_configurations':len(candidates),'seed_fits':len(candidates)*len(config['seeds']),
        'primary_seed':config['primary_seed'],'test_opened':False,'training_wall_seconds':time.perf_counter()-started}
    (OUT/'validation_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    freeze={'config_sha256':protocol['config_sha256'],'primary_seed':config['primary_seed'],
        'primary_experiments':config['primary_experiments'],'selection':result,
        'training_protocol_sha256':hashlib.sha256((OUT/'training_protocol.json').read_bytes()).hexdigest(),
        'lr_summary_sha256':hashlib.sha256((ROOT/'outputs/logistic_baselines/summary.json').read_bytes()).hexdigest(),
        'test_npz_sha256':data['checksums']['test.npz'],'test_opened':False}
    (OUT/'frozen_comparison.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    print(json.dumps({'seed_fits':result['seed_fits'],'training_wall_seconds':result['training_wall_seconds'],
        'frozen_before_test':True,'test_opened':False}),flush=True)


if __name__=='__main__':
    main()

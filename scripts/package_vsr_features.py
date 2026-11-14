"""Validate every observation and package raw arrays ready for detector training."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'outputs/vsr_full'
DEST=ROOT/'data/vsr/detector_ready'
LAYERS=[9,18,36]
CORE=['p_original','confidence','p_blind','image_sensitivity']
BEHAVIOR=CORE+['converse_disagreement','exclusive_violation','mirror_disagreement']


def arrays_for_split(rows):
    n=len(rows)
    core=np.empty((n,4),dtype=np.float32)
    behavior=np.full((n,7),np.nan,dtype=np.float32)
    hidden={layer:np.empty((n,2048),dtype=np.float32) for layer in LAYERS}
    for i,row in enumerate(rows):
        scores=row['scores']
        p=scores['original']['p_yes']
        blind=scores['blind']['p_yes']
        expected=[p,max(p,1-p),blind,abs(p-blind)]
        for name,value in zip(CORE,expected):
            if not math.isclose(row['features'][name],value,abs_tol=1e-7):
                raise ValueError('Core feature mismatch: '+row['example_id'])
        core[i]=expected
        behavior[i,:4]=expected
        if row['consistency_eligible']:
            extra=[abs(p-scores['converse']['p_yes']),max(0.,p+scores['exclusive']['p_yes']-1),abs(p-scores['mirrored_equivalent']['p_yes'])]
            behavior[i,4:]=extra
            for name,value in zip(BEHAVIOR[4:],extra):
                if not math.isclose(row['features'][name],value,abs_tol=1e-7):
                    raise ValueError('Consistency feature mismatch.')
        expected_modes={'original','blind'}|({'converse','exclusive','mirrored_equivalent'} if row['consistency_eligible'] else set())
        if set(scores)!=expected_modes:
            raise ValueError('Missing or unexpected scoring modes.')
        for mode,score in scores.items():
            expected_statement=row['statement'] if mode in ('original','blind') else row['variants'][mode]
            if score['statement']!=expected_statement:
                raise ValueError('Score belongs to another statement.')
            if not all(math.isfinite(score[k]) and 0<=score[k]<=1 for k in ('p_yes','p_no')) or abs(score['p_yes']+score['p_no']-1)>1e-5:
                raise ValueError('Invalid binary probability.')
        if row['prediction']!=int(p>=.5) or row['error_label']!=int(row['prediction']!=row['truth_label']):
            raise ValueError('Incorrect supervised error label.')
        saved=torch.load(ROOT/row['hidden_file'],map_location='cpu',weights_only=True)
        if saved.get('hidden_state_indices')!=LAYERS or saved.get('position')!='final prompt token, before answering':
            raise ValueError('Incorrect hidden-state alignment or layer indices.')
        for layer in LAYERS:
            vector=saved['vectors'][layer].float().numpy()
            if vector.shape!=(2048,) or not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError('Invalid hidden state.')
            hidden[layer][i]=vector
    result={'example_ids':np.array([r['example_id'] for r in rows]),
        'image_ids':np.array([r['image_id'] for r in rows]),
        'relations':np.array([r['relation'] for r in rows]),
        'truth_label':np.array([r['truth_label'] for r in rows],dtype=np.int8),
        'vlm_prediction':np.array([r['prediction'] for r in rows],dtype=np.int8),
        'y_error':np.array([r['error_label'] for r in rows],dtype=np.int8),
        'consistency_mask':np.array([r['consistency_eligible'] for r in rows],dtype=bool),
        'X_confidence':core[:,:2].copy(),'X_core':core,'X_behavior':behavior,
        **{f'H_layer{layer}':hidden[layer] for layer in LAYERS}}
    if not np.isfinite(core).all() or not np.isfinite(behavior[result['consistency_mask']]).all():
        raise ValueError('Non-finite usable features.')
    if not np.isnan(behavior[~result['consistency_mask'],4:]).all():
        raise ValueError('Unavailable consistency features must be explicit NaN.')
    return result


def main():
    manifest=ROOT/'data/vsr/extraction/manifest.jsonl'
    approval=json.loads((manifest.parent/'batch_approval.json').read_text(encoding='utf-8'))
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=approval['manifest_sha256']:
        raise ValueError('Approval fingerprint mismatch.')
    signature=json.loads((RAW/'signature.json').read_text(encoding='utf-8'))
    if signature['manifest_sha256']!=approval['manifest_sha256'] or signature['config']!=approval['config'] or signature['hidden_layers']!=LAYERS:
        raise ValueError('Extraction signature differs from approved inputs.')
    inputs=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    source={r['example_id']:r for r in map(json.loads,(ROOT/'data/vsr/processed/directional_candidates.jsonl').read_text(encoding='utf-8').splitlines())}
    records=[]
    for expected in inputs:
        path=RAW/'records'/f'{expected["example_id"]}.json'
        if not path.exists():
            raise ValueError('Extraction is incomplete: '+expected['example_id'])
        record=json.loads(path.read_text(encoding='utf-8'))
        if any(record[key]!=expected[key] for key in expected):
            raise ValueError('Cached row does not match approved manifest.')
        if source[record['example_id']]['truth_label']!=record['truth_label']:
            raise ValueError('Source truth label changed.')
        records.append(record)
    if len(records)!=941 or len({r['example_id'] for r in records})!=941:
        raise ValueError('Unexpected row count or duplicate examples.')
    sets=[{r['image_id'] for r in records if r['split']==split} for split in ('train','validation','test')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Image leakage across splits.')
    for image_id in {r['image_id'] for r in records}:
        row=next(r for r in records if r['image_id']==image_id)
        if hashlib.sha256((ROOT/row['image_path']).read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError('Source image changed.')
    DEST.mkdir(parents=True,exist_ok=True)
    splits={}
    for split,n in [('train',659),('validation',148),('test',134)]:
        rows=[r for r in records if r['split']==split]
        if len(rows)!=n:
            raise ValueError('Split count changed.')
        arrays=arrays_for_split(rows)
        np.savez_compressed(DEST/f'{split}.npz',**arrays)
        with np.load(DEST/f'{split}.npz',allow_pickle=False) as reloaded:
            for key,value in arrays.items():
                equal=np.array_equal(reloaded[key],value,equal_nan=True) if value.dtype.kind in 'fc' else np.array_equal(reloaded[key],value)
                if not equal:
                    raise ValueError('Array round-trip mismatch: '+key)
        columns=['example_id','image_id','split','relation','truth_label','prediction','error_label','consistency_eligible']+BEHAVIOR
        with (DEST/f'{split}_features.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns)
            writer.writeheader()
            for row in rows:
                flat={k:row[k] for k in columns if k not in BEHAVIOR}
                flat.update({k:row['features'].get(k,'') for k in BEHAVIOR})
                writer.writerow(flat)
        index=[{'row':i,'example_id':r['example_id'],'image_id':r['image_id'],'statement':r['statement'],
                'consistency_basis':r['consistency_basis']} for i,r in enumerate(rows)]
        (DEST/f'{split}_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
        splits[split]={'rows':n,'images':len({r['image_id'] for r in rows}),
            'consistency_rows':int(arrays['consistency_mask'].sum()),
            'core_shape':list(arrays['X_core'].shape),'hidden_shape_per_layer':list(arrays['H_layer18'].shape),
            'both_error_classes_present':len(np.unique(arrays['y_error']))==2}
        if split=='train':
            splits[split]['error_label_counts']={str(k):int((arrays['y_error']==k).sum()) for k in (0,1)}
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.suffix in ('.npz','.csv') or p.name.endswith('_index.json')}
    metadata={'ready_for_training':True,'total_rows':len(records),'total_images':len(set.union(*sets)),
        'splits':splits,'core_feature_names':CORE,'behavior_feature_names':BEHAVIOR,'hidden_layers':LAYERS,
        'hidden_size':2048,'hidden_position':'final prompt token, before answering',
        'target':'y_error = int(vlm_prediction != truth_label); 1 means incorrect',
        'truth_labels_changed':0,'preprocessing_fitted':False,'detector_trained':False,
        'consistency_policy':approval['transform_policy'],'manifest_sha256':approval['manifest_sha256'],
        'config':approval['config'],'checksums':checksums}
    (DEST/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    verification={'passed':True,'source_rows_retained':941,'duplicates':0,'image_overlap_between_splits':0,
        'hidden_vectors_verified':941*3,'hidden_values_finite':True,'all_core_features_finite':True,
        'consistency_missingness_matches_mask':True,'probability_checks_passed':True,
        'label_formula_checked':True,'source_labels_preserved':True,'npz_roundtrip_checked':True,
        'training_readiness':'All required files complete. No scaler, detector or threshold fitted.'}
    (DEST/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    card=['# Detector training data', '',
        'Raw inputs ready for detector training. All 941 original VSR statements and their source labels are retained. '
        'This array bundle contains no fitted preprocessing or models; training-run artifacts and status are stored separately.', '',
        '## Dataset and targets', '',
        '| Split | Rows | Images | Consistency-eligible rows |', '|---|---:|---:|---:|']
    for split,info in splits.items():
        card.append(f'| {split} | {info["rows"]} | {info["images"]} | {info["consistency_rows"]} |')
    card += ['',f'Training targets: {splits["train"]["error_label_counts"]["0"]} correct answers (0), '
        f'{splits["train"]["error_label_counts"]["1"]} incorrect answers (1). Validation/test accuracy is not analysed here.', '',
        '`y_error = int(vlm_prediction != truth_label)`. Original predictions are Yes when P(Yes) >= 0.5. '
        'Source truth labels and original predictions are metadata, never part of X.', '',
        '## Arrays in each split NPZ', '',
        '| Key | Dimensions | Meaning |','|---|---|---|',
        '| X_confidence | N × 2 | p_original, confidence |',
        '| X_core | N × 4 | p_original, confidence, p_blind, image_sensitivity |',
        '| X_behavior | N × 7 | Core plus converse disagreement, exclusive violation, mirror disagreement |',
        '| H_layer9 / H_layer18 / H_layer36 | N × 2048 each | Original-prompt hidden state at each selected decoder layer |',
        '| y_error | N | Target: 1 = original answer incorrect |',
        '| consistency_mask | N | Which rows support the 7-feature comparison |',
        '| example_ids / image_ids / relations | N | Alignment and metadata |', '',
        'The hidden state is saved at the final prompt token before answering, for every image–statement pair. '
        'Arrays are float32 conversions of captured bfloat16 vectors. No feature standardization has been fitted.', '',
        '## Consistency eligibility', '',
        approval['transform_policy'], '',
        'This gives 618 consistency-eligible rows (vertical relations), while all 941 rows support core and hidden-state detectors. '
        'Unavailable auxiliary columns are NaN, not zero. Use consistency_mask or the loader below; do not feed the full NaN-containing X_behavior array directly to a classifier. '
        'No earlier agent quality flag removes an original row. Exclusive statements have no automatically assigned truth label.', '',
        '## Load data without training', '',
        'From the project root:', '',
        '```python',
        'import sys',
        'sys.path.insert(0, "scripts")',
        'from load_detector_data import load_detector_data', '',
        'X_train, y_train, info = load_detector_data("train", "core")',
        'H_train, y_hidden, _ = load_detector_data("train", "hidden", layer=18)', '',
        '# Compare consistency features and confidence on exactly the same rows:',
        'B_train, y_behavior, b_info = load_detector_data(',
        '    "train", "behavior", cohort="consistency")',
        'C_train, y_confidence, c_info = load_detector_data(',
        '    "train", "confidence", cohort="consistency")',
        'assert (b_info["example_ids"] == c_info["example_ids"]).all()',
        '```', '',
        'Feature sets: confidence (2), core (4), behavior (7), hidden (2048), '
        'combined_core (2052), combined_behavior (2055). The last combines seven features and one chosen layer, and requires cohort="consistency". '
        'Use cohort="consistency" for every baseline in that ablation, including hidden-only comparisons.', '',
        'Fit StandardScaler and logistic regression using training data only. Choose regularization, hidden layer, feature groups '
        'and abstention threshold with validation data. Keep the test split for the final detector evaluation. '
        'The files preserve image separation and metadata row order; index JSON files link rows to original statements.', '',
        '## Reproduce / resume', '',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts\\extract_vsr_features.py',
        '.\\.venv\\Scripts\\python.exe scripts\\package_vsr_features.py',
        '.\\.venv\\Scripts\\python.exe scripts\\load_detector_data.py',
        '```', '',
        'Raw per-pair observations and individual hidden-state files are in outputs/vsr_full/. '
        'The signature pins model revision, processing settings, package versions and the scoring function. '
        'Complete cached records are validated before resumption. metadata.json lists artifact checksums; '
        'verification.json records the final integrity checks.', '']
    (DEST/'DATASET_CARD.md').write_text('\n'.join(card),encoding='utf-8')
    print(json.dumps({'ready_for_training':True,'splits':splits,'verified_hidden_vectors':941*3},indent=2))


if __name__=='__main__':
    main()

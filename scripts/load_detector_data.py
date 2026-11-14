"""Load aligned raw detector inputs and error targets; performs no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_FOLDER=ROOT/'data/vsr/detector_ready'
FEATURE_SETS={'confidence','core','behavior','hidden','combined_core','combined_behavior'}


def load_detector_data(split='train',feature_set='core',layer=18,cohort='all',folder=DEFAULT_FOLDER):
    """Return X, y_error, metadata. Consistency comparisons require cohort='consistency'."""
    if split not in ('train','validation','test') or feature_set not in FEATURE_SETS:
        raise ValueError('Unknown split or feature set.')
    if cohort not in ('all','consistency') or layer not in (9,18,36):
        raise ValueError('Unknown cohort or hidden layer.')
    if feature_set in ('behavior','combined_behavior') and cohort!='consistency':
        raise ValueError("Use cohort='consistency' for behavior or combined_behavior. Compare baselines on that same cohort.")
    folder=Path(folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    path=folder/f'{split}.npz'
    if not metadata.get('ready_for_training') or hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['checksums'][path.name]:
        raise ValueError('Dataset incomplete or artifact checksum changed.')
    with np.load(path,allow_pickle=False) as data:
        mask=data['consistency_mask'] if cohort=='consistency' else np.ones(len(data['y_error']),dtype=bool)
        if feature_set=='confidence':
            X=data['X_confidence'][mask]
        elif feature_set in ('core','behavior'):
            X=data['X_'+feature_set][mask]
        elif feature_set=='hidden':
            X=data[f'H_layer{layer}'][mask]
        else:
            key='X_core' if feature_set=='combined_core' else 'X_behavior'
            X=np.concatenate((data[key][mask],data[f'H_layer{layer}'][mask]),axis=1)
        y=data['y_error'][mask]
        info={'example_ids':data['example_ids'][mask],'image_ids':data['image_ids'][mask],
              'relations':data['relations'][mask],'split':split,'feature_set':feature_set,'cohort':cohort,'layer':layer}
    if len(X)!=len(y) or not np.isfinite(X).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid training arrays.')
    return X,y,info


if __name__=='__main__':
    X,y,_=load_detector_data()
    print(f'Training core inputs: {X.shape}; targets: {y.shape}; errors: {int(y.sum())}. No model trained.')

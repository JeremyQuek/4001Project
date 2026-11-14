"""Verify saved research artifacts without loading the VLM or fitting models."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-index',action='store_true',help='Also require identical staged bytes for frozen files.')
    args=parser.parse_args()
    checked={}

    def verify(relative,expected):
        path=ROOT/relative
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError(f'Checksum mismatch: {relative}')
        checked[path.relative_to(ROOT).as_posix()]=content

    data=read('data/vsr/detector_ready/metadata.json')
    for name,digest in data['checksums'].items():
        verify('data/vsr/detector_ready/'+name,digest)
    for source in read('data/vsr/raw/sources.json')['files']:
        filename='dev' if source['split']=='dev' else source['split']
        verify('data/vsr/raw/'+filename+'.jsonl',source['sha256'])
    approval=read('data/vsr/extraction/batch_approval.json')
    verify('data/vsr/extraction/manifest.jsonl',approval['manifest_sha256'])
    verify('data/vsr/processed/directional_candidates.jsonl',approval['source_sha256'])
    lr=read('outputs/logistic_baselines/summary.json')
    for group in lr['cohorts'].values():
        for model in group['models'].values():
            if model.get('trained'):
                verify(model['model_file'],model['model_sha256'])
    freeze=read('outputs/mlp_comparison/frozen_comparison.json')
    verify('configs/mlp_comparison.json',freeze['config_sha256'])
    verify('outputs/mlp_comparison/training_protocol.json',freeze['training_protocol_sha256'])
    verify('outputs/logistic_baselines/summary.json',freeze['lr_summary_sha256'])
    verify('data/vsr/detector_ready/test.npz',freeze['test_npz_sha256'])
    if read('outputs/mlp_comparison/validation_selection.json')!=freeze['selection']:
        raise ValueError('MLP selection differs from the pre-test frozen selection.')
    for model in freeze['selection']['models'].values():
        verify(model['scaler_file'],model['scaler_sha256'])
        for seed in model['seeds'].values():
            verify(seed['checkpoint'],seed['checkpoint_sha256'])
    final=read('outputs/mlp_comparison/test_summary.json')
    protocol=final['protocol']
    verify('outputs/mlp_comparison/frozen_comparison.json',protocol['frozen_comparison_sha256'])
    verify('scripts/evaluate_detector_comparison.py',protocol['evaluator_sha256'])
    verify('outputs/mlp_comparison/test_predictions.csv',final['test_predictions_sha256'])
    if read('outputs/mlp_comparison/test_evaluation_started.json')['protocol']!=protocol:
        raise ValueError('Test exposure marker and final evaluation differ.')
    if args.git_index:
        raw=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT)
        index={}
        for item in raw.split(b'\0'):
            if not item:
                continue
            meta,name=item.split(b'\t',1)
            mode,digest,stage=meta.decode().split()
            if stage!='0':
                raise ValueError('Git index contains an unresolved merge.')
            index[name.decode('utf-8')]=digest
        for name,content in checked.items():
            blob=b'blob '+str(len(content)).encode()+b'\0'+content
            if index.get(name)!=hashlib.sha1(blob).hexdigest():
                raise ValueError(f'Staged bytes differ or artifact missing: {name}')
        forbidden=[name for name in index if name.startswith(('.venv/','.cache/','data/vsr/images/'))]
        if forbidden:
            raise ValueError('Git index unexpectedly contains local environment, weights or original photos.')
    print(f'PASS: {len(checked)} unique frozen artifact checksums; source metadata, dataset, LR/MLP models, test protocol and predictions unchanged.')
    if args.git_index:
        print('PASS: frozen artifacts are staged byte-for-byte; local environment, VLM cache and original photos are excluded.')


if __name__=='__main__':
    main()

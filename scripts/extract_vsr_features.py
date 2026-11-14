"""Resumable frozen-VLM extraction; does not train or evaluate a detector."""
import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from PIL import Image,ImageOps
from run_vlm import load_vlm,score
from run_vsr_pilot import features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vsr_full'
LAYERS=[9,18,36]


def atomic_json(path,value):
    temp=path.with_suffix('.json.tmp')
    temp.write_bytes(json.dumps(value).encode())
    # Windows readers and antivirus can briefly deny replacement of a file.
    for attempt in range(12):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==11:
                raise
            time.sleep(min(.05*2**attempt,.8))


def validate_hidden(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('hidden_state_indices')!=LAYERS or saved.get('position')!='final prompt token, before answering':
        raise ValueError('Unexpected hidden state definition: '+str(path))
    for layer in LAYERS:
        value=saved['vectors'][layer]
        if tuple(value.shape)!=(2048,) or not torch.isfinite(value).all().item():
            raise ValueError('Invalid hidden vector: '+str(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-records',type=int,default=0,help='Optional bounded run; default processes all remaining records.')
    args=parser.parse_args()
    source=ROOT/'data/vsr/extraction/manifest.jsonl'
    content=source.read_bytes()
    rows=[json.loads(line) for line in content.decode().splitlines()]
    approval=json.loads((source.parent/'batch_approval.json').read_text(encoding='utf-8'))
    if hashlib.sha256(content).hexdigest()!=approval['manifest_sha256']:
        raise ValueError('Manifest differs from batch approval.')
    cfg=approval['config']
    signature={'manifest_sha256':approval['manifest_sha256'],'config':cfg,
        'score_source_sha256':hashlib.sha256(inspect.getsource(score).encode()).hexdigest(),
        'hidden_layers':LAYERS,'prompt_version':'run_vlm_statement_yes_no_v1',
        'packages':{p:version(p) for p in ('torch','torchvision','transformers','accelerate','bitsandbytes','Pillow','huggingface-hub')}}
    for folder in (OUT,OUT/'records',OUT/'hidden',OUT/'sessions'):
        folder.mkdir(parents=True,exist_ok=True)
    sigpath=OUT/'signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text(encoding='utf-8'))!=signature:
        raise ValueError('Existing extraction uses a different configuration.')
    atomic_json(sigpath,signature)
    pilot_dir=ROOT/'outputs/vsr_pilot_256'
    pilot_run=json.loads((pilot_dir/'run.json').read_text(encoding='utf-8'))
    pilot_signature=pilot_run['signature']
    pilot_ok=all(pilot_run.get(k)==cfg.get(k) for k in ('model_id','model_revision','quantization','compute_dtype','attention','batch_size','max_pixels','processor_use_fast','min_pixels'))
    pilot_ok &= pilot_signature['prompt_version']==signature['prompt_version'] and pilot_signature['save_hidden']
    pilot_ok &= all(pilot_run['packages'][p]==signature['packages'][p] for p in pilot_run['packages'])
    pilot={r['example_id']:r for r in map(json.loads,(pilot_dir/'results.jsonl').read_text(encoding='utf-8').splitlines())} if pilot_ok else {}
    old_images={r['image_id']:r['sha256'] for r in json.loads((ROOT/'outputs/pilot_image_download.json').read_text(encoding='utf-8'))['images']}
    done=[]
    for row in rows:
        path=OUT/'records'/f'{row["example_id"]}.json'
        if path.exists():
            result=json.loads(path.read_text(encoding='utf-8'))
            if result['image_sha256']!=row['image_sha256'] or result['statement']!=row['statement'] or result['truth_label']!=row['truth_label'] or result['consistency_eligible']!=row['consistency_eligible']:
                raise ValueError('Cached record mismatch.')
            validate_hidden(OUT/'hidden'/f'{row["example_id"]}.pt')
            done.append(row['example_id'])
    if len(done)==len(rows):
        print('All 941 records complete; extraction cache verified. No GPU work needed.',flush=True)
        return
    done=set(done)
    started=time.perf_counter()
    model=processor=None
    metadata={}
    new_records=new_calls=reused_calls=reused_hidden=0
    def ensure_model():
        nonlocal model,processor,metadata
        if model is None:
            model,processor,metadata=load_vlm(cfg['model_revision'],cfg['max_pixels'])
            if any(metadata[k]!=cfg[k] for k in ('model_id','model_revision','quantization','compute_dtype','attention','batch_size','max_pixels','processor_use_fast','min_pixels')):
                raise ValueError('Loaded model differs from frozen configuration.')
    def progress(split,batch=None):
        value={'completed':len(done),'total':len(rows),'current_split':split,'new_records':new_records,
            'new_calls':new_calls,'reused_calls':reused_calls,'reused_hidden':reused_hidden,
            'elapsed_seconds':round(time.perf_counter()-started,1),'complete':len(done)==len(rows)}
        atomic_json(OUT/'progress.json',value)
        if batch is not None:
            print(f'Batch {batch}/11 | {split} | total {len(done)}/941 | {new_calls} new calls, {reused_calls} reused | {value["elapsed_seconds"]}s',flush=True)
    plan=json.loads((source.parent/'batch_plan.json').read_text(encoding='utf-8'))
    byid={r['example_id']:r for r in rows}
    for batch in plan:
        for example_id in batch['example_ids']:
            if example_id in done:
                continue
            row=byid[example_id]
            image_path=ROOT/row['image_path']
            if hashlib.sha256(image_path.read_bytes()).hexdigest()!=row['image_sha256']:
                raise ValueError('Image changed: '+row['image_id'])
            with Image.open(image_path) as source_image:
                image=ImageOps.exif_transpose(source_image).convert('RGB')
            hidden_path=OUT/'hidden'/f'{example_id}.pt'
            cached=pilot.get(example_id)
            if cached and (cached['statement']!=row['statement'] or old_images.get(row['image_id'])!=row['image_sha256']):
                cached=None
            scores={}
            modes=['original','blind']+(['converse','exclusive','mirrored_equivalent'] if row['consistency_eligible'] else [])
            for mode in modes:
                statement=row['statement'] if mode in ('original','blind') else row['variants'][mode]
                cached_score=cached.get('scores',{}).get(mode) if cached else None
                if cached_score and cached_score['statement']==statement:
                    if mode=='original':
                        old_path=Path(cached_score['hidden_states_file'])
                        validate_hidden(old_path)
                        shutil.copyfile(old_path,hidden_path)
                        reused_hidden+=1
                    result=dict(cached_score)
                    reused_calls+=1
                else:
                    ensure_model()
                    variant_image=ImageOps.mirror(image) if mode=='mirrored_equivalent' else Image.new('RGB',image.size,(128,128,128)) if mode=='blind' else image
                    result=score(model,processor,variant_image,statement,hidden_path if mode=='original' else None)
                    new_calls+=1
                if mode=='original':
                    result['hidden_states_file']=str(hidden_path)
                if not (math.isfinite(result['p_yes']) and 0<=result['p_yes']<=1 and abs(result['p_yes']+result['p_no']-1)<1e-5):
                    raise ValueError('Invalid score.')
                scores[mode]=result
            validate_hidden(hidden_path)
            prediction=int(scores['original']['p_yes']>=.5)
            record=dict(row,scores=scores,features=features(scores),prediction=prediction,
                error_label=int(prediction!=row['truth_label']),hidden_file=str(hidden_path.relative_to(ROOT)))
            atomic_json(OUT/'records'/f'{example_id}.json',record)
            done.add(example_id)
            new_records+=1
            progress(row['split'])
            if new_records%25==0:
                print(f'Progress {len(done)}/941 | {row["split"]} | {round(time.perf_counter()-started)}s',flush=True)
            if args.max_new_records and new_records>=args.max_new_records:
                break
        progress(batch['split'],batch['batch'])
        if args.max_new_records and new_records>=args.max_new_records:
            break
    session={'started_unix':time.time()-(time.perf_counter()-started),'wall_seconds':round(time.perf_counter()-started,3),
        'new_records':new_records,'new_calls':new_calls,'reused_calls':reused_calls,'reused_hidden':reused_hidden,
        'complete':len(done)==len(rows),'completed':len(done),'model':metadata}
    atomic_json(OUT/'sessions'/f'{time.time_ns()}.json',session)
    print(json.dumps(session),flush=True)


if __name__=='__main__':
    main()

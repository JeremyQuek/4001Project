"""Record user batch approval and freeze non-visual extraction eligibility."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_vsr_audit import variants

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'data/vsr/processed/directional_candidates.jsonl'
    source_bytes = source.read_bytes()
    cfg = json.loads((ROOT/'configs/vsr_experiment.json').read_text(encoding='utf-8'))
    image_report = json.loads((ROOT/'outputs/directional_image_download.json').read_text(encoding='utf-8'))
    if image_report['failed'] or image_report['verified'] != 855:
        raise ValueError('Complete verified images are required.')
    hashes = {r['image_id']:r['sha256'] for r in image_report['images']}
    review_file = ROOT/'data/vsr/review/human_decisions.json'
    human = json.loads(review_file.read_text(encoding='utf-8')).get('decisions',{}) if review_file.exists() else {}
    rows = []
    for r in map(json.loads,source_bytes.decode().splitlines()):
        image = ROOT/r['image_path']
        actual = hashlib.sha256(image.read_bytes()).hexdigest()
        if actual != hashes[r['image_id']]:
            raise ValueError('Image bytes changed: '+r['image_id'])
        status = human.get(r['example_id'],{}).get('status','unreviewed')
        eligible = r['relation'] in ('above','below') or status=='approve'
        if status in ('original_only','set_aside'):
            eligible = False
        proposal = variants(r)
        # Auxiliary questions are diagnostics, not independently annotated targets.
        proposal['exclusive_truth'] = None
        proposal['converse_truth'] = r['truth_label'] if eligible else None
        proposal['mirrored_equivalent_truth'] = r['truth_label'] if eligible else None
        rows.append(dict(r,image_sha256=actual,consistency_eligible=eligible,variants=proposal,
            consistency_basis='explicit_human_review' if status=='approve' else 'vertical_relation_rule_v1' if eligible else 'horizontal_or_reviewed_transform_not_certified'))
    rows.sort(key=lambda r: (('train','validation','test').index(r['split']),r['image_id'],r['example_id']))
    sets = [{r['image_id'] for r in rows if r['split']==split} for split in ('train','validation','test')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Image leakage detected.')
    if len(rows)!=941 or len({r['example_id'] for r in rows})!=941:
        raise ValueError('Unexpected candidate count or duplicate example IDs.')
    folder=ROOT/'data/vsr/extraction'
    folder.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
    manifest=folder/'manifest.jsonl'
    if manifest.exists() and manifest.read_bytes()!=content:
        raise ValueError('Extraction manifest already exists with different inputs. Use a versioned directory for changes.')
    manifest.write_bytes(content)
    approval={'authorization':'User accepted all VSR source labels and explicitly requested preparation/extraction, stopping before detector training.',
        'source':'chat_batch_approval','recorded_at':datetime.now(timezone.utc).isoformat(),
        'original_examples':len(rows),'consistency_examples':sum(r['consistency_eligible'] for r in rows),
        'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'manifest_sha256':hashlib.sha256(content).hexdigest(),
        'source_labels_changed':0,'human_review_state_modified':False,
        'transform_policy':'All originals retained. Vertical consistency rules plus explicit human approvals; other auxiliary features unavailable. No numeric reference-frame codes inferred.',
        'config':cfg,'splits':{s:{'rows':sum(r['split']==s for r in rows),'consistency_rows':sum(r['split']==s and r['consistency_eligible'] for r in rows)} for s in ('train','validation','test')}}
    if not (folder/'batch_approval.json').exists():
        (folder/'batch_approval.json').write_text(json.dumps(approval,indent=2),encoding='utf-8')
    plan=[]
    for split in ('train','validation','test'):
        selected=[r for r in rows if r['split']==split]
        for start in range(0,len(selected),100):
            plan.append({'batch':len(plan)+1,'split':split,'rows':len(selected[start:start+100]),'example_ids':[r['example_id'] for r in selected[start:start+100]]})
    (folder/'batch_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'images':len(hashes),'consistency_rows':approval['consistency_examples'],'splits':approval['splits'],'batches':len(plan)},indent=2))


if __name__=='__main__':
    main()

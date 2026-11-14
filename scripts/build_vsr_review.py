"""Prepare a human review dataset without copying images or model outputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from build_vsr_audit import variants

ROOT = Path(__file__).resolve().parents[1]
SPEC = 'vsr-human-review-v1'


def main():
    source = ROOT / 'data/vsr/processed/directional_candidates.jsonl'
    raw = source.read_bytes()
    rows = sorted([json.loads(x) for x in raw.decode().splitlines()], key=lambda r: (r['image_id'], r['example_id']))
    images = {}
    for row in rows:
        image_id = row['image_id']
        if image_id in images:
            continue
        path = ROOT / row['image_path']
        content = path.read_bytes()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            images[image_id] = {'sha256': hashlib.sha256(content).hexdigest(),
                'width': im.width, 'height': im.height, 'bytes': len(content)}
    image_digest = hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest()
    identity = {'candidate_sha256': hashlib.sha256(raw).hexdigest(),
                'image_manifest_sha256': image_digest, 'review_spec': SPEC}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    examples = []
    for i, row in enumerate(rows, 1):
        proposal = variants(row)
        # Human review decides whether these proposed equivalences are usable.
        proposal['converse_truth'] = None
        proposal['mirrored_equivalent_truth'] = None
        examples.append(dict(row, review_number=i, batch=(i-1)//50+1, variants=proposal))
    payload = {'dataset_fingerprint': fingerprint, **identity,
        'statement_count': len(examples), 'image_count': len(images), 'batch_size': 50,
        'batches': (len(examples)+49)//50, 'relations': dict(Counter(r['relation'] for r in rows)),
        'splits': {split: {'statements': sum(r['split']==split for r in rows),
            'images':len({r['image_id'] for r in rows if r['split']==split})}
            for split in ('train','validation','test')}, 'images':images, 'rows':examples}
    out = ROOT / 'data/vsr/review'
    out.mkdir(parents=True, exist_ok=True)
    (out/'dataset.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('rows','images')}, indent=2))


if __name__ == '__main__':
    main()

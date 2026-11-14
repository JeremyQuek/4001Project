"""Score the audited training pilot once, caching all observations and features."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
from PIL import Image, ImageOps
from run_vlm import load_vlm, score

ROOT = Path(__file__).resolve().parents[1]
REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'


def features(scores):
    p = scores['original']['p_yes']
    blind = scores['blind']['p_yes']
    values = {
        'p_original': p, 'confidence': max(p, 1 - p),
        'p_blind': blind, 'image_sensitivity': abs(p - blind),
        'blind_support_for_original_answer': blind if p >= .5 else 1 - blind,
    }
    if 'converse' in scores:
        values.update(
            p_converse=scores['converse']['p_yes'], p_exclusive=scores['exclusive']['p_yes'],
            p_mirrored=scores['mirrored_equivalent']['p_yes'],
            converse_disagreement=abs(p - scores['converse']['p_yes']),
            exclusive_violation=max(0., p + scores['exclusive']['p_yes'] - 1),
            mirror_disagreement=abs(p - scores['mirrored_equivalent']['p_yes']),
        )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'outputs/vsr_audit/reviewed_pilot.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/vsr_pilot_256')
    parser.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--save-hidden', action='store_true')
    parser.add_argument('--original-only', action='store_true', help='For a resolution comparison on the same training images.')
    args = parser.parse_args()
    content = args.manifest.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines()]
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Only training pilot examples are allowed.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        'manifest_sha256': hashlib.sha256(content).hexdigest(), 'model_revision': REVISION,
        'max_pixels': args.max_pixels, 'save_hidden': args.save_hidden,
        'original_only': args.original_only, 'prompt_version': 'run_vlm_statement_yes_no_v1',
    }
    signature_path = args.output_dir / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Output directory belongs to a different run configuration.')
    signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    results_path = args.output_dir / 'results.jsonl'
    existing = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    done = {row['example_id'] for row in existing}
    if len(done) != len(existing) or not done <= {r['example_id'] for r in rows}:
        raise ValueError('Unexpected or duplicate cached results.')
    if len(done) == len(rows) and (args.output_dir / 'run.json').exists():
        print(f'All {len(rows)} records are already complete. Preserving the original run metadata and timings.', flush=True)
        return
    started = time.perf_counter()
    model, processor, metadata = load_vlm(REVISION, args.max_pixels)
    results = existing[:]
    with results_path.open('a', encoding='utf-8') as handle:
        for row in rows:
            if row['example_id'] in done:
                continue
            example_start = time.perf_counter()
            with Image.open(ROOT / row['image_path']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            hidden_path = None
            if args.save_hidden:
                (args.output_dir / 'hidden').mkdir(exist_ok=True)
                hidden_path = args.output_dir / 'hidden' / f'{row["example_id"]}.pt'
            scores = {'original': score(model, processor, image, row['statement'], hidden_path)}
            if not args.original_only:
                scores['blind'] = score(model, processor, Image.new('RGB', image.size, (128, 128, 128)), row['statement'])
                if row['consistency_eligible']:
                    for name in ('converse', 'exclusive', 'mirrored_equivalent'):
                        variant_image = ImageOps.mirror(image) if name == 'mirrored_equivalent' else image
                        scores[name] = score(model, processor, variant_image, row['variants'][name])
            prediction = int(scores['original']['p_yes'] >= .5)
            record = dict(row, prediction=prediction, error_label=int(prediction != row['truth_label']),
                scores=scores, features=features(scores) if not args.original_only else {},
                example_wall_seconds=round(time.perf_counter() - example_start, 3))
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            results.append(record)
            if len(results) % 10 == 0 or len(results) == len(rows):
                correct = sum(not r['error_label'] for r in results)
                print(f'{len(results)}/{len(rows)} completed; original correct={correct}/{len(results)}; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    metadata.update(signature=signature, examples=len(results),
        cached_records=len(existing), wall_time_scope='current invocation',
        correct=sum(not r['error_label'] for r in results), errors=sum(r['error_label'] for r in results),
        scored_variants=sum(len(r['scores']) for r in results),
        total_wall_seconds=round(time.perf_counter() - started, 3),
        total_example_wall_seconds=round(sum(r['example_wall_seconds'] for r in results), 3),
        max_peak_allocated_vram_gib=max(s['peak_allocated_vram_gib'] for r in results for s in r['scores'].values()))
    (args.output_dir / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if not args.original_only:
        flat = [dict(example_id=r['example_id'], image_id=r['image_id'], split=r['split'],
            truth_label=r['truth_label'], prediction=r['prediction'], error_label=r['error_label'],
            consistency_eligible=r['consistency_eligible'], **r['features']) for r in results]
        columns = list(dict.fromkeys(key for row in flat for key in row))
        with (args.output_dir / 'features.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()

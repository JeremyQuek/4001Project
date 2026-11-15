"""Stages 4-5: prompt the frozen VLM for every query and save p_yes plus hidden states.

Resumable: one small NPZ record per query, written atomically. A record is
reused only if its signature (frozen VLM config, library versions, statement,
image hash) matches, so an interrupted run continues where it stopped.

    python -I scripts/05_extract.py --pilot    # 200 training queries, timing report
    python -I scripts/05_extract.py            # everything in queries.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from common import EXTRACTION, IMAGES, OUTPUTS, PROCESSED, PROMPT_TEMPLATE, VLM, sha256_text, write_json
from vlm import load_image, load_vlm, score

RECORDS = EXTRACTION / 'records'


def record_path(query_id):
    return RECORDS / f'{query_id}.npz'


def signature(statement, image_sha, versions):
    """Everything that determines a VLM output for one query."""
    return sha256_text(repr((sorted(VLM.items(), key=str), PROMPT_TEMPLATE, versions, statement, image_sha)))


def load_record(query_id, expected_signature):
    """Return a finished record, or None if it is missing, damaged or from another config."""
    path = record_path(query_id)
    if not path.exists():
        return None
    try:
        with np.load(path) as saved:
            record = {key: saved[key] for key in saved.files}
    except Exception:
        return None
    if str(record['signature']) != expected_signature or record['hidden'].shape != (3, 2048):
        return None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Only the pilot training queries.')
    args = parser.parse_args()
    RECORDS.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(PROCESSED / 'queries.csv')
    manifest = pd.read_csv(PROCESSED / 'image_manifest.csv').set_index('image_id')
    missing = set(queries['image_id']) - set(manifest.index)
    if missing:
        raise RuntimeError(f'{len(missing)} sampled images are not downloaded; run stages 03/04 again.')
    if args.pilot:
        queries = queries[queries['pilot'] == 1]
        if set(queries['split']) != {'train'}:
            raise RuntimeError('The pilot must only use the training split.')

    model, processor, metadata = load_vlm()
    versions = (metadata['torch'], tuple(sorted(metadata['packages'].items())))
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    new = reused = 0
    forward_seconds = []
    cached_image = (None, None)  # the four variants of a relationship share one image
    for position, row in enumerate(queries.itertuples(), 1):
        image_sha = manifest.at[row.image_id, 'sha256']
        expected = signature(row.statement, image_sha, versions)
        if load_record(row.query_id, expected) is not None:
            reused += 1
            continue
        if cached_image[0] != row.image_id:
            cached_image = (row.image_id, load_image(IMAGES / manifest.at[row.image_id, 'file']))
        result = score(model, processor, cached_image[1], row.statement)
        forward_seconds.append(result['forward_seconds'])
        temporary = RECORDS / f'{row.query_id}.tmp.npz'
        np.savez(temporary, hidden=result['hidden'], p_yes=np.float32(result['p_yes']),
                 logit_yes=np.float32(result['logit_yes']), logit_no=np.float32(result['logit_no']),
                 yes_no_mass=np.float32(result['yes_no_mass']), top_token=result['top_token'],
                 input_tokens=result['input_tokens'], signature=expected)
        os.replace(temporary, record_path(row.query_id))
        new += 1
        if new % 100 == 0:
            rate = (time.perf_counter() - started) / new
            print(f'{position}/{len(queries)} done; {rate:.2f} s/query; '
                  f'about {rate * (len(queries) - position) / 60:.1f} min left', flush=True)
    wall = time.perf_counter() - started

    summary = {
        'mode': 'pilot' if args.pilot else 'full', 'queries': len(queries), 'new': new, 'reused': reused,
        'wall_seconds': round(wall, 1),
        'wall_seconds_per_new_query': round(wall / new, 4) if new else None,
        'forward_seconds_median': float(np.median(forward_seconds)) if new else None,
        'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
        'peak_reserved_vram_gib': round(torch.cuda.max_memory_reserved() / 2**30, 3),
        'vlm': metadata,
    }
    if args.pilot:
        # Pilot report (training split only): answer accuracy and Yes/No behaviour.
        records = [load_record(q.query_id, signature(q.statement, manifest.at[q.image_id, 'sha256'], versions))
                   for q in queries.itertuples()]
        p_yes = np.array([float(r['p_yes']) for r in records])
        prediction = (p_yes >= 0.5).astype(int)
        truth = queries['truth'].to_numpy()
        frame = queries.assign(correct=(prediction == truth).astype(int), said_yes=prediction)
        summary['pilot'] = {
            'accuracy': float((prediction == truth).mean()),
            'yes_rate': float(prediction.mean()),
            'errors': int((prediction != truth).sum()),
            'accuracy_by_variant': frame.groupby('variant')['correct'].mean().to_dict(),
            'accuracy_by_predicate': frame.groupby('canonical_predicate')['correct'].mean().to_dict(),
            'top_token_is_yes_or_no': float(np.mean([str(r['top_token']) in ('Yes', 'No') for r in records])),
            'min_yes_no_mass': float(min(float(r['yes_no_mass']) for r in records)),
        }
    write_json(OUTPUTS / ('pilot_summary.json' if args.pilot else 'extraction_summary.json'), summary)
    printable = {k: v for k, v in summary.items() if k != 'vlm'}
    print(printable, flush=True)


if __name__ == '__main__':
    main()

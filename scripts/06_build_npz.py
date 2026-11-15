"""Stage 6: pack the per-query records into the shared NPZ contract and write
the row tables, metadata with checksums, and DATASET_CARD.md.

Run with `python -I`. Needs a finished run of 05_extract.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import (CANONICAL, DETECTOR_READY, EXTRACTION, OUTPUTS, PROCESSED, PROMPT_TEMPLATE, VARIANTS,
                    VLM, read_json, sha256_file, sha256_text, write_json)

ROW_COLUMNS = ['row_idx', 'image_id', 'coco_id', 'rel_id', 'variant', 'canonical_predicate',
               'vg_predicate', 'subject', 'object', 'statement', 'truth', 'split']


def signature(statement, image_sha, versions):
    """Must stay identical to the signature in 05_extract.py."""
    return sha256_text(repr((sorted(VLM.items(), key=str), PROMPT_TEMPLATE, versions, statement, image_sha)))


def markdown_table(frame):
    """Small dependency-free Markdown table (index is the first column)."""
    header = '| ' + ' | '.join([frame.index.name or ''] + [str(c) for c in frame.columns]) + ' |'
    rule = '|---|' + '---:|' * len(frame.columns)
    body = ['| ' + ' | '.join([str(index)] + [str(v) for v in row]) + ' |'
            for index, row in zip(frame.index, frame.to_numpy())]
    return '\n'.join([header, rule] + body)


def main():
    DETECTOR_READY.mkdir(parents=True, exist_ok=True)
    queries = pd.read_csv(PROCESSED / 'queries.csv')
    manifest = pd.read_csv(PROCESSED / 'image_manifest.csv').set_index('image_id')
    run = read_json(OUTPUTS / 'extraction_summary.json')
    versions = (run['vlm']['torch'], tuple(sorted(run['vlm']['packages'].items())))

    checksums, counts = {}, {}
    described = []  # train and validation rows with outcomes, for the card
    for split in ('train', 'val', 'test'):
        part = queries[queries['split'] == split].reset_index(drop=True)
        part.insert(0, 'row_idx', np.arange(len(part)))
        hidden = np.empty((len(part), 3, 2048), dtype=np.float32)
        p_yes = np.empty(len(part), dtype=np.float32)
        for row in part.itertuples():
            with np.load(EXTRACTION / 'records' / f'{row.query_id}.npz') as record:
                expected = signature(row.statement, manifest.at[row.image_id, 'sha256'], versions)
                if str(record['signature']) != expected:
                    raise ValueError(f'Record {row.query_id} was made with a different configuration.')
                hidden[row.row_idx] = record['hidden']
                p_yes[row.row_idx] = record['p_yes']
        if not (np.isfinite(hidden).all() and np.isfinite(p_yes).all()):
            raise ValueError('Non-finite values in extracted features.')
        truth = part['truth'].to_numpy(dtype=np.int8)
        vlm_pred = (p_yes >= 0.5).astype(np.int8)                    # A4
        confidence = np.maximum(p_yes, 1 - p_yes).astype(np.float32)
        y_error = (vlm_pred != truth).astype(np.int8)                # A5 detector target
        arrays = {f'hidden_L{layer}': hidden[:, i] for i, layer in enumerate(VLM['hidden_layers'])}
        arrays.update(p_yes=p_yes, confidence=confidence, vlm_pred=vlm_pred, truth=truth, y_error=y_error)
        np.savez(DETECTOR_READY / f'{split}.npz', **arrays)
        part[ROW_COLUMNS].to_csv(DETECTOR_READY / f'{split}_rows.csv', index=False)
        for name in (f'{split}.npz', f'{split}_rows.csv'):
            checksums[name] = sha256_file(DETECTOR_READY / name)
        counts[split] = {'rows': len(part), 'relationships': int(part['rel_id'].nunique()),
                         'images': int(part['image_id'].nunique())}
        if split != 'test':  # test outcomes stay closed until the final evaluation
            counts[split].update(errors=int(y_error.sum()), error_rate=float(y_error.mean()),
                                 yes_rate=float(vlm_pred.mean()))
            described.append(part.assign(y_error=y_error))

    image_sets = {s: set(pd.read_csv(DETECTOR_READY / f'{s}_rows.csv')['image_id']) for s in counts}
    if image_sets['train'] & image_sets['val'] or image_sets['train'] & image_sets['test'] \
            or image_sets['val'] & image_sets['test']:
        raise ValueError('An image appears in more than one split.')

    write_json(DETECTOR_READY / 'metadata.json', {
        'checksums': checksums, 'counts': counts, 'vlm': run['vlm'],
        'sample': read_json(PROCESSED / 'sample_summary.json'),
        'feature_keys': ['hidden_L9', 'hidden_L18', 'hidden_L36', 'p_yes', 'confidence'],
        'metadata_keys_never_inputs': ['vlm_pred', 'truth', 'y_error'],
    })

    # ---- Dataset card ----
    split_table = pd.DataFrame(counts).T
    split_table.index.name = 'Split'
    for column in ('errors', 'error_rate', 'yes_rate'):
        split_table[column] = split_table[column].map(
            lambda v: 'closed' if pd.isna(v) else (f'{v:.3f}' if column != 'errors' else int(v)))
    by_predicate = queries.groupby(['canonical_predicate', 'split']).size().unstack()[['train', 'val', 'test']]
    by_predicate = by_predicate.reindex(CANONICAL)
    by_predicate.index.name = 'Canonical predicate'
    by_variant = queries.groupby(['variant', 'split']).size().unstack()[['train', 'val', 'test']].reindex(VARIANTS)
    by_variant.index.name = 'Variant'
    seen = pd.concat(described)
    error_predicate = seen.groupby(['canonical_predicate', 'split'])['y_error'].mean().unstack()[['train', 'val']]
    error_predicate = error_predicate.reindex(CANONICAL).map(lambda v: f'{v:.3f}')
    error_predicate.index.name = 'Canonical predicate'
    error_variant = seen.groupby(['variant', 'split'])['y_error'].mean().unstack()[['train', 'val']]
    error_variant = error_variant.reindex(VARIANTS).map(lambda v: f'{v:.3f}')
    error_variant.index.name = 'Variant'
    checksum_table = pd.DataFrame({'SHA-256': checksums})
    checksum_table.index.name = 'File'

    card = f"""# Dataset card: Visual Genome spatial-relation error detection

Generated by `scripts/06_build_npz.py`; do not edit by hand.

Each row is one Yes/No query to the frozen VLM (`{VLM['model_id']}`, revision
`{VLM['model_revision']}`, 4-bit NF4) about a spatial statement generated from one
Visual Genome relationship. Every relationship gives four statements (2 true, 2 false).
The detector target is `y_error = int(vlm_pred != truth)`.

## Rows per split

Split is by image ID (70/15/15, seed 42); no image is in two splits. Test outcomes are
not shown here because the test split stays closed until the final evaluation.

{markdown_table(split_table)}

## Queries per canonical predicate

{markdown_table(by_predicate)}

## Queries per variant

{markdown_table(by_variant)}

## VLM error rate (train and validation only)

{markdown_table(error_predicate)}

{markdown_table(error_variant)}

## Arrays in each `{{split}}.npz`

| Key | Shape / dtype | Meaning |
|---|---|---|
| `hidden_L9`, `hidden_L18`, `hidden_L36` | (N, 2048) float32 | Final-prompt-token hidden state |
| `p_yes` | (N,) float32 | Relative Yes support |
| `confidence` | (N,) float32 | max(p_yes, 1 - p_yes) |
| `vlm_pred` | (N,) int8 | 1 = Yes |
| `truth` | (N,) int8 | Statement truth from VG |
| `y_error` | (N,) int8 | `int(vlm_pred != truth)`, the detector target |

`{{split}}_rows.csv` has one line per NPZ row, in the same order (`row_idx`), with
`{', '.join(ROW_COLUMNS)}`.

`truth`, `vlm_pred`, IDs and split are metadata, never detector inputs. Load the data
with `scripts/load_detector_data.py`, which checks the checksums and row alignment.

## Known limitations

- Truth labels come from VG annotations without relabelling, so they contain noise.
- "S on O" converted to "O under S" (and similar converses) is not always natural English.
- An image can contain several objects with the same name, so a statement may be
  ambiguous even after contradictory annotations were dropped (see `ASSUMPTIONS.md`, A3b).

## Checksums

{markdown_table(checksum_table)}
"""
    (DETECTOR_READY / 'DATASET_CARD.md').write_text(card, encoding='utf-8')
    print(split_table.to_string())


if __name__ == '__main__':
    main()

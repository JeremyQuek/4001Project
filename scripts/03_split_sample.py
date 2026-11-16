"""Stage 3: split by image, sample relationships with a per-predicate cap,
and generate the four query variants for each sampled relationship.

All randomness comes from seeded hashes of image/relationship ids, so the
output is identical on every run and does not depend on row order.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import (CANONICAL, PROCESSED, SEED, SPLIT_FRACTIONS, make_statements, seeded_priority,
                    write_json)


def split_of(image_id):
    """70/15/15 by image id; every relationship and variant of an image shares its split."""
    value = seeded_priority('split', image_id)
    if value < SPLIT_FRACTIONS['train']:
        return 'train'
    return 'val' if value < SPLIT_FRACTIONS['train'] + SPLIT_FRACTIONS['val'] else 'test'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-predicate', type=int, default=250,
                        help='Relationships per canonical predicate over all splits.')
    parser.add_argument('--max-per-image', type=int, default=2)
    parser.add_argument('--pilot-relationships', type=int, default=50)
    args = parser.parse_args()

    table = pd.read_parquet(PROCESSED / 'spatial_relationships.parquet')
    total = len(table)
    # Drop relationships whose false variant is itself annotated as true (ASSUMPTIONS.md, A3b).
    conflicts = int(table['label_conflict'].sum())
    table = table[~table['label_conflict']]
    # Leave out images that are also used in the group's VSR experiment.
    vsr_images = int(table.loc[table['in_teammate_vsr'], 'image_id'].nunique())
    vsr_relationships = int(table['in_teammate_vsr'].sum())
    table = table[~table['in_teammate_vsr']].copy()
    # Images that failed download/decoding in an earlier run of 04_download_images.py are excluded,
    # so re-running this script replaces their relationships.
    failed_path = PROCESSED / 'failed_images.csv'
    failed = set(pd.read_csv(failed_path)['image_id']) if failed_path.exists() else set()
    table = table[~table['image_id'].isin(failed)].copy()

    table['split'] = table['image_id'].map(split_of)
    table['priority'] = table['rel_id'].map(lambda rel_id: seeded_priority('sample', rel_id))

    # Equal per-predicate quota in each split (70/15/15 of the cap).
    held_out = round(args.per_predicate * SPLIT_FRACTIONS['val'])
    quota = {'train': args.per_predicate - 2 * held_out, 'val': held_out, 'test': held_out}
    # Fill rare predicates first so the at-most-N-per-image rule cannot starve them.
    order = table['canonical_predicate'].value_counts().sort_values().index.tolist()
    per_image = Counter()
    chosen = []
    for split in ('train', 'val', 'test'):
        for predicate in order:
            pool = table[(table['split'] == split) & (table['canonical_predicate'] == predicate)]
            taken = 0
            for row in pool.sort_values('priority').itertuples():
                if taken == quota[split]:
                    break
                if per_image[row.image_id] >= args.max_per_image:
                    continue
                per_image[row.image_id] += 1
                chosen.append(row.rel_id)
                taken += 1
    sample = table[table['rel_id'].isin(chosen)].sort_values(['split', 'priority']).copy()

    # Pilot: N sampled training relationships chosen with an independent seeded key
    # (the sampling priority is not reused: it is smallest for abundant predicates).
    train_ids = sample.loc[sample['split'] == 'train', 'rel_id'].tolist()
    train_ids.sort(key=lambda rel_id: seeded_priority('pilot', rel_id))
    pilot_ids = set(train_ids[:args.pilot_relationships])

    rows = []
    for relation in sample.itertuples():
        for variant, statement, truth in make_statements(
                relation.subject, relation.canonical_predicate, relation.object):
            rows.append({
                'query_id': f'{relation.rel_id}_{variant}', 'image_id': relation.image_id,
                'coco_id': relation.coco_id, 'rel_id': relation.rel_id, 'variant': variant,
                'canonical_predicate': relation.canonical_predicate,
                'vg_predicate': relation.vg_predicate, 'subject': relation.subject,
                'object': relation.object, 'statement': statement, 'truth': truth,
                'split': relation.split, 'pilot': int(relation.rel_id in pilot_ids)})
    queries = pd.DataFrame(rows)
    queries.to_csv(PROCESSED / 'queries.csv', index=False)

    images = pd.read_parquet(PROCESSED / 'image_data.parquet')
    needed = images[images['image_id'].isin(sample['image_id'])].merge(
        sample[['image_id', 'split']].drop_duplicates(), on='image_id')
    needed.to_csv(PROCESSED / 'sample_images.csv', index=False)

    counts = sample.groupby(['canonical_predicate', 'split']).size().unstack(fill_value=0)
    counts = counts.reindex(CANONICAL)[['train', 'val', 'test']]
    summary = {
        'seed': SEED, 'per_predicate': args.per_predicate, 'max_per_image': args.max_per_image,
        'quota_per_predicate': quota, 'filtered_relationships': total,
        'dropped_label_conflict': conflicts,
        'left_out_teammate_vsr_images': vsr_images,
        'left_out_teammate_vsr_relationships': vsr_relationships,
        'left_out_failed_images': len(failed),
        'eligible_relationships': len(table), 'eligible_images': int(table['image_id'].nunique()),
        'sampled_relationships': len(sample), 'sampled_images': int(sample['image_id'].nunique()),
        'queries': len(queries), 'pilot_relationships': len(pilot_ids),
        'relationships_per_predicate_split': {p: counts.loc[p].to_dict() for p in counts.index},
    }
    write_json(PROCESSED / 'sample_summary.json', summary)
    print(counts.to_string())
    print(f'\n{len(sample)} relationships, {sample["image_id"].nunique()} images, {len(queries)} queries; '
          f'pilot = {len(pilot_ids)} training relationships.')
    print(f'Dropped {conflicts} label-conflict relationships; left out {vsr_images} VSR images '
          f'({vsr_relationships} relationships); {len(failed)} failed images excluded.')


if __name__ == '__main__':
    main()

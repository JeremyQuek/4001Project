"""Stream relationships.json once, map predicates, apply the minimal
validity filters, and cache a compact Parquet table (ASSUMPTIONS.md, A1 and A3).

"""
import zipfile
from collections import Counter

import ijson
import pandas as pd

from common import (CANONICAL, OPPOSITE, PREDICATE_FORMS, PROCESSED, RAW, SURFACE_TO_CANONICAL,
                    VSR_MANIFEST, clean_name, write_json)


def entity_name(entity):
    """VG 1.4 stores either `name` or a `names` list; use the first listed name."""
    if entity.get('name') is not None:
        return clean_name(entity['name'])
    names = entity.get('names') or ['']
    return clean_name(names[0])


def stream_items(archive_name):
    """Yield top-level JSON array items straight from the zip (no extraction)."""
    with zipfile.ZipFile(RAW / archive_name) as archive:
        with archive.open(archive.infolist()[0]) as handle:
            yield from ijson.items(handle, 'item')


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Image metadata: download URL and COCO id for every VG image.
    images = pd.DataFrame(
        {'image_id': int(item['image_id']), 'url': item['url'],
         'coco_id': -1 if item.get('coco_id') is None else int(item['coco_id'])}
        for item in stream_items('image_data.json.zip'))
    images.to_parquet(PROCESSED / 'image_data.parquet', index=False)
    image_ids = set(images['image_id'])

    counts = Counter()          # filter bookkeeping
    surface_counts = Counter()  # matches per accepted VG surface form (before the filters)
    seen = set()                # (image, subject, canonical predicate, object) for duplicates
    rows = []
    for item in stream_items('relationships.json.zip'):
        image_id = int(item['image_id'])
        for relation in item['relationships']:
            counts['relationships_total'] += 1
            surface = clean_name(relation['predicate'])
            canonical = SURFACE_TO_CANONICAL.get(surface)
            if canonical is None:
                continue
            counts['predicate_matched'] += 1
            surface_counts[surface] += 1
            subject, obj = entity_name(relation['subject']), entity_name(relation['object'])
            # Validity filters, applied in this order; each relationship is counted once.
            if not subject or not obj:
                counts['dropped_empty_name'] += 1
                continue
            if subject == obj:
                counts['dropped_identical_names'] += 1
                continue
            key = (image_id, subject, canonical, obj)
            if key in seen:
                counts['dropped_duplicate'] += 1
                continue
            seen.add(key)
            if image_id not in image_ids:
                counts['dropped_image_not_in_metadata'] += 1
                continue
            rows.append({'rel_id': int(relation['relationship_id']), 'image_id': image_id,
                         'canonical_predicate': canonical, 'vg_predicate': surface,
                         'subject': subject, 'object': obj})
    table = pd.DataFrame(rows)
    if table['rel_id'].duplicated().any():
        raise ValueError('relationship_id is not unique; it cannot be used as rel_id.')
    table = table.merge(images[['image_id', 'coco_id']], on='image_id', how='left')

    # Mark VG images that are also used in the group's VSR experiment (matched by COCO id).
    if VSR_MANIFEST.exists():
        vsr_coco = set(pd.read_csv(VSR_MANIFEST, dtype=str)['image_id'].astype(int))
        table['in_teammate_vsr'] = table['coco_id'].isin(vsr_coco)
    else:
        vsr_coco = None
        table['in_teammate_vsr'] = False

    # Diagnostic only (nothing is dropped): relationships whose generated FALSE
    # statement is itself annotated as TRUE elsewhere in the same image, which
    # happens when an image has several objects with the same name.
    kept = set(zip(table['image_id'], table['subject'], table['canonical_predicate'], table['object']))
    swap_conflict = [(i, o, p, s) in kept for i, s, p, o in
                     zip(table['image_id'], table['subject'], table['canonical_predicate'], table['object'])]
    opposite_conflict = [(i, s, OPPOSITE[p], o) in kept for i, s, p, o in
                         zip(table['image_id'], table['subject'], table['canonical_predicate'], table['object'])]
    table['label_conflict'] = [a or b for a, b in zip(swap_conflict, opposite_conflict)]

    table.to_parquet(PROCESSED / 'spatial_relationships.parquet', index=False)

    per_canonical = table.groupby('canonical_predicate').agg(
        relationships=('rel_id', 'size'), images=('image_id', 'nunique'),
        label_conflicts=('label_conflict', 'sum'), in_teammate_vsr=('in_teammate_vsr', 'sum'))
    summary = {
        'filter_counts': dict(counts),
        'kept_relationships': len(table), 'kept_images': int(table['image_id'].nunique()),
        'surface_form_matches_before_filters': {
            canon: {form: surface_counts[form] for form in forms} for canon, forms in PREDICATE_FORMS.items()},
        'kept_per_canonical': {canon: {k: int(v) for k, v in per_canonical.loc[canon].items()}
                               for canon in CANONICAL if canon in per_canonical.index},
        'kept_per_surface_form': {k: int(v) for k, v in table['vg_predicate'].value_counts().items()},
        'label_conflicts_total': int(table['label_conflict'].sum()),
        'teammate_manifest_available': vsr_coco is not None,
        'teammate_vsr_images': None if vsr_coco is None else len(vsr_coco),
        'vg_images_also_in_teammate_vsr': int(table.loc[table['in_teammate_vsr'], 'image_id'].nunique()),
    }
    write_json(PROCESSED / 'filter_summary.json', summary)

    print('Filter counts:', dict(counts))
    print('\nMatches per accepted VG surface form (before the filters):')
    for canon, forms in PREDICATE_FORMS.items():
        print(f'  {canon}: ' + ', '.join(f'{form}={surface_counts[form]}' for form in forms))
    print('\nKept after the filters:')
    print(per_canonical.reindex(CANONICAL).to_string())
    print(f'\nTotal kept: {len(table)} relationships in {table["image_id"].nunique()} images; '
          f'{summary["vg_images_also_in_teammate_vsr"]} of those images are also VSR images.')


if __name__ == '__main__':
    main()

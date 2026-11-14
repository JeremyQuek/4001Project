"""Apply recorded pilot visual review without modifying VSR truth labels."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_vsr_audit import variants

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/vsr_audit'


def main():
    source = ROOT / 'data/vsr/processed/pilot_candidates.jsonl'
    content = source.read_bytes()
    config = json.loads((ROOT / 'configs/vsr_pilot_audit.json').read_text())
    if hashlib.sha256(content).hexdigest() != config['source_manifest_sha256']:
        raise ValueError('Pilot manifest changed: recorded reviews must be rechecked.')
    rows = [json.loads(line) for line in content.decode().splitlines()]
    reviews = {r['example_id']: r for r in config['reviews']}
    if len(reviews) != len(rows) or set(reviews) != {r['example_id'] for r in rows}:
        raise ValueError('Missing or duplicate review records.')
    results = []
    for row in rows:
        review = reviews[row['example_id']]
        if review['status'] not in ('no_issue_found', 'needs_review'):
            raise ValueError('Unknown review status.')
        ready = review['status'] == 'no_issue_found'
        proposal = variants(row)
        if not ready:
            proposal['converse_truth'] = None
            proposal['mirrored_equivalent_truth'] = None
        results.append(dict(row, audit_index=review['audit_index'], audit_status=review['status'],
            audit_category=review['category'], audit_notes=review['notes'],
            consistency_eligible=ready, variants=proposal))
    OUT.mkdir(parents=True, exist_ok=True)
    ready = [r for r in results if r['consistency_eligible']]
    flagged = [r for r in results if not r['consistency_eligible']]
    for name, subset in [('reviewed_pilot', results), ('consistency_ready_pilot', ready), ('flagged_pilot', flagged)]:
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in subset), encoding='utf-8')
    fields = ['audit_index', 'example_id', 'image_id', 'statement', 'truth_label',
              'reference_frame', 'audit_status', 'audit_category', 'audit_notes', 'consistency_eligible']
    with (OUT / 'review.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    summary = {
        'reviewer': config['reviewer'], 'method': config['method'],
        'source_manifest_sha256': config['source_manifest_sha256'],
        'reviewed': len(results), 'consistency_ready': len(ready), 'flagged': len(flagged),
        'flag_categories': dict(Counter(r['audit_category'] for r in flagged)),
        'ready_relations': dict(Counter(r['relation'] for r in ready)),
        'ready_truth_labels': dict(Counter(str(r['truth_label']) for r in ready)),
        'truth_labels_changed': 0, 'all_training_split': all(r['split'] == 'train' for r in results),
        'vlm_predictions_used_for_review': False,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        '# VSR pilot: first-pass visual audit', '',
        f'Reviewed {len(results)} statements across {len({r["image_id"] for r in results})} training images. '
        f'{len(ready)} have no immediate issue; {len(flagged)} are held for closer review.', '',
        'Reviewer: Codex visual inspection. This is one agent-assisted review, not independent human annotation or inter-rater validation. '
        'All original VSR labels are preserved. A flag indicates uncertainty, not proof that an annotation is wrong. '
        'No VLM predictions were used to select or flag examples.', '',
        '## Transformation rules', '',
        '- Converse: swap subject/object and swap left/right or above/below. Horizontal equivalence requires a consistent viewer-relative interpretation.',
        '- Exclusive: replace the relation with its opposite, but do not assign a new truth label. False original statements can have false opposites.',
        '- Mirror: horizontally reflect pixels, swap left/right in the statement, and leave above/below unchanged. Perspective-sensitive cases remain held.',
        '- Blind: use the original statement with an equally sized grey image; this diagnostic has no independent ground-truth answer.', '',
        '## Files and use', '',
        '`reviewed_pilot.jsonl` retains all 100 examples and review flags. '
        '`consistency_ready_pilot.jsonl` is a provisional subset for initial consistency feature checks. '
        '`flagged_pilot.jsonl` records held cases. `review.csv` opens as a review table in Excel. '
        'To revise decisions reproducibly, edit `configs/vsr_pilot_audit.json` and rerun `scripts/finalize_vsr_audit.py`.', '',
        'Run original-answer evaluation on the full pilot with VSR labels. For consistency ablations, compare all feature groups '
        'on the same reviewed subset; do not compare a baseline on 100 examples with a verifier on the smaller subset. '
        'Report both counts and the selection criteria. This review does not relabel or delete the source dataset.', '',
        'The numeric reference-frame codes are preserved without assuming their meanings. The two highlighted perspective '
        'cases are flagged using image evidence and uncertain converse semantics.', '',
        '## Cases requiring closer review', '',
        '| Audit # | Statement | VSR truth | Issue |', '|---|---|---:|---|',
    ]
    for row in flagged:
        target = (ROOT / row['image_path']).as_posix()
        lines.append(f'| [{row["audit_index"]:03}]({target}) | {row["statement"]} | {row["truth_label"]} | {row["audit_notes"]} |')
    lines += ['', '## Contact sheets', '']
    for page in range(1, 11):
        lines.append(f'- [Examples {(page-1)*10+1:03}–{page*10:03}]({(OUT / f"contact_{page:02}.jpg").as_posix()})')
    lines += ['', f'[Original/mirror comparison examples]({(OUT / "mirror_examples.jpg").as_posix()})', '',
        '## Next step', '',
        'Run a GPU pilot that preserves all original answers/labels and records five diagnostic scores where the transformations '
        'are eligible. Keep held cases separate when interpreting logical inconsistencies. Choose final processing settings '
        'using training pilot data before extracting validation/test features.', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

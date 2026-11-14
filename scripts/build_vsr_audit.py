"""Build numbered contact sheets and transformation proposals for the pilot."""
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/vsr_audit'
INVERSE = {'left of': 'right of', 'right of': 'left of', 'above': 'below', 'below': 'above'}


def variants(row):
    subject, obj, relation = row['subject'], row['object'], row['relation']
    inverse = INVERSE[relation]
    swapped = f'{obj[0].upper() + obj[1:]} is {inverse} {subject[0].lower() + subject[1:]}.'
    opposite = f'{subject} is {inverse} {obj}.'
    horizontal = relation in ('left of', 'right of')
    return {
        'original': row['statement'], 'converse': swapped, 'exclusive': opposite,
        'mirrored_equivalent': opposite if horizontal else row['statement'],
        'blind': row['statement'],
        'converse_truth': row['truth_label'],
        'mirrored_equivalent_truth': row['truth_label'],
        'exclusive_truth': None,
        'equivalence_condition': 'viewer-relative frame; requires visual review' if horizontal else 'same two object referents and vertical interpretation',
    }


def main():
    rows = [json.loads(line) for line in (ROOT / 'data/vsr/processed/pilot_candidates.jsonl').read_text().splitlines()]
    OUT.mkdir(parents=True, exist_ok=True)
    normal = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 20)
    proposals = []
    for index, row in enumerate(rows, 1):
        if row['split'] != 'train':
            raise ValueError('Pilot audit must use training examples only.')
        proposal = dict(row, audit_index=index, proposed_variants=variants(row))
        if INVERSE[INVERSE[row['relation']]] != row['relation']:
            raise ValueError('Invalid inverse mapping.')
        proposals.append(proposal)
    for start in range(0, len(rows), 10):
        canvas = Image.new('RGB', (1000, 2100), '#e5e7eb')
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(rows[start:start + 10]):
            x, y = (offset % 2) * 500, (offset // 2) * 420
            draw.rectangle((x + 4, y + 4, x + 496, y + 416), fill='white')
            header = f'{start + offset + 1:03} | VSR={row["truth_label"]} | {row["image_id"]}'
            draw.text((x + 12, y + 10), header, font=bold, fill='black')
            for line_index, line in enumerate(textwrap.wrap(row['statement'], width=48)):
                draw.text((x + 12, y + 38 + line_index * 22), line, font=normal, fill='black')
            with Image.open(ROOT / row['image_path']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((476, 320))
                canvas.paste(image, (x + (500 - image.width) // 2, y + 92 + (320 - image.height) // 2))
        canvas.save(OUT / f'contact_{start // 10 + 1:02}.jpg', quality=94)
    (OUT / 'proposals.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in proposals), encoding='utf-8')
    print(f'Created {len(rows)} proposals and {(len(rows) + 9) // 10} contact sheets in {OUT}')


if __name__ == '__main__':
    main()

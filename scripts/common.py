"""Shared constants and helpers for the Visual Genome spatial-relation pipeline.

Everything that defines the dataset (predicate mapping, variants, split seed) and
the frozen VLM configuration lives here so each stage reads the same values.
"""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

# Set VG_PIPELINE_ROOT to run the pipeline in another folder (e.g. on synthetic data); normally unset.
ROOT = Path(os.environ.get('VG_PIPELINE_ROOT') or Path(__file__).resolve().parents[1])
RAW = ROOT / 'data/vg/raw'              # downloaded archives, read as data only
PROCESSED = ROOT / 'data/vg/processed'  # parsed tables, splits, queries
IMAGES = ROOT / 'data/vg/images'        # only the images the sample needs
EXTRACTION = ROOT / 'data/vg/extraction'
DETECTOR_READY = ROOT / 'data/vg/detector_ready'
OUTPUTS = ROOT / os.environ.get('VG_OUTPUT_DIR', 'outputs')
VSR_MANIFEST = ROOT / 'data/vsr/processed/directional_images.csv'

SEED = 42
SPLIT_FRACTIONS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# Official links, read from https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
# (relationships and objects: version 1.4; image meta data: version 1.2, the latest).
VG_BASE = 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/'
VG_FILES = ['relationships.json.zip', 'objects.json.zip', 'image_data.json.zip']

# Exact lowercase VG surface forms accepted for each canonical predicate (ASSUMPTIONS.md, A1).
PREDICATE_FORMS = {
    'to the left of': ['left of', 'to left of', 'to the left of', 'on left side of', 'on the left of'],
    'to the right of': ['right of', 'to right of', 'to the right of', 'on right side of', 'on the right of'],
    'above': ['above', 'over'],
    'below': ['below', 'beneath', 'underneath'],
    'in front of': ['in front of', 'front of'],
    'behind': ['behind', 'in back of'],
    'on': ['on', 'on top of', 'sitting on', 'lying on', 'standing on', 'resting on'],
    'under': ['under'],
}
SURFACE_TO_CANONICAL = {form: canon for canon, forms in PREDICATE_FORMS.items() for form in forms}
CANONICAL = list(PREDICATE_FORMS)
OPPOSITE = {
    'to the left of': 'to the right of', 'to the right of': 'to the left of',
    'above': 'below', 'below': 'above',
    'in front of': 'behind', 'behind': 'in front of',
    'on': 'under', 'under': 'on',
}
VARIANTS = ['orig', 'converse', 'swap', 'opposite']

# Frozen VLM configuration; the group's VSR experiment uses the same values.
VLM = {
    'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'model_revision': '66285546d2b821cf421d4f5eb2576359d3770cd3',
    'quantization': '4-bit NF4 double quantization; visual and lm_head excluded',
    'compute_dtype': 'torch.bfloat16',
    'attention': 'sdpa',
    'batch_size': 1,
    'max_pixels': 200704,
    'min_pixels': 3136,
    'processor_use_fast': True,
    'prompt_version': 'run_vlm_statement_yes_no_v1',
    'hidden_layers': [9, 18, 36],
    'hidden_position': 'final prompt token, before answering',
}
PROMPT_TEMPLATE = 'Statement: "{statement}"\nIs this statement true in the image? Answer Yes or No.'


def clean_name(text):
    """Lowercase and collapse whitespace (used for VG names and predicates)."""
    return re.sub(r'\s+', ' ', str(text).lower()).strip()


def make_statements(subject, predicate, obj):
    """The four query variants and their truth labels for one positive VG relationship."""
    opposite = OPPOSITE[predicate]
    return [
        ('orig', f'The {subject} is {predicate} the {obj}.', 1),
        ('converse', f'The {obj} is {opposite} the {subject}.', 1),
        ('swap', f'The {obj} is {predicate} the {subject}.', 0),
        ('opposite', f'The {subject} is {opposite} the {obj}.', 0),
    ]


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def extraction_settings(metadata):
    """Settings stored with cached VLM outputs, without hardware or timing fields."""
    return json.dumps({'vlm': VLM, 'prompt': PROMPT_TEMPLATE, 'torch': metadata['torch'],
                       'packages': metadata['packages']}, sort_keys=True)


def record_matches(record, statement, image_sha, settings):
    """Only reuse a cached answer for the same input and model settings."""
    expected = {'statement': statement, 'image_sha': image_sha, 'settings': settings}
    return all(key in record and str(record[key]) == value for key, value in expected.items())


def seeded_priority(*parts, seed=SEED):
    """Deterministic pseudo-random sort key in [0, 1) that does not depend on row order.

    Used for the image split and for sampling, so a larger sample is always a
    superset of a smaller one and completed VLM calls can be reused.
    """
    digest = hashlib.sha256('|'.join(map(str, (seed,) + parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def write_json(path, value):
    """Atomic JSON write so an interrupted run never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

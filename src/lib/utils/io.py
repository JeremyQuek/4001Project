"""Small, resumable read/write helpers. Long GPU stages append as they go so a
Colab disconnect only loses the work since the last flush."""
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def done_keys(path, key):
    """Set of `key` values already present in a jsonl file (for resuming)."""
    return {r[key] for r in read_jsonl(path)}

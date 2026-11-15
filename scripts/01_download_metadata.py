"""Stage 2a: download the Visual Genome metadata archives into data/vg/raw/.

Only the official links are used. Archives are kept zipped and are only ever
read as data (never extracted over the project or executed).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from common import RAW, ROOT, TEAMMATE_MANIFEST, VG_BASE, VG_FILES, sha256_file, write_json


def download(url, target):
    """Stream to a .part file, then rename, so a partial download is never used."""
    part = target.with_name(target.name + '.part')
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        expected = int(response.headers.get('Content-Length', 0))
        with part.open('wb') as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
    if expected and part.stat().st_size != expected:
        raise RuntimeError(f'Incomplete download: {url}')
    part.rename(target)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    record = {'source_page': 'https://homes.cs.washington.edu/~ranjay/visualgenome/api.html', 'files': {}}
    for name in VG_FILES:
        target = RAW / name
        if not target.exists():
            print(f'Downloading {name}...', flush=True)
            download(VG_BASE + name, target)
        record['files'][name] = {'url': VG_BASE + name, 'bytes': target.stat().st_size,
                                 'sha256': sha256_file(target)}
        print(f'{name}: {target.stat().st_size / 2**20:.1f} MiB', flush=True)
    write_json(RAW / 'sources.json', record)

    # Teammate's VSR image list (branch origin/isaac of the shared repository); used
    # only to keep VSR images out of this dataset so cross-evaluation stays clean.
    if not TEAMMATE_MANIFEST.exists():
        shown = subprocess.run(
            ['git', 'show', 'origin/isaac:data/vsr/processed/directional_images.csv'],
            cwd=ROOT, capture_output=True)
        if shown.returncode == 0:
            TEAMMATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            TEAMMATE_MANIFEST.write_bytes(shown.stdout)
    print('Teammate VSR manifest:', 'available' if TEAMMATE_MANIFEST.exists() else 'NOT available')


if __name__ == '__main__':
    main()

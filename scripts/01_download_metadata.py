"""Download the Visual Genome metadata archives into data/vg/raw/.

Uses the official links. Archives stay zipped and are read directly.
"""
import subprocess

import requests

from common import RAW, ROOT, VSR_MANIFEST, VG_BASE, VG_FILES, sha256_file, write_json


def download(url, target):
    # write to .part first so a broken download isn't picked up
    part = target.with_name(target.name + '.part')
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        expected = int(response.headers.get('Content-Length', 0))
        with part.open('wb') as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
    if expected and part.stat().st_size != expected:
        raise RuntimeError(f'download incomplete: {url}')
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

    # Image list of the group's VSR experiment, used only to keep those images out of
    # this dataset. The file ships with the code; inside the group's git repository it
    # can also be restored from the branch that holds the VSR experiment.
    if not VSR_MANIFEST.exists():
        shown = subprocess.run(
            ['git', 'show', 'origin/isaac:data/vsr/processed/directional_images.csv'],
            cwd=ROOT, capture_output=True)
        if shown.returncode == 0:
            VSR_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            VSR_MANIFEST.write_bytes(shown.stdout)
    print('VSR image list:', 'available' if VSR_MANIFEST.exists() else 'NOT available')


if __name__ == '__main__':
    main()

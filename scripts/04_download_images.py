"""Download only the images the sample needs (URL from VG image_data.json).

Each file is fully decoded and hashed before it is accepted. Failures are
retried, then logged to failed_images.csv; re-run 03_split_sample.py afterwards
to replace their relationships.
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

from common import IMAGES, PROCESSED, sha256_file

RETRIES = 3


def image_path(image_id):
    return IMAGES / f'{int(image_id)}.jpg'


def check_image(source):
    # full decode, catches truncated files
    with Image.open(source) as image:
        image.convert('RGB').load()
        return image.size


def fetch(row):
    target = image_path(row.image_id)
    error = ''
    if not target.exists():
        for attempt in range(RETRIES):
            try:
                response = requests.get(row.url, timeout=30)
                response.raise_for_status()
                check_image(io.BytesIO(response.content))
                part = target.with_name(target.name + '.part')
                part.write_bytes(response.content)
                part.rename(target)
                error = ''
                break
            except Exception as problem:  # network, HTTP or decode failure
                error = f'{type(problem).__name__}: {problem}'
                time.sleep(2 * (attempt + 1))
    if not target.exists():
        return {'image_id': row.image_id, 'url': row.url, 'error': error}
    try:
        width, height = check_image(target)
    except Exception as problem:
        target.unlink()
        return {'image_id': row.image_id, 'url': row.url, 'error': f'decode: {problem}'}
    return {'image_id': row.image_id, 'url': row.url, 'coco_id': row.coco_id, 'split': row.split,
            'file': target.name, 'width': width, 'height': height, 'sha256': sha256_file(target)}


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    needed = pd.read_csv(PROCESSED / 'sample_images.csv')
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(fetch, needed.itertuples()))
    good = [r for r in results if 'error' not in r]
    bad = [r for r in results if 'error' in r]
    pd.DataFrame(good).to_csv(PROCESSED / 'image_manifest.csv', index=False)
    # Append to the failure log so earlier failures stay excluded on re-sampling.
    failed_path = PROCESSED / 'failed_images.csv'
    if bad:
        old = pd.read_csv(failed_path) if failed_path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(bad)]).drop_duplicates('image_id').to_csv(failed_path, index=False)
    print(f'{len(good)} images verified, {len(bad)} failed.')
    if bad:
        print('Failures were logged; re-run 03_split_sample.py then this script to replace them.')


if __name__ == '__main__':
    main()

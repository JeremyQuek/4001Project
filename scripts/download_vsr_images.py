"""Download only images listed in a VSR manifest and verify decoded pixels."""
import argparse
import csv
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = (ROOT / 'data/vsr/images').resolve()


def inspect_image(content):
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        return {'width': image.width, 'height': image.height, 'format': image.format}


def download(row):
    target = (ROOT / row['image_path']).resolve()
    if target.parent != IMAGE_ROOT or not re.fullmatch(r'\d{12}\.jpg', target.name):
        raise ValueError('Unexpected image destination: ' + str(target))
    parsed = urlparse(row['image_url'])
    if parsed.hostname != 'images.cocodataset.org' or parsed.scheme not in ('http', 'https'):
        raise ValueError('Unexpected image URL: ' + row['image_url'])
    if Path(parsed.path).name != target.name:
        raise ValueError('Image URL and destination filename differ.')
    result = {'image_id': row['image_id'], 'image_path': str(target), 'source_url': row['image_url']}
    if target.exists():
        try:
            content = target.read_bytes()
            info = inspect_image(content)
            return dict(result, status='cached', bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), **info)
        except (OSError, ValueError):
            pass  # Replace invalid cached files only after a valid download succeeds.
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = Request(row['image_url'], headers={'User-Agent': 'VSR-research-image-download/1.0'})
            with urlopen(request, timeout=30) as response:
                content = response.read(20 * 1024 * 1024 + 1)
            if len(content) > 20 * 1024 * 1024:
                raise ValueError('Image exceeds 20 MiB download limit.')
            info = inspect_image(content)
            temporary = target.with_suffix('.jpg.part')
            temporary.write_bytes(content)
            temporary.replace(target)
            return dict(result, status='downloaded', bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), **info)
        except Exception as error:
            if attempt == 2:
                return dict(result, status='failed', error=str(error))
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/vsr/processed/pilot_images.csv')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/pilot_image_download.json')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    with args.manifest.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Expected a non-empty manifest with unique image IDs.')
    print(f'Checking/downloading {len(rows)} selected images with {args.workers} workers...', flush=True)
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(download, row) for row in rows]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            if result['status'] == 'failed':
                print(f'FAILED {result["image_id"]}: {result["error"]}', flush=True)
            elif len(results) % 10 == 0 or len(results) == len(rows):
                print(f'Verified {len(results)}/{len(rows)}', flush=True)
    failed = [r for r in results if r['status'] == 'failed']
    summary = {
        'manifest': str(args.manifest.resolve()), 'image_directory': str(IMAGE_ROOT),
        'requested': len(rows), 'verified': len(results) - len(failed), 'failed': len(failed),
        'downloaded': sum(r['status'] == 'downloaded' for r in results),
        'cached': sum(r['status'] == 'cached' for r in results),
        'total_bytes': sum(r.get('bytes', 0) for r in results),
        'elapsed_seconds': round(time.perf_counter() - start, 2),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(summary, images=sorted(results, key=lambda r: r['image_id'])), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

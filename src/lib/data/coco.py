"""COCO val2014 loading, POPE exclusion, image sampling/splitting and image download."""
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

INSTANCES = "annotations/instances_val2014.json"
CAPTIONS = "annotations/captions_val2014.json"


def download(url, dest, chunk=1 << 20):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))
    tmp.rename(dest)
    return dest


def ensure_annotations(data_dir, url):
    """Download the 2014 annotation zip once and extract only the two val2014 files we use."""
    data_dir = Path(data_dir)
    if (data_dir / INSTANCES).exists() and (data_dir / CAPTIONS).exists():
        return
    zpath = download(url, data_dir / "annotations_trainval2014.zip")
    with zipfile.ZipFile(zpath) as z:
        for member in (INSTANCES, CAPTIONS):
            z.extract(member, data_dir)


def ensure_synonyms(data_dir, url):
    return download(url, Path(data_dir) / "chair_synonyms.txt")


def load_coco(data_dir):
    """Returns (images, categories_by_image, captions_by_image).

    images: {image_id: {"file_name", "coco_url", "height", "width"}}
    categories_by_image: {image_id: set of COCO category names from instance annotations}
    captions_by_image: {image_id: [human caption, ...]}
    """
    data_dir = Path(data_dir)
    with open(data_dir / INSTANCES) as f:
        inst = json.load(f)
    with open(data_dir / CAPTIONS) as f:
        caps = json.load(f)

    images = {im["id"]: {k: im[k] for k in ("file_name", "coco_url", "height", "width")} for im in inst["images"]}
    id_to_name = {c["id"]: c["name"] for c in inst["categories"]}
    cats = {i: set() for i in images}
    for a in inst["annotations"]:
        cats[a["image_id"]].add(id_to_name[a["category_id"]])
    captions = {i: [] for i in images}
    for a in caps["annotations"]:
        captions[a["image_id"]].append(a["caption"])
    return images, cats, captions


def coco_id_from_filename(name):
    # COCO_val2014_000000310196.jpg -> 310196
    return int(Path(name).stem.split("_")[-1])


def load_pope_image_ids(data_dir, urls):
    """Image ids used by POPE (COCO splits). Files are JSON-lines with an "image" filename field."""
    ids = set()
    for url in urls:
        path = download(url, Path(data_dir) / "pope" / Path(url).name)
        with open(path) as f:
            for line in f:
                if line.strip():
                    ids.add(coco_id_from_filename(json.loads(line)["image"]))
    return ids


def sample_and_split(candidate_ids, n, fractions, seed):
    """Sample n image ids and split them by image into train/val/test."""
    ids = sorted(candidate_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n]
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def image_path(data_dir, file_name):
    return Path(data_dir) / "images" / "val2014" / file_name


def download_images(data_dir, images, ids, workers=16):
    """Download only the images we sampled, via each image's coco_url (skips files already present)."""
    def fetch(i):
        im = images[i]
        dest = image_path(data_dir, im["file_name"])
        if dest.exists() and dest.stat().st_size > 0:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                r = requests.get(im["coco_url"], timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
                return True
            except requests.RequestException:
                continue
        return False

    with ThreadPoolExecutor(workers) as pool:
        ok = list(tqdm(pool.map(fetch, ids), total=len(ids), desc="images"))
    failed = [i for i, good in zip(ids, ok) if not good]
    return failed

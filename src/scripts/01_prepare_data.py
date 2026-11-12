"""Stage 01: download COCO annotations + CHAIR synonyms, exclude POPE images, sample and split
images by image id, compute CHAIR ground truth per image, and download only the sampled photos.

Writes: <run>/splits.json, <run>/images.json, <data>/images/val2014/*.jpg
"""
from lib.config import base_parser, config_from_args
from lib.data import coco
from lib.data.chair import Chair
from lib.utils.io import write_json


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    d = cfg.data

    coco.ensure_annotations(cfg.data_dir, d.annotations_url)
    syn = coco.ensure_synonyms(cfg.data_dir, d.synonyms_url)
    images, cats, caps = coco.load_coco(cfg.data_dir)

    # Only images with at least one instance annotation (CHAIR needs object ground truth).
    candidates = {i for i in images if cats[i]}
    n_pope = 0
    if d.exclude_pope:
        pope = coco.load_pope_image_ids(cfg.data_dir, d.pope_urls)
        n_pope = len(candidates & pope)
        candidates -= pope
    splits = coco.sample_and_split(candidates, d.n_images, d.split, cfg.seed)

    chair = Chair(syn)
    meta = {}
    for split, ids in splits.items():
        for i in ids:
            meta[str(i)] = {
                "file_name": images[i]["file_name"],
                "coco_url": images[i]["coco_url"],
                "split": split,
                "coco_categories": sorted(cats[i]),
                "gt": sorted(chair.ground_truth(cats[i], caps[i])),
            }
    write_json(cfg.run_dir / "splits.json", {**splits, "excluded_pope_images": n_pope})
    write_json(cfg.run_dir / "images.json", meta)

    all_ids = [i for ids in splits.values() for i in ids]
    failed = coco.download_images(cfg.data_dir, images, all_ids, d.download_workers)
    print({k: len(v) for k, v in splits.items()}, f"excluded {n_pope} POPE images")
    if failed:
        print(f"WARNING: {len(failed)} images failed to download; rerun this script to retry: {failed[:10]}")


if __name__ == "__main__":
    main()

"""
Builds a stratified subset of train2017 instead of downloading the full
118,287-image split -- ran out of local disk quota on the target server for
the full ~18GB download (see Journel/Week4.md for the pivot).

Sampling is proportional-to-frequency per category, not uniform-random over
images: COCO's category distribution is heavily skewed (a handful of
categories like "person" appear in tens of thousands of images, others like
"toaster" or "hair drier" appear in a few dozen), so a blind uniform sample
over images would systematically underrepresent rare classes. Per category,
sampled roughly `TRAIN_SUBSET_SIZE * (images containing that category / total
train2017 images)` images, rarest categories first so their quota is filled
before an overlapping, more common category's sampling could crowd them out
(the same image containing several categories only needs to be picked once).

Only instances_train2017.json is needed to build the sample (already
downloaded as part of the shared annotations zip, see dataset_setup.py) --
no train2017 images are touched until the exact image_id list is fixed. That
list is written to config.TRAIN_SUBSET_IMAGE_IDS_PATH so the subset is
reproducible and documented, not just an ephemeral in-memory sample re-drawn
differently on every run.

Usage:
    python build_train_subset.py
"""

import json
import os
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from dataset_setup import ensure_annotations

# Individual per-image HTTP downloads are latency-bound, not bandwidth-bound --
# a thread pool gets real throughput out of them where a sequential loop
# would take hours for 20,000 images.
DOWNLOAD_WORKERS = 32


def stratified_sample_image_ids(coco, target_size, seed):
    """
    Returns a sorted list of `target_size` image ids (or as close to it as
    overlap between categories allows), sampled from `coco` (a parsed
    instances_train2017.json) so each category's representation in the
    sample tracks its frequency across the full split.
    """
    category_to_images = defaultdict(set)
    for ann in coco["annotations"]:
        category_to_images[ann["category_id"]].add(ann["image_id"])

    all_image_ids = [img["id"] for img in coco["images"]]
    total_images = len(all_image_ids)

    rng = random.Random(seed)
    sampled = set()

    # Rarest categories first, so a common category's larger quota can't
    # exhaust the pool of images a rare category also happens to appear in
    # before the rare category gets its own turn.
    categories_by_rarity = sorted(category_to_images.items(), key=lambda kv: len(kv[1]))
    for category_id, image_ids in categories_by_rarity:
        proportion = len(image_ids) / total_images
        quota = max(1, round(proportion * target_size))
        candidates = list(image_ids - sampled)
        rng.shuffle(candidates)
        sampled.update(candidates[:quota])

    # Category quotas round independently and images overlap categories, so
    # the union usually undershoots target_size slightly -- top up randomly
    # from whatever's left. Overshoot (rarer) gets trimmed the same way.
    if len(sampled) < target_size:
        remaining = [i for i in all_image_ids if i not in sampled]
        rng.shuffle(remaining)
        sampled.update(remaining[:target_size - len(sampled)])
    elif len(sampled) > target_size:
        sampled = set(rng.sample(sorted(sampled), target_size))

    return sorted(sampled)


def load_or_build_subset_ids():
    """
    Returns the subset's image ids, reading config.TRAIN_SUBSET_IMAGE_IDS_PATH
    if it already exists (reproducible re-run) or sampling + writing it fresh.
    """
    if os.path.exists(config.TRAIN_SUBSET_IMAGE_IDS_PATH):
        with open(config.TRAIN_SUBSET_IMAGE_IDS_PATH, "r") as f:
            image_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"[build_train_subset] reusing existing subset list ({len(image_ids)} ids) "
              f"from {config.TRAIN_SUBSET_IMAGE_IDS_PATH}")
        return image_ids

    ensure_annotations()
    print(f"[build_train_subset] sampling {config.TRAIN_SUBSET_SIZE} images from "
          f"{config.annotations_json('train2017')}")
    with open(config.annotations_json("train2017"), "r") as f:
        coco = json.load(f)

    image_ids = stratified_sample_image_ids(coco, config.TRAIN_SUBSET_SIZE, config.TRAIN_SUBSET_SEED)

    os.makedirs(os.path.dirname(config.TRAIN_SUBSET_IMAGE_IDS_PATH), exist_ok=True)
    with open(config.TRAIN_SUBSET_IMAGE_IDS_PATH, "w") as f:
        f.write("\n".join(str(i) for i in image_ids) + "\n")
    print(f"[build_train_subset] wrote {len(image_ids)} sampled image ids to "
          f"{config.TRAIN_SUBSET_IMAGE_IDS_PATH}")

    return image_ids


def _file_name_for(image_id, coco_images_by_id):
    entry = coco_images_by_id.get(image_id)
    if entry is not None:
        return entry["file_name"]
    return f"{image_id:012d}.jpg"  # COCO's own naming convention, fallback if not found in json


def _download_one(image_id, dest_path):
    url = config.COCO_TRAIN_IMAGE_URL_TEMPLATE.format(image_id=image_id)
    urllib.request.urlretrieve(url, dest_path)


def ensure_train_subset():
    """
    Ensures config.images_dir("train2017") contains the sampled subset's
    images (downloaded individually and in parallel, not via the full
    train2017.zip) and config.annotations_json("train2017") exists.
    Idempotent -- images already on disk are not re-downloaded.
    """
    image_ids = load_or_build_subset_ids()

    images_dir = config.images_dir("train2017")
    os.makedirs(images_dir, exist_ok=True)

    with open(config.annotations_json("train2017"), "r") as f:
        coco = json.load(f)
    coco_images_by_id = {img["id"]: img for img in coco["images"]}

    to_download = []
    num_skipped = 0
    for image_id in image_ids:
        file_name = _file_name_for(image_id, coco_images_by_id)
        dest_path = os.path.join(images_dir, file_name)
        if os.path.exists(dest_path):
            num_skipped += 1
        else:
            to_download.append((image_id, dest_path))

    print(f"[build_train_subset] {num_skipped} already present, downloading {len(to_download)} "
          f"with {DOWNLOAD_WORKERS} parallel workers")

    num_downloaded = 0
    num_failed = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_one, image_id, dest_path): image_id
                   for image_id, dest_path in to_download}
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                future.result()
                num_downloaded += 1
                if num_downloaded % 1000 == 0:
                    print(f"[build_train_subset] downloaded {num_downloaded}/{len(to_download)} images so far")
            except Exception as e:
                num_failed += 1
                print(f"[build_train_subset] WARNING: failed to download image_id={image_id}: {e}")

    print(f"[build_train_subset] done -- {num_downloaded} downloaded, {num_failed} failed, "
          f"{num_skipped} already present, {len(image_ids)} total target in {images_dir}")


if __name__ == "__main__":
    ensure_train_subset()

"""
Downloads + unpacks COCO images and instances_<split>.json annotations for a
given split into config.DATASET_DIR. Idempotent -- skips whatever's already
present, safe to call at the top of every run.
"""

import os
import zipfile
import urllib.request

import config


def _download(url, dest_path):
    print(f"[download] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)


def _unzip(zip_path, dest_dir):
    print(f"[unzip] {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def ensure_annotations():
    """Both instances_train2017.json and instances_val2017.json live in one zip."""
    needed = [config.annotations_json(split) for split in config.SPLITS]
    if all(os.path.exists(p) for p in needed):
        print(f"[dataset_setup] annotations already present, skipping download")
        return

    os.makedirs(config.DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(config.DATASET_DIR, "annotations_trainval2017.zip")
    if not os.path.exists(zip_path):
        _download(config.COCO_ANNOTATIONS_URL, zip_path)
    _unzip(zip_path, config.DATASET_DIR)
    os.remove(zip_path)

    for p in needed:
        assert os.path.exists(p), f"missing {p} after annotations setup"


def ensure_coco_split(split):
    """
    Ensures config.images_dir(split) (full image set, via the official zip)
    and its annotations json exist. Only for splits with a full-zip download
    URL in config.COCO_IMAGES_URL (val2017) -- train2017 uses a stratified
    subset instead, see build_train_subset.ensure_train_subset().
    """
    os.makedirs(config.DATASET_DIR, exist_ok=True)

    images_dir = config.images_dir(split)
    expected_count = config.EXPECTED_IMAGE_COUNT[split]
    if os.path.isdir(images_dir) and len(os.listdir(images_dir)) >= expected_count:
        print(f"[dataset_setup] {split} images already present ({images_dir}), skipping download")
    else:
        zip_path = os.path.join(config.DATASET_DIR, f"{split}.zip")
        if not os.path.exists(zip_path):
            _download(config.COCO_IMAGES_URL[split], zip_path)
        _unzip(zip_path, config.DATASET_DIR)
        os.remove(zip_path)

    ensure_annotations()

    assert os.path.isdir(images_dir), f"missing {images_dir} after setup"
    assert os.path.exists(config.annotations_json(split)), f"missing annotations for {split} after setup"


if __name__ == "__main__":
    for split in config.SPLITS:
        if split in config.COCO_IMAGES_URL:
            ensure_coco_split(split)
        else:
            from build_train_subset import ensure_train_subset
            ensure_train_subset()

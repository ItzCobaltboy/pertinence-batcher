"""
Real COCO images for the timing sweep, from the same dataset/ folders
yolo-analysis already downloaded (val2017 + the train2017 subset). Nothing is
downloaded here: a missing folder is a hard error.

Preprocessing matches ultralytics' predict() for a batch of mixed-shape
images, which is what yolo-analysis ran: cv2.imread (BGR), LetterBox to a
640x640 square (auto=False, pad value 114), BGR->RGB, HWC->CHW, float32 / 255.

Everything is loaded, letterboxed and copied to the device BEFORE timing
starts, so disk, JPEG decode, resize and host-to-device copies never land in
the timed region.
"""

import os

import numpy as np

import settings

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def build_pool_manifest(splits, per_split, seed, dataset_root=None):
    """Seeded sample of image files per split. Saved once per session so every
    run of that session draws from the same pool."""
    rng = np.random.default_rng(seed)
    entries = []
    for split in splits:
        folder = settings.images_dir(split, dataset_root)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"image folder {folder} does not exist. This sweep never downloads data: "
                f"run yolo-analysis first or pass --dataset-root.")
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
        if len(files) == 0:
            raise FileNotFoundError(f"no images in {folder}")
        take = min(per_split, len(files))
        picked = rng.choice(len(files), size=take, replace=False)
        for i in sorted(picked):
            entries.append({"split": split, "file": files[i]})
    return {"splits": list(splits), "per_split": per_split, "seed": seed,
            "dataset_root": dataset_root, "images": entries}


def batches_for_run(manifest, batch_size, rng_key, max_batches):
    """Pick which pool images go into this run's pre-built batches. Different
    runs (different rng_key) get different images; within a batch no image
    repeats. Returns a list of lists of pool indices."""
    pool = len(manifest["images"])
    n_batches = max(1, min(max_batches, pool // batch_size))
    need = n_batches * batch_size
    rng = np.random.default_rng(rng_key)
    if need <= pool:
        order = rng.permutation(pool)[:need]
    else:  # pool smaller than one batch (tiny test pools only)
        order = rng.choice(pool, size=need, replace=True)
    return [order[i * batch_size:(i + 1) * batch_size].tolist() for i in range(n_batches)]


def load_letterboxed(manifest, indices, img_size, dataset_root=None):
    """uint8 array (N, 3, img_size, img_size), RGB, letterboxed like predict()."""
    import cv2
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(img_size, img_size), auto=False, stride=32)
    out = np.empty((len(indices), 3, img_size, img_size), dtype=np.uint8)
    for row, index in enumerate(indices):
        entry = manifest["images"][index]
        path = os.path.join(settings.images_dir(entry["split"], dataset_root), entry["file"])
        image = cv2.imread(path)
        if image is None:
            raise IOError(f"could not read image {path}")
        boxed = letterbox(image=image)
        out[row] = boxed[..., ::-1].transpose(2, 0, 1)
    return out


def device_batches(manifest, batch_indices, img_size, device, dataset_root=None):
    """Pre-built float32 batches already sitting on the device."""
    import torch

    unique = sorted({i for batch in batch_indices for i in batch})
    loaded = load_letterboxed(manifest, unique, img_size, dataset_root)
    row_of = {index: row for row, index in enumerate(unique)}
    batches = []
    for batch in batch_indices:
        array = np.ascontiguousarray(loaded[[row_of[i] for i in batch]])
        tensor = torch.from_numpy(array).to(device).float().div_(255.0)
        batches.append(tensor.contiguous())
    return batches

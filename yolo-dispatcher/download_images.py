"""
Makes sure every image referenced by the three ground-truth CSVs exists under
config.DATASET_DIR, downloading only the missing ones individually from
images.cocodataset.org (no 18GB train2017.zip). On a machine that already ran
yolo-analysis, DATASET_DIR defaults to its dataset/ folder and nothing is
downloaded.

Run: python download_images.py   (stdlib + pandas only)
"""

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config

DOWNLOAD_WORKERS = 32
RETRIES = 4


def _download(image_path):
    split, file_name = image_path.split("/")
    url = config.COCO_IMAGE_URL_TEMPLATE.format(split=split, image_id=int(os.path.splitext(file_name)[0]))
    dest = os.path.join(config.DATASET_DIR, image_path)
    tmp = dest + ".part"
    for attempt in range(RETRIES):
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)   # never leaves a truncated .jpg behind
            return None
        except Exception as e:  # noqa: BLE001 - report, retry, then give up on this one image
            if attempt == RETRIES - 1:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return f"{image_path}: {e}"
            time.sleep(2 ** attempt)


def ensure_images():
    paths = []
    for csv in (config.TRAIN_GROUND_TRUTH_CSV, config.VAL_GROUND_TRUTH_CSV, config.FINAL_VAL_GROUND_TRUTH_CSV):
        if not os.path.exists(csv):
            raise FileNotFoundError(f"{csv} missing: run label_data.py first")
        paths.extend(pd.read_csv(csv, usecols=["image_path"])["image_path"].tolist())

    missing = [p for p in paths if not os.path.exists(os.path.join(config.DATASET_DIR, p))]
    print(f"COCO images dir: {config.DATASET_DIR}")
    print(f"{len(paths)} images referenced, {len(paths) - len(missing)} present, {len(missing)} to download")
    if not missing:
        return

    for split in {p.split("/")[0] for p in missing}:
        os.makedirs(os.path.join(config.DATASET_DIR, split), exist_ok=True)

    failures = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(_download, p) for p in missing]
        for done, future in enumerate(as_completed(futures), 1):
            error = future.result()
            if error:
                failures.append(error)
            if done % 1000 == 0 or done == len(missing):
                print(f"  {done}/{len(missing)} downloaded ({len(failures)} failed)")

    if failures:
        print("\n".join(failures[:20]))
        raise RuntimeError(f"{len(failures)} images failed to download; re-run to retry just those")


if __name__ == "__main__":
    ensure_images()

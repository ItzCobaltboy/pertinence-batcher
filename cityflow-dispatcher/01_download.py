# Downloads CityFlowV2 (AI City 2022 Track 1) from Google Drive with gdown (resumable),
# extracts videos + GT + camera tables into data/cityflow/AIC22_Track1/, then checks every
# video decodes and writes a table of scenario/camera/fps/resolution/frames/GT.
import argparse
import os
import shutil
import zipfile

import cv2
import pandas as pd

import config
from common import paths, videos

WANTED_FILES = ("vdo.avi", "gt.txt", "ReadMe.txt", "list_cam.txt")


def download(zip_path):
    import gdown

    if os.path.exists(zip_path) and os.path.getsize(zip_path) == config.ZIP_SIZE:
        print("zip already downloaded:", zip_path)
        return
    os.makedirs(paths.CITYFLOW, exist_ok=True)
    free_gb = shutil.disk_usage(paths.CITYFLOW).free / 1e9
    print(f"downloading {config.ZIP_NAME} ({config.ZIP_SIZE / 1e9:.1f} GB), {free_gb:.0f} GB free on D:")
    gdown.download(id=config.DATASET_FILE_ID, output=zip_path, quiet=False, resume=True)
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) != config.ZIP_SIZE:
        raise SystemExit("Download did not finish with the expected size (Drive quota?). "
                         "Run again later to resume; not hunting for mirrors.")


def extract(zip_path):
    """Extracts only what is needed: videos, GT files, cam_timestamp/cam_framenum tables."""
    target = paths.extracted_dir()
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        print("zip has", len(names), "files; first 5:", names[:5])
        wanted = [n for n in names
                  if os.path.basename(n) in WANTED_FILES or n.startswith(("cam_timestamp", "cam_framenum"))]
        todo = [n for n in wanted if not os.path.exists(os.path.join(target, n))]
        print(f"extracting {len(todo)} of {len(wanted)} wanted files to {target}")
        for i, name in enumerate(todo, 1):
            z.extract(name, target)
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def check_videos():
    scenarios = sum(config.SPLITS.values(), []) + config.STREAM_ONLY_SCENARIOS
    rows = []
    for cam in videos.find_cameras(scenarios):
        fps, header_count, width, height = videos.video_info(cam["video_path"])
        capture = cv2.VideoCapture(cam["video_path"])
        ok, _ = capture.read()          # prove the first frame decodes
        capture.release()
        rows.append({"scenario": cam["scenario"], "camera": cam["camera"], "folder": cam["folder"],
                     "video": os.path.relpath(cam["video_path"], paths.CITYFLOW), "fps": fps,
                     "width": width, "height": height, "frames_header": header_count,
                     "has_gt": cam["gt_path"] is not None, "decodes": ok})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    zip_path = os.path.join(paths.CITYFLOW, config.ZIP_NAME)
    download(zip_path)
    extract(zip_path)
    table = check_videos()
    out_dir = paths.result_dir("01_download")
    table.to_csv(os.path.join(out_dir, "cameras.csv"), index=False)
    print(table.to_string(index=False))
    print("all videos decode:", bool(table["decodes"].all()), "| cameras:", len(table))
    print("saved", out_dir)


if __name__ == "__main__":
    main()

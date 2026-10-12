"""
Downloads AICity22_Track1_MTMC_Tracking.zip (~16.9 GB) from Google Drive with
gdown. Skips it if the zip is already there at the right size; resumes a
partial download.

Run: python download.py [--data-dir DIR]
"""

import argparse
import os
import shutil
import sys

import dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=dataset.DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Install it with:\n\n    pip install --user gdown\n")
        sys.exit(1)

    os.makedirs(args.data_dir, exist_ok=True)
    path = dataset.zip_path(args.data_dir)
    size_gb = dataset.ZIP_SIZE / 1e9

    if os.path.exists(path):
        if os.path.getsize(path) == dataset.ZIP_SIZE:
            print(f"Already downloaded: {path}")
            return
        sys.exit(f"{path} exists but has the wrong size ({os.path.getsize(path)} bytes, "
                 f"expected {dataset.ZIP_SIZE}). Delete it and run again.")

    free_gb = shutil.disk_usage(args.data_dir).free / 1e9
    print(f"Downloading {dataset.ZIP_NAME} ({size_gb:.1f} GB) to {args.data_dir} "
          f"({free_gb:.1f} GB free)")
    if free_gb < size_gb:
        sys.exit("Not enough free disk for the zip.")

    gdown.download(id=dataset.DATASET_FILE_ID, output=path, quiet=False, resume=True)

    if not os.path.exists(path) or os.path.getsize(path) != dataset.ZIP_SIZE:
        sys.exit("Download did not finish with the expected size; run again to resume.")
    print(f"Done: {path}")


if __name__ == "__main__":
    main()

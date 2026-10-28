# Builds the pool tables (per-frame recall of each model vs yolov8l, ideal label) for CF_TAG's fine-tuned
# weights, then prints/saves frame-correct rates and the ideal split next to the first (v1) pool.
# Needs the GPU: it runs the fine-tuned models over the frames (detections are cached in data/detections/).
import argparse
import os

import numpy as np
import pandas as pd

import config
from common import paths, pool, report


def describe(version, split, table):
    row = {"split": split, "version": version, "frames": len(table)}
    for letter in config.MODEL_LETTERS:
        row[f"correct_{letter}"] = round(float((table[f"recall_{letter}"] >= config.RECALL_THRESHOLD).mean()), 4)
    for letter in config.MODEL_LETTERS:
        row[f"ideal_{letter}"] = round(float((table["ideal"] == letter).mean()), 4)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()
    rows = []
    for split in args.splits:
        new = pool.pool_table(split)
        if paths.TAG:
            old = pd.read_csv(os.path.join(paths.data_root(), "dispatch", f"{split}_pool.csv"))
            rows.append(describe("v1", split, old))
        rows.append(describe(paths.TAG.strip("_") or "v1", split, new))
    table = pd.DataFrame(rows)
    out_dir = paths.result_dir("pool")
    table.to_csv(os.path.join(out_dir, "pool_compare.csv"), index=False)
    report.write_summary(out_dir, "pool tables (frame-correct rate and ideal split)",
                         ["fine-tuned weights " + paths.FINETUNED, "cached yolov8l detections"], ["pool_compare.csv"],
                         table, ["frame-correct = box recall vs yolov8l >= 0.80; ideal = cheapest correct model"])
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

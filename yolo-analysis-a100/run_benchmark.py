"""
Single entry point for the YOLOv8 / COCO multi-label class-recall benchmark.
Run this and only this script:

    python run_benchmark.py

What it does, for each split in config.SPLITS (val2017, train2017):
  1. Ensures that split's COCO images + instances_<split>.json are present
     under dataset/ -- val2017 via the full official zip (dataset_setup.py),
     train2017 via a stratified subset instead of the full 118,287-image
     split (build_train_subset.py; see Journel/Week4.md for why: local disk
     budget on the target server).
  2. Loads ground-truth per-image class sets from the annotations, restricted
     to images actually present on disk (coco_gt.py).
  3. For each model in config.MODEL_POOL: runs batched inference over every
     image in the split (skipped if a raw-prediction cache already exists),
     caching raw detections to results/<split>/raw_predictions/<model>.json
     (inference.py).
  4. Scores every model's cached predictions against ground truth as
     multi-label classification (score.py) and writes one flat CSV across all
     (model, image) pairs to config.output_csv(split).

train2017 is included alongside val2017 so the dispatcher's eventual labeling
step has train-side ground truth to train the routing head on, not just the
held-out val numbers used to pick a correctness definition.

This is data-gathering only -- no dispatcher/labeling logic here. The
resulting CSVs are what the next step (deciding recall vs. exact-match vs.
some threshold as the detection-pool correctness definition) will be based on.
"""

import csv
import os

import config
from dataset_setup import ensure_coco_split
from build_train_subset import ensure_train_subset
from coco_gt import load_ground_truth
from inference import run_model_inference
from score import score_model


def _ensure_split_dataset(split):
    if split in config.COCO_IMAGES_URL:
        ensure_coco_split(split)
    else:
        ensure_train_subset()


def run_split(split):
    print(f"[run_benchmark] === split: {split} ===")

    print(f"[run_benchmark] {split} step 1/4: ensuring COCO {split} dataset is present")
    _ensure_split_dataset(split)

    print(f"[run_benchmark] {split} step 2/4: loading ground truth")
    image_id_to_file_name, image_id_to_gt_classes, _category_id_to_name = load_ground_truth(split)
    file_name_to_image_id = {v: k for k, v in image_id_to_file_name.items()}
    print(f"[run_benchmark] {split}: {len(image_id_to_file_name)} images, "
          f"{sum(len(c) for c in image_id_to_gt_classes.values())} total gt class instances")

    print(f"[run_benchmark] {split} step 3/4: running inference for {len(config.MODEL_POOL)} models: {config.MODEL_POOL}")
    for model_name in config.MODEL_POOL:
        run_model_inference(model_name, split, file_name_to_image_id)

    print(f"[run_benchmark] {split} step 4/4: scoring all models against ground truth")
    os.makedirs(config.results_dir(split), exist_ok=True)
    fieldnames = [
        "image_id", "model", "num_gt_classes", "num_pred_classes",
        "recall", "exact_match", "gt_classes", "pred_classes",
    ]
    output_csv = config.output_csv(split)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in config.MODEL_POOL:
            rows = score_model(model_name, split, image_id_to_file_name, image_id_to_gt_classes)
            for row in rows:
                row["model"] = model_name
                writer.writerow(row)
            print(f"[run_benchmark] {split}/{model_name}: {len(rows)} rows written")

    print(f"[run_benchmark] {split} done -- {output_csv}")


def main():
    for split in config.SPLITS:
        run_split(split)
    print("[run_benchmark] all splits done")


if __name__ == "__main__":
    main()

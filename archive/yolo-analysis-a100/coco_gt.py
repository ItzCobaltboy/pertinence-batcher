"""
Loads ground-truth data straight out of instances_<split>.json -- no
pycocotools dependency, just the three arrays this benchmark actually needs
(images, annotations, categories). Bounding boxes are intentionally dropped:
this pass scores multi-label classification only (which classes are present),
not detection.
"""

import json
import os
from collections import defaultdict

import config


def load_ground_truth(split):
    """
    Returns ground truth restricted to images actually present in
    config.images_dir(split) -- for val2017 (the full split downloaded) this
    is every image in the annotations json, but for train2017 (a stratified
    subset, not the full 118,287-image split -- see build_train_subset.py)
    the json still lists every image in the official split, so filtering
    against what's actually on disk keeps un-downloaded images out of
    scoring. Without this filter, score.py would treat an image no model was
    ever run on as "predicted nothing" instead of excluding it.

    Returns:
        image_id_to_file_name: dict[int, str]
        image_id_to_gt_classes: dict[int, set[str]]  (deduplicated category names)
        category_id_to_name: dict[int, str]
    """
    with open(config.annotations_json(split), "r") as f:
        coco = json.load(f)

    category_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    on_disk = set(os.listdir(config.images_dir(split)))
    image_id_to_file_name = {
        img["id"]: img["file_name"] for img in coco["images"] if img["file_name"] in on_disk
    }

    image_id_to_gt_classes = defaultdict(set)
    for ann in coco["annotations"]:
        if ann["image_id"] not in image_id_to_file_name:
            continue
        class_name = category_id_to_name[ann["category_id"]]
        image_id_to_gt_classes[ann["image_id"]].add(class_name)

    # Images with zero annotated objects (rare, but possible) still get an
    # entry with an empty set rather than being absent from the dict.
    for image_id in image_id_to_file_name:
        image_id_to_gt_classes.setdefault(image_id, set())

    return image_id_to_file_name, dict(image_id_to_gt_classes), category_id_to_name

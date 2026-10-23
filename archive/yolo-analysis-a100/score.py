"""
Scores one model's cached raw predictions against ground truth as multi-label
classification: dedup predicted boxes down to a set of predicted classes,
compare against the deduplicated ground-truth class set for that image.

Kept separate from inference.py on purpose -- this is the part that changes
when the correctness definition gets revisited later (exact-match,
confidence-thresholded, IoU-aware, ...), while the cached raw predictions
stay valid across that change.
"""

import json

import config
from inference import raw_predictions_path


def score_model(model_name, split, image_id_to_file_name, image_id_to_gt_classes):
    """
    Returns a list of row dicts, one per image, with columns matching the
    benchmark CSV schema (minus the `model` column, added by the caller):
    image_id, num_gt_classes, num_pred_classes, recall, exact_match,
    gt_classes, pred_classes.
    """
    with open(raw_predictions_path(model_name, split), "r") as f:
        predictions_by_image_id = json.load(f)

    rows = []
    for image_id in image_id_to_file_name:
        gt_classes = image_id_to_gt_classes[image_id]
        detections = predictions_by_image_id.get(str(image_id), [])
        pred_classes = {d["class_name"] for d in detections}

        intersection = pred_classes & gt_classes
        recall = (len(intersection) / len(gt_classes)) if gt_classes else 1.0
        exact_match = pred_classes == gt_classes

        rows.append({
            "image_id": image_id,
            "num_gt_classes": len(gt_classes),
            "num_pred_classes": len(pred_classes),
            "recall": recall,
            "exact_match": exact_match,
            "gt_classes": config.CLASS_LIST_SEPARATOR.join(sorted(gt_classes)),
            "pred_classes": config.CLASS_LIST_SEPARATOR.join(sorted(pred_classes)),
        })

    return rows

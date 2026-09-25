"""Derives router probabilities / a routing trace from the real
YOLO/COCO recall benchmark (`yolo-analysis/results/<split>/
coco_class_recall_benchmark.csv`, long format: image_id, model,
recall, ...).

Label per image = cheapest model (in `model_order`, default
yolov8n/s/m/l) whose recall >= `threshold` (default 0.80, the
correctness definition settled on in Journel/Week4.md's
"[DECISION] Correctness definition for the YOLO pool" entry). If no
model clears the threshold for an image, it's labeled with the largest
model (best effort, matching that entry's intent).

This recomputes the label from the raw long-format CSV every call,
with `threshold` as a parameter -- it deliberately does NOT read the
repo's `*_minimum_match.csv` files, which the journal notes get
overwritten by whatever threshold was last run there.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

DEFAULT_MODEL_ORDER = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]


def label_images_from_recall_csv(
    path: str,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    threshold: float = 0.80,
) -> Dict[str, str]:
    """Returns {image_id: chosen_model}, one entry per image_id found
    in the CSV. `model_order` must be cheapest-first."""
    recall_by_image: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "model", "recall"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {required}, got {reader.fieldnames}")
        for row in reader:
            recall_by_image[row["image_id"]][row["model"]] = float(row["recall"])

    labels: Dict[str, str] = {}
    for image_id, per_model in recall_by_image.items():
        chosen = None
        for model in model_order:
            if model in per_model and per_model[model] >= threshold:
                chosen = model
                break
        if chosen is None:
            # none pass -> largest model in model_order that we actually
            # have a score for
            for model in reversed(model_order):
                if model in per_model:
                    chosen = model
                    break
        if chosen is not None:
            labels[image_id] = chosen
    return labels


def label_counts_to_weights(
    labels: Dict[str, str], model_order: Sequence[str] = DEFAULT_MODEL_ORDER
) -> List[float]:
    """Turns {image_id: model} into per-model empirical frequencies,
    ordered by `model_order` -- feed straight into WeightedRandomRouter
    or StickyRouter's base_weights."""
    counts = {m: 0 for m in model_order}
    for m in labels.values():
        if m in counts:
            counts[m] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no labeled images matched model_order")
    return [counts[m] / total for m in model_order]


def labels_to_trace(
    labels: Dict[str, str], model_order: Sequence[str] = DEFAULT_MODEL_ORDER
) -> List[int]:
    """Turns {image_id: model} into a queue-id trace (image_id sorted
    ascending for a deterministic, reproducible order), one entry per
    image, model index within `model_order` as the queue id. Feed
    straight into TraceRouter."""
    index = {m: i for i, m in enumerate(model_order)}
    ordered_ids = sorted(labels.keys(), key=lambda x: int(x))
    return [index[labels[i]] for i in ordered_ids if labels[i] in index]

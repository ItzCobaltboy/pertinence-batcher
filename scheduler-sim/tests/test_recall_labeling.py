"""Router label derivation on a tiny hand-made recall CSV, matching the
long-format schema of yolo-analysis/results/<split>/coco_class_recall_benchmark.csv
(image_id, model, recall, ...)."""
import csv
import tempfile

from scheduler_sim.routers.recall_labeling import (
    label_counts_to_weights,
    label_images_from_recall_csv,
    labels_to_trace,
)

MODEL_ORDER = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]


def _write_csv(rows):
    f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
    writer = csv.DictWriter(f, fieldnames=["image_id", "model", "recall"])
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name


def test_cheapest_passing_model_is_chosen():
    rows = [
        # image 1: nano already clears 0.80 -> nano
        {"image_id": "1", "model": "yolov8n.pt", "recall": 0.90},
        {"image_id": "1", "model": "yolov8s.pt", "recall": 0.95},
        {"image_id": "1", "model": "yolov8m.pt", "recall": 0.98},
        {"image_id": "1", "model": "yolov8l.pt", "recall": 1.00},
        # image 2: nano fails, small clears -> small
        {"image_id": "2", "model": "yolov8n.pt", "recall": 0.50},
        {"image_id": "2", "model": "yolov8s.pt", "recall": 0.85},
        {"image_id": "2", "model": "yolov8m.pt", "recall": 0.90},
        {"image_id": "2", "model": "yolov8l.pt", "recall": 0.95},
        # image 3: nobody clears 0.80 -> largest model (fallback)
        {"image_id": "3", "model": "yolov8n.pt", "recall": 0.10},
        {"image_id": "3", "model": "yolov8s.pt", "recall": 0.20},
        {"image_id": "3", "model": "yolov8m.pt", "recall": 0.30},
        {"image_id": "3", "model": "yolov8l.pt", "recall": 0.40},
    ]
    path = _write_csv(rows)
    labels = label_images_from_recall_csv(path, model_order=MODEL_ORDER, threshold=0.80)
    assert labels == {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt",
        "3": "yolov8l.pt",
    }


def test_threshold_changes_the_label():
    rows = [
        {"image_id": "1", "model": "yolov8n.pt", "recall": 0.70},
        {"image_id": "1", "model": "yolov8s.pt", "recall": 0.85},
        {"image_id": "1", "model": "yolov8m.pt", "recall": 0.95},
        {"image_id": "1", "model": "yolov8l.pt", "recall": 1.00},
    ]
    path = _write_csv(rows)
    loose = label_images_from_recall_csv(path, model_order=MODEL_ORDER, threshold=0.65)
    strict = label_images_from_recall_csv(path, model_order=MODEL_ORDER, threshold=0.90)
    assert loose["1"] == "yolov8n.pt"
    assert strict["1"] == "yolov8m.pt"


def test_label_counts_to_weights_and_trace():
    rows = [
        {"image_id": "1", "model": "yolov8n.pt", "recall": 0.90},
        {"image_id": "2", "model": "yolov8n.pt", "recall": 0.10},
        {"image_id": "2", "model": "yolov8s.pt", "recall": 0.85},
        {"image_id": "3", "model": "yolov8n.pt", "recall": 0.85},
    ]
    path = _write_csv(rows)
    labels = label_images_from_recall_csv(path, model_order=MODEL_ORDER, threshold=0.80)
    # image 1 -> nano, image 2 -> small, image 3 -> nano
    weights = label_counts_to_weights(labels, model_order=MODEL_ORDER)
    assert weights == [2 / 3, 1 / 3, 0.0, 0.0]

    trace = labels_to_trace(labels, model_order=MODEL_ORDER)
    # sorted by image_id: 1 (nano=idx0), 2 (small=idx1), 3 (nano=idx0)
    assert trace == [0, 1, 0]

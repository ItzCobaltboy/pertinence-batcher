"""
Runs one YOLOv8 checkpoint over every image in a given COCO split and caches
the raw, per-image detections (boxes + classes + confidences) to disk. Boxes
are kept in the cache even though this first scoring pass only uses the class
list -- so a future correctness definition (exact-match, confidence-
thresholded, IoU-aware, whatever) can be re-scored straight from the cache
instead of re-running inference on all 4 models (x2 splits) again.

Caches per (split, model) to results/<split>/raw_predictions/<model>.json,
keyed by image_id (string, since JSON object keys are always strings).
"""

import json
import os

import torch
from ultralytics import YOLO

import config


def _resolve_device():
    if config.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print(f"[inference] {config.DEVICE} requested but CUDA unavailable, falling back to cpu")
        return "cpu"
    return config.DEVICE


def raw_predictions_path(model_name, split):
    stem = os.path.splitext(model_name)[0]
    return os.path.join(config.raw_predictions_dir(split), f"{stem}.json")


def run_model_inference(model_name, split, file_name_to_image_id):
    """
    Runs `model_name` over every image in config.images_dir(split), batched
    per config.INFERENCE_BATCH_SIZE. Writes {image_id(str): [detection, ...]}
    to raw_predictions_path(model_name, split), where each detection is
    {"class_id": int, "class_name": str, "confidence": float,
     "bbox_xyxy": [x1, y1, x2, y2]}.

    Skips inference entirely if the cache file already exists.
    """
    out_path = raw_predictions_path(model_name, split)
    if os.path.exists(out_path):
        print(f"[inference] {split}/{model_name}: cache already exists at {out_path}, skipping")
        return out_path

    os.makedirs(config.raw_predictions_dir(split), exist_ok=True)
    device = _resolve_device()

    print(f"[inference] {model_name}: loading checkpoint (weights dir: {config.MODELS_DIR})")
    weights_path = os.path.join(config.MODELS_DIR, model_name)
    model = YOLO(weights_path if os.path.exists(weights_path) else model_name)

    images_dir = config.images_dir(split)
    print(f"[inference] {split}/{model_name}: running batched inference over {images_dir} "
          f"(batch={config.INFERENCE_BATCH_SIZE}, device={device})")
    results_stream = model.predict(
        source=images_dir,
        batch=config.INFERENCE_BATCH_SIZE,
        imgsz=config.IMG_SIZE,
        conf=config.CONF_THRESHOLD,
        device=device,
        stream=True,
        verbose=False,
        save=False,
    )

    predictions_by_image_id = {}
    num_processed = 0
    for result in results_stream:
        file_name = os.path.basename(result.path)
        image_id = file_name_to_image_id.get(file_name)
        if image_id is None:
            print(f"[inference] WARNING: {file_name} not found in annotations, skipping")
            continue

        detections = []
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(boxes.conf[i].item()),
                    "bbox_xyxy": [float(v) for v in boxes.xyxy[i].tolist()],
                })

        predictions_by_image_id[str(image_id)] = detections
        num_processed += 1
        if num_processed % 2000 == 0:
            print(f"[inference] {split}/{model_name}: {num_processed} images done")

    print(f"[inference] {split}/{model_name}: {num_processed} images total, writing cache to {out_path}")
    with open(out_path, "w") as f:
        json.dump(predictions_by_image_id, f)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return out_path

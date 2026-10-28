# Runs a YOLO model on image files (or arrays) and returns road-class boxes per image.
# cached_detections() keeps the boxes of one model on one split on disk so reruns are fast.
import os
import pickle

from ultralytics import YOLO

import config
from common import paths


def load_model(weights):
    return YOLO(weights)


def weights_for(variant, letter, smoke=False):
    """variant 'ft' = fine-tuned weights, anything else = stock COCO weights."""
    if variant == "ft":
        return paths.finetuned_path(letter, smoke)
    return paths.weights_path(letter)


def detect(model, images, batch=config.DETECT_BATCH):
    """images: list of file paths or BGR arrays. Returns a list of {"boxes","cls","conf"}
    dicts (numpy, xyxy in pixels), road classes only."""
    out = []
    for start in range(0, len(images), batch):
        results = model.predict(images[start:start + batch], imgsz=config.IMG_SIZE, conf=config.DETECT_CONF,
                                classes=list(config.ROAD_CLASSES), device=0, verbose=False)
        for r in results:
            out.append({"boxes": r.boxes.xyxy.cpu().numpy(),
                        "cls": r.boxes.cls.cpu().numpy().astype(int),
                        "conf": r.boxes.conf.cpu().numpy()})
    return out


def cached_detections(variant, letter, split, image_paths, smoke=False):
    cache_dir = os.path.join(paths.data_root(smoke), "detections")
    os.makedirs(cache_dir, exist_ok=True)
    tag = paths.TAG if variant == "ft" else ""   # only fine-tuned detections depend on the experiment tag
    cache = os.path.join(cache_dir, f"{variant}{tag}_{letter}_{split}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    dets = detect(load_model(weights_for(variant, letter, smoke)), image_paths)
    with open(cache, "wb") as f:
        pickle.dump(dets, f)
    return dets

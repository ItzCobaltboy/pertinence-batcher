"""
Builds the CityFlow training/eval data for the dispatcher: for every
FRAME_STRIDE-th frame of each selected camera, it runs

  - the dispatcher's extractor (yolo-dispatcher's frozen yolov8n backbone,
    same preprocessing as cityflow-routing/route.py) -> 448-d embedding;
  - all four YOLOv8 models (n/s/m/l) on the frame -> boxes of road classes;

then scores n/s/m/l by box recall against yolov8l's boxes and labels the
frame with the cheapest model reaching RECALL_THRESHOLD (see cf_config).

Writes, per camera (resumable: a camera with a labels file is skipped):
  data/labels/<scenario>_<camera>.csv.gz   one row per sampled frame
  data/embeddings/<scenario>_<camera>.npz  float32 [n, 448] + frame_idx, same order
Frames are read straight from vdo.avi, never written to disk.

Run: python label.py                        (TRAIN_SCENARIOS + EVAL_SCENARIOS)
     python label.py --scenarios S01        (subset)
"""

import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

import cf_config as C
import matching

LETTERS = [name[-1] for name in C.MODEL_NAMES]       # n, s, m, l
REF_INDEX = C.MODEL_NAMES.index(C.REFERENCE_MODEL)


# ---------------------------------------------------------------- models

def preprocess_bgr(frame_bgr):
    """Identical to cityflow-routing/route.py preprocess_bgr: OpenCV BGR ->
    RGB -> PIL -> yolo-dispatcher IMAGE_TRANSFORM (Letterbox 640, /255)."""
    import cv2
    from PIL import Image

    return C.yd.IMAGE_TRANSFORM(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))


def build_embedder(device):
    import torch

    extractor = C.yd.build_feature_extractor(device).to(device).eval()

    def embed(frames):
        batch = torch.stack([preprocess_bgr(f) for f in frames]).to(device)
        with torch.no_grad():
            return extractor(batch).float().cpu().numpy()
    return embed


def resolve_weights(model_name):
    """Same lookup as yolo_backbone.load_detection_model: an existing .pt in
    yolo-dispatcher's search dirs, else a path in its models/ dir (ultralytics
    downloads there)."""
    file_name = f"{model_name}.pt"
    for d in C.yd.WEIGHT_SEARCH_DIRS:
        if os.path.exists(os.path.join(d, file_name)):
            return os.path.join(d, file_name)
    os.makedirs(C.yd.MODELS_DIR, exist_ok=True)
    return os.path.join(C.yd.MODELS_DIR, file_name)


def build_detectors(device):
    """{model_name: detect(frames_bgr) -> [ {boxes, cls, conf} per frame ]}.
    ultralytics treats numpy frames as BGR, which is what OpenCV decodes."""
    from ultralytics import YOLO

    detectors = {}
    for name in C.MODEL_NAMES:
        model = YOLO(resolve_weights(name))

        def detect(frames, model=model):
            results = model.predict(frames, imgsz=C.IMG_SIZE, conf=C.DETECT_CONF, classes=C.DETECT_CLASSES,
                                    device=device, verbose=False)
            return [{"boxes": r.boxes.xyxy.cpu().numpy(),
                     "cls": r.boxes.cls.cpu().numpy().astype(int),
                     "conf": r.boxes.conf.cpu().numpy()} for r in results]
        detectors[name] = detect
    return detectors


# ---------------------------------------------------------------- labelling

def label_batch(frames, embed, detectors):
    """Returns (embeddings float32 [k, 448], list of per-frame dicts)."""
    embeddings = embed(frames)
    detections = {name: detectors[name](frames) for name in C.MODEL_NAMES}
    rows = []
    for k in range(len(frames)):
        ref = detections[C.REFERENCE_MODEL][k]
        recalls = [matching.recall_vs_reference(ref, detections[name][k], C.MATCH_IOU) for name in C.MODEL_NAMES]
        ideal = matching.cheapest_correct(recalls, C.RECALL_THRESHOLD)
        row = {"n_ref": len(ref["boxes"])}
        for letter, name, recall in zip(LETTERS, C.MODEL_NAMES, recalls):
            row[f"boxes_{letter}"] = len(detections[name][k]["boxes"])
            row[f"recall_{letter}"] = round(float(recall), 4)
        row["ideal_label"] = ideal
        row["ideal_model"] = C.MODEL_NAMES[ideal]
        rows.append(row)
    return embeddings.astype(np.float32), rows


def write_atomic(path, write_fn):
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
    write_atomic(path, write)


def label_camera(cam, embed, detectors, stride, batch_size, labels_dir, embeddings_dir, log):
    """Labels one camera. The labels CSV is written last, so its presence
    means the camera is complete."""
    stem = f"{cam['scenario']}_{cam['camera']}"
    capture, fps, header_count = C.cf_dataset.open_video(cam["video_path"])
    started = time.time()

    all_emb, all_rows, frame_ids = [], [], []
    frames, ids = [], []

    def flush():
        emb, rows = label_batch(frames, embed, detectors)
        all_emb.append(emb)
        all_rows.extend(rows)
        frame_ids.extend(ids)
        frames.clear()
        ids.clear()

    for frame_idx, frame in C.cf_dataset.iter_frames(capture):
        if (frame_idx - 1) % stride:
            continue
        frames.append(frame)
        ids.append(frame_idx)
        if len(frames) == batch_size:
            flush()
    if frames:
        flush()
    capture.release()

    if not frame_ids:
        log(f"  {stem}: no frames decoded, skipped")
        return None

    frame_idx = np.array(frame_ids, dtype=np.int64)
    embeddings = np.concatenate(all_emb)
    emb_path = os.path.join(embeddings_dir, f"{stem}.npz")

    def write_npz(tmp):
        with open(tmp, "wb") as f:
            np.savez(f, embeddings=embeddings, frame_idx=frame_idx)
    write_atomic(emb_path, write_npz)

    table = pd.DataFrame(all_rows)
    table.insert(0, "frame_idx", frame_idx)
    table.insert(0, "fps", fps)
    table.insert(0, "role", C.role_of(cam["scenario"]))
    table.insert(0, "camera", cam["camera"])
    table.insert(0, "scenario", cam["scenario"])
    table.insert(0, "split", cam["split"])
    labels_path = os.path.join(labels_dir, f"{stem}.csv.gz")
    write_atomic(labels_path, lambda tmp: table.to_csv(tmp, index=False, compression="gzip"))

    seconds = time.time() - started
    split = table["ideal_model"].value_counts(normalize=True).reindex(C.MODEL_NAMES, fill_value=0)
    log(f"  {stem}: {len(table)} frames (every {stride}th of ~{header_count}) in {seconds:.0f} s, "
        f"labels " + " ".join(f"{l}={v:.0%}" for l, v in zip(LETTERS, split.values)))
    return {"frames_labelled": len(table), "fps": fps, "frames_header": header_count,
            "seconds": round(seconds, 1), "label_split": dict(zip(C.MODEL_NAMES, map(float, split.values)))}


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, message):
        line = f"{datetime.datetime.now():%H:%M:%S}  {message}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def main(argv=None, embed=None, detectors=None):
    """embed/detectors can be injected (tests); by default they are the real
    extractor and the four YOLOv8 models on cuda:0 if available."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=C.TRAIN_SCENARIOS + C.EVAL_SCENARIOS)
    parser.add_argument("--video-data-dir", default=C.VIDEO_DATA_DIR,
                        help="where cityflow-routing/extract.py put the videos")
    parser.add_argument("--data-dir", default=C.DATA_DIR, help="output: labels/ and embeddings/")
    parser.add_argument("--stride", type=int, default=C.FRAME_STRIDE)
    parser.add_argument("--batch", type=int, default=C.DETECT_BATCH)
    args = parser.parse_args(argv)

    for scenario in args.scenarios:
        C.role_of(scenario)   # refuses scenarios outside train/eval lists

    labels_dir = os.path.join(args.data_dir, "labels")
    embeddings_dir = os.path.join(args.data_dir, "embeddings")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    log = Logger(os.path.join(args.data_dir, "label.log"))

    root = C.cf_dataset.extracted_dir(args.video_data_dir)
    cameras = C.cf_dataset.find_cameras(root, args.scenarios)
    found = sorted({c["scenario"] for c in cameras})
    missing = [s for s in args.scenarios if s not in found]
    if missing:
        sys.exit(f"No videos for {missing} under {root}. From cityflow-routing/: "
                 f"python extract.py --scenarios {' '.join(missing)}")

    todo = [c for c in cameras
            if not os.path.exists(os.path.join(labels_dir, f"{c['scenario']}_{c['camera']}.csv.gz"))]
    log(f"{len(cameras)} cameras in {args.scenarios}, {len(cameras) - len(todo)} already labelled, "
        f"stride {args.stride}, reference {C.REFERENCE_MODEL}, recall >= {C.RECALL_THRESHOLD} at IoU {C.MATCH_IOU}")
    if not todo:
        log("Nothing to do.")
        return

    device = None
    if embed is None or detectors is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device != "cpu" else ""))
    embed = embed or build_embedder(device)
    detectors = detectors or build_detectors(device)

    manifest = {}
    manifest_path = os.path.join(args.data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest.update({
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "train_scenarios": C.TRAIN_SCENARIOS, "eval_scenarios": C.EVAL_SCENARIOS,
        "frame_stride": args.stride, "reference_model": C.REFERENCE_MODEL,
        "recall_threshold": C.RECALL_THRESHOLD, "match_iou": C.MATCH_IOU,
        "detect_classes": C.DETECT_CLASSES, "detect_conf": C.DETECT_CONF, "img_size": C.IMG_SIZE,
        "model_names": C.MODEL_NAMES, "host": platform.node(),
    })
    manifest.setdefault("cameras", {})

    for cam in todo:
        entry = label_camera(cam, embed, detectors, args.stride, args.batch, labels_dir, embeddings_dir, log)
        if entry:
            manifest["cameras"][f"{cam['scenario']}_{cam['camera']}"] = entry
            write_json(manifest_path, manifest)
    log(f"Done. Labels in {labels_dir}")


if __name__ == "__main__":
    main()

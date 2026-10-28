# Builds the train/val/test frame set: every 5th frame per camera as JPEG, pseudo-labels from
# frozen yolov8l (YOLO txt format), manifest.csv, ultralytics yaml, and (test split) the official
# GT boxes for the sampled frames plus 6 overlay images to see how complete GT is.
# Resumable per camera. --smoke does the first camera of each split, 100 frames, into data/smoke/.
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml

import config
from common import detect, paths, report, videos

SMOKE_FRAMES = 100


def write_label_file(path, dets, width, height):
    """YOLO format: class x_center y_center w h, all normalised. Class id = COCO id."""
    with open(path, "w") as f:
        for box, cls in zip(dets["boxes"], dets["cls"]):
            x1, y1, x2, y2 = box
            f.write(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")


def label_camera(model, cam, split, smoke, part_path):
    """Saves the sampled frames of one camera and their pseudo-labels, writes the camera's
    manifest part last (its presence means the camera is done)."""
    img_dir, lab_dir = paths.images_dir(split, smoke), paths.labels_dir(split, smoke)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    fps, _, width, height = videos.video_info(cam["video_path"])
    max_frames = SMOKE_FRAMES if smoke else None

    rows = []
    pending = []                     # (frame_idx, image_path) waiting for detection

    def flush():
        detections = detect.detect(model, [p for _, p in pending])
        for (frame_idx, image_path), dets in zip(pending, detections):
            label_path = os.path.join(lab_dir, os.path.basename(image_path)[:-4] + ".txt")
            write_label_file(label_path, dets, width, height)
            rows.append({"split": split, "scenario": cam["scenario"], "camera": cam["camera"],
                         "frame_idx": frame_idx, "time_sec": round((frame_idx - 1) / fps, 3),
                         "path": image_path, "width": width, "height": height,
                         "n_boxes_l": len(dets["boxes"]),
                         "classes_l": " ".join(str(c) for c in dets["cls"])})
        pending.clear()

    for frame_idx, frame in videos.sampled_frames(cam["video_path"], max_frames=max_frames):
        name = f"{cam['scenario']}_{cam['camera']}_{frame_idx:05d}.jpg"
        image_path = os.path.join(img_dir, name)
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        pending.append((frame_idx, image_path))
        if len(pending) == config.DETECT_BATCH:
            flush()
    if pending:
        flush()
    pd.DataFrame(rows).to_csv(part_path, index=False)
    return len(rows)


def read_gt(gt_path, frame_ids):
    """MOT-style gt.txt: frame,id,x,y,w,h,... -> rows for the wanted frames only (xyxy)."""
    gt = pd.read_csv(gt_path, header=None, usecols=range(6), names=["frame", "id", "x", "y", "w", "h"])
    gt = gt[gt["frame"].isin(frame_ids)].copy()
    gt["x1"], gt["y1"] = gt["x"], gt["y"]
    gt["x2"], gt["y2"] = gt["x"] + gt["w"], gt["y"] + gt["h"]
    return gt[["frame", "id", "x1", "y1", "x2", "y2"]]


def write_gt_test(manifest, smoke, out_dir):
    """Official GT boxes of the sampled test frames + 6 overlay images."""
    test = manifest[manifest["split"] == "test"]
    cameras = {c["camera"]: c for c in videos.find_cameras(config.SPLITS["test"])}
    parts = []
    for camera, rows in test.groupby("camera"):
        gt_path = cameras[camera]["gt_path"]
        if gt_path is None:
            continue
        gt = read_gt(gt_path, set(rows["frame_idx"]))
        gt.insert(0, "camera", camera)
        parts.append(gt.rename(columns={"frame": "frame_idx"}))
    if not parts:
        return None
    gt_all = pd.concat(parts)
    gt_all.to_csv(paths.gt_test_boxes(smoke), index=False)

    check_dir = os.path.join(out_dir, "gt_check")
    os.makedirs(check_dir, exist_ok=True)
    picks = test.iloc[np.linspace(0, len(test) - 1, 6).astype(int)]
    for _, row in picks.iterrows():
        image = cv2.imread(row["path"])
        boxes = gt_all[(gt_all["camera"] == row["camera"]) & (gt_all["frame_idx"] == row["frame_idx"])]
        for _, b in boxes.iterrows():
            cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 0), 2)
        cv2.putText(image, f"GT boxes: {len(boxes)}  yolov8l boxes: {row['n_boxes_l']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(check_dir, os.path.basename(row["path"])), image)
    return gt_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model = detect.load_model(paths.weights_path(config.REFERENCE_LETTER))
    parts_dir = os.path.join(paths.data_root(args.smoke), "manifest_parts")
    os.makedirs(parts_dir, exist_ok=True)

    manifest_parts = []
    for split, scenarios in config.SPLITS.items():
        cameras = videos.find_cameras(scenarios)
        if not cameras:
            raise SystemExit(f"no videos for {split} {scenarios}; run 01_download.py")
        if args.smoke:
            cameras = cameras[:1]
        for cam in cameras:
            part_path = os.path.join(parts_dir, f"{split}_{cam['scenario']}_{cam['camera']}.csv")
            if os.path.exists(part_path) and not args.force:
                print("skip (done):", part_path)
            else:
                n = label_camera(model, cam, split, args.smoke, part_path)
                print(f"{split} {cam['scenario']} {cam['camera']}: {n} frames", flush=True)
            manifest_parts.append(pd.read_csv(part_path))

    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest.to_csv(paths.manifest_path(args.smoke), index=False)

    names = {int(k): v for k, v in model.names.items()}
    data_yaml = {"path": paths.data_root(args.smoke).replace("\\", "/"), "train": "images/train",
                 "val": "images/val", "test": "images/test", "names": names}
    with open(paths.dataset_yaml(args.smoke), "w") as f:
        yaml.safe_dump(data_yaml, f)

    out_dir = paths.result_dir("02_dataset", args.smoke)
    gt = write_gt_test(manifest, args.smoke, out_dir)

    def empty_frac(x):
        return round(float((x == 0).mean()), 3)

    per_camera = (manifest.groupby(["split", "scenario", "camera"])
                  .agg(frames=("path", "count"), boxes=("n_boxes_l", "sum"), empty_frac=("n_boxes_l", empty_frac))
                  .reset_index())
    per_camera.to_csv(os.path.join(out_dir, "per_camera.csv"), index=False)
    per_split = (manifest.groupby("split")
                 .agg(frames=("path", "count"), boxes=("n_boxes_l", "sum"), empty_frac=("n_boxes_l", empty_frac))
                 .reset_index())
    classes = pd.Series([int(c) for s in manifest["classes_l"].dropna() for c in str(s).split()])
    hist = classes.map(config.ROAD_CLASSES).value_counts().rename_axis("class").reset_index(name="pseudo_boxes")
    hist.to_csv(os.path.join(out_dir, "class_histogram.csv"), index=False)

    extra = ["Cameras are never shared between splits: scenarios are disjoint, so no camera is in two splits.",
             "Class histogram of pseudo-labels: " + ", ".join(f"{r['class']}={r['pseudo_boxes']}" for _, r in hist.iterrows())]
    flags = []
    if gt is not None:
        n_l = int(manifest[manifest["split"] == "test"]["n_boxes_l"].sum())
        extra.append(f"Test split: {len(gt)} official GT boxes vs {n_l} yolov8l pseudo-boxes on the same frames "
                     f"(overlays in gt_check/).")
        flags.append(f"GT completeness: {len(gt)} GT boxes vs {n_l} yolov8l boxes on the same test frames.")
    else:
        flags.append("No GT file found for the test split.")
    report.write_summary(out_dir, "02 make dataset", ["data/cityflow (videos)", "models/yolov8l.pt"],
                         [paths.manifest_path(args.smoke), paths.dataset_yaml(args.smoke), "per_camera.csv",
                          "class_histogram.csv"], per_split, flags, extra)
    print(per_split.to_string(index=False))
    print("saved", out_dir)


if __name__ == "__main__":
    main()

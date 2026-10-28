# Compares original vs fine-tuned yolov8 n, s, m on the test split (val for reference):
# box recall/precision vs yolov8l, mAP vs pseudo-labels, frame-correct rate, recall on official GT
# vehicle boxes, per-camera table, and the ideal-label split (cheapest correct model) of both pools.
# Detections are cached in data/detections/. --smoke uses the small dataset and smoke weights.
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import config
from common import detect, paths, report
from common.boxes import count_matches
from common.labels import ideal_letter

COMPARE_SPLITS = ["test", "val"]
VEHICLE_CLASSES = [2, 3, 5, 7]       # official GT is vehicles only


def frame_rows(manifest_split, ref_dets, dets, variant, letter, split):
    """One row per frame: matched boxes, counts and recall vs the reference."""
    rows = []
    for (_, m), ref, pred in zip(manifest_split.iterrows(), ref_dets, dets):
        matched = count_matches(ref, pred, config.MATCH_IOU)
        n_ref, n_pred = len(ref["boxes"]), len(pred["boxes"])
        rows.append({"split": split, "camera": m["camera"], "frame_idx": m["frame_idx"], "variant": variant,
                     "model": letter, "matched": matched, "n_ref": n_ref, "n_pred": n_pred,
                     "recall": 1.0 if n_ref == 0 else matched / n_ref})
    return rows


def gt_recall(manifest_split, dets, gt):
    """Recall on official GT boxes: class-agnostic over vehicle classes, extra detections ignored
    (GT is incomplete, so precision against it means nothing). Micro-average over all GT boxes."""
    matched, total = 0, 0
    for (_, m), pred in zip(manifest_split.iterrows(), dets):
        boxes = gt[(gt["camera"] == m["camera"]) & (gt["frame_idx"] == m["frame_idx"])]
        ref = {"boxes": boxes[["x1", "y1", "x2", "y2"]].to_numpy(), "cls": np.zeros(len(boxes), dtype=int)}
        keep = np.isin(pred["cls"], VEHICLE_CLASSES)
        vehicle_pred = {"boxes": pred["boxes"][keep], "cls": np.zeros(int(keep.sum()), dtype=int), "conf": pred["conf"][keep]}
        matched += count_matches(ref, vehicle_pred, config.MATCH_IOU)
        total += len(boxes)
    return matched / total if total else float("nan")


def map_scores(variant, letter, split, smoke):
    model = YOLO(detect.weights_for(variant, letter, smoke))
    result = model.val(data=paths.dataset_yaml(smoke), split=split, imgsz=config.IMG_SIZE, batch=16, workers=0,
                       device=0, plots=False, verbose=False, project=os.path.join(paths.MODELS, "runs_val"),
                       name=f"{variant}_{letter}_{split}", exist_ok=True)
    return result.box.map50, result.box.map


def ideal_split(frames, variant):
    """Fraction of frames whose cheapest correct model is n, s, m, l for one pool.
    variant 'orig' = stock n,s,m; 'ft' = fine-tuned n,s,m. l is always the stock l."""
    pivot = {}
    for letter in config.MODEL_LETTERS:
        use = "orig" if letter == "l" else variant
        rows = frames[(frames["variant"] == use) & (frames["model"] == letter)]
        pivot[letter] = rows.set_index("frame_idx")["recall"]
    table = pd.DataFrame(pivot).dropna()
    labels = [ideal_letter(dict(row)) for _, row in table.iterrows()]
    return pd.Series(labels).value_counts(normalize=True).reindex(config.MODEL_LETTERS, fill_value=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(paths.manifest_path(args.smoke))
    gt_path = paths.gt_test_boxes(args.smoke)
    gt = pd.read_csv(gt_path) if os.path.exists(gt_path) else None

    all_rows, overall = [], []
    for split in COMPARE_SPLITS:
        m = manifest[manifest["split"] == split].sort_values(["camera", "frame_idx"]).reset_index(drop=True)
        images = list(m["path"])
        ref_dets = detect.cached_detections("orig", "l", split, images, args.smoke)
        for variant in ["orig", "ft"]:
            for letter in config.FINETUNED_LETTERS:
                print(f"{split} {variant} yolov8{letter}", flush=True)
                dets = detect.cached_detections(variant, letter, split, images, args.smoke)
                rows = frame_rows(m, ref_dets, dets, variant, letter, split)
                all_rows += rows
                table = pd.DataFrame(rows)
                map50, map5095 = map_scores(variant, letter, split, args.smoke)
                overall.append({
                    "split": split, "variant": variant, "model": f"yolov8{letter}",
                    "recall_vs_l": table["matched"].sum() / table["n_ref"].sum(),
                    "precision_vs_l": table["matched"].sum() / max(table["n_pred"].sum(), 1),
                    "mAP50": map50, "mAP50_95": map5095,
                    "frame_correct": (table["recall"] >= config.RECALL_THRESHOLD).mean(),
                    "gt_recall": gt_recall(m, dets, gt) if (gt is not None and split == "test") else float("nan")})
        # reference row: stock l on official GT
        if gt is not None and split == "test":
            overall.append({"split": split, "variant": "orig", "model": "yolov8l", "gt_recall": gt_recall(m, ref_dets, gt)})
        all_rows += frame_rows(m, ref_dets, ref_dets, "orig", "l", split)

    frames = pd.DataFrame(all_rows)
    out_dir = paths.result_dir("04_compare", args.smoke)
    overall = pd.DataFrame(overall).round(4)
    overall.to_csv(os.path.join(out_dir, "overall.csv"), index=False)

    frames["correct"] = frames["recall"] >= config.RECALL_THRESHOLD
    per_camera = (frames[frames["model"] != "l"].groupby(["split", "camera", "model", "variant"])
                  .agg(recall=("recall", "mean"), frame_correct=("correct", "mean")).round(4).reset_index())
    per_camera.to_csv(os.path.join(out_dir, "per_camera.csv"), index=False)

    split_rows = []
    for split in COMPARE_SPLITS:
        for pool, variant in [("original", "orig"), ("fine-tuned", "ft")]:
            share = ideal_split(frames[frames["split"] == split], variant)
            split_rows.append({"split": split, "pool": pool, **{f"yolov8{l}": round(v, 4) for l, v in share.items()}})
    ideal = pd.DataFrame(split_rows)
    ideal.to_csv(os.path.join(out_dir, "ideal_split.csv"), index=False)

    fig, axes = plt.subplots(1, len(COMPARE_SPLITS), figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, COMPARE_SPLITS):
        part = ideal[ideal["split"] == split].set_index("pool")[[f"yolov8{l}" for l in config.MODEL_LETTERS]]
        part.T.plot.bar(ax=ax, rot=0)
        ax.set_title(f"ideal-label split, {split}")
        ax.set_ylabel("fraction of frames")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ideal_split.png"), dpi=150)

    report.write_summary(out_dir, "04 compare", ["models/yolov8*.pt, models/finetuned/", "data/manifest.csv", "data/gt_test_boxes.csv"],
                         ["overall.csv", "per_camera.csv", "ideal_split.csv", "ideal_split.png"],
                         overall, ["gt_recall is class-agnostic over car/motorcycle/bus/truck, micro-averaged; GT is incomplete."],
                         ["Ideal-label split (fraction of frames per cheapest correct model):", "", report.table_to_text(ideal)])
    print(overall.to_string(index=False))
    print(ideal.to_string(index=False))
    print("saved", out_dir)


if __name__ == "__main__":
    main()

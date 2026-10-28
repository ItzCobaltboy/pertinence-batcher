# Builds the inputs the vendored PERTINENCE code expects (same CSV and npz formats as
# archive/yolo-dispatcher/data) from the CityFlow pool tables and the cached embeddings.
# Roles: train (FC training), test (NSGA-II fitness), final_val (reported once, always S02).
#   PERT_MODE=s05   : S05 cameras are split in two folds, A trains the heads, B is the fitness set.
#                     No frame the fine-tuned models trained on is used to make a label.
#   PERT_MODE=train : train split trains the heads, S05 is the fitness set (the old COCO-style protocol).
import os

import numpy as np
import pandas as pd

import config
import pertinence_config as pc
from common import paths


def split_by_camera(table, embeddings):
    """S05 into two disjoint camera folds: sorted cameras, even positions -> A, odd positions -> B."""
    cameras = sorted(table["camera"].unique())
    fold_a = cameras[0::2]
    fold_b = cameras[1::2]
    mask_a = table["camera"].isin(fold_a).to_numpy()
    mask_b = table["camera"].isin(fold_b).to_numpy()
    return ((table[mask_a].reset_index(drop=True), embeddings[mask_a], fold_a),
            (table[mask_b].reset_index(drop=True), embeddings[mask_b], fold_b))


def ground_truth(table):
    """image_path, per-model recall and correct flags, label = cheapest correct model (yolov8l is always
    correct, recall_l is 1.0 by definition, so every frame gets a real label)."""
    out = pd.DataFrame({"image_path": table["path"], "scenario": table["scenario"], "camera": table["camera"],
                        "frame_idx": table["frame_idx"], "time_sec": table["time_sec"]})
    correct = np.zeros((len(table), pc.NUM_CLASSES), dtype=bool)
    for k, letter in enumerate(config.MODEL_LETTERS):
        recall = table[f"recall_{letter}"].to_numpy()
        out[f"{pc.MODEL_NAMES[k]}_recall"] = recall
        correct[:, k] = recall >= config.RECALL_THRESHOLD
    for k in range(pc.NUM_CLASSES):
        out[f"{pc.MODEL_NAMES[k]}_correct"] = correct[:, k].astype(int)
    label = np.where(correct.any(axis=1), correct.argmax(axis=1), pc.NO_CORRECT_MODEL_LABEL)
    out["label"] = label
    ideal_check = table["ideal"].map({l: i for i, l in enumerate(config.MODEL_LETTERS)}).to_numpy()
    assert (label == ideal_check).all(), "label differs from the pool table's ideal column"
    return out


def save_role(name, table, embeddings, csv_path, npz_path):
    gt = ground_truth(table)
    assert len(gt) == len(embeddings), (name, len(gt), len(embeddings))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    gt.to_csv(csv_path, index=False)
    np.savez(npz_path, embeddings=embeddings.astype(np.float32), labels=gt["label"].to_numpy().astype(np.int64))
    share = gt["label"].value_counts(normalize=True).reindex(range(pc.NUM_CLASSES), fill_value=0).round(3).tolist()
    print(f"{name:10s} frames {len(gt):6d} cameras {gt['camera'].nunique():3d} ideal split n/s/m/l {share} "
          f"emb mean {embeddings.mean():.3f} std {embeddings.std():.3f} max {embeddings.max():.2f}")
    return {"role": name, "frames": len(gt), "cameras": gt["camera"].nunique(),
            "ideal_n": share[0], "ideal_s": share[1], "ideal_m": share[2], "ideal_l": share[3]}


def load(split):
    """Cached pool table (data/dispatch<TAG>/<split>_pool.csv) and its cached embeddings, in table order.
    Same read as pool.split_embeddings (one npz per camera, groups in table order), but without importing
    ultralytics, so this script runs anywhere numpy and pandas do."""
    table = pd.read_csv(os.path.join(paths.data_root(), "dispatch" + paths.TAG, f"{split}_pool.csv"))
    parts = []
    for (sp, scenario, camera), rows in table.groupby(["split", "scenario", "camera"], sort=False):
        cache = os.path.join(paths.data_root(), "embeddings", f"{sp}_{scenario}_{camera}.npz")
        saved = np.load(cache)
        assert (saved["frame_idx"] == rows["frame_idx"].to_numpy()).all(), cache
        parts.append(saved["embeddings"])
    embeddings = np.concatenate(parts)
    assert len(embeddings) == len(table), (split, len(embeddings), len(table))
    return table, embeddings


def main():
    print("mode", pc.MODE, "tag", repr(pc.TAG), "-> data in", pc.DATA_DIR)
    final_table, final_emb = load("test")
    val_table, val_emb = load("val")
    rows = []
    if pc.MODE == "s05":
        (a_table, a_emb, cams_a), (b_table, b_emb, cams_b) = split_by_camera(val_table, val_emb)
        print("S05 fold A (head training):", cams_a)
        print("S05 fold B (NSGA-II fitness):", cams_b)
        train_role, fitness_role = (a_table, a_emb), (b_table, b_emb)
    else:
        train_table, train_emb = load("train")
        train_role, fitness_role = (train_table, train_emb), (val_table, val_emb)
    rows.append(save_role("train", *train_role, pc.TRAIN_GROUND_TRUTH_CSV, pc.TRAIN_EMBEDDINGS_NPZ))
    rows.append(save_role("test", *fitness_role, pc.VAL_GROUND_TRUTH_CSV, pc.VAL_EMBEDDINGS_NPZ))
    rows.append(save_role("final_val", final_table, final_emb, pc.FINAL_VAL_GROUND_TRUTH_CSV, pc.FINAL_VAL_EMBEDDINGS_NPZ))
    pd.DataFrame(rows).to_csv(os.path.join(pc.DATA_DIR, "roles.csv"), index=False)


if __name__ == "__main__":
    main()

"""
Builds the three ground-truth CSVs the shared dispatcher/ and
dispatcher_analysis/ code consume, straight from yolo-analysis' committed
per-model, per-image recall benchmark (no inference here).

Schema (same as the CIFAR-100 track, plus image_id and raw recalls):

    image_path,image_id,<model>_recall...,<model>_correct...,label

- image_path: <split>/<image_id:012d>.jpg, relative to config.DATASET_DIR
- <model>_correct: recall >= config.RECALL_THRESHOLD (Week4 [DECISION])
- label: argmin_j { cost_j | model_j correct }, models in cost-ascending
  order; images no model gets right are kept and labelled
  config.NO_CORRECT_MODEL_LABEL (the largest model)

Splits: train = the train2017 subset; val2017 is split, stratified by
label, into test (NSGA-II fitness, paper's "test set") and final_val
(reported once, paper's "validation set"). See config.py's "Splits".

Run: python label_data.py   (numpy + pandas only)
"""

import os

import numpy as np
import pandas as pd

import config


def build_split(split):
    """One row per image of `split`, sorted by image_id."""
    bench = pd.read_csv(config.benchmark_csv(split), usecols=["image_id", "model", "recall"])
    recall = bench.pivot(index="image_id", columns="model", values="recall")

    missing = [f for f in config.MODEL_FILES if f not in recall.columns]
    if missing:
        raise ValueError(f"{config.benchmark_csv(split)} has no rows for {missing}")
    recall = recall[config.MODEL_FILES]
    if recall.isna().any().any():
        raise ValueError(f"{split}: some images are missing a model's recall")
    recall = recall.sort_index()

    correct = (recall.values >= config.RECALL_THRESHOLD).astype(np.int64)

    # models are listed cost-ascending, so the first correct column is the
    # cheapest correct model
    any_correct = correct.any(axis=1)
    labels = np.where(any_correct, correct.argmax(axis=1), config.NO_CORRECT_MODEL_LABEL)

    out = pd.DataFrame({
        "image_path": [f"{split}/{image_id:012d}.jpg" for image_id in recall.index],
        "image_id": recall.index.values,
    })
    for j, name in enumerate(config.MODEL_NAMES):
        out[f"{name}_recall"] = recall.values[:, j]
    for j, name in enumerate(config.MODEL_NAMES):
        out[f"{name}_correct"] = correct[:, j]
    out["label"] = labels
    return out, int((~any_correct).sum())


def stratified_split(df, fraction, seed):
    """Returns (rest, held_out): `fraction` of every label class goes to
    held_out, drawn with a fixed seed. Both keep image_id order."""
    rng = np.random.RandomState(seed)
    held_out_idx = []
    for _, group in df.groupby("label", sort=True):
        n = int(round(len(group) * fraction))
        held_out_idx.extend(rng.choice(group.index.values, size=n, replace=False).tolist())
    mask = df.index.isin(held_out_idx)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def _report(name, df, no_correct):
    dist = df["label"].value_counts(normalize=True).sort_index()
    dist_text = "  ".join(f"{config.MODEL_NAMES[c]} {100 * p:.1f}%" for c, p in dist.items())
    acc_text = "  ".join(f"{m} {100 * df[f'{m}_correct'].mean():.1f}%" for m in config.MODEL_NAMES)
    print(f"{name:>9}: {len(df):>6} images | label: {dist_text}")
    print(f"{'':>9}  per-model correct: {acc_text} | no model correct: {no_correct}")


def run_labeling():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    print(f"Correct = class-set recall >= {config.RECALL_THRESHOLD}; "
          f"no-correct images -> {config.MODEL_NAMES[config.NO_CORRECT_MODEL_LABEL]}\n")

    train_df, train_none = build_split(config.TRAIN_SPLIT)
    heldout_df, _ = build_split(config.HELDOUT_SPLIT)
    test_df, final_val_df = stratified_split(heldout_df, config.FINAL_VAL_FRACTION, config.SPLIT_SEED)

    def none_count(df):
        return int((df[[f"{m}_correct" for m in config.MODEL_NAMES]].sum(axis=1) == 0).sum())

    train_df.to_csv(config.TRAIN_GROUND_TRUTH_CSV, index=False)
    test_df.to_csv(config.VAL_GROUND_TRUTH_CSV, index=False)
    final_val_df.to_csv(config.FINAL_VAL_GROUND_TRUTH_CSV, index=False)

    _report("train", train_df, train_none)
    _report("test", test_df, none_count(test_df))
    _report("final_val", final_val_df, none_count(final_val_df))
    print(f"\nWritten to {config.DATA_DIR}")


if __name__ == "__main__":
    run_labeling()

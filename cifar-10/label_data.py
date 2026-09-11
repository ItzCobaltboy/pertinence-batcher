"""
Builds the CIFAR-10 ground-truth CSVs consumed by the shared dispatcher/ and
dispatcher_analysis/ code, matching the schema of the ImageNette track's
data/{train,val}_ground_truth.csv exactly:

    image_path,<model1>_correct,<model2>_correct,<model3>_correct,<model4>_correct,label

- image_path: path to a cached per-image .png under dataset/images/,
  relative to config.DATASET_DIR
- <model>_correct: 1/0, whether that model's own prediction matches the
  CIFAR-10 ground-truth class
- label: argmin_j { MAdds_j | model_j is correct } (dispatcher's ideal
  routing target, 0..3 indexing MODEL_NAMES in config.py's MAdds order:
  shufflenetv2_x0_5 < resnet20 < resnet32 < vgg11_bn)

Split methodology: uses CIFAR-10's own official train/test division AS-IS,
with no re-mixing.

  - Official train (50,000 images) is the pool that labeling and dispatcher
    FC training see — written to data/train_ground_truth.csv.
  - Official test (10,000 images) is written to data/val_ground_truth.csv
    ("val" here means this pipeline's held-out evaluation split, matching
    the ImageNette track's naming — it is CIFAR-10's official test set, NOT
    a further re-split of anything). Labeling and FC training never touch
    it; it is used by NSGA-II's per-individual fitness evaluation
    (dispatcher/fitness.py) and by dispatcher_analysis/'s final eval.

Using the official division (rather than re-mixing train+test into one pool
and re-splitting) means the chenyaofo/pytorch-cifar-models checkpoints in
models/ — trained on the official 50k train set — never see train-time-seen
images in what's reported as held-out evaluation.

Every raw image in both official splits goes through this script, and every
one of them ends up in the output CSV — nothing is dropped or subsampled,
including images where NO pool model got them right. Those "impossible"
images are routed to the highest-cost model (the last entry in
MODEL_NAMES_BY_MADDS, i.e. vgg11_bn) instead of being dropped or given a
sentinel label — when nothing is correct, the biggest/most capable model in
the pool is the most defensible fallback target, and it keeps every row a
normal, valid training example (a real class 0..3, not a special case every
downstream consumer would otherwise need to filter around). Counts of these
are printed per split so they're visible, and eda/ground_truth_eda.py
reports the oracle rate ("at least one model correct") directly from the
correctness columns, independent of `label`.
"""

import os
import sys

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision.datasets import CIFAR10

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "models"))
import config  # noqa: E402
from model_loader import load_model, MODEL_NAMES_BY_MADDS, MADDS_M  # noqa: E402


def download_and_dump_images():
    """Downloads CIFAR-10 train+test via torchvision (no-op if already
    downloaded) and dumps every image as a PNG under dataset/images/,
    returning (train_df, test_df) — each a DataFrame of (image_path,
    class_label) for its own official split. Idempotent: skips PNGs that
    already exist."""
    images_dir = os.path.join(config.DATASET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    train_set = CIFAR10(root=config.DATASET_DIR, train=True, download=True)
    test_set = CIFAR10(root=config.DATASET_DIR, train=False, download=True)

    split_frames = {}
    for split_name, dataset in [("train", train_set), ("test", test_set)]:
        rows = []
        for i in range(len(dataset)):
            img, label = dataset[i]
            filename = f"{split_name}_{i:05d}.png"
            abs_path = os.path.join(images_dir, filename)
            if not os.path.exists(abs_path):
                img.save(abs_path)
            # image_path is stored relative to config.DATASET_DIR, resolved
            # by the shared dispatcher/dispatcher_analysis embeddings.py
            # via os.path.join(config.DATASET_DIR, image_path) — not
            # relative to whatever directory a script happens to run from.
            rows.append({"image_path": os.path.join("images", filename),
                         "abs_image_path": abs_path,
                         "class_label": int(label)})
        split_frames[split_name] = pd.DataFrame(rows)

    return split_frames["train"], split_frames["test"]


@torch.no_grad()
def compute_correctness(df, device, split_label):
    """Runs all 4 CIFAR models over every image in df, returns df augmented
    with one <model>_correct column per model plus the argmin-MAdds label.
    NO rows are dropped — images where no model is correct are routed to
    the highest-cost model instead of being removed. Returns (out_df,
    impossible_count, total_count) — split_label is only used for progress
    printing."""
    models = {name: load_model(name, device=device) for name in MODEL_NAMES_BY_MADDS}

    n = len(df)
    correct_cols = {name: np.zeros(n, dtype=np.int64) for name in MODEL_NAMES_BY_MADDS}

    batch_size = 256
    paths = df["abs_image_path"].tolist()
    class_labels = df["class_label"].tolist()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for p in paths[start:end]:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(config.IMAGE_TRANSFORM(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        batch_labels = torch.tensor(class_labels[start:end])

        for name, model in models.items():
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu()
            correct_cols[name][start:end] = (preds == batch_labels).numpy().astype(np.int64)

        if (start // batch_size) % 20 == 0:
            print(f"  [{split_label}] correctness pass: {end}/{n}")

    out_df = df.copy()
    for name in MODEL_NAMES_BY_MADDS:
        out_df[f"{name}_correct"] = correct_cols[name]

    # label = argmin-MAdds among correct models; for images where no model
    # is correct (kept, not dropped), route to the highest-cost model
    # (last index) instead — the most defensible fallback, and it keeps
    # every row a normal, valid training target.
    madds = np.array(MADDS_M)
    correct_matrix = np.stack([correct_cols[name] for name in MODEL_NAMES_BY_MADDS], axis=1)  # (n, 4)
    any_correct = correct_matrix.sum(axis=1) > 0
    highest_cost_model_idx = len(MODEL_NAMES_BY_MADDS) - 1

    labels = np.full(n, highest_cost_model_idx, dtype=np.int64)
    for i in range(n):
        if any_correct[i]:
            correct_idxs = np.where(correct_matrix[i] == 1)[0]
            labels[i] = correct_idxs[np.argmin(madds[correct_idxs])]
    out_df["label"] = labels

    impossible = int((~any_correct).sum())

    final_cols = ["image_path"] + [f"{name}_correct" for name in MODEL_NAMES_BY_MADDS] + ["label"]
    return out_df[final_cols], impossible, n


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Downloading/loading CIFAR-10's official train (50k) + test (10k) splits "
          "and dumping images (no re-mixing)...")
    train_raw, test_raw = download_and_dump_images()
    print(f"Official train: {len(train_raw)} images | Official test: {len(test_raw)} images")

    print("\nComputing per-model correctness + labels for the official TRAIN split "
          "(this is the only split labeling/search/FC-training ever sees)...")
    train_df, train_impossible, train_total = compute_correctness(train_raw, device, "train")
    print(f"Train: {train_total} images written, {train_impossible} routed to the "
          f"highest-cost model ({100 * train_impossible / train_total:.2f}%, no pool "
          f"model correct — kept, not dropped)")

    print("\nComputing per-model correctness + labels for the official TEST split "
          "(held out — used only once, by dispatcher_analysis's final eval)...")
    val_df, val_impossible, val_total = compute_correctness(test_raw, device, "test/val")
    print(f"Val (official test): {val_total} images written, {val_impossible} routed to "
          f"the highest-cost model ({100 * val_impossible / val_total:.2f}%, no pool "
          f"model correct — kept, not dropped)")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_df.to_csv(config.TRAIN_GROUND_TRUTH_CSV, index=False)
    val_df.to_csv(config.VAL_GROUND_TRUTH_CSV, index=False)

    print("\nWrote:")
    print(f"  {config.TRAIN_GROUND_TRUTH_CSV}")
    print(f"  {config.VAL_GROUND_TRUTH_CSV}")
    print("\nDone. Every raw image is in these CSVs — the oracle rate ('at least one "
          "model correct') is computed directly from the <model>_correct columns by "
          "eda/ground_truth_eda.py, independent of label.")


if __name__ == "__main__":
    main()

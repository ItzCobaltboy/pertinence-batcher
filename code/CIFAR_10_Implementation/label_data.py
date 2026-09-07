"""
Builds the CIFAR-10 ground-truth CSVs consumed by dispatcher/ and
dispatcher_analysis/, matching the schema of the original ImageNette
pipeline's data/{train,val}_ground_truth.csv exactly:

    image_path,<model1>_correct,<model2>_correct,<model3>_correct,<model4>_correct,label

- image_path: path to a cached per-image .png under dataset/images/
- <model>_correct: 1/0, whether that model's own prediction matches the
  CIFAR-10 ground-truth class
- label: argmin_j { MAdds_j | model_j is correct } (dispatcher's ideal
  routing target, 0..3 indexing MODEL_NAMES in dispatcher/src/constants.py's
  MAdds order: resnet20 < resnet32 < shufflenetv2_x2_0 < vgg16_bn)

Split methodology (see Journel log [DECISION]): combines CIFAR-10's official
train (50k) + test (10k) = 60k images into one pool, then draws a FRESH
class-stratified 70:30 train:val split - NOT the official train/test
division - since this project's own 70:30 split is what "ground truth" means
elsewhere in the pipeline, and the official CIFAR-10 test set was also used
by chenyaofo to select/report the checkpoints' accuracy, so reusing it as-is
here would double-count it as both "held-out" and "reported-accuracy" data.

Images with NO model correct are dropped (no valid argmin-cheapest-correct
label exists) - same behavior as the ImageNette pipeline's raw->cleaned CSV
step (9469 -> 8940 dropped rows there).
"""

import os
import sys

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "models"))
from model_loader import load_model, MODEL_NAMES_BY_MADDS, MADDS_M  # noqa: E402

DATASET_DIR = os.path.join(_THIS_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
DISPATCHER_DATA_DIR = os.path.join(_THIS_DIR, "dispatcher", "data")
ANALYSIS_DATA_DIR = os.path.join(_THIS_DIR, "dispatcher_analysis", "data")

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

RANDOM_SEED = 42
VAL_FRACTION = 0.30


def download_and_dump_images():
    """Downloads CIFAR-10 train+test via torchvision and dumps every image
    as a PNG under dataset/images/, returning a DataFrame of
    (image_path, class_label) for all 60k images. Idempotent: skips PNGs
    that already exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    train_set = CIFAR10(root=DATASET_DIR, train=True, download=True)
    test_set = CIFAR10(root=DATASET_DIR, train=False, download=True)

    rows = []
    idx = 0
    for split_name, dataset in [("train", train_set), ("test", test_set)]:
        for i in range(len(dataset)):
            img, label = dataset[i]
            filename = f"{split_name}_{i:05d}.png"
            path = os.path.join(IMAGES_DIR, filename)
            if not os.path.exists(path):
                img.save(path)
            # csv_image_path is stored relative to code/CIFAR_10_Implementation/
            # <dispatcher|dispatcher_analysis>/ (main.py's cwd), matching the
            # original pipeline's "./../dataset/..." convention. abs_image_path
            # is used internally here (this script's own cwd may differ).
            rows.append({"image_path": os.path.join("..", "dataset", "images", filename),
                         "abs_image_path": path,
                         "class_label": int(label)})
            idx += 1

    return pd.DataFrame(rows)


def stratified_70_30_split(all_images_df):
    """Fresh class-stratified 70:30 train:val split over the combined 60k
    pool (NOT CIFAR's own train/test division - see module docstring)."""
    train_df, val_df = train_test_split(
        all_images_df,
        test_size=VAL_FRACTION,
        stratify=all_images_df["class_label"],
        random_state=RANDOM_SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


@torch.no_grad()
def compute_correctness(df, device):
    """Runs all 4 CIFAR models over every image in df, returns df augmented
    with one <model>_correct column per model plus the argmin-MAdds label,
    with all-models-wrong rows dropped."""
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
            batch_imgs.append(TRANSFORM(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        batch_labels = torch.tensor(class_labels[start:end])

        for name, model in models.items():
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu()
            correct_cols[name][start:end] = (preds == batch_labels).numpy().astype(np.int64)

        if (start // batch_size) % 20 == 0:
            print(f"  correctness pass: {end}/{n}")

    out_df = df.copy()
    for name in MODEL_NAMES_BY_MADDS:
        out_df[f"{name}_correct"] = correct_cols[name]

    # label = argmin-MAdds among correct models; drop rows where none correct
    madds = np.array(MADDS_M)
    correct_matrix = np.stack([correct_cols[name] for name in MODEL_NAMES_BY_MADDS], axis=1)  # (n, 4)
    any_correct = correct_matrix.sum(axis=1) > 0

    labels = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if any_correct[i]:
            correct_idxs = np.where(correct_matrix[i] == 1)[0]
            labels[i] = correct_idxs[np.argmin(madds[correct_idxs])]
    out_df["label"] = labels

    dropped = int((~any_correct).sum())
    out_df = out_df[any_correct].reset_index(drop=True)

    final_cols = ["image_path"] + [f"{name}_correct" for name in MODEL_NAMES_BY_MADDS] + ["label"]
    return out_df[final_cols], dropped


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Downloading/loading CIFAR-10 train+test and dumping images...")
    all_images_df = download_and_dump_images()
    print(f"Total combined pool: {len(all_images_df)} images")

    train_raw, val_raw = stratified_70_30_split(all_images_df)
    print(f"Stratified 70:30 split -> train {len(train_raw)}, val {len(val_raw)}")

    print("Computing per-model correctness + labels for train split...")
    train_df, train_dropped = compute_correctness(train_raw, device)
    print(f"Train: {len(train_df)} kept, {train_dropped} dropped (no model correct)")

    print("Computing per-model correctness + labels for val split...")
    val_df, val_dropped = compute_correctness(val_raw, device)
    print(f"Val: {len(val_df)} kept, {val_dropped} dropped (no model correct)")

    os.makedirs(DISPATCHER_DATA_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DATA_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(DISPATCHER_DATA_DIR, "train_ground_truth.csv"), index=False)
    train_df.to_csv(os.path.join(ANALYSIS_DATA_DIR, "train_ground_truth.csv"), index=False)
    val_df.to_csv(os.path.join(ANALYSIS_DATA_DIR, "val_ground_truth.csv"), index=False)

    print("\nWrote:")
    print(f"  {os.path.join(DISPATCHER_DATA_DIR, 'train_ground_truth.csv')}")
    print(f"  {os.path.join(ANALYSIS_DATA_DIR, 'train_ground_truth.csv')}")
    print(f"  {os.path.join(ANALYSIS_DATA_DIR, 'val_ground_truth.csv')}")
    print("\nDone.")


if __name__ == "__main__":
    main()

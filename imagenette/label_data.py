"""
Builds the ImageNette ground-truth CSVs consumed by the shared dispatcher/ and
dispatcher_analysis/ code, matching cifar-10/label_data.py's schema exactly:

    image_path,<model1>_correct,<model2>_correct,<model3>_correct,<model4>_correct,label

- image_path: path to the raw ImageNette .JPEG under dataset/imagenette2-320/,
  relative to config.DATASET_DIR
- <model>_correct: 1/0, whether that model's own prediction (over ImageNet's
  full 1000 classes) matches the image's true ImageNet class index
- label: argmin_j { FLOPs_j | model_j is correct } (dispatcher's ideal routing
  target, 0..3 indexing MODEL_NAMES in config.py's FLOPs order: resnet18 <
  resnet34 < resnet50 < resnet152)

Running this script overwrites data/{train,val}_ground_truth.csv with freshly
computed values. Anything trained against a prior version of those CSVs
(e.g. results/nsga2/pareto_front.csv) should be re-checked against fresh
output from this script if the numbers differ.

Split methodology: uses ImageNette's own official train/val folders as-is,
walked directly from dataset/imagenette2-320/{train,val}/ — no re-mixing,
since ImageNette's own division needs none.

Every raw image in both folders goes through this script, and every one of
them ends up in the output CSV — nothing is dropped or subsampled, including
images where NO pool model got them right. Those "impossible" images are
routed to the highest-cost model (the last entry in MODEL_NAMES, i.e.
resnet152) instead of being dropped or given a sentinel label — when
nothing is correct, the biggest/most capable model in the pool is the most
defensible fallback target, and it keeps every row a normal, valid training
example (a real class 0..3, not a special case every downstream consumer
would otherwise need to filter around). Counts of these are printed per
split so they're visible, and eda/ground_truth_eda.py reports the oracle
rate ("at least one model correct") directly from the correctness columns,
independent of `label`.

Gotcha: ImageNette folder names are ImageNet synset IDs (e.g. n01440764), not
0-9 class indices — config.IMAGENETTE_LABEL_MAP remaps them to the real
1000-class ImageNet index every pretrained model here expects. Getting this
wrong looks like every model scoring ~9-10% ("random guess") accuracy.
"""

import os
import sys

import numpy as np
import torch
import pandas as pd
from PIL import Image
import torchvision.models as tvm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import config  # noqa: E402

# name -> constructor, ordered to match config.MODEL_NAMES / config.MODEL_COST
# (FLOPs ascending). Full 1000-class ImageNet head — NOT the classifier-
# stripped embedding extractor config.build_feature_extractor() returns.
_MODEL_BUILDERS = {
    "resnet18": lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT),
    "resnet34": lambda: tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT),
    "resnet50": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT),
    "resnet152": lambda: tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT),
}


def load_models(device):
    """Loads all 4 pool models (ImageNet-pretrained, full 1000-class head)
    in eval mode, keyed by config.MODEL_NAMES."""
    models = {}
    for name in config.MODEL_NAMES:
        model = _MODEL_BUILDERS[name]().to(device)
        model.eval()
        models[name] = model
    return models


def list_split_images(split_name):
    """Returns a DataFrame of (image_path, imagenet_class) for every image
    under dataset/imagenette2-320/<split_name>/. image_path is relative to
    config.DATASET_DIR; imagenet_class is the REAL 1000-class ImageNet index
    (remapped from the synset folder name via config.IMAGENETTE_LABEL_MAP —
    NOT any 0-9 numbering)."""
    split_dir = os.path.join(config.DATASET_DIR, "imagenette2-320", split_name)
    rows = []
    for synset_id in sorted(os.listdir(split_dir)):
        synset_dir = os.path.join(split_dir, synset_id)
        if not os.path.isdir(synset_dir):
            continue
        imagenet_class = config.IMAGENETTE_LABEL_MAP[synset_id]
        for filename in sorted(os.listdir(synset_dir)):
            image_path = os.path.join("imagenette2-320", split_name, synset_id, filename)
            rows.append({"image_path": image_path, "imagenet_class": imagenet_class})
    return pd.DataFrame(rows)


@torch.no_grad()
def compute_correctness(df, models, device, split_label):
    """Runs all 4 pool models over every image in df, returns df augmented
    with one <model>_correct column per model plus the argmin-FLOPs label.
    NO rows are dropped — images where no model is correct are routed to
    the highest-cost model instead of being removed. Returns (out_df,
    impossible_count, total_count)."""
    n = len(df)
    correct_cols = {name: np.zeros(n, dtype=np.int64) for name in config.MODEL_NAMES}

    batch_size = 64
    paths = df["image_path"].tolist()
    imagenet_classes = df["imagenet_class"].tolist()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for p in paths[start:end]:
            full_path = os.path.join(config.DATASET_DIR, p)
            img = Image.open(full_path).convert("RGB")
            batch_imgs.append(config.IMAGE_TRANSFORM(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        batch_labels = torch.tensor(imagenet_classes[start:end])

        for name, model in models.items():
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu()
            correct_cols[name][start:end] = (preds == batch_labels).numpy().astype(np.int64)

        if (start // batch_size) % 20 == 0:
            print(f"  [{split_label}] correctness pass: {end}/{n}")

    out_df = df.copy()
    for name in config.MODEL_NAMES:
        out_df[f"{name}_correct"] = correct_cols[name]

    # label = argmin-cost among correct models; for images where no model
    # is correct (kept, not dropped), route to the highest-cost model
    # (last index) instead — the most defensible fallback, and it keeps
    # every row a normal, valid training target.
    cost = np.array(config.MODEL_COST)
    correct_matrix = np.stack([correct_cols[name] for name in config.MODEL_NAMES], axis=1)  # (n, 4)
    any_correct = correct_matrix.sum(axis=1) > 0
    highest_cost_model_idx = len(config.MODEL_NAMES) - 1

    labels = np.full(n, highest_cost_model_idx, dtype=np.int64)
    for i in range(n):
        if any_correct[i]:
            correct_idxs = np.where(correct_matrix[i] == 1)[0]
            labels[i] = correct_idxs[np.argmin(cost[correct_idxs])]
    out_df["label"] = labels

    impossible = int((~any_correct).sum())

    final_cols = ["image_path"] + [f"{name}_correct" for name in config.MODEL_NAMES] + ["label"]
    return out_df[final_cols], impossible, n


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Listing ImageNette's official train/val folders (no re-mixing needed)...")
    train_raw = list_split_images("train")
    val_raw = list_split_images("val")
    print(f"Official train: {len(train_raw)} images | Official val: {len(val_raw)} images")

    print("Loading 4 pool models (ImageNet-pretrained)...")
    models = load_models(device)

    print("\nComputing per-model correctness + labels for the official TRAIN split "
          "(this is the only split labeling/search/FC-training ever sees)...")
    train_df, train_impossible, train_total = compute_correctness(train_raw, models, device, "train")
    print(f"Train: {train_total} images written, {train_impossible} routed to the "
          f"highest-cost model ({100 * train_impossible / train_total:.2f}%, no pool "
          f"model correct — kept, not dropped)")

    print("\nComputing per-model correctness + labels for the official VAL split "
          "(held out — used only once, by dispatcher_analysis's final eval)...")
    val_df, val_impossible, val_total = compute_correctness(val_raw, models, device, "val")
    print(f"Val: {val_total} images written, {val_impossible} routed to the "
          f"highest-cost model ({100 * val_impossible / val_total:.2f}%, no pool "
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

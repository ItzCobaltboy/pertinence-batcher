"""
Evaluator — Step 1D of the dispatcher pipeline.

Loads the trained dispatcher FC checkpoint and evaluates on the full
cleaned dataset (no resampling) to get real per-image routing accuracy.

Metrics:
  - Overall routing accuracy  : % images routed to correct cheapest model
  - Per-class recall          : did dispatcher find label-1/2/3 images?
  - Confusion matrix          : what gets misrouted to what
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as tvm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

CLEANED_CSV     = "./results/cleaned-data/dispatcher_labels_clean.csv"
CHECKPOINT_PATH = "./results/checkpoints/dispatcher_fc.pt"
OUTPUT_DIR      = "./results/eval/"
MODELS          = ["resnet18", "resnet34", "resnet50", "resnet152"]
NUM_CLASSES     = 4

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Dataset (same as trainer, no sampler) ────────────────────────────────────

class DispatcherDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model (same backbone, load FC from checkpoint) ───────────────────────────

class DispatcherModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone  = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        self.feat = nn.Sequential(*list(backbone.children())[:-1])
        self.fc   = nn.Linear(512, NUM_CLASSES)
        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.feat(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    cleaned_csv: str     = CLEANED_CSV,
    checkpoint_path: str = CHECKPOINT_PATH,
    output_dir: str      = OUTPUT_DIR,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DispatcherModel().to(device)
    model.fc.load_state_dict(ckpt["fc_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained with scheme={ckpt['weight_scheme']}, epochs={ckpt['epochs']}\n")

    dataset = DispatcherDataset(cleaned_csv, transform=TRANSFORM)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    n = len(all_labels)
    correct_mask = all_preds == all_labels
    under_mask   = all_preds < all_labels   # cheaper model predicted → accuracy risk
    over_mask    = all_preds > all_labels   # pricier model predicted → FLOPs waste

    # ── Overall routing accuracy ──
    overall_acc = 100.0 * correct_mask.sum() / n
    print(f"Overall routing accuracy : {overall_acc:.2f}%  ({correct_mask.sum()}/{n})")
    print(f"  Correct        : {correct_mask.sum():>5}  ({100*correct_mask.sum()/n:.1f}%)")
    print(f"  Underestimated : {under_mask.sum():>5}  ({100*under_mask.sum()/n:.1f}%)  ← accuracy risk")
    print(f"  Overestimated  : {over_mask.sum():>5}  ({100*over_mask.sum()/n:.1f}%)  ← FLOPs waste")

    # ── Per-class breakdown ──
    print("\nPer-class routing breakdown (true label rows):")
    header = f"  {'Class':<12} {'N':>5}  {'Correct':>9}  {'Under':>9}  {'Over':>9}  {'Recall':>8}"
    print(header)
    print(f"  {'-'*65}")
    for cls in range(NUM_CLASSES):
        mask = all_labels == cls
        nc   = mask.sum()
        if nc == 0:
            continue
        c = (all_preds[mask] == cls).sum()
        u = (all_preds[mask] <  cls).sum()
        o = (all_preds[mask] >  cls).sum()
        print(f"  [{cls}] {MODELS[cls]:<8} {nc:>5}  "
              f"{c:>5} ({100*c/nc:4.1f}%)  "
              f"{u:>5} ({100*u/nc:4.1f}%)  "
              f"{o:>5} ({100*o/nc:4.1f}%)  "
              f"{100*c/nc:6.1f}%")

    # ── Full classification report ──
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds,
                                target_names=MODELS, digits=3))

    os.makedirs(output_dir, exist_ok=True)

    # ── Confusion matrix plot ──
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion matrix (counts)",   "d"),
        (axes[1], cm_norm, "Confusion matrix (row %)",    ".1f"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(MODELS, rotation=20)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(MODELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True label")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = f"{data[i,j]:{fmt}}" + ("%" if fmt == ".1f" else "")
                ax.text(j, i, val, ha="center", va="center", fontsize=8,
                        color="white" if data[i,j] > data.max()*0.6 else "black")

    plt.suptitle(f"Dispatcher FC — routing accuracy {overall_acc:.1f}%", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix → {cm_path}")

    # ── Save per-image predictions ──
    df = dataset.df.copy()
    df["predicted"] = all_preds
    df["correct_routing"]  = (all_preds == all_labels).astype(int)
    df["underestimated"]   = (all_preds < all_labels).astype(int)
    df["overestimated"]    = (all_preds > all_labels).astype(int)
    pred_path = os.path.join(output_dir, "dispatcher_predictions.csv")
    df.to_csv(pred_path, index=False)
    print(f"Predictions CSV  → {pred_path}")

    return overall_acc, all_preds, all_labels


if __name__ == "__main__":
    evaluate()

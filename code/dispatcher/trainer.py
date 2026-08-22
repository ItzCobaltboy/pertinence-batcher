"""
Trainer — Step 1C of the dispatcher pipeline.

Trains a single FC layer (Linear 512→4) on top of a frozen ResNet18 backbone
to predict the cheapest correct model for each image.

Architecture:
  - Feature extractor : ResNet18 (pretrained, fc head stripped) → 512-dim embedding
  - Classifier        : Linear(512 → 4), the only trainable component

Training details:
  - Input  : results/cleaned-data/dispatcher_labels_clean.csv  (8940 images)
  - Loss   : CrossEntropy * asymmetric penalty matrix P
             (underestimation penalized harder than overestimation)
  - Imbalance: INS / ISNS / ENS sample weighting (selectable via WEIGHT_SCHEME)
  - Optimizer: Adam, lr=1e-3
  - Epochs : 30

Output: results/checkpoints/dispatcher_fc.pt
"""

import os
import sys
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

CLEANED_CSV     = "./results/cleaned-data/dispatcher_labels_clean.csv"
CHECKPOINT_DIR  = "./results/checkpoints/"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dispatcher_fc.pt")
LOG_DIR         = "./results/logs/"

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE    = 64
EPOCHS        = 30
LR            = 1e-3
WEIGHT_SCHEME = "ISNS"   # "INS" | "ISNS" | "ENS"
ENS_BETA      = 0.9999   # used only when WEIGHT_SCHEME == "ENS"
NUM_CLASSES   = 4

# ── Penalty matrix P (4×4) ───────────────────────────────────────────────────
# P[true_label, pred_label] — multiplied into CrossEntropy when pred != true.
# Diagonal must be 0 (correct prediction → no penalty scaling).
# Underestimation (pred < true) → high penalty (accuracy risk).
# Overestimation  (pred > true) → low penalty  (FLOPs waste only).
#
#              pred 0   pred 1   pred 2   pred 3
PENALTY = torch.tensor([
    [0.0,    0.5,     0.5,     0.5  ],   # true 0 (RN18)
    [2.0,    0.0,     0.5,     0.5  ],   # true 1 (RN34)
    [3.0,    2.0,     0.0,     0.5  ],   # true 2 (RN50)
    [4.0,    3.0,     2.0,     0.0  ],   # true 3 (RN152)
], dtype=torch.float32)

# ── Transform ────────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────────────────────────

class DispatcherDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
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

# ── Sample weights ───────────────────────────────────────────────────────────

def compute_sample_weights(labels: np.ndarray, scheme: str, beta: float = 0.9999) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    if scheme == "INS":
        class_w = 1.0 / counts
    elif scheme == "ISNS":
        class_w = 1.0 / np.sqrt(counts)
    elif scheme == "ENS":
        class_w = (1.0 - beta) / (1.0 - beta ** counts)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")
    class_w = class_w / class_w.sum() * NUM_CLASSES  # normalise so mean weight ≈ 1
    return class_w[labels]

# ── Model ────────────────────────────────────────────────────────────────────

class DispatcherModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone    = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # strip the classification head — keep everything up to avgpool
        self.feat   = nn.Sequential(*list(backbone.children())[:-1])
        self.fc     = nn.Linear(512, NUM_CLASSES)

        # freeze backbone
        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.feat(x)          # (B, 512, 1, 1)
        x = x.flatten(start_dim=1)  # (B, 512)
        return self.fc(x)          # (B, 4)

# ── Custom loss ───────────────────────────────────────────────────────────────

class PenalizedCrossEntropy(nn.Module):
    """
    Per-sample loss:
      L_k = 0                                      if pred == true
      L_k = CrossEntropy(logits, true) * P[true, pred]   otherwise
    """
    def __init__(self, penalty: torch.Tensor):
        super().__init__()
        self.register_buffer("P", penalty)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds    = logits.argmax(dim=1)
        ce_loss  = self.ce(logits, targets)                  # (B,)
        penalty  = self.P[targets, preds]                    # (B,)
        is_wrong = (preds != targets).float()                # 0 if correct
        return (ce_loss * penalty * is_wrong).mean()

# ── Training loop ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, weight_scheme: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{weight_scheme}_{run_id}.log")

    logger = logging.getLogger("dispatcher_trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger


def train(
    cleaned_csv: str     = CLEANED_CSV,
    checkpoint_path: str = CHECKPOINT_PATH,
    weight_scheme: str   = WEIGHT_SCHEME,
    epochs: int          = EPOCHS,
    batch_size: int      = BATCH_SIZE,
    lr: float            = LR,
):
    log = setup_logging(LOG_DIR, weight_scheme)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    log.info(f"Scheme : {weight_scheme}  |  Epochs: {epochs}  |  LR: {lr}")

    dataset = DispatcherDataset(cleaned_csv, transform=TRANSFORM)
    labels  = dataset.df["label"].to_numpy()

    sample_weights = compute_sample_weights(labels, scheme=weight_scheme,
                                            beta=ENS_BETA)
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(dataset),
        replacement = True,
    )

    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=sampler, num_workers=4, pin_memory=True)

    model     = DispatcherModel().to(device)
    criterion = PenalizedCrossEntropy(PENALTY).to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    log.info(f"Trainable params : {sum(p.numel() for p in model.fc.parameters()):,}")
    log.info(f"Dataset size     : {len(dataset)}")
    log.info(f"Batches/epoch    : {len(loader)}")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0

        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == targets).sum().item()

        avg_loss = total_loss / len(dataset)
        acc      = 100.0 * correct / len(dataset)
        marker   = "  ← best" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
        log.info(f"Epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  acc={acc:.1f}%{marker}")

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    torch.save({
        "fc_state_dict": model.fc.state_dict(),
        "weight_scheme": weight_scheme,
        "epochs":        epochs,
        "penalty":       PENALTY,
    }, checkpoint_path)
    log.info(f"Checkpoint saved → {checkpoint_path}")
    return model


if __name__ == "__main__":
    train()

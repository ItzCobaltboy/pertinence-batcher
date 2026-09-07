"""
Computes (and caches) CIFAR resnet20 embeddings for the training set.

An "embedding" is the c.EMBEDDING_DIM-number output of resnet20 (chenyaofo/
pytorch-cifar-models, the cheapest pool model) with its final classification
layer removed — the input to the dispatcher's FC head. The backbone is
frozen, so every individual in the NSGA-II search reuses the exact same
embeddings — computed once, cached to disk.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import constants as c

sys.path.insert(0, c.MODELS_DIR_SHARED)
from model_loader import load_model, strip_classifier_head  # noqa: E402


class ImageDataset(Dataset):
    """Loads (image, label) pairs from a dataframe with image_path/label columns."""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = c.IMAGE_TRANSFORM(image)
        label = int(row["label"])
        return image, label


def compute_embeddings(dataframe, device):
    """Runs frozen resnet20 (classifier head removed) over every image in
    dataframe. Returns (embeddings, labels) as numpy arrays."""
    dataset = ImageDataset(dataframe)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=32)

    backbone = load_model("resnet20", device=None)  # keep on CPU until stripped
    feature_extractor, measured_dim = strip_classifier_head(backbone, "resnet20")
    assert measured_dim == c.EMBEDDING_DIM, (
        f"resnet20 embedding dim changed: measured {measured_dim}, "
        f"constants.py says {c.EMBEDDING_DIM}"
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = feature_extractor(images)
            output = output.reshape(output.shape[0], -1)
            all_embeddings.append(output.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    del feature_extractor, loader
    torch.cuda.empty_cache()

    return embeddings, labels


def load_or_compute_train_embeddings(device):
    """Loads cached train embeddings if present, otherwise computes them
    from data/train_ground_truth.csv and caches the result."""
    import pandas as pd

    if os.path.exists(c.TRAIN_EMBEDDINGS_NPZ):
        cached = np.load(c.TRAIN_EMBEDDINGS_NPZ)
        return cached["embeddings"], cached["labels"]

    dataframe = pd.read_csv(c.TRAIN_GROUND_TRUTH_CSV)
    embeddings, labels = compute_embeddings(dataframe, device)

    os.makedirs(os.path.dirname(c.TRAIN_EMBEDDINGS_NPZ), exist_ok=True)
    np.savez(c.TRAIN_EMBEDDINGS_NPZ, embeddings=embeddings, labels=labels)

    return embeddings, labels

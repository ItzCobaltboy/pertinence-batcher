"""
Loads the cached ResNet18 embeddings for train/val from embeddings_cache/.
A fresh-compute fallback (frozen ResNet18, classifier head removed) is
included so this pipeline still works standalone if a cache file is missing.
"""

import os
import numpy as np

import constants as c


def _compute_embeddings(dataframe, device):
    """Runs frozen ResNet18 (classifier head removed) over every image in
    dataframe. Returns (embeddings, labels) as numpy arrays."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    class ImageDataset(Dataset):
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

    loader = DataLoader(ImageDataset(dataframe), batch_size=64, shuffle=False, num_workers=0)

    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = feature_extractor(images).reshape(images.shape[0], -1)
            all_embeddings.append(output.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def _load_or_compute(cache_path, ground_truth_csv, device):
    """Loads cache_path if present, otherwise computes embeddings from
    ground_truth_csv and writes the cache."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["embeddings"], cached["labels"]

    import pandas as pd
    dataframe = pd.read_csv(ground_truth_csv)
    embeddings, labels = _compute_embeddings(dataframe, device)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, labels=labels)
    return embeddings, labels


def load_train_embeddings(device):
    """Returns (embeddings, labels) for the train set."""
    return _load_or_compute(c.TRAIN_EMBEDDINGS_NPZ, c.TRAIN_GROUND_TRUTH_CSV, device)


def load_val_embeddings(device):
    """Returns (embeddings, labels) for the held-out val set."""
    return _load_or_compute(c.VAL_EMBEDDINGS_NPZ, c.VAL_GROUND_TRUTH_CSV, device)

"""
Computes (and caches) ResNet18 embeddings for a set of images.

An "embedding" here is the 512-number output of ResNet18 with its final
classification layer removed — a compressed summary of the image, used as
input to the tiny dispatcher head (Linear 512 -> 4).
"""

import os
import torch
import torch.nn as nn
import torchvision.models as tvm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import constants as c


class ImageDataset(Dataset):
    """Loads one image at a time from a dataframe with an 'image_path' and 'label' column."""

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
    """
    Runs every image in `dataframe` through a frozen ResNet18 (head removed)
    and returns:
      embeddings: numpy array, shape (num_images, 512)
      labels:     numpy array, shape (num_images,)
    Order matches the order of rows in `dataframe`.
    """
    dataset = ImageDataset(dataframe)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    # drop the final classification layer, keep everything up to avgpool
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    all_embeddings = []
    all_labels = []

    print(f"  computing embeddings for {len(dataframe)} images...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = feature_extractor(images)            # shape (batch, 512, 1, 1)
            output = output.reshape(output.shape[0], -1)   # shape (batch, 512)
            all_embeddings.append(output.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    del feature_extractor, loader
    torch.cuda.empty_cache()

    return embeddings, labels


def load_or_compute_embeddings(dataframe, cache_path, device):
    """
    If a cached .npz file already exists at `cache_path`, loads it instead of
    recomputing. Otherwise computes embeddings and saves them for next time.
    """
    if os.path.exists(cache_path):
        print(f"  loading cached embeddings from {cache_path}")
        cached = np.load(cache_path)
        return cached["embeddings"], cached["labels"]

    embeddings, labels = compute_embeddings(dataframe, device)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, labels=labels)
    print(f"  saved embeddings cache -> {cache_path}")

    return embeddings, labels

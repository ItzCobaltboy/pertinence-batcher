"""
Computes (and caches) the dispatcher's embedding-extractor output for the
train and held-out val sets.

An "embedding" is the config.EMBEDDING_DIM-number output of this dataset's
embedding-extractor backbone (built by config.build_feature_extractor —
kept per-dataset since the backbone-loading mechanism itself differs
between tracks, not just its constants) with its final classification
layer removed — the input to the dispatcher's FC head. The backbone is
frozen, so every individual in the NSGA-II search reuses the exact same
embeddings — computed once, cached to disk.

Val embeddings are needed here (not just in dispatcher_analysis/) because
fitness evaluation during the search itself runs on the held-out set, per
the paper's own methodology ("we perform the fitness evaluation for each
individual on the test set") — see fitness.py's module docstring.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageDataset(Dataset):
    """Loads (image, label) pairs from a dataframe with image_path/label
    columns. image_path is stored relative to config.DATASET_DIR."""

    def __init__(self, dataframe, config):
        self.dataframe = dataframe
        self.config = config

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        full_path = os.path.join(self.config.DATASET_DIR, row["image_path"])
        image = Image.open(full_path).convert("RGB")
        image = self.config.IMAGE_TRANSFORM(image)
        label = int(row["label"])
        return image, label


def compute_embeddings(dataframe, device, config):
    """Runs the frozen embedding-extractor backbone over every image in
    dataframe. Returns (embeddings, labels) as numpy arrays."""
    dataset = ImageDataset(dataframe, config)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=config.EMBEDDING_NUM_WORKERS)

    feature_extractor = config.build_feature_extractor(device)
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


def _load_or_compute(cache_path, ground_truth_csv, device, config):
    """Loads cache_path if present, otherwise computes embeddings from
    ground_truth_csv and writes the cache."""
    import pandas as pd

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["embeddings"], cached["labels"]

    dataframe = pd.read_csv(ground_truth_csv)
    embeddings, labels = compute_embeddings(dataframe, device, config)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, labels=labels)

    return embeddings, labels


def load_or_compute_train_embeddings(device, config):
    """Loads cached train embeddings if present, otherwise computes them
    from config.TRAIN_GROUND_TRUTH_CSV and caches the result."""
    return _load_or_compute(config.TRAIN_EMBEDDINGS_NPZ, config.TRAIN_GROUND_TRUTH_CSV, device, config)


def load_or_compute_val_embeddings(device, config):
    """Loads cached val embeddings if present, otherwise computes them
    from config.VAL_GROUND_TRUTH_CSV and caches the result. Used for
    per-individual fitness evaluation during the search itself — see
    module docstring."""
    return _load_or_compute(config.VAL_EMBEDDINGS_NPZ, config.VAL_GROUND_TRUTH_CSV, device, config)

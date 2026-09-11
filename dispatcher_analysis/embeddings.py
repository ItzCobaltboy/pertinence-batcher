"""
Loads the cached embeddings for train/val from config's embeddings_cache
paths. A fresh-compute fallback (the frozen, dataset-specific embedding-
extractor backbone from config.build_feature_extractor, classifier head
removed) is included so this pipeline still works standalone if a cache
file is missing.
"""

import os
import numpy as np


def _compute_embeddings(dataframe, device, config):
    """Runs the frozen embedding-extractor backbone over every image in
    dataframe. Returns (embeddings, labels) as numpy arrays."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    class ImageDataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            full_path = os.path.join(config.DATASET_DIR, row["image_path"])
            image = Image.open(full_path).convert("RGB")
            image = config.IMAGE_TRANSFORM(image)
            label = int(row["label"])
            return image, label

    loader = DataLoader(ImageDataset(dataframe), batch_size=64, shuffle=False,
                         num_workers=config.EMBEDDING_NUM_WORKERS)

    feature_extractor = config.build_feature_extractor(device).to(device)
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


def _load_or_compute(cache_path, ground_truth_csv, device, config):
    """Loads cache_path if present, otherwise computes embeddings from
    ground_truth_csv and writes the cache."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["embeddings"], cached["labels"]

    import pandas as pd
    dataframe = pd.read_csv(ground_truth_csv)
    embeddings, labels = _compute_embeddings(dataframe, device, config)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, labels=labels)
    return embeddings, labels


def load_train_embeddings(device, config):
    """Returns (embeddings, labels) for the train set."""
    return _load_or_compute(config.TRAIN_EMBEDDINGS_NPZ, config.TRAIN_GROUND_TRUTH_CSV, device, config)


def load_val_embeddings(device, config):
    """Returns (embeddings, labels) for the held-out val set."""
    return _load_or_compute(config.VAL_EMBEDDINGS_NPZ, config.VAL_GROUND_TRUTH_CSV, device, config)

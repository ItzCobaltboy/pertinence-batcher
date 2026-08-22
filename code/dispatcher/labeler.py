"""
Labeler — Step 1A of the dispatcher pipeline.

For every image in the ImageNette train set, finds the cheapest model in the
pool that classifies it correctly. That model's index becomes the dispatcher
training label for that image.

    label(x) = argmin_j { FLOPs_j | model_j(x) == correct }

If no model gets it right → label 3 (ResNet152, most expensive fallback).

Output: results/dispatcher_labels.csv
  Columns: image_path, resnet18_correct, resnet34_correct,
           resnet50_correct, resnet152_correct, label
"""

import os
import torch
import numpy as np
import pandas as pd

from utils import POOL_MODELS, load_pool_models, get_imagenette_loader


def generate_labels(dataset_path: str, output_path: str) -> pd.DataFrame:
    """
    Args:
        dataset_path: dataset root (expects imagenette2-320/train/ inside)
        output_path:  path to write dispatcher_labels.csv
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    pool = load_pool_models(device)

    loader, image_paths = get_imagenette_loader(dataset_path, split="train")
    n_batches = len(loader)
    print(f"\nLabeling {len(image_paths)} train images across {n_batches} batches...")

    correctness = {m["name"]: [] for m in POOL_MODELS}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels_np = labels.numpy()

            for m in POOL_MODELS:
                name = m["name"]
                logits = pool[name](images)
                predicted = logits.argmax(dim=1).cpu().numpy()
                correctness[name].extend((predicted == labels_np).astype(int).tolist())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
                print(f"  {batch_idx + 1}/{n_batches} batches")

    # Assign label = index of cheapest correct model
    labels_out = []
    for i in range(len(image_paths)):
        label = len(POOL_MODELS) - 1  # fallback: most expensive
        for j, m in enumerate(POOL_MODELS):
            if correctness[m["name"]][i] == 1:
                label = j
                break
        labels_out.append(label)

    df = pd.DataFrame({
        "image_path": image_paths,
        **{f"{m['name']}_correct": correctness[m["name"]] for m in POOL_MODELS},
        "label": labels_out,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}")
    print(f"Total images: {len(df)}")
    print("Label distribution (0=RN18, 1=RN34, 2=RN50, 3=RN152):")
    dist = df["label"].value_counts().sort_index()
    for lbl, count in dist.items():
        print(f"  [{lbl}] {POOL_MODELS[lbl]['name']:>8}: {count:>5} ({100*count/len(df):.1f}%)") # type: ignore

    return df

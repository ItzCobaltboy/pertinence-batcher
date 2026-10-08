"""
Trains one dispatcher FC head (Linear EMBEDDING_DIM -> NUM_CLASSES) and
predicts with it.

Class imbalance is handled entirely inside the loss (loss.py /
class_weights.py), so the DataLoader here is a plain shuffled loader —
no sampler.
"""

import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from loss import penalized_loss


def train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device, config, seed=None):
    """Train Linear(EMBEDDING_DIM, NUM_CLASSES) on the given embeddings/labels
    under the penalized loss. Returns the trained weight matrix W
    (NUM_CLASSES, EMBEDDING_DIM) and bias b (NUM_CLASSES,) as plain numpy
    arrays."""
    if seed is not None:
        torch.manual_seed(seed)   # seeds FC init + DataLoader shuffle (CPU and CUDA)

    embeddings_tensor = torch.from_numpy(train_embeddings)
    labels_tensor = torch.from_numpy(train_labels)
    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    fc = nn.Linear(config.EMBEDDING_DIM, config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(fc.parameters(), lr=config.LEARNING_RATE)

    fc.train()
    for epoch in range(config.FC_EPOCHS):
        for batch_embeddings, batch_labels in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = fc(batch_embeddings)
            loss = penalized_loss(logits, batch_labels, penalty_matrix, class_weights)
            loss.backward()
            optimizer.step()

    W = fc.weight.detach().cpu().numpy()
    b = fc.bias.detach().cpu().numpy()

    # free GPU memory immediately — this trains once per NSGA-II individual (2500+ times/run)
    del fc, optimizer, loader
    torch.cuda.empty_cache()

    return W, b


def chromosome_seed(chromosome):
    """Deterministic RNG seed for one chromosome's FC training. The search
    (fitness.py), the post-search save (save_models.py) and
    dispatcher_analysis/build_models.py all retrain the same chromosome;
    seeding init + shuffle from the chromosome itself makes those runs
    produce the same weights, so the model evaluated at the end is the one
    the search actually scored. Hashes the float64 bytes, so the chromosome
    has to round-trip pareto_front.csv at full precision (save_results.py
    writes it unrounded)."""
    data = np.asarray(chromosome, dtype=np.float64).tobytes()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def predict(embeddings, W, b):
    """Plain numpy prediction: argmax(embeddings @ W.T + b)."""
    logits = embeddings.dot(W.T) + b
    return np.argmax(logits, axis=1)

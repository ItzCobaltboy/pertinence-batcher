"""
Trains one dispatcher FC head (Linear 512 -> 4) and predicts with it.

No sampler here — a plain shuffled DataLoader, since class imbalance is now
handled entirely inside the loss (see loss.py / class_weights.py).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import constants as c
from loss import penalized_loss


def train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device):
    """
    Returns the trained weight matrix W (4, 512) and bias b (4,) as plain
    numpy arrays.
    """
    embeddings_tensor = torch.from_numpy(train_embeddings)
    labels_tensor = torch.from_numpy(train_labels)
    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=c.BATCH_SIZE, shuffle=True)

    fc = nn.Linear(512, c.NUM_CLASSES).to(device)
    optimizer = optim.Adam(fc.parameters(), lr=c.LEARNING_RATE)

    fc.train()
    for epoch in range(c.FC_EPOCHS):
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

    del fc, optimizer, loader
    torch.cuda.empty_cache()

    return W, b


def predict(embeddings, W, b):
    """Plain numpy prediction: argmax(embeddings @ W.T + b)."""
    logits = embeddings.dot(W.T) + b
    return np.argmax(logits, axis=1)

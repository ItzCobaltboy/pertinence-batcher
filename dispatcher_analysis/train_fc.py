"""
Trains one dispatcher FC head (Linear EMBEDDING_DIM -> NUM_CLASSES) and
predicts with it. Kept identical to dispatcher/dispatcher_model.py, renamed
here since this is the only training this pipeline does.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from loss import penalized_loss


def train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device, config):
    """Returns the trained weight matrix W (NUM_CLASSES, EMBEDDING_DIM) and
    bias b (NUM_CLASSES,) as numpy arrays."""
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

    del fc, optimizer, loader
    torch.cuda.empty_cache()

    return W, b


def predict(embeddings, W, b):
    """Plain numpy prediction: argmax(embeddings @ W.T + b)."""
    logits = embeddings.dot(W.T) + b
    return np.argmax(logits, axis=1)

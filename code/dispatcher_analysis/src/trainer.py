"""
Trains one dispatcher config (a Linear 512 -> 4 head) on the train
embeddings, using its penalty matrix + class-weighted sampling, and
predicts routing labels for any set of embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import constants as c
from config_utils import build_sample_weights


def penalized_loss(logits, targets, penalty_matrix_np):
    """
    Per-sample loss: 0 if prediction is correct, else
    CrossEntropy(logits, target) * penalty_matrix[true_class, predicted_class]

    Note: targets/predictions are pulled to the CPU as plain numpy ONCE per
    batch (not once per sample) before the loop, so this stays a simple,
    readable loop without forcing a slow GPU sync on every single image.
    """
    cross_entropy = nn.functional.cross_entropy(logits, targets, reduction="none")
    predictions = torch.argmax(logits, dim=1)

    targets_np = targets.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()

    penalties = np.zeros(len(targets_np), dtype=np.float32)
    for i in range(len(targets_np)):
        true_class = targets_np[i]
        pred_class = predictions_np[i]
        penalties[i] = penalty_matrix_np[true_class, pred_class]

    penalties_tensor = torch.from_numpy(penalties).to(logits.device)
    is_wrong = (predictions != targets).float()
    loss_per_sample = cross_entropy * penalties_tensor * is_wrong
    return loss_per_sample.mean()


def train_dispatcher_config(train_embeddings, train_labels, penalty_matrix, alpha, device):
    """
    Trains a Linear(512 -> 4) head on the train embeddings. Returns the
    trained weight matrix W (4, 512) and bias b (4,) as plain numpy arrays.
    """
    sample_weights = build_sample_weights(train_labels, alpha)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_embeddings_tensor = torch.from_numpy(train_embeddings)
    train_labels_tensor = torch.from_numpy(train_labels)
    train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, sampler=sampler)

    fc = nn.Linear(512, c.NUM_CLASSES).to(device)
    optimizer = optim.Adam(fc.parameters(), lr=c.LEARN_RATE)

    fc.train()
    for epoch in range(c.FC_EPOCHS):
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = fc(batch_embeddings)
            loss = penalized_loss(logits, batch_labels, penalty_matrix)
            loss.backward()
            optimizer.step()

    fc.eval()
    W = fc.weight.detach().cpu().numpy()   # shape (4, 512)
    b = fc.bias.detach().cpu().numpy()     # shape (4,)

    del fc, optimizer, train_loader
    torch.cuda.empty_cache()

    return W, b


def predict_with_weights(embeddings, W, b):
    """
    Plain numpy prediction: logits = embeddings @ W.T + b, then pick the
    class with the highest logit for every image.
    """
    logits = embeddings.dot(W.T) + b         # shape (num_images, 4)
    predictions = np.argmax(logits, axis=1)  # shape (num_images,)
    return predictions

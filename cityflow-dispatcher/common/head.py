# The dispatcher head: standardise the 448-d embedding, one linear layer 448 -> 4.
# Loss = class-weighted cross-entropy, scaled up when the head puts probability on models smaller
# than the ideal one (underestimation): loss_i = w[y_i] * CE_i * (1 + (lam - 1) * P(smaller than y_i)).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

LETTER_INDEX = {l: i for i, l in enumerate(config.MODEL_LETTERS)}


def class_weights(labels, scheme):
    """Per-class weight, normalised so the average weight over samples is 1."""
    counts = np.bincount(labels, minlength=4).astype(float)
    freq = counts / counts.sum()
    if scheme == "none":
        w = np.ones(4)
    elif scheme == "inv":
        w = 1.0 / np.maximum(freq, 1e-6)
    elif scheme == "sqrt_inv":
        w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    else:
        raise ValueError(scheme)
    return w / (w * freq).sum()


def fit_head(x_train, y_train, scheme, lam, epochs=300, lr=0.01, seed=config.SEED):
    """Full-batch Adam on the GPU. Returns a dict with weights and the standardisation stats."""
    torch.manual_seed(seed)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x = torch.tensor((x_train - mean) / std, dtype=torch.float32).cuda()
    y = torch.tensor(y_train, dtype=torch.long).cuda()
    w = torch.tensor(class_weights(y_train, scheme), dtype=torch.float32).cuda()
    layer = nn.Linear(x.shape[1], 4).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=1e-4)
    smaller = (torch.arange(4).cuda()[None, :] < y[:, None]).float()      # [n, 4]: classes cheaper than the ideal
    for _ in range(epochs):
        logits = layer(x)
        ce = F.cross_entropy(logits, y, reduction="none")
        p_under = (F.softmax(logits, dim=1) * smaller).sum(1)
        loss = (w[y] * ce * (1 + (lam - 1) * p_under)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"weight": layer.weight.detach().cpu(), "bias": layer.bias.detach().cpu(), "mean": mean, "std": std}


def predict_proba(head, x):
    z = torch.tensor((x - head["mean"]) / head["std"], dtype=torch.float32)
    return F.softmax(z @ head["weight"].T + head["bias"], dim=1).numpy()

"""
Per-sample penalized loss for training the dispatcher FC head. Exact copy
of code/Dispatcher/src/loss.py.

  L_k = 0                                                     if pred == true
  L_k = CrossEntropy(logits, true) * P[true, pred] * class_weight[true]   otherwise
"""

import torch
import numpy as np


def penalized_loss(logits, targets, penalty_matrix_np, class_weights_np):
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    predictions = torch.argmax(logits, dim=1)

    targets_np = targets.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()

    combined_weight = np.zeros(len(targets_np), dtype=np.float32)
    for i in range(len(targets_np)):
        true_class = targets_np[i]
        pred_class = predictions_np[i]
        combined_weight[i] = penalty_matrix_np[true_class, pred_class] * class_weights_np[true_class]

    combined_weight_tensor = torch.from_numpy(combined_weight).to(logits.device)
    is_wrong = (predictions != targets).float()

    loss_per_sample = cross_entropy * combined_weight_tensor * is_wrong
    return loss_per_sample.mean()

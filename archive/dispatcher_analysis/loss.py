"""
Per-sample loss for training the dispatcher FC head:

  L_k = 0                                                     if pred == true
  L_k = CrossEntropy(logits, true) * P[true, pred] * class_weight[true]   otherwise

Two multiplicative corrections stack on top of plain cross-entropy:
  - P[true, pred]      the penalty matrix (this chromosome) — asymmetric
                        cost for underestimating vs overestimating
  - class_weight[true]  the chromosome's INS/ISNS/ENS class weight — boosts
                        minority-class errors
"""

import torch


def penalized_loss(logits, targets, penalty_matrix_np, class_weights_np):
    """Computes the batch-mean penalized loss described above."""
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    predictions = torch.argmax(logits, dim=1)

    # per-sample weight = penalty(true, pred) * class_weight(true), gathered
    # on-device in one indexing op (P[true, true] = 0 already zeroes the loss
    # for correct predictions; is_wrong below keeps that explicit)
    penalty_matrix = torch.as_tensor(penalty_matrix_np, dtype=logits.dtype, device=logits.device)
    class_weights = torch.as_tensor(class_weights_np, dtype=logits.dtype, device=logits.device)
    combined_weight = penalty_matrix[targets, predictions] * class_weights[targets]
    is_wrong = (predictions != targets).to(logits.dtype)   # zeroes out the loss for correct predictions

    loss_per_sample = cross_entropy * combined_weight * is_wrong
    return loss_per_sample.mean()

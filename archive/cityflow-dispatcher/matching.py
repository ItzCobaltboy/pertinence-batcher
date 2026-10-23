"""
Box recall of one detector's output against a reference detector's output
(numpy only). Used to decide which models are "correct" on a CCTV frame.

Matching is class-aware and greedy by confidence, the usual detection
convention: predictions of a class are taken in descending confidence,
each claims the unmatched reference box of the same class with the highest
IoU, if that IoU is >= the threshold. Extra predictions are not penalised
(recall, not precision).
"""

import numpy as np


def box_iou(a, b):
    """a (N, 4), b (M, 4), xyxy. Returns (N, M) IoU."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def recall_vs_reference(ref, pred, iou_threshold):
    """ref, pred: dicts with 'boxes' (N, 4) xyxy, 'cls' (N,), and pred also
    'conf' (N,). Returns the fraction of reference boxes matched, 1.0 when
    the reference found nothing (nothing to miss)."""
    ref_boxes = np.asarray(ref["boxes"]).reshape(-1, 4)
    ref_cls = np.asarray(ref["cls"]).reshape(-1)
    if len(ref_boxes) == 0:
        return 1.0
    pred_boxes = np.asarray(pred["boxes"]).reshape(-1, 4)
    pred_cls = np.asarray(pred["cls"]).reshape(-1)
    pred_conf = np.asarray(pred["conf"]).reshape(-1)

    matched = 0
    for c in np.unique(ref_cls):
        r = ref_boxes[ref_cls == c]
        keep = pred_cls == c
        if not keep.any():
            continue
        order = np.argsort(-pred_conf[keep], kind="stable")
        p = pred_boxes[keep][order]
        iou = box_iou(p, r)
        used = np.zeros(len(r), dtype=bool)
        for i in range(len(p)):
            candidates = np.where(used, -1.0, iou[i])
            j = int(np.argmax(candidates))
            if candidates[j] >= iou_threshold:
                used[j] = True
                matched += 1
    return matched / len(ref_boxes)


def cheapest_correct(recalls, threshold):
    """recalls: per model, cost-ascending. Index of the first model whose
    recall >= threshold; the last (largest) model if none is."""
    for i, r in enumerate(recalls):
        if r >= threshold:
            return i
    return len(recalls) - 1

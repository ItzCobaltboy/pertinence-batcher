# Box IoU and greedy class-aware matching of one detector's boxes against a reference set.
# Recall = matched / reference boxes, precision = matched / predicted boxes.
# A "dets" dict is {"boxes": (N,4) xyxy, "cls": (N,), "conf": (N,)}; the reference needs no conf.
import numpy as np


def box_iou(a, b):
    """a (N,4), b (M,4), xyxy -> (N,M) IoU."""
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


def count_matches(ref, pred, iou_threshold):
    """Greedy by confidence, same class only: each prediction claims the free reference
    box with the highest IoU if that IoU >= threshold. Returns the number of matches."""
    ref_boxes = np.asarray(ref["boxes"]).reshape(-1, 4)
    ref_cls = np.asarray(ref["cls"]).reshape(-1)
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
    return matched


def box_recall(ref, pred, iou_threshold):
    """Fraction of reference boxes found. 1.0 when the reference has no boxes (nothing to miss)."""
    n_ref = len(ref["boxes"])
    if n_ref == 0:
        return 1.0
    return count_matches(ref, pred, iou_threshold) / n_ref


def box_precision(ref, pred, iou_threshold):
    """Fraction of predicted boxes that match a reference box. 1.0 when nothing was predicted."""
    n_pred = len(pred["boxes"])
    if n_pred == 0:
        return 1.0
    return count_matches(ref, pred, iou_threshold) / n_pred

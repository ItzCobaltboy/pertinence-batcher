"""
Plain-loop classification metrics, computed against ideal_label (the
argmin-cheapest-correct reference column cached in the predictions CSVs) —
this is routing-quality relative to the ideal target class, a different
question from alpha_sys (was the DISPATCHED model itself correct on this
image). Both are useful; don't conflate them.
"""

import numpy as np

import constants as c


def confusion_matrix(true_labels, predicted_labels):
    """Rows = true (ideal) class, columns = predicted class."""
    cm = np.zeros((c.NUM_CLASSES, c.NUM_CLASSES), dtype=int)
    for true_val, pred_val in zip(true_labels, predicted_labels):
        cm[true_val, pred_val] += 1
    return cm


def recall_per_class(cm):
    """Of images whose ideal label is class c, what fraction got routed to c?"""
    recall = np.zeros(c.NUM_CLASSES)
    for class_idx in range(c.NUM_CLASSES):
        row_total = cm[class_idx, :].sum()
        recall[class_idx] = cm[class_idx, class_idx] / row_total if row_total > 0 else np.nan
    return recall


def precision_per_class(cm):
    """Of images routed to class c, what fraction have ideal label c?"""
    precision = np.zeros(c.NUM_CLASSES)
    for class_idx in range(c.NUM_CLASSES):
        col_total = cm[:, class_idx].sum()
        precision[class_idx] = cm[class_idx, class_idx] / col_total if col_total > 0 else np.nan
    return precision

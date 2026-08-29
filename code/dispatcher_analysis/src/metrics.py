"""
Plain-loop metric functions: confusion matrix, accuracy, recall, precision,
and the under/correct/over routing breakdown.
"""

import numpy as np

import constants as c


def compute_confusion_matrix(true_labels, predicted_labels):
    """
    Builds a 4x4 confusion matrix by simple counting.
    Rows = true class, columns = predicted class.
    """
    cm = np.zeros((c.NUM_CLASSES, c.NUM_CLASSES), dtype=int)
    for true_val, pred_val in zip(true_labels, predicted_labels):
        cm[true_val, pred_val] += 1
    return cm


def compute_accuracy(cm):
    correct_count = 0
    for i in range(c.NUM_CLASSES):
        correct_count += cm[i, i]
    total_count = cm.sum()
    return correct_count / total_count


def compute_recall_per_class(cm):
    """Of images that truly belong to class c, what fraction were predicted c?"""
    recall = np.zeros(c.NUM_CLASSES)
    for class_idx in range(c.NUM_CLASSES):
        row_total = cm[class_idx, :].sum()
        if row_total > 0:
            recall[class_idx] = cm[class_idx, class_idx] / row_total
        else:
            recall[class_idx] = np.nan
    return recall


def compute_precision_per_class(cm):
    """Of images predicted as class c, what fraction truly belong to c?"""
    precision = np.zeros(c.NUM_CLASSES)
    for class_idx in range(c.NUM_CLASSES):
        col_total = cm[:, class_idx].sum()
        if col_total > 0:
            precision[class_idx] = cm[class_idx, class_idx] / col_total
        else:
            precision[class_idx] = np.nan
    return precision


def compute_under_correct_over(cm):
    """
    For each true class row, splits predictions into three buckets:
      under   = predicted a CHEAPER model than needed (column index < row index)
      correct = predicted exactly the right model (column index == row index)
      over    = predicted a MORE EXPENSIVE model than needed (column index > row index)
    Returns a list of (under, correct, over) tuples, one per true class.
    """
    results = []
    for true_class in range(c.NUM_CLASSES):
        under = 0
        correct = 0
        over = 0
        for pred_class in range(c.NUM_CLASSES):
            count = cm[true_class, pred_class]
            if pred_class < true_class:
                under += count
            elif pred_class == true_class:
                correct += count
            else:
                over += count
        results.append((under, correct, over))
    return results


def compute_average_flops(predicted_labels):
    total_flops = 0.0
    for pred in predicted_labels:
        total_flops += c.FLOPS_G[pred]
    return total_flops / len(predicted_labels)

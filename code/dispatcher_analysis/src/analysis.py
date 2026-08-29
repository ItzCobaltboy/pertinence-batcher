"""
Reads a predictions CSV (one column per config) and computes accuracy,
confusion matrix, recall, precision, and under/correct/over counts for
every config.
"""

import os
import numpy as np
import pandas as pd

import constants as c
from metrics import (
    compute_confusion_matrix,
    compute_accuracy,
    compute_recall_per_class,
    compute_precision_per_class,
    compute_under_correct_over,
    compute_average_flops,
)


def analyze_predictions(predictions_csv, summary_csv, cm_npz):
    df = pd.read_csv(predictions_csv)
    true_labels = df["true_label"].values

    prediction_columns = [col for col in df.columns if col.startswith("pred_")]
    print(f"  found {len(prediction_columns)} configs in {predictions_csv}")

    summary_rows = []
    all_confusion_matrices = []
    all_individual_ids = []

    for column_name in prediction_columns:
        individual_id = int(column_name.replace("pred_", ""))
        predicted_labels = df[column_name].values

        cm = compute_confusion_matrix(true_labels, predicted_labels)
        accuracy = compute_accuracy(cm)
        recall = compute_recall_per_class(cm)
        precision = compute_precision_per_class(cm)
        under_correct_over = compute_under_correct_over(cm)
        avg_flops = compute_average_flops(predicted_labels)

        row = {
            "individual": individual_id,
            "accuracy": accuracy,
            "avg_flops_G": avg_flops,
        }
        for class_idx in range(c.NUM_CLASSES):
            model_name = c.MODEL_NAMES[class_idx]
            row[f"recall_{model_name}"] = recall[class_idx]
            row[f"precision_{model_name}"] = precision[class_idx]
            under, correct, over = under_correct_over[class_idx]
            row[f"under_{model_name}"] = under
            row[f"correct_{model_name}"] = correct
            row[f"over_{model_name}"] = over

        summary_rows.append(row)
        all_confusion_matrices.append(cm)
        all_individual_ids.append(individual_id)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("accuracy", ascending=False)

    os.makedirs(os.path.join(c.PROJECT_ROOT, "results"), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"  saved -> {summary_csv}")

    np.savez(cm_npz,
             confusion_matrices=np.array(all_confusion_matrices),
             individual_ids=np.array(all_individual_ids))
    print(f"  saved -> {cm_npz}")


def run_analysis():
    print("\nAnalyzing train predictions...")
    analyze_predictions(c.TRAIN_PREDICTIONS_CSV, c.TRAIN_SUMMARY_CSV, c.TRAIN_CM_NPZ)

    print("\nAnalyzing val predictions...")
    analyze_predictions(c.VAL_PREDICTIONS_CSV, c.VAL_SUMMARY_CSV, c.VAL_CM_NPZ)

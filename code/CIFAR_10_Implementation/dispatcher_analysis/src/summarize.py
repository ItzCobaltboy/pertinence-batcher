"""
Turns cached predictions into one summary row per Pareto individual, per
split: alpha_sys, avg_flops_G, and per-class recall/precision.

alpha_sys (Eq. 3 of the PERTINENCE paper) is the fraction of images where
the model the dispatcher actually picked classifies correctly — looked up
directly from each ground-truth CSV's own <model>_correct columns, indexed
by the predicted model. This is NOT exact-match against ideal_label (the
argmin-cheapest-correct reference column also cached in the predictions
CSV) — exact-match punishes overestimation (a bigger, still-correct model)
as hard as an actual misclassification, which alpha_sys does not. Don't
conflate the two.

Per-class recall/precision, by contrast, ARE computed against ideal_label —
a different, complementary question ("how well does this config route
images to their ideal target class"), independent of alpha_sys.
"""

import os
import numpy as np
import pandas as pd

import constants as c
from metrics import confusion_matrix, recall_per_class, precision_per_class


def _correctness_matrix(ground_truth_csv):
    """Loads the (num_images, NUM_MODELS) boolean correctness matrix from a
    ground-truth CSV's <model>_correct columns."""
    columns = [f"{name}_correct" for name in c.MODEL_NAMES]
    return pd.read_csv(ground_truth_csv)[columns].values.astype(bool)


def _summarize_split(predictions_df, correctness_matrix, out_csv):
    """Computes alpha_sys, avg_flops_G, and per-class recall/precision for
    every pred_<id> column in predictions_df, sorted by alpha_sys
    descending, and writes the result to out_csv."""
    ideal_labels = predictions_df["ideal_label"].values
    prediction_columns = [col for col in predictions_df.columns if col.startswith("pred_")]
    flops_array = np.array(c.MADDS_M)
    image_indices = np.arange(len(predictions_df))

    rows = []
    for column_name in prediction_columns:
        individual_id = int(column_name.replace("pred_", ""))
        predicted_labels = predictions_df[column_name].values

        alpha_sys = correctness_matrix[image_indices, predicted_labels].mean()
        avg_flops_G = flops_array[predicted_labels].mean()

        cm = confusion_matrix(ideal_labels, predicted_labels)
        recall = recall_per_class(cm)
        precision = precision_per_class(cm)

        row = {
            "individual": individual_id,
            "alpha_sys": alpha_sys,
            "avg_flops_G": avg_flops_G,
        }
        for class_idx, model_name in enumerate(c.MODEL_NAMES):
            row[f"recall_{model_name}"] = recall[class_idx]
            row[f"precision_{model_name}"] = precision[class_idx]

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("alpha_sys", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    return summary_df


def summarize(train_predictions_df, val_predictions_df):
    """Summarizes both splits and writes results/{train,val}_summary.csv.
    Returns (train_summary, val_summary)."""
    train_correctness = _correctness_matrix(c.TRAIN_GROUND_TRUTH_CSV)
    val_correctness = _correctness_matrix(c.VAL_GROUND_TRUTH_CSV)

    print("\nSummarizing train predictions...")
    train_summary = _summarize_split(train_predictions_df, train_correctness, c.TRAIN_SUMMARY_CSV)
    print(f"Saved -> {c.TRAIN_SUMMARY_CSV}")

    print("\nSummarizing val predictions...")
    val_summary = _summarize_split(val_predictions_df, val_correctness, c.VAL_SUMMARY_CSV)
    print(f"Saved -> {c.VAL_SUMMARY_CSV}")

    return train_summary, val_summary

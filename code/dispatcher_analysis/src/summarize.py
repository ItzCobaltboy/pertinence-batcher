"""
Turns cached predictions into one summary row per Pareto individual, per
split: alpha_sys and avg_flops_G.

alpha_sys (Eq. 3 of the PERTINENCE paper) is the fraction of images where
the model the dispatcher actually picked classifies correctly — looked up
directly from each ground-truth CSV's own <model>_correct columns, indexed
by the predicted model. This is NOT exact-match against ideal_label (the
argmin-cheapest-correct reference column also cached in the predictions
CSV) — that was the earlier, incorrect metric this project used, and it
punished overestimation (a bigger, still-correct model) as hard as an
actual misclassification. Don't reintroduce that confusion here.
"""

import os
import numpy as np
import pandas as pd

import constants as c


def _correctness_matrix(ground_truth_csv):
    columns = [f"{name}_correct" for name in c.MODEL_NAMES]
    return pd.read_csv(ground_truth_csv)[columns].values.astype(bool)


def _summarize_split(predictions_df, correctness_matrix, out_csv):
    prediction_columns = [col for col in predictions_df.columns if col.startswith("pred_")]
    flops_array = np.array(c.FLOPS_G)
    image_indices = np.arange(len(predictions_df))

    rows = []
    for column_name in prediction_columns:
        individual_id = int(column_name.replace("pred_", ""))
        predicted_labels = predictions_df[column_name].values

        alpha_sys = correctness_matrix[image_indices, predicted_labels].mean()
        avg_flops_G = flops_array[predicted_labels].mean()

        rows.append({
            "individual": individual_id,
            "alpha_sys": alpha_sys,
            "avg_flops_G": avg_flops_G,
        })

    summary_df = pd.DataFrame(rows).sort_values("alpha_sys", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    return summary_df


def summarize(train_predictions_df, val_predictions_df):
    train_correctness = _correctness_matrix(c.TRAIN_GROUND_TRUTH_CSV)
    val_correctness = _correctness_matrix(c.VAL_GROUND_TRUTH_CSV)

    print("\nSummarizing train predictions...")
    train_summary = _summarize_split(train_predictions_df, train_correctness, c.TRAIN_SUMMARY_CSV)
    print(f"Saved -> {c.TRAIN_SUMMARY_CSV}")

    print("\nSummarizing val predictions...")
    val_summary = _summarize_split(val_predictions_df, val_correctness, c.VAL_SUMMARY_CSV)
    print(f"Saved -> {c.VAL_SUMMARY_CSV}")

    return train_summary, val_summary

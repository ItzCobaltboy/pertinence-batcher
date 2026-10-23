"""
Turns cached predictions into one summary row per Pareto individual, per
split: alpha_sys, accuracy_exact_match_vs_ideal, avg_model_cost, and
per-class recall/precision.

Three different, easily-confused numbers show up in this pipeline's
output — spelled out here explicitly since mixing them up has caused real
confusion in project meetings before:

  - alpha_sys (Eq. 3 of the PERTINENCE paper): the fraction of images where
    the model the dispatcher ACTUALLY PICKED classifies correctly — looked
    up directly from each ground-truth CSV's own <model>_correct columns,
    indexed by the predicted model. Overestimating (routing to a bigger,
    still-correct model) costs nothing on this metric, only avg_model_cost.

  - avg_model_cost includes config.DISPATCHER_OVERHEAD_COST (the feature
    extractor's + FC layer's own forward-pass cost) on top of the
    dispatched model's own cost, matching the paper's Eq. 4 definition of
    this objective and how dispatcher/fitness.py computes it during the
    search — see that module's docstring for the full reasoning. It's the
    same fixed constant added to every individual, so it doesn't change
    which individuals are non-dominated, only the absolute number.

  - accuracy_exact_match_vs_ideal (this column): the fraction of images
    where the dispatcher's routing decision was an EXACT MATCH to
    ideal_label (the argmin-cheapest-correct reference label). This is
    strictly stricter than alpha_sys — it punishes overestimation exactly
    as hard as an actual misclassification, since routing to any model
    other than the ideal one counts as wrong here, even a bigger-but-
    still-correct one.

  - recall_<model> / precision_<model>: per-class classification metrics,
    ALSO computed against ideal_label — "how well does this config route
    images to their ideal target class," broken down per class rather than
    collapsed into one number like accuracy_exact_match_vs_ideal.

alpha_sys and accuracy_exact_match_vs_ideal will diverge whenever the
dispatcher overestimates a lot (routes to bigger-than-needed models) — a
high-alpha_sys, low-accuracy config is one that's "safe" (rarely picks a
model that gets it wrong) but not "sharp" (rarely picks the ideal,
cheapest-correct model). Don't conflate the two, and don't compare either
against another pipeline's numbers without checking which definition was
used.

"Impossible" images (no pool model correct) are never dropped by
label_data.py — they're routed to the highest-cost model instead of being
given a sentinel label, so ideal_label is always a real class here, no
special-casing needed in this module.
"""

import os
import numpy as np
import pandas as pd

from metrics import confusion_matrix, recall_per_class, precision_per_class


def correctness_matrix_from_csv(ground_truth_csv, config):
    """Loads the (num_images, NUM_MODELS) boolean correctness matrix from a
    ground-truth CSV's <model>_correct columns."""
    columns = [f"{name}_correct" for name in config.MODEL_NAMES]
    return pd.read_csv(ground_truth_csv)[columns].values.astype(bool)


def _summarize_split(predictions_df, correctness_matrix, out_csv, config):
    """Computes alpha_sys, accuracy_exact_match_vs_ideal (see module
    docstring for how this differs from alpha_sys), avg_model_cost, and
    per-class recall/precision for every pred_<id> column in
    predictions_df, sorted by alpha_sys descending, and writes the result
    to out_csv."""
    ideal_labels = predictions_df["ideal_label"].values
    prediction_columns = [col for col in predictions_df.columns if col.startswith("pred_")]
    cost_array = np.array(config.MODEL_COST)
    image_indices = np.arange(len(predictions_df))

    rows = []
    for column_name in prediction_columns:
        individual_id = int(column_name.replace("pred_", ""))
        predicted_labels = predictions_df[column_name].values

        alpha_sys = correctness_matrix[image_indices, predicted_labels].mean()
        avg_model_cost = cost_array[predicted_labels].mean() + config.DISPATCHER_OVERHEAD_COST
        accuracy_exact_match_vs_ideal = (predicted_labels == ideal_labels).mean()

        cm = confusion_matrix(ideal_labels, predicted_labels, config)
        recall = recall_per_class(cm, config)
        precision = precision_per_class(cm, config)

        row = {
            "individual": individual_id,
            "alpha_sys": alpha_sys,
            "accuracy_exact_match_vs_ideal": accuracy_exact_match_vs_ideal,
            "avg_model_cost": avg_model_cost,
        }
        for class_idx, model_name in enumerate(config.MODEL_NAMES):
            row[f"recall_{model_name}"] = recall[class_idx]
            row[f"precision_{model_name}"] = precision[class_idx]

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("alpha_sys", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    return summary_df


def summarize(train_predictions_df, val_predictions_df, config):
    """Summarizes both splits and writes results/{train,val}_summary.csv.
    Returns (train_summary, val_summary)."""
    train_correctness = correctness_matrix_from_csv(config.TRAIN_GROUND_TRUTH_CSV, config)
    val_correctness = correctness_matrix_from_csv(config.VAL_GROUND_TRUTH_CSV, config)

    print("\nSummarizing train predictions...")
    train_summary = _summarize_split(train_predictions_df, train_correctness, config.TRAIN_SUMMARY_CSV, config)
    print(f"Saved -> {config.TRAIN_SUMMARY_CSV}")

    print("\nSummarizing val predictions...")
    val_summary = _summarize_split(val_predictions_df, val_correctness, config.VAL_SUMMARY_CSV, config)
    print(f"Saved -> {config.VAL_SUMMARY_CSV}")

    return train_summary, val_summary

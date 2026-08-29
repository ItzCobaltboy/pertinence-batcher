"""
Phase 1: for every config on the Pareto front, train a fresh dispatcher head
on the train set, then predict on both the train set and the val set. Saves
every prediction to disk so the analysis phase never needs to retrain.
"""

import os
import torch
import pandas as pd

import constants as c
from embeddings import load_or_compute_embeddings
from config_utils import build_penalty_matrix
from trainer import train_dispatcher_config, predict_with_weights


def run_training_and_prediction():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_df = pd.read_csv(c.TRAIN_GROUND_TRUTH_CSV)
    val_df = pd.read_csv(c.VAL_GROUND_TRUTH_CSV)
    pareto_df = pd.read_csv(c.PARETO_FRONT_CSV)
    print(f"  train images   : {len(train_df)}")
    print(f"  val images     : {len(val_df)}")
    print(f"  pareto configs : {len(pareto_df)}")

    print("\nTrain embeddings:")
    train_embeddings, train_labels = load_or_compute_embeddings(
        train_df, c.TRAIN_EMBEDDINGS_NPZ, device)

    print("\nVal embeddings:")
    val_embeddings, val_labels = load_or_compute_embeddings(
        val_df, c.VAL_EMBEDDINGS_NPZ, device)

    # these two tables will hold one prediction column per config
    train_predictions_df = train_df[["image_path", "label"]].copy()
    train_predictions_df = train_predictions_df.rename(columns={"label": "true_label"})

    val_predictions_df = val_df[["image_path", "label"]].copy()
    val_predictions_df = val_predictions_df.rename(columns={"label": "true_label"})

    print(f"\nTraining and predicting {len(pareto_df)} configs...\n")
    for i in range(len(pareto_df)):
        pareto_row = pareto_df.iloc[i]
        individual_id = int(pareto_row["individual"])

        penalty_matrix = build_penalty_matrix(pareto_row)
        alpha = float(pareto_row["alpha"])

        W, b = train_dispatcher_config(train_embeddings, train_labels, penalty_matrix, alpha, device)

        train_preds = predict_with_weights(train_embeddings, W, b)
        val_preds = predict_with_weights(val_embeddings, W, b)

        column_name = f"pred_{individual_id}"
        train_predictions_df[column_name] = train_preds
        val_predictions_df[column_name] = val_preds

        train_correct = (train_preds == train_labels).sum() / len(train_labels)
        val_correct = (val_preds == val_labels).sum() / len(val_labels)
        print(f"  [{i+1:>2}/{len(pareto_df)}] individual={individual_id:>3}  "
              f"train_acc={100*train_correct:.1f}%  val_acc={100*val_correct:.1f}%")

    os.makedirs(os.path.join(c.PROJECT_ROOT, "predictions"), exist_ok=True)
    train_predictions_df.to_csv(c.TRAIN_PREDICTIONS_CSV, index=False)
    val_predictions_df.to_csv(c.VAL_PREDICTIONS_CSV, index=False)
    print(f"\nSaved -> {c.TRAIN_PREDICTIONS_CSV}")
    print(f"Saved -> {c.VAL_PREDICTIONS_CSV}")

"""
For every individual on the Pareto front, loads its FC weights from
model_cache/ (freshly retrained by build_models.py earlier in the same run)
and predicts on both the train and val embeddings. Saves every prediction
to disk once, so the summary step (and any future per-class analysis)
never needs to touch the models again.
"""

import os
import numpy as np
import pandas as pd

import constants as c


def _predict(embeddings, W, b):
    """Plain numpy prediction: argmax(embeddings @ W.T + b)."""
    logits = embeddings.dot(W.T) + b
    return np.argmax(logits, axis=1)


def _predict_all_individuals(embeddings, individual_ids):
    """Loads each individual's saved (W, b) from model_cache/ and predicts
    on embeddings. Returns {individual_id: predictions}."""
    predictions = {}
    for individual_id in individual_ids:
        path = os.path.join(c.MODEL_CACHE_DIR, f"individual_{individual_id}.npz")
        saved = np.load(path)
        predictions[individual_id] = _predict(embeddings, saved["W"], saved["b"])
    return predictions


def _save(dataframe, predictions, out_csv):
    """Writes image_path, ideal_label, and one pred_<id> column per
    individual to out_csv."""
    out_df = dataframe[["image_path", "label"]].copy()
    out_df = out_df.rename(columns={"label": "ideal_label"})
    for individual_id, preds in predictions.items():
        out_df[f"pred_{individual_id}"] = preds

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def cache_predictions(train_embeddings, val_embeddings):
    """Predicts with every Pareto individual on train + val embeddings and
    caches both to predictions/{train,val}_predictions.csv. Returns
    (train_predictions_df, val_predictions_df)."""
    pareto_df = pd.read_csv(c.PARETO_FRONT_CSV)
    individual_ids = pareto_df["individual"].tolist()
    print(f"Predicting with {len(individual_ids)} saved Pareto individuals...")

    train_df = pd.read_csv(c.TRAIN_GROUND_TRUTH_CSV)
    val_df = pd.read_csv(c.VAL_GROUND_TRUTH_CSV)

    train_predictions = _predict_all_individuals(train_embeddings, individual_ids)
    val_predictions = _predict_all_individuals(val_embeddings, individual_ids)

    train_out = _save(train_df, train_predictions, c.TRAIN_PREDICTIONS_CSV)
    val_out = _save(val_df, val_predictions, c.VAL_PREDICTIONS_CSV)
    print(f"Saved -> {c.TRAIN_PREDICTIONS_CSV}")
    print(f"Saved -> {c.VAL_PREDICTIONS_CSV}")

    return train_out, val_out

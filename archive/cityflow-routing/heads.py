"""
Loads yolo-dispatcher's Pareto front (one Linear(448, 4) FC head per point)
and applies every head to a batch of embeddings. numpy only.

Formats, as written by the shared dispatcher code:
  results/nsga2/pareto_front.csv           one row per point, key `individual`
  results/eval/model_cache/individual_<id>.npz   W (4, 448), b (4,), chromosome, scheme
      (retrained by dispatcher_analysis/build_models.py, seeded from the
      chromosome: the weights the test/final_val numbers were computed with)
  results/nsga2/models/individual_<id>.npz       same fields, saved by
      dispatcher/save_models.py (fallback if model_cache is missing)
  results/eval/{test,final_val}_summary.csv      individual, alpha_sys, avg_model_cost, ...

A head routes an image to argmax(emb @ W.T + b); index i is MODEL_NAMES[i]
(yolov8n, yolov8s, yolov8m, yolov8l), the same as dispatcher_model.predict.
"""

import hashlib
import os

import numpy as np
import pandas as pd


def _weights_path(yolo_results, point_id):
    for sub in [("eval", "model_cache"), ("nsga2", "models")]:
        path = os.path.join(yolo_results, *sub, f"individual_{point_id}.npz")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no weights for Pareto point {point_id} under {yolo_results}")


def _summary(yolo_results, split):
    path = os.path.join(yolo_results, "eval", f"{split}_summary.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} missing, its accuracy/cost columns will be empty")
        return pd.DataFrame(columns=["individual", "alpha_sys", "avg_model_cost"])
    return pd.read_csv(path)[["individual", "alpha_sys", "avg_model_cost"]]


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_pareto(yolo_results):
    """Returns (points, heads): points is a DataFrame with point_id,
    test_alpha_sys, test_cost_gflops, final_val_alpha_sys,
    final_val_cost_gflops, scheme, weights_path, weights_sha256; heads is
    {point_id: (W float32 (4, 448), b float32 (4,))}."""
    front = pd.read_csv(os.path.join(yolo_results, "nsga2", "pareto_front.csv"))
    points = pd.DataFrame({"point_id": front["individual"].astype(int)})
    points["scheme"] = front["scheme"] if "scheme" in front else ""

    for split in ["test", "final_val"]:
        summary = _summary(yolo_results, split).rename(columns={
            "individual": "point_id",
            "alpha_sys": f"{split}_alpha_sys",
            "avg_model_cost": f"{split}_cost_gflops"})
        points = points.merge(summary, on="point_id", how="left")

    heads = {}
    paths, hashes = [], []
    for point_id in points["point_id"]:
        path = _weights_path(yolo_results, point_id)
        saved = np.load(path)
        heads[point_id] = (saved["W"].astype(np.float32), saved["b"].astype(np.float32))
        paths.append(os.path.abspath(path))
        hashes.append(file_sha256(path))
    points["weights_path"] = paths
    points["weights_sha256"] = hashes

    columns = ["point_id", "test_alpha_sys", "test_cost_gflops", "final_val_alpha_sys",
               "final_val_cost_gflops", "scheme", "weights_path", "weights_sha256"]
    return points[columns].sort_values("point_id").reset_index(drop=True), heads


def apply_heads(embeddings, heads):
    """embeddings: float32 (n, 448). Returns {point_id: (probs (n, 4), chosen (n,))}
    with chosen = argmax of the logits (the model index) and probs = softmax."""
    out = {}
    for point_id, (W, b) in heads.items():
        logits = embeddings @ W.T + b
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        out[point_id] = (probs, np.argmax(logits, axis=1))
    return out

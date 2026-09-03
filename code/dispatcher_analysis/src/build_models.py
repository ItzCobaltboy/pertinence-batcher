"""
Retrains every Pareto individual's FC head directly from its chromosome
(pareto_front.csv) and INS class weights — never trusted as a copied
cache. Overwrites model_cache/individual_<id>.npz every run.

Same reasoning as save_models.py in code/Dispatcher: fresh init + shuffle
means these weights won't be bit-identical to whatever pymoo saw during the
search, but with the same chromosome, hyperparameters, and class weights,
results should be very close. Retraining all 50 individuals costs a few
minutes (~3-5s each) — cheap insurance against ever evaluating a stale or
mismatched weight file.
"""

import os
import numpy as np
import pandas as pd

import constants as c
from penalty_matrix import build_penalty_matrix
from class_weights import compute_ins_class_weights
from train_fc import train_fc


def build_models(train_embeddings, train_labels, device):
    pareto_df = pd.read_csv(c.PARETO_FRONT_CSV)
    chromosome_columns = [col for col in pareto_df.columns if col.startswith("P_")]

    class_weights = compute_ins_class_weights(train_labels)
    os.makedirs(c.MODEL_CACHE_DIR, exist_ok=True)

    print(f"Retraining {len(pareto_df)} Pareto individuals from their chromosomes "
          f"({c.FC_EPOCHS} epochs each)...")
    for i, row in pareto_df.iterrows():
        individual_id = int(row["individual"])
        chromosome = row[chromosome_columns].values.astype(np.float32)

        penalty_matrix = build_penalty_matrix(chromosome)
        W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device)

        path = os.path.join(c.MODEL_CACHE_DIR, f"individual_{individual_id}.npz")
        np.savez(path, W=W, b=b, chromosome=chromosome)

        if (i + 1) % 10 == 0 or i == len(pareto_df) - 1:
            print(f"  trained {i + 1}/{len(pareto_df)}")

    print(f"All {len(pareto_df)} models retrained -> {c.MODEL_CACHE_DIR}")

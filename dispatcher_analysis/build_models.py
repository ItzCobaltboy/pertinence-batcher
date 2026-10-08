"""
Retrains every Pareto individual's FC head directly from its chromosome
(pareto_front.csv), overwriting model_cache/individual_<id>.npz every run
— never a copied cache.

FC init + shuffle are seeded from the chromosome (train_fc.chromosome_seed,
same function as dispatcher/dispatcher_model.py), and pareto_front.csv stores
chromosomes unrounded, so on the same device this reproduces the weights the
search scored (up to GPU nondeterminism). Retraining every
individual costs a few minutes (~3-5s each) — cheap insurance against ever
evaluating a stale or mismatched weight file.
"""

import os
import numpy as np
import pandas as pd

from penalty_matrix import build_penalty_matrix
from class_weights import compute_class_weights
from weighting_scheme import decode_scheme, searches_weighting_scheme
from train_fc import train_fc, chromosome_seed


def build_models(train_embeddings, train_labels, device, config):
    """Reads pareto_front.csv, retrains an FC head per individual, and
    writes model_cache/individual_<id>.npz (W, b, chromosome, scheme) for
    each. Each individual's own weighting scheme (decoded from its saved
    chromosome — "INS" for every individual on tracks that don't search
    it) decides which class weights that individual retrains with,
    matching what fitness.py used for it during the original search."""
    # round_trip: the seed is hashed from these exact float64 genes
    pareto_df = pd.read_csv(config.PARETO_FRONT_CSV, float_precision="round_trip")
    penalty_columns = [col for col in pareto_df.columns if col.startswith("P_")]
    searches_scheme = searches_weighting_scheme(config)

    os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)

    print(f"Retraining {len(pareto_df)} Pareto individuals from their chromosomes "
          f"({config.FC_EPOCHS} epochs each)...")
    for i, row in pareto_df.iterrows():
        individual_id = int(row["individual"])
        chromosome = row[penalty_columns].values.astype(np.float64)
        if searches_scheme:
            # pareto_front.csv's own `scheme_gene` column carries the raw
            # gene value save_results.py recorded — appending it here
            # reconstructs the exact chromosome decode_scheme() expects
            # (penalty genes followed by the scheme gene), rather than
            # re-deriving the scheme from the already-decoded `scheme`
            # column string.
            chromosome = np.append(chromosome, np.float64(row["scheme_gene"]))

        penalty_matrix = build_penalty_matrix(chromosome, config)
        scheme = decode_scheme(chromosome, config)
        class_weights = compute_class_weights(train_labels, scheme, config)
        W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device, config,
                        seed=chromosome_seed(chromosome))

        path = os.path.join(config.MODEL_CACHE_DIR, f"individual_{individual_id}.npz")
        np.savez(path, W=W, b=b, chromosome=chromosome, scheme=scheme)

        if (i + 1) % 10 == 0 or i == len(pareto_df) - 1:
            print(f"  trained {i + 1}/{len(pareto_df)}")

    print(f"All {len(pareto_df)} models retrained -> {config.MODEL_CACHE_DIR}")

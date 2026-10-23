"""
Saves the final Pareto front (chromosomes + objectives) to a CSV — one row
per non-dominated individual, penalty matrix values spelled out as columns.
"""

import os
import pandas as pd

from weighting_scheme import decode_scheme, searches_weighting_scheme


def save_pareto_front(chromosomes, objectives, logger, config):
    """Writes config.PARETO_FRONT_CSV with columns: individual, alpha_sys,
    avg_model_cost, P_<true><pred> for every off-diagonal penalty entry,
    plus a `scheme` column (and the raw `scheme_gene` value) when this
    track searches the weighting scheme — sorted by alpha_sys descending.
    Values are written unrounded: dispatcher_analysis rebuilds each
    chromosome from this CSV and seeds its FC retraining from the exact
    float64 genes (a rounded scheme gene could also flip bins).
    avg_model_cost's unit is dataset-specific — see config.MODEL_COST_UNIT."""
    searches_scheme = searches_weighting_scheme(config)
    rows = []
    for individual_id in range(len(chromosomes)):
        chromosome = chromosomes[individual_id]
        alpha_sys_loss, avg_model_cost = objectives[individual_id]
        alpha_sys = 1.0 - alpha_sys_loss

        row = {
            "individual": individual_id,
            "alpha_sys": float(alpha_sys),
            "avg_model_cost": float(avg_model_cost),
        }

        gene_index = 0
        for true_class in range(config.NUM_CLASSES):
            for pred_class in range(config.NUM_CLASSES):
                if true_class != pred_class:
                    row[f"P_{true_class}{pred_class}"] = float(chromosome[gene_index])
                    gene_index += 1

        if searches_scheme:
            row["scheme"] = decode_scheme(chromosome, config)
            row["scheme_gene"] = float(chromosome[gene_index])

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("alpha_sys", ascending=False)
    os.makedirs(config.NSGA2_DIR, exist_ok=True)
    df.to_csv(config.PARETO_FRONT_CSV, index=False)
    logger.info(f"Pareto front -> {config.PARETO_FRONT_CSV}  ({len(df)} individuals, "
                f"cost unit: {config.MODEL_COST_UNIT})")

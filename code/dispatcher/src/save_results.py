"""
Saves the final Pareto front (chromosomes + objectives) to a CSV — one row
per non-dominated individual, penalty matrix values spelled out as columns.
"""

import os
import pandas as pd

import constants as c


def save_pareto_front(chromosomes, objectives, logger):
    rows = []
    for individual_id in range(len(chromosomes)):
        chromosome = chromosomes[individual_id]
        alpha_sys_loss, avg_flops_G = objectives[individual_id]
        alpha_sys = 1.0 - alpha_sys_loss

        row = {
            "individual": individual_id,
            "alpha_sys": round(float(alpha_sys), 4),
            "avg_flops_G": round(float(avg_flops_G), 4),
        }

        gene_index = 0
        for true_class in range(c.NUM_CLASSES):
            for pred_class in range(c.NUM_CLASSES):
                if true_class != pred_class:
                    row[f"P_{true_class}{pred_class}"] = round(float(chromosome[gene_index]), 4)
                    gene_index += 1

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("alpha_sys", ascending=False)
    os.makedirs(c.NSGA2_DIR, exist_ok=True)
    df.to_csv(c.PARETO_FRONT_CSV, index=False)
    logger.info(f"Pareto front -> {c.PARETO_FRONT_CSV}  ({len(df)} individuals)")

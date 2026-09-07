"""
Matplotlib outputs:
  - Pareto scatter: 1 - alpha_sys vs avg_flops_G (both minimized), train and
    val side by side, non-dominated points highlighted and connected — a
    visual check of whether the front NSGA-II found on train is still a
    real Pareto front once independently recomputed here, and whether it
    still holds up on held-out val.
  - Confusion matrices (ideal_label vs predicted) for a few representative
    individuals off the val-recomputed front: cheapest, highest-alpha_sys,
    and a middle pick.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import constants as c
from metrics import confusion_matrix


def _non_dominated_mask(x, y):
    """
    Both x (avg_flops_G) and y (1 - alpha_sys) are minimized. A point is
    non-dominated if no other point is <= it on both axes with a strict
    improvement on at least one.
    """
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = x[j] <= x[i] and y[j] <= y[i]
            strictly_better = x[j] < x[i] or y[j] < y[i]
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return ~dominated


def plot_pareto_scatter(train_summary, val_summary, out_path):
    """Saves a side-by-side train/val Pareto scatter (1-alpha_sys vs
    avg_flops_G) with non-dominated points highlighted, to out_path."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, summary, title in [(axes[0], train_summary, "Train"), (axes[1], val_summary, "Val")]:
        x = summary["avg_flops_G"].values
        y = 1.0 - summary["alpha_sys"].values
        non_dominated = _non_dominated_mask(x, y)

        ax.scatter(x[~non_dominated], y[~non_dominated], color="lightgray",
                   label="dominated", zorder=2)
        order = np.argsort(x[non_dominated])
        ax.plot(x[non_dominated][order], y[non_dominated][order], "o-", color="crimson",
                label="Pareto front", zorder=3)

        ax.set_xlabel("avg FLOPs (G)")
        ax.set_ylabel("1 - alpha_sys")
        ax.set_title(f"{title}: {non_dominated.sum()}/{len(summary)} non-dominated")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Pareto front | 1-alpha_sys vs avg FLOPs")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_confusion_matrices(val_predictions_df, val_summary, individual_ids, out_path):
    """Saves one confusion-matrix heatmap (predicted vs. ideal_label) per
    individual in individual_ids, side by side, to out_path."""
    ideal_labels = val_predictions_df["ideal_label"].values

    fig, axes = plt.subplots(1, len(individual_ids), figsize=(5.5 * len(individual_ids), 5))
    if len(individual_ids) == 1:
        axes = [axes]

    for ax, individual_id in zip(axes, individual_ids):
        predicted_labels = val_predictions_df[f"pred_{individual_id}"].values
        cm = confusion_matrix(ideal_labels, predicted_labels)

        row = val_summary[val_summary["individual"] == individual_id].iloc[0]

        im = ax.imshow(cm, cmap="Blues")
        for i in range(c.NUM_CLASSES):
            for j in range(c.NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xticks(range(c.NUM_CLASSES))
        ax.set_yticks(range(c.NUM_CLASSES))
        ax.set_xticklabels(c.MODEL_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(c.MODEL_NAMES)
        ax.set_xlabel("Predicted (routed to)")
        ax.set_ylabel("Ideal label")
        ax.set_title(f"individual {individual_id}\nalpha_sys={100*row['alpha_sys']:.1f}%  "
                      f"flops={row['avg_flops_G']:.2f}G")

    fig.suptitle("Confusion matrices vs ideal_label (val set)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")

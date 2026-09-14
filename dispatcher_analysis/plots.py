"""
Matplotlib outputs:
  - Pareto scatter: 1 - alpha_sys vs avg_model_cost (both minimized), train
    and val side by side, non-dominated points highlighted and connected —
    a visual check of whether the front NSGA-II found on train is still a
    real Pareto front once independently recomputed here, and whether it
    still holds up on held-out val.
  - Accuracy-vs-cost: the paper's own Fig. 6-10 plotting convention
    directly — accuracy (%) on the y-axis (higher better), cost on the
    x-axis (lower better), PERTINENCE's dispatcher configs as one series
    and each individual pool model's own standalone accuracy/cost ("SOTA
    CNNs" in the paper's legend) as a second series, with the Pareto front
    across BOTH series connected. This is a different framing from the
    scatter above (which only ever plots PERTINENCE's own points against
    each other, never compares against a standalone SOTA model) — use this
    one when the point is showing PERTINENCE's actual value proposition
    over just picking one fixed model, the way the paper's own figures do.
  - Confusion matrices (ideal_label vs predicted) for a few representative
    individuals off the val-recomputed front: cheapest, highest-alpha_sys,
    and a middle pick.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from metrics import confusion_matrix


def _non_dominated_mask(x, y, minimize_y=True):
    """
    x (cost) is always minimized. y is minimized if minimize_y (e.g.
    1-alpha_sys), maximized otherwise (e.g. raw accuracy %) — a point is
    non-dominated if no other point is at least as good on both axes with
    a strict improvement on at least one.
    """
    n = len(x)
    y_better_or_equal = (lambda a, b: a <= b) if minimize_y else (lambda a, b: a >= b)
    y_strictly_better = (lambda a, b: a < b) if minimize_y else (lambda a, b: a > b)

    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = x[j] <= x[i] and y_better_or_equal(y[j], y[i])
            strictly_better = x[j] < x[i] or y_strictly_better(y[j], y[i])
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return ~dominated


def plot_pareto_scatter(train_summary, val_summary, out_path, config):
    """Saves a side-by-side train/val Pareto scatter (1-alpha_sys vs
    avg_model_cost) with non-dominated points highlighted, to out_path."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, summary, title in [(axes[0], train_summary, "Train"), (axes[1], val_summary, "Val")]:
        x = summary["avg_model_cost"].values
        y = 1.0 - summary["alpha_sys"].values
        non_dominated = _non_dominated_mask(x, y)

        ax.scatter(x[~non_dominated], y[~non_dominated], color="lightgray",
                   label="dominated", zorder=2)
        order = np.argsort(x[non_dominated])
        ax.plot(x[non_dominated][order], y[non_dominated][order], "o-", color="crimson",
                label="Pareto front", zorder=3)

        ax.set_xlabel(f"avg model cost ({config.MODEL_COST_UNIT})")
        ax.set_ylabel("1 - alpha_sys")
        ax.set_title(f"{title}: {non_dominated.sum()}/{len(summary)} non-dominated")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Pareto front | 1-alpha_sys vs avg model cost")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_accuracy_vs_cost(summary, correctness_matrix, out_path, config, title="PERTINENCE vs SOTA CNNs"):
    """Saves one accuracy(%)-vs-cost plot matching the paper's own Fig.
    6-10 style exactly: PERTINENCE's dispatcher configs (black dots), each
    individual pool model's own standalone accuracy/cost (green dots,
    "SOTA CNNs" in the paper's legend), and the Pareto front computed
    across both series together (red dashed line + markers).

    `correctness_matrix` is the (num_images, num_models) boolean array for
    whichever split `summary` was computed against — the same one
    summarize.py's `correctness_matrix_from_csv` builds from a ground-truth CSV's
    <model>_correct columns. A SOTA model's own point uses ONLY its own
    MODEL_COST, no DISPATCHER_OVERHEAD_COST — a standalone deployment of
    that model never runs the feature extractor or FC head, so it doesn't
    pay for either."""
    pertinence_x = summary["avg_model_cost"].values
    pertinence_y = 100.0 * summary["alpha_sys"].values

    sota_x = np.array(config.MODEL_COST, dtype=float)
    sota_y = 100.0 * correctness_matrix.mean(axis=0)

    all_x = np.concatenate([pertinence_x, sota_x])
    all_y = np.concatenate([pertinence_y, sota_y])
    non_dominated = _non_dominated_mask(all_x, all_y, minimize_y=False)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.scatter(pertinence_x, pertinence_y, color="black", label="PERTINENCE", zorder=3, s=25)
    ax.scatter(sota_x, sota_y, color="mediumseagreen", label="SOTA CNNs", zorder=3, s=60)
    for name, x, y in zip(config.MODEL_NAMES, sota_x, sota_y):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    order = np.argsort(all_x[non_dominated])
    ax.plot(all_x[non_dominated][order], all_y[non_dominated][order], "x--", color="crimson",
            label="Pareto Front", zorder=2, markersize=8)

    ax.set_xlabel(f"{config.MODEL_COST_UNIT.replace('-M', '')} per image - lower the better")
    ax.set_ylabel("Accuracy (%) - higher the better")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_confusion_matrices(val_predictions_df, val_summary, individual_ids, out_path, config):
    """Saves one confusion-matrix heatmap (predicted vs. ideal_label) per
    individual in individual_ids, side by side, to out_path."""
    ideal_labels = val_predictions_df["ideal_label"].values

    fig, axes = plt.subplots(1, len(individual_ids), figsize=(5.5 * len(individual_ids), 5))
    if len(individual_ids) == 1:
        axes = [axes]

    for ax, individual_id in zip(axes, individual_ids):
        predicted_labels = val_predictions_df[f"pred_{individual_id}"].values
        cm = confusion_matrix(ideal_labels, predicted_labels, config)

        row = val_summary[val_summary["individual"] == individual_id].iloc[0]

        ax.imshow(cm, cmap="Blues")
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xticks(range(config.NUM_CLASSES))
        ax.set_yticks(range(config.NUM_CLASSES))
        ax.set_xticklabels(config.MODEL_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(config.MODEL_NAMES)
        ax.set_xlabel("Predicted (routed to)")
        ax.set_ylabel("Ideal label")
        ax.set_title(f"individual {individual_id}\nalpha_sys={100*row['alpha_sys']:.1f}%  "
                      f"cost={row['avg_model_cost']:.2f} {config.MODEL_COST_UNIT}")

    fig.suptitle("Confusion matrices vs ideal_label (val set)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")

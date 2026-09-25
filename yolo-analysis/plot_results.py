"""
Plots the per-model recall and exact_match distributions from each split's
results/<split>/coco_class_recall_benchmark.csv (produced by
run_benchmark.py), and derives the cross-model exact_match agreement
structure from the same data. Runs once per split in config.SPLITS
(val2017, train2017), writing outputs under that split's own results/<split>/.

4x2 grid: one row per model in config.MODEL_POOL, recall histogram in the
left column, exact_match bar (True/False counts) in the right column. This
is the first look at how much the four YOLOv8 variants actually disagree,
image by image -- the thing the correctness-definition decision (plain
recall vs. exact-match vs. a threshold) gets made from.

Also pivots the long-format CSV into one row per image with each model's
exact_match as its own column, and counts every True/False permutation
across the model pool (2^len(MODEL_POOL) = 16 combinations for 4 models) --
this is the actual cross-model disagreement structure the correctness
definition depends on, not just each model's own marginal distribution.

Usage:
    python plot_results.py
"""

import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd

import config


def distributions_plot_path(split):
    return os.path.join(config.results_dir(split), "recall_exact_match_distributions.png")


def per_image_exact_match_csv(split):
    return os.path.join(config.results_dir(split), "per_image_exact_match.csv")


def permutation_counts_csv(split):
    return os.path.join(config.results_dir(split), "exact_match_permutation_counts.csv")


def plot_distributions(df, output_path):
    models = config.MODEL_POOL
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.2 * len(models)))

    for row_idx, model_name in enumerate(models):
        model_df = df[df["model"] == model_name]

        ax_recall = axes[row_idx, 0]
        ax_recall.hist(model_df["recall"], bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
        ax_recall.set_title(f"{model_name} -- recall distribution")
        ax_recall.set_xlabel("recall")
        ax_recall.set_ylabel("num images")
        ax_recall.set_xlim(0, 1)

        ax_exact = axes[row_idx, 1]
        counts = model_df["exact_match"].value_counts().reindex([True, False], fill_value=0)
        bars = ax_exact.bar(["exact_match", "not exact_match"], counts.values, color=["#55A868", "#C44E52"])
        ax_exact.set_title(f"{model_name} -- exact_match counts")
        ax_exact.set_ylabel("num images")
        for bar, count in zip(bars, counts.values):
            ax_exact.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          str(count), ha="center", va="bottom")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {output_path}")


def build_per_image_exact_match(df):
    """
    Pivots the long-format (image_id, model, exact_match, ...) CSV into one
    row per image_id with each model's exact_match as its own boolean column.
    """
    pivoted = df.pivot(index="image_id", columns="model", values="exact_match")
    pivoted = pivoted[config.MODEL_POOL]  # fixed column order matching MODEL_POOL
    pivoted = pivoted.reset_index()
    return pivoted

def build_per_image_minimum_match(df):
    """
    Pivots the long-format (image_id, model, partial_match, ...) CSV into one
    row per image_id with each model's partial_match as its own boolean column.
    """

    df["partial_match"] = df["recall"] >= config.MINIMUM_RECALL_THRESHOLD

    pivoted = df.pivot(index="image_id", columns="model", values="partial_match")
    pivoted = pivoted[config.MODEL_POOL]  # fixed column order matching MODEL_POOL
    pivoted = pivoted.reset_index()
    return pivoted


def compute_permutation_counts(pivoted):
    """
    Counts every True/False combination of exact_match across all models in
    config.MODEL_POOL -- 2^len(MODEL_POOL) rows (16 for a 4-model pool), each
    with the count of images landing in that exact combination. Combinations
    with zero images are still included, with count 0.

    Rows are in truth-table order (all False first, all True last), with the
    first model in config.MODEL_POOL as the most significant bit, rather than
    sorted by count -- so the same combination sits on the same row across
    every split and threshold, and tables can be compared side by side.
    """
    models = config.MODEL_POOL
    observed_counts = pivoted.groupby(models, dropna=False).size()

    rows = []
    for combo in itertools.product([False, True], repeat=len(models)):
        row = dict(zip(models, combo))
        row["count"] = int(observed_counts.get(combo, 0))
        rows.append(row)

    counts_df = pd.DataFrame(rows)
    return counts_df


def run_split(split):
    print(f"[plot_results] === split: {split} ===")
    df = pd.read_csv(config.output_csv(split))

    plot_distributions(df, distributions_plot_path(split))

    pivoted = build_per_image_exact_match(df)
    pivoted_min = build_per_image_minimum_match(df)
    out_csv = per_image_exact_match_csv(split)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pivoted.to_csv(out_csv, index=False)
    pivoted_min.to_csv(out_csv.replace(".csv", "_minimum_match.csv"), index=False)
    print(f"[plot_results] wrote {out_csv} ({len(pivoted)} images)")

    print(f"[plot_results] wrote {out_csv.replace('.csv', '_minimum_match.csv')} ({len(pivoted_min)} images)")

    counts_df = compute_permutation_counts(pivoted)
    min_counts_df = compute_permutation_counts(pivoted_min)
    counts_path = permutation_counts_csv(split)
    counts_df.to_csv(counts_path, index=False)
    min_counts_df.to_csv(counts_path.replace(".csv", "_minimum_match.csv"), index=False)
    print(f"[plot_results] wrote {counts_path} ({len(counts_df)} permutations)")
    print(counts_df.to_string(index=False))


def main():
    for split in config.SPLITS:
        run_split(split)


if __name__ == "__main__":
    main()

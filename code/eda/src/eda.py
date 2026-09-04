"""
EDA on model_analysis's benchmark results — separate from, and does not
touch, code/dispatcher's own EDA.

Analyses:
  1. Accuracy per model, grouped by precision (TRT sweep)
  2. Latency per model, grouped by precision (TRT sweep)
  3. Model size per model, grouped by precision (TRT sweep)
  4. Accuracy vs latency scatter, colored by precision, eager baseline
     overlaid as a reference point per model

Plots saved to results/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants as c

PRECISION_COLORS = {
    "fp32": "#4c72b0",
    "fp16": "#55a868",
    "int8": "#c44e52",
    "fp8":  "#8172b2",
}
EAGER_COLOR = "#333333"


def load_results():
    trt_df = pd.read_csv(c.BENCHMARK_RESULTS_CSV)
    eager_df = pd.read_csv(c.EAGER_BASELINE_CSV)
    return trt_df, eager_df


def print_summary_table(trt_df, eager_df):
    print("\n-- Eager fp32 baseline --")
    print(eager_df.to_string(index=False))
    print("\n-- Torch-TensorRT benchmark summary --")
    print(trt_df.to_string(index=False))


def plot_grouped_bar(trt_df, value_column, ylabel, title, filename):
    model_names = c.MODEL_NAMES
    precisions = c.PRECISIONS

    x_positions = np.arange(len(model_names))
    n_precisions = len(precisions)
    bar_width = 0.8 / n_precisions
    # center offset: precision 0 goes leftmost, last precision goes rightmost,
    # the whole group of bars stays centered on each model's x position
    center_offset = (n_precisions - 1) / 2.0

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, precision in enumerate(precisions):
        values = []
        for model_name in model_names:
            row = trt_df[(trt_df["model"] == model_name) & (trt_df["precision"] == precision)]
            if len(row) == 0:
                values.append(0)
            else:
                values.append(row[value_column].values[0])

        offset = (i - center_offset) * bar_width
        ax.bar(x_positions + offset, values, width=bar_width,
               label=precision, color=PRECISION_COLORS[precision])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(c.RESULTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_accuracy_vs_latency(trt_df, eager_df):
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in trt_df.iterrows():
        color = PRECISION_COLORS[row["precision"]]
        ax.scatter(row["latency_ms"], row["accuracy"], color=color, s=80)
        label = f"{row['model']}-{row['precision']}"
        ax.annotate(label, (row["latency_ms"], row["accuracy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    for _, row in eager_df.iterrows():
        ax.scatter(row["latency_ms"], row["accuracy"], color=EAGER_COLOR,
                    marker="x", s=100)
        label = f"{row['model']}-eager"
        ax.annotate(label, (row["latency_ms"], row["accuracy"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=7)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Latency, all models x precisions (+ eager fp32 baseline)")

    # legend built manually since color already encodes precision, not a plotted series
    for precision, color in PRECISION_COLORS.items():
        ax.scatter([], [], color=color, label=precision)
    ax.scatter([], [], color=EAGER_COLOR, marker="x", label="eager fp32")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(c.RESULTS_DIR, "accuracy_vs_latency.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def run_eda():
    trt_df, eager_df = load_results()
    print_summary_table(trt_df, eager_df)

    plot_grouped_bar(trt_df, "accuracy", "Top-1 Accuracy (%)",
                      "Accuracy by model and precision", "accuracy_by_precision.png")
    plot_grouped_bar(trt_df, "latency_ms", "Latency (ms)",
                      "Latency by model and precision", "latency_by_precision.png")
    plot_grouped_bar(trt_df, "model_size_mb", "Model size (MB)",
                      "Model size by model and precision", "size_by_precision.png")
    plot_accuracy_vs_latency(trt_df, eager_df)

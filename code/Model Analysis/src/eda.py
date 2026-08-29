"""
EDA on the Torch-TensorRT benchmark results — separate from and does not
touch code/Dispatcher/eda.py (dispatcher labeling EDA).

Analyses:
  1. Accuracy per model, grouped by precision
  2. Latency per model, grouped by precision
  3. Model size per model, grouped by precision
  4. Accuracy vs latency scatter, colored by precision

Plots saved to results/eda/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants as c

EDA_OUTPUT_DIR = os.path.join(c.RESULTS_DIR, "eda")

PRECISION_COLORS = {
    "fp32": "#4c72b0",
    "fp16": "#55a868",
    "int8": "#c44e52",
    "fp8":  "#8172b2",
}


def load_results():
    return pd.read_csv(c.BENCHMARK_RESULTS_CSV)


def print_summary_table(df):
    print("\n-- Benchmark summary --")
    print(df.to_string(index=False))


def plot_grouped_bar(df, value_column, ylabel, title, filename):
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
            row = df[(df["model"] == model_name) & (df["precision"] == precision)]
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

    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(EDA_OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_accuracy_vs_latency(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        color = PRECISION_COLORS[row["precision"]]
        ax.scatter(row["latency_ms"], row["accuracy"], color=color, s=80)
        label = f"{row['model']}-{row['precision']}"
        ax.annotate(label, (row["latency_ms"], row["accuracy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Latency, all models x precisions")

    # legend built manually since color already encodes precision, not a plotted series
    for precision, color in PRECISION_COLORS.items():
        ax.scatter([], [], color=color, label=precision)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(EDA_OUTPUT_DIR, "accuracy_vs_latency.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def run_eda():
    df = load_results()
    print_summary_table(df)

    plot_grouped_bar(df, "accuracy", "Top-1 Accuracy (%)",
                      "Accuracy by model and precision", "accuracy_by_precision.png")
    plot_grouped_bar(df, "latency_ms", "Latency (ms)",
                      "Latency by model and precision", "latency_by_precision.png")
    plot_grouped_bar(df, "model_size_mb", "Model size (MB)",
                      "Model size by model and precision", "size_by_precision.png")
    plot_accuracy_vs_latency(df)

"""
EDA — Step 1B of the dispatcher pipeline.

Analyses dispatcher_labels.csv to understand:
  1. Label-3 split: RN152 actually correct vs. no model gets it right (noise)
  2. Per-class label distribution across 10 ImageNette classes
  3. Model correctness heatmap: which classes trip which models
  4. Agreement anomalies: images where a smaller model succeeds but larger fails
  5. Cumulative correctness: how much of the dataset each model "covers"

Saves plots to results/eda/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

LABELS_CSV   = "./results/dispatcher_labels.csv"
OUTPUT_DIR   = "./results/eda/"
MODELS       = ["resnet18", "resnet34", "resnet50", "resnet152"]
MODEL_COLS   = [f"{m}_correct" for m in MODELS]
FLOPS        = [1.824, 3.679, 4.134, 11.604]

SYNSET_NAMES = {
    "n01440764": "tench",
    "n02102040": "eng. springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "french horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["synset"] = df["image_path"].apply(lambda p: Path(p).parent.name)
    df["class_name"] = df["synset"].map(SYNSET_NAMES)
    return df


# ── 1. Label-3 split ────────────────────────────────────────────────────────

def label3_split(df: pd.DataFrame):
    l3 = df[df["label"] == 3]
    rn152_correct = l3["resnet152_correct"].sum()
    nobody_correct = (l3["resnet152_correct"] == 0).sum()

    print("\n── Label-3 split ──")
    print(f"  Total label-3 images : {len(l3)}")
    print(f"  RN152 gets it right  : {rn152_correct} ({100*rn152_correct/len(l3):.1f}%)  ← hard but solvable")
    print(f"  Nobody gets it right : {nobody_correct} ({100*nobody_correct/len(l3):.1f}%)  ← noise in training data")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["RN152 correct\n(hard images)", "Nobody correct\n(noise)"],
           [rn152_correct, nobody_correct],
           color=["#4c72b0", "#dd8452"])
    ax.set_title("Label-3 breakdown")
    ax.set_ylabel("Images")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label3_split.png"), dpi=150)
    plt.close()


# ── 2. Label distribution overall ───────────────────────────────────────────

def label_distribution(df: pd.DataFrame):
    dist = df["label"].value_counts().sort_index()
    labels = [f"[{i}] {MODELS[i]}" for i in dist.index]

    print("\n── Label distribution ──")
    for lbl, count in dist.items():
        print(f"  [{lbl}] {MODELS[lbl]:>8}: {count:>5} ({100*count/len(df):.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, dist.values, color=["#4c72b0","#55a868","#c44e52","#8172b2"])
    ax.set_title("Dispatcher label distribution (train set)")
    ax.set_ylabel("Images")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, count in zip(bars, dist.values):
        pct = 100 * count / len(df)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=150)
    plt.close()


# ── 3. Per-class label distribution ─────────────────────────────────────────

def per_class_label_dist(df: pd.DataFrame):
    pivot = (df.groupby(["class_name", "label"])
               .size()
               .unstack(fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100  # normalise to %

    print("\n── Per-class label distribution (%) ──")
    print(pivot.round(1).to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4c72b0","#55a868","#c44e52","#8172b2"]
    bottom = np.zeros(len(pivot))
    for i, col in enumerate(sorted(pivot.columns)):
        vals = pivot[col].values if col in pivot.columns else np.zeros(len(pivot))
        ax.bar(pivot.index, vals, bottom=bottom,
               label=f"[{col}] {MODELS[col]}", color=colors[i])
        bottom += vals
    ax.set_title("Label distribution by ImageNette class")
    ax.set_ylabel("% of class images")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_label_dist.png"), dpi=150)
    plt.close()


# ── 4. Model correctness heatmap ─────────────────────────────────────────────

def correctness_heatmap(df: pd.DataFrame):
    heat = df.groupby("class_name")[MODEL_COLS].mean() * 100

    print("\n── Model accuracy by class (%) ──")
    print(heat.round(1).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=50, vmax=100)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=20)
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index)
    for i in range(len(heat)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{heat.values[i,j]:.0f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if heat.values[i,j] > 65 else "white")
    plt.colorbar(im, ax=ax, label="Top-1 accuracy (%)")
    ax.set_title("Model correctness by ImageNette class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correctness_heatmap.png"), dpi=150)
    plt.close()


# ── 5. Cumulative coverage ───────────────────────────────────────────────────

def cumulative_coverage(df: pd.DataFrame):
    n = len(df)
    coverages = []
    for col in MODEL_COLS:
        # images covered by this model OR any cheaper model
        covered = df[MODEL_COLS[:MODEL_COLS.index(col)+1]].any(axis=1).sum()
        coverages.append(100 * covered / n)

    print("\n── Cumulative coverage (% images at least one model gets right) ──")
    for m, c in zip(MODELS, coverages):
        print(f"  Up to {m:>8}: {c:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(MODELS, coverages, marker="o", linewidth=2, color="#4c72b0")
    ax.set_ylim(80, 101)
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_title("Coverage: % images correctly classified by cheapest model up to X")
    for m, c in zip(MODELS, coverages):
        ax.annotate(f"{c:.1f}%", (m, c), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_coverage.png"), dpi=150)
    plt.close()


# ── 6. Anomalies: smaller model right, larger wrong ─────────────────────────

def anomalies(df: pd.DataFrame):
    print("\n── Anomalies: smaller correct, larger wrong ──")
    pairs = [
        ("resnet18",  "resnet34"),
        ("resnet34",  "resnet50"),
        ("resnet50",  "resnet152"),
        ("resnet18",  "resnet152"),
    ]
    for small, large in pairs:
        n = ((df[f"{small}_correct"] == 1) & (df[f"{large}_correct"] == 0)).sum()
        print(f"  {small} correct but {large} wrong: {n} images ({100*n/len(df):.2f}%)")


# ── 7. All 16 correctness combinations ──────────────────────────────────────

def combination_counts(df: pd.DataFrame):
    combo = df[MODEL_COLS].astype(str).agg("".join, axis=1)
    dist = combo.value_counts().sort_index()

    print("\n── Correctness combinations (RN18 RN34 RN50 RN152) ──")
    print(f"  {'Pattern':<20} {'Count':>6}  {'%':>6}  Meaning")
    print(f"  {'-'*55}")
    for pattern, count in dist.items():
        bits = list(pattern)
        correct = [MODELS[i] for i, b in enumerate(bits) if b == "1"]
        wrong   = [MODELS[i] for i, b in enumerate(bits) if b == "0"]
        meaning = ("all correct" if not wrong
                   else "none correct" if not correct
                   else f"{correct[-1]} ceiling")
        label = f"RN18={bits[0]} RN34={bits[1]} RN50={bits[2]} RN152={bits[3]}"
        print(f"  {label:<20} {count:>6}  {100*count/len(df):>5.1f}%  {meaning}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {LABELS_CSV}...")
    df = load(LABELS_CSV)
    print(f"Loaded {len(df)} images.")

    label_distribution(df)
    label3_split(df)
    per_class_label_dist(df)
    correctness_heatmap(df)
    cumulative_coverage(df)
    anomalies(df)
    combination_counts(df)

    print(f"\nPlots saved → {OUTPUT_DIR}")

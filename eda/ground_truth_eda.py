"""
Ground-truth EDA — runs directly on the labeled ground-truth CSVs
(<model>_correct columns + label column) that dispatcher/ and
dispatcher_analysis/ consume, for both the train and held-out ("val")
split. Generic: reads whichever dataset's config module it's pointed at
(imagenette/config.py or cifar-10/config.py), same as dispatcher/ and
dispatcher_analysis/.

This is a different EDA from archive/model_analysis/eda/, which compares
model_analysis's own quantization-precision benchmark CSVs and has nothing
to do with dataset ground truth.

Both label_data.py scripts keep every raw image in their output CSV,
including ones where no pool model is correct ("impossible" images — these
get routed to the highest-cost model rather than dropped or given a
sentinel label, see either script's module docstring). Since `label` alone
can't distinguish "genuinely routed to the biggest model" from
"impossible, fell back to the biggest model," the oracle rate here is
computed straight from the <model>_correct columns instead — "at least one
model correct" doesn't need `label` at all.

Reports, per split:
  1. Per-model standalone accuracy (mean of each <model>_correct column,
     over ALL images including impossible ones — this is genuinely how
     often that model is right on the real population).
  2. Theoretical maximum ("oracle") accuracy: the fraction of images where
     at least one pool model is correct.
  3. Class balance: the distribution of the routing target (`label`)
     across the 4 pool models — note this folds impossible images in with
     genuine highest-cost-model routes; the oracle/impossible count from
     point 2 is the way to see that split out.
  4. Pairwise model error overlap (Jaccard index on each pair's wrong-image
     sets) — how independent the models' mistakes are, not just how often
     each is individually right.
"""

import numpy as np
import pandas as pd


def _load_ground_truth(csv_path, config):
    df = pd.read_csv(csv_path)
    correctness_columns = [f"{name}_correct" for name in config.MODEL_NAMES]
    return df, correctness_columns


def per_model_accuracy(df, correctness_columns):
    """Mean of each <model>_correct column — standalone accuracy per model
    on this split, over every image (impossible ones included — they
    correctly pull every model's accuracy down slightly, since by
    definition all four are wrong on them)."""
    return df[correctness_columns].mean()


def oracle_accuracy(df, correctness_columns):
    """Fraction of images where at least one pool model is correct,
    computed directly from the correctness columns (not from `label`,
    which can't distinguish an impossible image from a genuine
    highest-cost-model route — see module docstring)."""
    return df[correctness_columns].any(axis=1).mean()


def class_balance(df, config):
    """Distribution of the routing target (`label`) across the pool, as
    counts and fractions — is one model's "territory" wildly over- or
    under-represented in this split. Note impossible images are folded
    into the highest-cost model's count here (see module docstring); use
    the oracle rate (point 2) to see how many of those are genuine vs.
    fallback routes."""
    counts = df["label"].value_counts().reindex(range(config.NUM_CLASSES), fill_value=0)
    fractions = counts / len(df)
    return pd.DataFrame({
        "model": config.MODEL_NAMES,
        "count": counts.values,
        "fraction": fractions.values,
    })


def pairwise_error_overlap(df, config):
    """Jaccard index of each pair of models' WRONG-image sets: |wrong_A n
    wrong_B| / |wrong_A u wrong_B|. Low overlap means the models tend to
    fail on different images (complementary errors) rather than all
    missing the same hard images — relevant to how much headroom a smarter
    dispatcher could capture beyond the single best model. Impossible
    images (wrong for every model) necessarily land in every pair's
    intersection, so they push overlap slightly upward — a real property
    of the split, not an artifact to filter out."""
    wrong_sets = {}
    for name in config.MODEL_NAMES:
        wrong_sets[name] = set(np.where(df[f"{name}_correct"].values == 0)[0])

    rows = []
    for i, name_a in enumerate(config.MODEL_NAMES):
        for name_b in config.MODEL_NAMES[i + 1:]:
            wrong_a, wrong_b = wrong_sets[name_a], wrong_sets[name_b]
            union = wrong_a | wrong_b
            jaccard = len(wrong_a & wrong_b) / len(union) if union else float("nan")
            rows.append({"model_a": name_a, "model_b": name_b, "jaccard_wrong_overlap": jaccard})
    return pd.DataFrame(rows)


def _print_split_report(split_name, csv_path, config):
    df, correctness_columns = _load_ground_truth(csv_path, config)
    n = len(df)
    impossible_count = int((~df[correctness_columns].any(axis=1)).sum())

    print(f"\n{'=' * 70}")
    print(f"{split_name.upper()} split — {csv_path}")
    print(f"{n} images total, including {impossible_count} impossible ones "
          f"(no pool model correct, routed to the highest-cost model) -- "
          f"nothing dropped from this CSV")
    print(f"{'=' * 70}")

    print("\n-- 1. Per-model standalone accuracy --")
    accuracy = per_model_accuracy(df, correctness_columns)
    for column_name, value in accuracy.items():
        print(f"  {column_name:<28} {100 * value:.2f}%")

    print("\n-- 2. Theoretical maximum (oracle) accuracy --")
    oracle_rate = oracle_accuracy(df, correctness_columns)
    print(f"  {100 * oracle_rate:.2f}% ({n - impossible_count}/{n} images have at least "
          f"one correct model; {impossible_count} are impossible for this pool)")

    best_model_column = accuracy.idxmax()
    best_model_accuracy = accuracy.max()
    headroom = oracle_rate - best_model_accuracy
    print(f"  Best single model: {best_model_column.replace('_correct', '')} at "
          f"{100 * best_model_accuracy:.2f}% -> oracle headroom above it: "
          f"{100 * headroom:.2f} points")

    print("\n-- 3. Class balance (routing-target distribution across the pool) --")
    balance = class_balance(df, config)
    print(balance.to_string(index=False))
    print(f"  (note: {impossible_count} impossible images are folded into "
          f"{config.MODEL_NAMES[-1]}'s count above, alongside genuine routes there)")

    print("\n-- 4. Pairwise model error overlap (Jaccard on wrong-image sets) --")
    overlap = pairwise_error_overlap(df, config)
    print(overlap.to_string(index=False))

    return {
        "split": split_name,
        "n": n,
        "impossible_count": impossible_count,
        "per_model_accuracy": accuracy,
        "oracle_rate": oracle_rate,
        "class_balance": balance,
        "pairwise_error_overlap": overlap,
    }


def run_eda(config):
    """Runs the ground-truth EDA on both the train and val splits for
    whatever dataset `config` points at, printing a report for each and
    returning both as a dict."""
    results = {
        "train": _print_split_report("train", config.TRAIN_GROUND_TRUTH_CSV, config),
        "val": _print_split_report("val", config.VAL_GROUND_TRUTH_CSV, config),
    }
    return results

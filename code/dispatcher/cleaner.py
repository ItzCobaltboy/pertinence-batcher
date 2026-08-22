"""
Cleaner — Step 1A-post of the dispatcher pipeline.

Reads dispatcher_labels.csv and drops images where no model in the pool
classifies them correctly (pattern 0000). These are unlearnable noise samples
— including them in FC training would add a false label-3 signal with no
useful gradient.

Output: results/cleaned/dispatcher_labels_clean.csv
  Same columns as dispatcher_labels.csv, with noise rows removed.
"""

import os
import pandas as pd

LABELS_CSV   = "./results/dispatcher_labels.csv"
CLEANED_DIR  = "./results/cleaned-data/"
CLEANED_CSV  = os.path.join(CLEANED_DIR, "dispatcher_labels_clean.csv")
MODEL_COLS   = ["resnet18_correct", "resnet34_correct",
                "resnet50_correct", "resnet152_correct"]


def clean_labels(
    input_path: str  = LABELS_CSV,
    output_path: str = CLEANED_CSV,
) -> pd.DataFrame:
    """
    Args:
        input_path:  path to raw dispatcher_labels.csv
        output_path: where to write the cleaned CSV
    Returns:
        cleaned DataFrame
    """
    df = pd.read_csv(input_path)
    n_total = len(df)

    noise_mask = (df[MODEL_COLS].sum(axis=1) == 0)
    n_noise    = noise_mask.sum()
    df_clean   = df[~noise_mask].reset_index(drop=True)

    print(f"\n── Cleaner ──")
    print(f"  Total images   : {n_total}")
    print(f"  Noise (0000)   : {n_noise}  ({100*n_noise/n_total:.1f}%)")
    print(f"  Retained       : {len(df_clean)}  ({100*len(df_clean)/n_total:.1f}%)")

    print("\n  Label distribution after cleaning:")
    dist = df_clean["label"].value_counts().sort_index()
    models = ["resnet18", "resnet34", "resnet50", "resnet152"]
    for lbl, count in dist.items():
        print(f"    [{lbl}] {models[lbl]:>8}: {count:>5}  ({100*count/len(df_clean):.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    return df_clean


if __name__ == "__main__":
    clean_labels()

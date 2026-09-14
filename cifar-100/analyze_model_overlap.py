"""
Analyzes how much a CIFAR-100 model subset's errors overlap vs. don't —
i.e. whether they're wrong on the same images (redundant) or different
images (complementary, giving a dispatcher real headroom above any single
model's own accuracy). Shared across every variant (fig9c/, fig9d/, ...) —
takes ground-truth CSV paths on the command line rather than hardcoding a
model list, since each variant's CSVs carry a different set of
<model>_correct columns depending on that variant's own pool.

Restricts to genuinely held-out images only (filename prefix 'test_' -
CIFAR-100's own official test set, which no pool checkpoint was trained
on). Both a variant's test_ground_truth.csv (the 7,000-image NSGA-II
fitness split) and final_val_ground_truth.csv (the 3,000-image final
validation split) are stratified subsets of this same official test set —
see label_data.py's module docstring for why the validation slice
specifically had to come from test rather than train. A variant's
train_ground_truth.csv is the full, untouched official 50k train set,
filtered out here by its 'train_' filename prefix.

Usage: python analyze_model_overlap.py <csv1> [csv2] ...
       (no default — pass e.g. fig9c/data/test_ground_truth.csv
       fig9c/data/final_val_ground_truth.csv)
"""

import sys
import itertools

import pandas as pd


def main():
    csv_paths = sys.argv[1:]
    if not csv_paths:
        print(__doc__)
        sys.exit(1)

    heldout_parts = []
    model_cols = None
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        these_model_cols = [c for c in df.columns if c.endswith("_correct")]
        if model_cols is None:
            model_cols = these_model_cols
        elif set(model_cols) != set(these_model_cols):
            print(f"WARNING: {csv_path} has different <model>_correct columns "
                  f"({these_model_cols}) than the first CSV ({model_cols}) — "
                  f"pass CSVs from the same variant only.")

        orig_split = df["image_path"].str.extract(r"/(train|test)_")[0]
        part = df[orig_split == "test"]
        print(f"{csv_path}: {len(part)} held-out / {len(df)} total rows")
        heldout_parts.append(part)

    heldout = pd.concat(heldout_parts, ignore_index=True)
    heldout = heldout.drop_duplicates(subset="image_path").reset_index(drop=True)
    print(f"\nCombined held-out (genuinely unseen, deduped) images: {len(heldout)} "
          f"(of 10,000 official CIFAR-100 test images)\n")

    correct = heldout[model_cols].astype(bool)

    print("Per-model accuracy on held-out set:")
    for col in model_cols:
        print(f"  {col:30s} {correct[col].mean():.4f}")
    best_single = correct.mean().max()
    print(f"  {'best single model':30s} {best_single:.4f}\n")

    any_correct = correct.any(axis=1).mean()
    all_correct = correct.all(axis=1).mean()
    print(f"any_correct (oracle ceiling):          {any_correct:.4f}")
    print(f"all_correct (all {len(model_cols)} agree+correct):        {all_correct:.4f}")
    print(f"oracle gain over best single model:    {any_correct - best_single:+.4f}\n")

    print("Pairwise error overlap (Jaccard of WRONG-image sets - 0 = fully")
    print("complementary errors, 1 = identical errors):")
    wrong = ~correct
    for a, b in itertools.combinations(model_cols, 2):
        set_a = set(heldout.index[wrong[a]])
        set_b = set(heldout.index[wrong[b]])
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else float("nan")
        print(f"  {a:26s} vs {b:26s} jaccard={jaccard:.3f}  "
              f"(|A|={len(set_a)}, |B|={len(set_b)}, |A&B|={len(set_a & set_b)})")
    print()

    print(f"Correctness-pattern breakdown (2^{len(model_cols)}={2**len(model_cols)} combinations, "
          f"1=correct/0=wrong, column order = {', '.join(model_cols)}):")
    pattern = correct.astype(int).astype(str).agg("".join, axis=1)
    counts = pattern.value_counts().sort_index()
    for pat, n in counts.items():
        print(f"  {pat}  {n:5d}  ({n / len(heldout):.1%})")
    print()

    print("Images only ONE model gets right (that model's unique catches):")
    for col in model_cols:
        others = [c for c in model_cols if c != col]
        unique = correct[col] & ~correct[others].any(axis=1)
        print(f"  only {col:26s} correct: {unique.sum()} images ({unique.mean():.2%})")


if __name__ == "__main__":
    main()

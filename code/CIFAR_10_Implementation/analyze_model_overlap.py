"""
Analyzes how much the CIFAR-10 pool models' errors overlap vs. don't -
i.e. whether they're wrong on the same images (redundant) or different
images (complementary, giving a dispatcher real headroom above any single
model's own accuracy).

Restricts to genuinely held-out images only (filename prefix 'test_' -
CIFAR-10's own official test set, which no pool model was trained on) even
if the ground-truth CSV passed in still has train-split rows mixed in from
before the leakage fix (see Journel/Week2.md) - so this is safe to run
against either the old or the corrected val_ground_truth.csv.

Accepts multiple CSVs (e.g. both train_ground_truth.csv and
val_ground_truth.csv from a pre-leakage-fix run, where official-test-set
images ended up scattered across both) and pools their held-out rows
together, deduped by image_path, for a bigger/more complete sample of the
true 10k official test set.

Usage: python analyze_model_overlap.py [csv1] [csv2] ...
       (defaults to both dispatcher_analysis/data/{train,val}_ground_truth.csv)
"""

import sys
import itertools

import pandas as pd

DEFAULT_CSVS = [
    "dispatcher_analysis/data/train_ground_truth.csv",
    "dispatcher_analysis/data/val_ground_truth.csv",
]

MODEL_COLS = [
    "resnet20_correct",
    "resnet32_correct",
    "shufflenetv2_x2_0_correct",
    "vgg16_bn_correct",
]


def main():
    csv_paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CSVS

    heldout_parts = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        orig_split = df["image_path"].str.extract(r"/(train|test)_")[0]
        part = df[orig_split == "test"]
        print(f"{csv_path}: {len(part)} held-out / {len(df)} total rows")
        heldout_parts.append(part)

    heldout = pd.concat(heldout_parts, ignore_index=True)
    heldout = heldout.drop_duplicates(subset="image_path").reset_index(drop=True)
    print(f"\nCombined held-out (genuinely unseen, deduped) images: {len(heldout)} "
          f"(of 10,000 official CIFAR-10 test images)\n")

    correct = heldout[MODEL_COLS].astype(bool)

    print("Per-model accuracy on held-out set:")
    for col in MODEL_COLS:
        print(f"  {col:30s} {correct[col].mean():.4f}")
    best_single = correct.mean().max()
    print(f"  {'best single model':30s} {best_single:.4f}\n")

    any_correct = correct.any(axis=1).mean()
    all_correct = correct.all(axis=1).mean()
    print(f"any_correct (oracle ceiling):        {any_correct:.4f}")
    print(f"all_correct (all 4 agree+correct):   {all_correct:.4f}")
    print(f"oracle gain over best single model:  {any_correct - best_single:+.4f}\n")

    print("Pairwise error overlap (Jaccard of WRONG-image sets - 0 = fully")
    print("complementary errors, 1 = identical errors):")
    wrong = ~correct
    for a, b in itertools.combinations(MODEL_COLS, 2):
        set_a = set(heldout.index[wrong[a]])
        set_b = set(heldout.index[wrong[b]])
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else float("nan")
        print(f"  {a:26s} vs {b:26s} jaccard={jaccard:.3f}  "
              f"(|A|={len(set_a)}, |B|={len(set_b)}, |A&B|={len(set_a & set_b)})")
    print()

    print("Correctness-pattern breakdown (2^4=16 combinations, 1=correct/0=wrong,")
    print("column order = resnet20, resnet32, shufflenetv2_x2_0, vgg16_bn):")
    pattern = correct.astype(int).astype(str).agg("".join, axis=1)
    counts = pattern.value_counts().sort_index()
    for pat, n in counts.items():
        print(f"  {pat}  {n:5d}  ({n / len(heldout):.1%})")
    print()

    print("Images only ONE model gets right (that model's unique catches):")
    for col in MODEL_COLS:
        others = [c for c in MODEL_COLS if c != col]
        unique = correct[col] & ~correct[others].any(axis=1)
        print(f"  only {col:26s} correct: {unique.sum()} images ({unique.mean():.2%})")


if __name__ == "__main__":
    main()

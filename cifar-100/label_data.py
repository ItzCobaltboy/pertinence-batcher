"""
Builds the CIFAR-100 ground-truth CSVs consumed by the shared dispatcher/ and
dispatcher_analysis/ code. Shared across every CIFAR-100 dispatcher variant
(fig9c/, fig9d/, ...) — takes a `config` argument the same way the shared
dispatcher/dispatcher_analysis/eda packages do, so each variant's own
config.py (its own model subset, its own data/ output directory) drives one
call to `run_labeling(config)` without this file needing to know how many
variants exist or what any of them are called.

Schema written to each of the three CSVs:

    image_path,<model1>_correct,...,<modelN>_correct,label

- image_path: path to a cached per-image .png under config.DATASET_DIR,
  shared across every variant (the raw images don't depend on which model
  subset a given variant dispatches to)
- <model>_correct: 1/0, whether that model's own prediction matches the
  CIFAR-100 ground-truth class, one column per config.MODEL_NAMES entry
- label: argmin_j { MAdds_j | model_j(x) is correct }, indexed into
  config.MODEL_NAMES in config.MODEL_COST's MAdds-ascending order — the
  dispatcher's ideal routing target for this variant's own pool

Three-way split, not the other two tracks' two-way train/val split
=====================================================================
The PERTINENCE paper (Section II-C, page 7) evaluates a MOEA individual's
fitness on what it calls the "test set" during the search itself, then
separately reports final numbers on what it calls the "validation set" —
"a validation set that was not used during model training or the MOEA
process." Confusingly, the paper's own terminology is backwards from what
the name would suggest: its "test set" is touched constantly (every fitness
eval, every generation), and its "validation set" is the one genuinely held
out until the very end. This file follows the paper's naming for the CSVs
it writes (test_ground_truth.csv, final_val_ground_truth.csv) precisely so
that backwardness stays visible rather than getting silently absorbed into
the other tracks' "val means held-out" convention — see each variant's
config.py's "Split naming" comment for how these map onto the shared
dispatcher/ package's config.VAL_* hook.

Where the split point actually goes
====================================
CIFAR-100 only ships an official train (50,000) / test (10,000) division.
An early version of this file carved the paper's "validation set" out of
the official TRAIN set, on the reasoning that a validation split is usually
carved from train. That was wrong, and the EDA on that data proved it: the
chenyaofo pool checkpoints (models/model_loader.py) are pretrained on the
*entire* official 50k train set, so any slice taken from train — no matter
how carefully stratified — is already memorized by every pool model. The
resulting "validation" slice measured 100% oracle accuracy with most
routing classes completely empty; a final-eval pass against it would report
near-ceiling numbers that say nothing about how the dispatcher generalizes.

The fix is to carve BOTH the paper's "test set" and its "validation set" out
of the official TEST split instead, since that's the only 10,000 images the
pool checkpoints never saw during their own pretraining. The official test
set has exactly 100 images per CIFAR-100 class, split class-stratified 70/30
with a fixed seed (config.FINAL_VAL_SEED):

  - 7,000 images (70/class) -> paper's "test set" -> config.VAL_GROUND_TRUTH_CSV.
    Used only for NSGA-II's per-individual fitness during the search
    (dispatcher/fitness.py) — never touched by FC training.
  - 3,000 images (30/class) -> paper's "validation set" ->
    config.FINAL_VAL_GROUND_TRUTH_CSV. Used only once, by
    dispatcher_analysis/'s final evaluation pass, after the NSGA-II search
    and all FC training are completely done.
  - The full, untouched official train set (50,000) -> paper's "training
    set" -> config.TRAIN_GROUND_TRUTH_CSV. The only pool labeling and
    dispatcher FC training ever see.

The 70/30 split point isn't something the paper specifies (it doesn't give
a split size or mechanism for any of its three datasets, for either the
train/test/validation boundary in general or this specific test/validation
carve) — it's a project-level choice, made and confirmed once, shared by
every variant that imports this file (the split itself doesn't depend on
which model subset a variant dispatches to, only the correctness columns
computed against that split do).

Every raw image in all three resulting splits goes through the same
labeling rule, and none are dropped — including images where NO pool model
in this variant's subset got them right. Those "impossible" images are
routed to the highest-MAdds model in config.MODEL_NAMES rather than dropped
or given a sentinel label, identical to both other tracks' convention — see
either track's label_data.py for the same reasoning. Counts are printed per
split.
"""

import os

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision.datasets import CIFAR100


def download_and_dump_images(config):
    """Downloads CIFAR-100 train+test via torchvision (no-op if already
    downloaded) and dumps every image as a PNG under config.DATASET_DIR,
    returning (train_df, test_df) — each a DataFrame of (image_path,
    class_label) for its own official split. Idempotent: skips PNGs that
    already exist. Shared across every variant, since the raw images and
    their filenames don't depend on a variant's model subset — safe to
    point every variant's config.DATASET_DIR at the same folder."""
    images_dir = os.path.join(config.DATASET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    train_set = CIFAR100(root=config.DATASET_DIR, train=True, download=True)
    test_set = CIFAR100(root=config.DATASET_DIR, train=False, download=True)

    split_frames = {}
    for split_name, dataset in [("train", train_set), ("test", test_set)]:
        rows = []
        for i in range(len(dataset)):
            img, label = dataset[i]
            filename = f"{split_name}_{i:05d}.png"
            abs_path = os.path.join(images_dir, filename)
            if not os.path.exists(abs_path):
                img.save(abs_path)
            # image_path is stored relative to config.DATASET_DIR, resolved
            # by the shared dispatcher/dispatcher_analysis embeddings.py
            # via os.path.join(config.DATASET_DIR, image_path) — not
            # relative to whatever directory a script happens to run from.
            rows.append({"image_path": os.path.join("images", filename),
                         "abs_image_path": abs_path,
                         "class_label": int(label)})
        split_frames[split_name] = pd.DataFrame(rows)

    return split_frames["train"], split_frames["test"]


def carve_test_and_validation_split(test_df, config):
    """Splits the official 10k test set into (fitness_test_df,
    final_val_df): a class-stratified config.FINAL_VAL_FRACTION_OF_TEST
    slice held out as the paper's "validation set" (touched only by the
    final eval pass, at the very end), and the remainder as the paper's
    "test set" (NSGA-II fitness during the search). Stratification is exact
    here since CIFAR-100's official test set has precisely 100 images per
    class — config.FINAL_VAL_FRACTION_OF_TEST * 100 divides evenly for any
    fraction that is a multiple of 1/100, which 0.30 (30/class) is. Doesn't
    depend on a variant's model subset, only on the split-point config
    (FINAL_VAL_FRACTION_OF_TEST, FINAL_VAL_SEED), which every variant
    shares."""
    rng = np.random.RandomState(config.FINAL_VAL_SEED)

    final_val_indices = []
    for class_label, group in test_df.groupby("class_label"):
        n_holdout = int(round(len(group) * config.FINAL_VAL_FRACTION_OF_TEST))
        chosen = rng.choice(group.index.values, size=n_holdout, replace=False)
        final_val_indices.extend(chosen.tolist())

    final_val_indices = set(final_val_indices)
    final_val_df = test_df.loc[test_df.index.isin(final_val_indices)].reset_index(drop=True)
    fitness_test_df = test_df.loc[~test_df.index.isin(final_val_indices)].reset_index(drop=True)

    return fitness_test_df, final_val_df


@torch.no_grad()
def compute_correctness(df, device, split_label, config):
    """Runs every model in config.MODEL_NAMES over every image in df,
    returns df augmented with one <model>_correct column per model plus the
    argmin-MAdds label (using config.MODEL_COST, in the same order as
    config.MODEL_NAMES). NO rows are dropped — images where no model in
    this variant's subset is correct are routed to the highest-MAdds model
    instead of being removed. Returns (out_df, impossible_count,
    total_count) — split_label is only used for progress printing."""
    from model_loader import load_model

    models = {name: load_model(name, device=device) for name in config.MODEL_NAMES}

    n = len(df)
    correct_cols = {name: np.zeros(n, dtype=np.int64) for name in config.MODEL_NAMES}

    batch_size = 256
    paths = df["abs_image_path"].tolist()
    class_labels = df["class_label"].tolist()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for p in paths[start:end]:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(config.IMAGE_TRANSFORM(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        batch_labels = torch.tensor(class_labels[start:end])

        for name, model in models.items():
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu()
            correct_cols[name][start:end] = (preds == batch_labels).numpy().astype(np.int64)

        if (start // batch_size) % 20 == 0:
            print(f"  [{split_label}] correctness pass: {end}/{n}")

    out_df = df.copy()
    for name in config.MODEL_NAMES:
        out_df[f"{name}_correct"] = correct_cols[name]

    # label = argmin-MAdds among correct models; for images where no model
    # is correct (kept, not dropped), route to the highest-cost model
    # (last index, since config.MODEL_NAMES/MODEL_COST are MAdds-ascending)
    # instead — the most defensible fallback, and it keeps every row a
    # normal, valid training target.
    madds = np.array(config.MODEL_COST)
    correct_matrix = np.stack([correct_cols[name] for name in config.MODEL_NAMES], axis=1)  # (n, num_models)
    any_correct = correct_matrix.sum(axis=1) > 0
    highest_cost_model_idx = len(config.MODEL_NAMES) - 1

    labels = np.full(n, highest_cost_model_idx, dtype=np.int64)
    for i in range(n):
        if any_correct[i]:
            correct_idxs = np.where(correct_matrix[i] == 1)[0]
            labels[i] = correct_idxs[np.argmin(madds[correct_idxs])]
    out_df["label"] = labels

    impossible = int((~any_correct).sum())

    final_cols = ["image_path"] + [f"{name}_correct" for name in config.MODEL_NAMES] + ["label"]
    return out_df[final_cols], impossible, n


def _report(label_df, impossible, total, csv_path):
    label_df.to_csv(csv_path, index=False)
    print(f"{total} images written to {csv_path}, {impossible} routed to the "
          f"highest-cost model ({100 * impossible / total:.2f}%, no pool model "
          f"correct — kept, not dropped)")


def run_labeling(config):
    """End-to-end labeling for one variant's model subset (config.MODEL_NAMES).
    Downloads/dumps the shared raw images (idempotent, safe to call from
    every variant), carves the shared test/validation split (idempotent
    given the same seed), then computes this variant's own correctness
    columns and writes its own three ground-truth CSVs."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model subset for this variant: {config.MODEL_NAMES}")

    print("\nDownloading/loading CIFAR-100's official train (50k) + test (10k) splits "
          "and dumping images...")
    train_raw, test_raw = download_and_dump_images(config)
    print(f"Official train: {len(train_raw)} images | Official test: {len(test_raw)} images")

    print(f"\nCarving a class-stratified {config.FINAL_VAL_FRACTION_OF_TEST:.0%} slice out of the "
          f"official TEST set as the paper's held-out 'validation set' (seed={config.FINAL_VAL_SEED}); "
          f"the official TRAIN set stays whole and untouched as the paper's 'training set'...")
    fitness_test_raw, final_val_raw = carve_test_and_validation_split(test_raw, config)
    print(f"NSGA-II fitness pool (paper's 'test set'): {len(fitness_test_raw)} images | "
          f"Final validation (paper's 'validation set'): {len(final_val_raw)} images")

    print("\nComputing per-model correctness + labels for the official TRAIN split "
          "(paper's 'training set' — the only split labeling/FC-training ever sees)...")
    train_df, train_impossible, train_total = compute_correctness(train_raw, device, "train", config)

    print("\nComputing per-model correctness + labels for the test-fitness split "
          "(paper's 'test set' — used only for NSGA-II fitness during the search)...")
    test_df, test_impossible, test_total = compute_correctness(fitness_test_raw, device, "test", config)

    print("\nComputing per-model correctness + labels for the final-validation split "
          "(paper's 'validation set' — touched only once, by the final eval pass)...")
    final_val_df, final_val_impossible, final_val_total = compute_correctness(
        final_val_raw, device, "final_val", config)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    print()
    _report(train_df, train_impossible, train_total, config.TRAIN_GROUND_TRUTH_CSV)
    _report(test_df, test_impossible, test_total, config.VAL_GROUND_TRUTH_CSV)
    _report(final_val_df, final_val_impossible, final_val_total, config.FINAL_VAL_GROUND_TRUTH_CSV)

    print("\nDone. The oracle rate ('at least one model correct') is computed directly "
          "from the <model>_correct columns by eda/ground_truth_eda.py, independent of label.")

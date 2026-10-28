# Fine-tunes yolov8 n, s, m from stock COCO weights on the CityFlow train split (pseudo-labels from
# yolov8l), validating on the val split every epoch. yolov8l is the teacher and is not trained.
# Saves best weights to models/finetuned/ and the training curve to results/03_finetune/.
# --smoke: 1 epoch on the small dataset from 02_make_dataset.py --smoke.
import argparse
import os
import shutil
import time

import pandas as pd
from ultralytics import YOLO

import config
from common import paths, report


def dataset_yaml_for(args):
    """The dataset yaml, with the val list thinned to every Nth frame when --val-every > 1. Validation runs
    after every epoch, and on all 14,775 val frames it took several minutes of each epoch. Only the
    per-epoch validation (which picks best.pt) uses the thinner list; training data is untouched."""
    base = paths.dataset_yaml(args.smoke)
    if args.val_every <= 1:
        return base
    manifest = pd.read_csv(paths.manifest_path(args.smoke))
    val = manifest[manifest["split"] == "val"].sort_values(["camera", "frame_idx"])
    subset = val.iloc[::args.val_every]
    list_path = os.path.join(paths.data_root(args.smoke), f"val_every{args.val_every}.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(subset["path"]) + "\n")
    lines = []
    for line in open(base).read().splitlines():
        if line.startswith("val:"):
            line = "val: " + list_path.replace("\\", "/")
        lines.append(line)
    new_yaml = os.path.join(paths.data_root(args.smoke), f"cityflow_val_every{args.val_every}.yaml")
    with open(new_yaml, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"validating on {len(subset)} of {len(val)} val frames per epoch")
    return new_yaml


def train_once(letter, args, batch):
    run_dir = os.path.join(paths.MODELS, "runs" + paths.TAG + ("_smoke" if args.smoke else ""))
    model = YOLO(paths.weights_path(letter))
    # optimizer is set explicitly: with the default "auto", ultralytics ignores lr0 and picks its own.
    # warmup_bias_lr: ultralytics starts the bias group at 0.1 during warmup, a value meant for SGD. With
    # AdamW at lr0=1e-4 that is a ~1000x spike on the biases in epochs 1-2 (seen in the v1 logs, lr/pg2).
    model.train(data=dataset_yaml_for(args), epochs=args.epochs, imgsz=config.IMG_SIZE,
                lr0=args.lr, optimizer="AdamW", warmup_bias_lr=args.warmup_bias_lr,
                batch=batch, workers=args.workers, seed=config.SEED,
                device=0, project=run_dir, name=letter, exist_ok=True, plots=False, verbose=False)
    return os.path.join(run_dir, letter)


def train_with_oom_retry(letter, args):
    """Halves the batch size on CUDA out-of-memory. Never touches image size or dataset."""
    batch = args.batch
    while True:
        try:
            return train_once(letter, args, batch), batch
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or batch == 1:
                raise
            batch //= 2
            print(f"CUDA OOM, retrying with batch {batch}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", help="n, s, m or all")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR0)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1,
                        help="0.1 = ultralytics default (what v1 used); set it equal to --lr to remove the bias spike")
    parser.add_argument("--val-every", type=int, default=1, help="validate on every Nth val frame (speed only)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1

    letters = config.FINETUNED_LETTERS if args.model == "all" else [args.model]
    out_dir = paths.result_dir("03_finetune", args.smoke)
    os.makedirs(os.path.dirname(paths.finetuned_path("n", args.smoke)), exist_ok=True)

    rows = []
    for letter in letters:
        target = paths.finetuned_path(letter, args.smoke)
        if os.path.exists(target) and not args.force:
            print("skip (done):", target)
            continue
        start = time.time()
        run_path, batch_used = train_with_oom_retry(letter, args)
        seconds = time.time() - start
        shutil.copy(os.path.join(run_path, "weights", "best.pt"), target)
        shutil.copy(os.path.join(run_path, "results.csv"), os.path.join(out_dir, f"yolov8{letter}_curve.csv"))
        rows.append({"model": f"yolov8{letter}", "epochs": args.epochs, "batch": batch_used, "lr0": args.lr,
                     "total_min": round(seconds / 60, 1), "min_per_epoch": round(seconds / 60 / args.epochs, 2)})
        print(rows[-1], flush=True)

    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(out_dir, "timing.csv"), mode="a", index=False,
                     header=not os.path.exists(os.path.join(out_dir, "timing.csv")))
        report.write_summary(out_dir, "03 finetune", ["data/images, data/labels (pseudo-labels from yolov8l)"],
                             [paths.finetuned_path(l, args.smoke) for l in letters] + ["yolov8*_curve.csv", "timing.csv"],
                             table, [f"optimizer set to AdamW (ultralytics 'auto' would ignore lr0={args.lr}); "
                                     f"seed {config.SEED}, warmup_bias_lr {args.warmup_bias_lr}, val every {args.val_every}"])
    print("saved", out_dir)


if __name__ == "__main__":
    main()

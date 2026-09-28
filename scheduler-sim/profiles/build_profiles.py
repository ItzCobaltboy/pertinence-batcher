"""Builds the two example RuntimeProfile CSVs shipped with
scheduler-sim:

- resnet_example.csv: real 2-model matrix (resnet18, resnet50, fp32,
  total latency column) read straight from
  archive/quantization_experiments/results/exp2_batch_size_sweep.csv.
- synthetic_4model.csv: a clearly-labeled SYNTHETIC 4-model matrix
  standing in for YOLOv8 n/s/m/l until real batched-inference curves
  are measured. Small models kept roughly affine, larger ones convex,
  per the shape difference documented in Journel/Week4.md.

Run once from the repo root (or scheduler-sim/); paths are relative to
the repo root and resolved from this file's location, no hardcoded
absolute paths.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE_CSV = REPO_ROOT / "archive" / "quantization_experiments" / "results" / "exp2_batch_size_sweep.csv"

BATCH_SIZES = [1, 4, 8, 16, 32]


def build_resnet_example():
    rows = {}
    with open(SOURCE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["precision"] != "fp32":
                continue
            model = r["model"]
            b = int(r["batch_size"])
            rows.setdefault(model, {})[b] = float(r["latency_ms_total"])

    out_path = HERE / "resnet_example.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + BATCH_SIZES)
        for model in ["resnet18", "resnet50"]:
            writer.writerow([model] + [rows[model][b] for b in BATCH_SIZES])
    print(f"wrote {out_path}")


def build_synthetic_4model():
    """SYNTHETIC placeholder for YOLOv8 n/s/m/l, until real batched
    curves are measured. Not real hardware data -- see the header
    comment this writes and profiles/README.md.

    Shape: yolov8n/s kept close to affine (small fixed overhead + a
    small, near-constant per-image cost); yolov8m/l made convex
    (per-image cost worsens with batch size), matching the
    affine-vs-convex split observed on the real resnet18/resnet50
    curves in Journel/Week4.md.
    """
    out_path = HERE / "synthetic_4model.csv"

    def affine(base, per_item, b):
        return round(base + per_item * b, 3)

    def convex(base, per_item, growth, b):
        # per-item cost grows mildly with b, i.e. total is superlinear
        return round(base + per_item * b * (1 + growth * (b - 1)), 3)

    models = {
        # (kind, base_ms, per_item_ms, growth)
        "yolov8n_synth": ("affine", 1.2, 0.35, 0.0),
        "yolov8s_synth": ("affine", 1.8, 0.55, 0.0),
        "yolov8m_synth": ("convex", 3.0, 1.10, 0.02),
        "yolov8l_synth": ("convex", 4.5, 1.90, 0.035),
    }

    with open(out_path, "w", newline="") as f:
        f.write(
            "# SYNTHETIC runtime matrix -- placeholder for real YOLOv8 n/s/m/l batched-inference\n"
            "# curves, which have not been measured yet (see Journel/Week5.md). Do NOT treat these\n"
            "# numbers as real hardware measurements; swap this file out once the real sweep exists.\n"
        )
        writer = csv.writer(f)
        writer.writerow(["model"] + BATCH_SIZES)
        for model, (kind, base, per_item, growth) in models.items():
            if kind == "affine":
                vals = [affine(base, per_item, b) for b in BATCH_SIZES]
            else:
                vals = [convex(base, per_item, growth, b) for b in BATCH_SIZES]
            writer.writerow([model] + vals)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build_resnet_example()
    build_synthetic_4model()

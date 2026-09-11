# eda

Analysis of `../` (this folder is nested inside model_analysis/, not a sibling)'s benchmark results — plotting only.

**Does**: reads `eager_baseline.csv` and `torch_tensorrt_benchmark.csv` (live from
`../results/`, not copied) and produces comparison plots.
**Does not**: run any benchmark itself, and does not touch `../dispatcher/`'s own EDA —
separate pipeline, separate concern.

## How to run

```
cd archive/model_analysis/eda
python main.py
```

Requires `../model_analysis/main.py` to have been run first (both CSVs must exist).

## Inputs

- `../model_analysis/results/eager_baseline.csv`
- `../model_analysis/results/torch_tensorrt_benchmark.csv`

## Outputs

- `results/accuracy_by_precision.png`, `results/latency_by_precision.png`,
  `results/size_by_precision.png` — grouped bar charts per model, per TRT precision.
- `results/accuracy_vs_latency.png` — scatter of every (model, precision) point, with
  the eager fp32 baseline overlaid per model for reference.

## Why

Model pool selection is narrated in `../../../Journel/Week0.md`; the Torch-TensorRT FP16
result in `../../../Journel/Week1.md`.

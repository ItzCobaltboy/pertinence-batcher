# eda

Analysis of `../model_analysis/`'s benchmark results — plotting only.

**Does**: reads `eager_baseline.csv` and `torch_tensorrt_benchmark.csv` (live from
`../model_analysis/results/`, not copied) and produces comparison plots.
**Does not**: run any benchmark itself, and does not touch `../dispatcher/`'s own EDA —
separate pipeline, separate concern.

## How to run

```
cd code/eda
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

Model pool selection and the Torch-TensorRT FP16 result are narrated in
`../../Journel/Week0.md`.

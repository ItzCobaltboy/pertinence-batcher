# quantization_experiments

Week2 follow-up: does *real, calibrated* PTQ beat the FP16 baseline from `../model_analysis/`?
Standalone — not wired into the dispatcher pool or NSGA-II runs.

**Does**: `exp1_ptq_calibration.py` runs proper calibrated INT8 PTQ (`modelopt
mtq.quantize()` + calibration dataloader + torch-tensorrt compile) and compares against
the uncalibrated FP16/INT8 baseline. `exp2_batch_size_sweep.py` sweeps batch sizes
1/4/8/16/32 across fp32/fp16/uncalibrated-int8 on resnet18 + resnet50.
**Does not**: feed back into `../../dispatcher/`'s model pool — results are exploratory
only, pool stays FP32 (see root README's dead-ends).

Archived 2026-09 (`../../Journel/Week3.md`) — done/exploratory work, not part of the
active pipeline. Moved here unmodified from the old `code/quantization_experiments/`.

## How to run

```
cd archive/quantization_experiments
python exp1_ptq_calibration.py
python exp2_batch_size_sweep.py
```

`exp1_resnet152_only.py` is a standalone re-run of just resnet152 (kept from the
original session, useful if only that model needs re-checking).

## Requirements

Same as root `requirements.txt`, notably `nvidia-modelopt` for calibration and
`torch-tensorrt`/`tensorrt-cu13` for compilation. GPU required; INT8 calibration needs
a real CUDA device (not tested on CPU).

## Outputs

- `results/exp1_ptq_calibration.csv` — calibrated PTQ accuracy/latency per model.
- `results/exp2_batch_size_sweep.csv` — latency per (model, precision, batch size).

## Why

Full results (5–7× speedup over FP16 on RN18/34/50, batching giving ~5× free
per-image speedup) and the `torch_tensorrt.save()` crash workaround are narrated in
`../../Journel/Week2.md`.

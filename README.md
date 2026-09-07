# pertinence-batcher

Re-implementation and batching extension of **PERTINENCE** — a runtime method for opportunistic dynamic execution of neural networks, selecting the lightest model that correctly handles each input.

**Supervisors**: Prof. Gayathri Ananthanarayanan (IIT Dharwad) · Dr. Marcello Traiola (INRIA)

---

## What this is

The original PERTINENCE paper assumes single-image inference. This project extends it to **batched mixed-complexity inference**: a dispatcher routes each image in a batch to the most efficient model that can handle it, runs sub-batches per model, and reassembles results.

**Model pool**: FP32 ResNet18 / ResNet34 / ResNet50 / ResNet152, evaluated on ImageNette.  
Latency spread: ~1.7–8.5 ms · Accuracy spread: 78–91% Top-1.

## Repo layout

```
code/
  model_analysis/            Step 0 -- model pool selection + Torch-TensorRT precision benchmark
  eda/                       Step 0b -- plots on model_analysis's benchmark CSVs
  dispatcher/                Step 1-2 -- NSGA-II search for the dispatcher's penalty matrix / weighting
  dispatcher_analysis/       Step 3 -- evaluation of the dispatcher's Pareto front (train + held-out val)
  dataset/                   ImageNette (10-class ImageNet subset, 224px)
  quantization_experiments/  standalone -- calibrated PTQ + batch-size-sweep experiments
  CIFAR_10_Implementation/   standalone -- reproduces the paper's own CIFAR-10 setup end to end
Journel/                     work session logs (narrative "why" record)
```

## Quickstart

Each pipeline is self-contained (`main.py`/scripts + `src/`) and has its own README with
run instructions, inputs, outputs, and requirements. Main dispatcher pipeline, run in order:

1. [`code/model_analysis/`](code/model_analysis/README.md) — model pool + precision benchmark
2. [`code/eda/`](code/eda/README.md) — plots on the benchmark results
3. [`code/dispatcher/`](code/dispatcher/README.md) — NSGA-II Pareto search
4. [`code/dispatcher_analysis/`](code/dispatcher_analysis/README.md) — evaluate the resulting front

Standalone tracks, independent of the above and of each other:

- [`code/quantization_experiments/`](code/quantization_experiments/README.md) — calibrated PTQ vs. FP16 baseline
- [`code/CIFAR_10_Implementation/`](code/CIFAR_10_Implementation/README.md) — paper's exact CIFAR-10 setup, reproduced

## Requirements

```
pip install -r code/requirements.txt
```

CUDA GPU required (all pipelines default to `cuda:0`, fall back to CPU only for
non-benchmark code paths). `requirements.txt` is pinned for CUDA 13.x
(`torch-tensorrt`/`tensorrt-cu13`) — built/tested on Nvidia Blackwell + Ampere (A100).
`pytorch-cifar-models` (used only by `CIFAR_10_Implementation/`) isn't pip-installable;
pulled via `torch.hub.load(...)` on first run instead (needs network access once, then
cached).

## Status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Model pool benchmarking + quantization exploration | ✅ Done |
| 1 | Dispatcher — labeling, training, EDA | ✅ Done |
| 2 | NSGA-II Pareto search over dispatcher configurations | ✅ Done (3 full runs) |
| 3 | Dispatcher evaluation — Pareto-front eval on train + held-out val | ✅ Done |
| 4 | Batching extension (route sub-batches per model, reassemble) | ⬜ Not started |
| — | Calibrated PTQ quantization experiments | ✅ Done (proven, not integrated into pool) |
| — | CIFAR-10 reproduction of the paper's exact setup | ✅ Done |

See `Journel/` for the full narrative.

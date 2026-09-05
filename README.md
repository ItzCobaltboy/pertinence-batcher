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
  model_analysis/       Step 0 -- model pool selection + Torch-TensorRT precision benchmark (no plots)
  eda/                   Step 0b -- plots on model_analysis's benchmark CSVs
  dispatcher/            Step 1-2 -- NSGA-II search for the dispatcher's penalty matrix / weighting
  dispatcher_analysis/   Step 3 -- evaluation of the dispatcher's Pareto front (train + held-out val)
  dataset/               ImageNette (10-class ImageNet subset, 224px)
Journel/                 work session logs (narrative "why" record)
```

## Quickstart

Each pipeline is self-contained (`main.py` + `src/`) and has its own README with run
instructions, inputs, and outputs. Run in this order:

1. [`code/model_analysis/`](code/model_analysis/README.md) — model pool + precision benchmark
2. [`code/eda/`](code/eda/README.md) — plots on the benchmark results
3. [`code/dispatcher/`](code/dispatcher/README.md) — NSGA-II Pareto search
4. [`code/dispatcher_analysis/`](code/dispatcher_analysis/README.md) — evaluate the resulting front

> Heads up! Requirements.txt is pinned for CUDA 13.x, needs Nvidia Blackwell series device
## Status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Model pool benchmarking + quantization exploration | ✅ Done |
| 1 | Dispatcher — labeling, training, EDA | ✅ Done |
| 2 | NSGA-II Pareto search over dispatcher configurations | ✅ Done (3 full runs) |
| 3 | Dispatcher evaluation — Pareto-front eval on train + held-out val | ✅ Done |
| 4 | Batching extension (route sub-batches per model, reassemble) | ⬜ Not started |

## Key findings so far

- Quantization (torchao weight-only, torchao dynamic, ONNX INT8 + CUDA) showed no meaningful Pareto improvement for ResNet CNN inference — dropped. Pool is FP32-only.
- Torch-TensorRT compilation gives a real ~2x latency win (kernel fusion), independent of precision; real INT8/FP8 quantization is not yet implemented (needs explicit calibration).
- The dispatcher is trained with NSGA-II to jointly search a penalty matrix and class-weighting scheme, evaluated against `alpha_sys` (fraction of images the dispatched model itself classifies correctly, per the PERTINENCE paper's Eq. 3).

See `Journel/` for the full narrative.

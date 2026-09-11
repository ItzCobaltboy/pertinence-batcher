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
dispatcher/                 Step 1-2 -- shared, generic NSGA-II search code (no dataset specifics)
dispatcher_analysis/        Step 3 -- shared, generic Pareto-front evaluation code
eda/                        ground-truth EDA -- shared, generic (per-model/oracle accuracy, class balance)
imagenette/                 ImageNette track -- config, entry points, dataset, results
cifar-10/                   CIFAR-10 track -- config, entry points, labeling script, models, dataset, results
archive/
  model_analysis/           done -- model pool selection + Torch-TensorRT precision benchmark
  quantization_experiments/ done -- calibrated PTQ + batch-size-sweep experiments
Journel/                    work session logs (narrative "why" record)
```

`dispatcher/`, `dispatcher_analysis/`, and `eda/` hold logic only, parameterized by a
`config` module (`imagenette/config.py` or `cifar-10/config.py`) passed in explicitly by
that track's entry-point scripts — see `dispatcher/README.md` for the exact mechanism.
This is one shared copy of the code across both datasets, not a per-dataset copy.

## Quickstart

Each shared folder and each track has its own README with run instructions, inputs,
outputs, and what's specific to it. Per track, run in order:

```
python imagenette/label_data.py         # optional -- existing ground-truth CSVs already work, see imagenette/README.md
python imagenette/run_eda.py
python imagenette/run_dispatcher.py
python imagenette/run_dispatcher_analysis.py
```
```
python cifar-10/label_data.py           # CIFAR-10 only -- builds the ground-truth CSVs first
python cifar-10/run_eda.py
python cifar-10/run_dispatcher.py
python cifar-10/run_dispatcher_analysis.py
```

- [`dispatcher/`](dispatcher/README.md) — NSGA-II Pareto search
- [`dispatcher_analysis/`](dispatcher_analysis/README.md) — evaluate the resulting front
- [`eda/`](eda/README.md) — ground-truth EDA (per-model/oracle accuracy, class balance)
- [`imagenette/`](imagenette/README.md) — ImageNette track specifics
- [`cifar-10/`](cifar-10/README.md) — CIFAR-10 track specifics

Archived, done work (kept for reference, not part of the active pipeline):

- [`archive/model_analysis/`](archive/model_analysis/README.md) — model pool selection +
  Torch-TensorRT precision benchmark (plus its own benchmark-comparison EDA, nested at
  `archive/model_analysis/eda/`)
- [`archive/quantization_experiments/`](archive/quantization_experiments/README.md) —
  calibrated PTQ vs. FP16 baseline

## Requirements

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

CUDA GPU required (all pipelines default to `cuda:0`, fall back to CPU only for
non-benchmark code paths). `requirements.txt` is pinned for CUDA 13.x
(`torch-tensorrt`/`tensorrt-cu13`) — built/tested on Nvidia Blackwell + Ampere (A100).
`pytorch-cifar-models` (used only by `cifar-10/`) isn't pip-installable; pulled via
`torch.hub.load(...)` on first run instead (needs network access once, then cached).

No committed virtualenv — set one up locally: `python -m venv venv` then the pip
install above.

## Status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Model pool benchmarking + quantization exploration | ✅ Done |
| 1 | Dispatcher — labeling, training, EDA | ✅ Done |
| 2 | NSGA-II Pareto search over dispatcher configurations | ✅ Implemented — no current front reflects the latest fitness/pool changes, needs a re-run |
| 3 | Dispatcher evaluation — Pareto-front eval on train + held-out val | ✅ Implemented — depends on step 2's front, so also needs a re-run |
| 4 | Batching extension (route sub-batches per model, reassemble) | ⬜ Not started |
| — | Calibrated PTQ quantization experiments | ✅ Done (proven, not integrated into pool) |
| — | CIFAR-10 track | 🔁 Uses a substitute model pool (not the paper's exact resnet8/resnet14/shufflenetv2_x0_5/vgg16_bn), official train/test split, labeling/search/eval need a re-run |

See `Journel/` for the full narrative.

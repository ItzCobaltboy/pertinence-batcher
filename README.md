# pertinence-batcher

Re-implementation and batching extension of **PERTINENCE** — a runtime method for opportunistic dynamic execution of neural networks, selecting the lightest model that correctly handles each input.

**Supervisors**: Prof. Gayathri Ananthanarayanan (IIT Dharwad) · Dr. Marcello Traiola (INRIA)

---

## What this is

The original PERTINENCE paper assumes single-image inference. This project extends it to **batched mixed-complexity inference**: a dispatcher routes each image in a batch to the most efficient model that can handle it, runs sub-batches per model, and reassembles results.

**Active track — CIFAR-100**: two thin sub-tracks (`fig9c/`, `fig9d/`) reproducing specific figures from the paper's own CIFAR-100 exploration (Fig. 9(c)/9(d), page 8). Model pool per sub-track drawn from `shufflenetv2_x0_5` / `mobilenetv2_x0_75` / `mobilenetv2_x1_4` / `repvgg_a2`.

## Repo layout

```
dispatcher/                 Step 1-2 -- shared, generic NSGA-II search code (no dataset specifics)
dispatcher_analysis/        Step 3 -- shared, generic Pareto-front evaluation code
eda/                        ground-truth EDA -- shared, generic (per-model/oracle accuracy, class balance)
cifar-100/                  CIFAR-100 track -- shared infra + two self-contained sub-tracks
  fig9c/                    Fig. 9(c) reproduction -- shufflenetv2_x0_5/mobilenetv2_x0_75/repvgg_a2
  fig9d/                    Fig. 9(d) reproduction -- shufflenetv2_x0_5/mobilenetv2_x1_4/repvgg_a2
  models/                   pretrained checkpoints + loader
archive/
  model_analysis/           done -- model pool selection + Torch-TensorRT precision benchmark
  quantization_experiments/ done -- calibrated PTQ + batch-size-sweep experiments
yolo-analysis/               YOLOv8 n/s/m/l on COCO -- class-recall benchmark, self-contained
scheduler-sim/               discrete-event simulator for the N-model/single-accelerator batch
                              scheduling problem (Journel/Week4.md's formal problem definition).
                              Plain OOP in 3 core files: sim.py (Job, Queue, Profile, Simulator and
                              the Workload/Scheduler base classes), workloads.py, schedulers.py.
                              Self-contained, see scheduler-sim/README.md
Journel/                    work session logs (narrative "why" record)
```

`dispatcher/`, `dispatcher_analysis/`, and `eda/` hold logic only, parameterized by a
`config` module (`cifar-100/fig9c/config.py` or `cifar-100/fig9d/config.py`) passed in
explicitly by that sub-track's entry-point scripts — see `dispatcher/README.md` for the
exact mechanism. This is one shared copy of the code across sub-tracks, not a per-track copy.

## Quickstart

Each shared folder and each sub-track has its own README with run instructions, inputs,
outputs, and what's specific to it. Per sub-track, run in order:

```
python cifar-100/label_data.py --track fig9c    # builds the ground-truth CSVs, see cifar-100/README.md
python cifar-100/fig9c/run_eda.py
python cifar-100/fig9c/run_dispatcher.py
python cifar-100/fig9c/run_dispatcher_analysis.py
```
```
python cifar-100/label_data.py --track fig9d
python cifar-100/fig9d/run_eda.py
python cifar-100/fig9d/run_dispatcher.py
python cifar-100/fig9d/run_dispatcher_analysis.py
```

- [`dispatcher/`](dispatcher/README.md) — NSGA-II Pareto search
- [`dispatcher_analysis/`](dispatcher_analysis/README.md) — evaluate the resulting front
- [`eda/`](eda/README.md) — ground-truth EDA (per-model/oracle accuracy, class balance)
- [`cifar-100/`](cifar-100/README.md) — CIFAR-100 track specifics (shared infra + both sub-tracks)
- [`yolo-analysis/`](yolo-analysis/README.md) — YOLOv8 n/s/m/l on COCO, class-recall benchmark
- [`scheduler-sim/`](scheduler-sim/README.md): discrete-event simulator for the batch
  scheduling problem. Subclass Workload (how jobs arrive and which queue gets them) or
  Scheduler (which queue and batch size runs next); both share the same Queue objects

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
Pretrained checkpoints for the CIFAR-100 pool aren't all pip-installable; pulled via
`torch.hub.load(...)` on first run instead (needs network access once, then cached).

No committed virtualenv — set one up locally: `python -m venv venv` then the pip
install above.

## Status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Model pool benchmarking + quantization exploration | ✅ Done |
| 1 | Dispatcher — labeling, training, EDA | ✅ Done |
| 2 | NSGA-II Pareto search over dispatcher configurations | ✅ Implemented |
| 3 | Dispatcher evaluation — Pareto-front eval on train + held-out val/test | ✅ Implemented |
| 4 | Batching extension (route sub-batches per model, reassemble) | ⬜ Not started |
| — | Calibrated PTQ quantization experiments | ✅ Done (proven, not integrated into pool) |
| — | CIFAR-100 track (`fig9c`/`fig9d`) | ✅ Both sub-tracks smoke-tested end to end; full-hyperparameter NSGA-II searches not yet run |
| 5 | YOLOv8 n/s/m/l on COCO — class-recall benchmark + correctness definition | ✅ Done |
| 6 | Scheduler discrete-event simulator (`scheduler-sim/`) | ✅ Built, tested, example sweeps run — see `scheduler-sim/README.md` |
| 7 | Scheduler validated experimentally with PERTINENCE in the loop | ⬜ Not started |

See `Journel/` for the full narrative.

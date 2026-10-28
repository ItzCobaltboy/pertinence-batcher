# cityflow-dispatcher

Can small YOLOv8 models fine-tuned on CityFlow CCTV frames replace yolov8l often enough that a
dispatcher can route frames between them? Spec: `research/spec_cityflow_dispatcher.md`.
Everything stays inside this folder on D: (venv in `.venv/`, data in `data/`, weights in `models/`).

Part A (built, see `results/REPORT_A.md`): download, dataset, fine-tune, compare.
Part B (dispatcher, evaluation, stream analysis): built and run, results in `results/05_*`, `06_*`, `07_stream`.

## Run order (from this folder)

```
python setup_env.py                        # creates .venv, torch cu128, proves sm_120 works, runs preflight
.venv\Scripts\python 01_download.py        # ~17 GB zip, resumable, extracts videos + GT
.venv\Scripts\python 02_make_dataset.py --smoke   # then without --smoke
.venv\Scripts\python 03_finetune.py --smoke       # then without --smoke (n, s, m, 10 epochs each)
.venv\Scripts\python 04_compare.py --smoke        # then without --smoke (about 4 h, val mAP is CPU-bound)
.venv\Scripts\python 05_build_dispatcher.py      # --smoke first; head sweep, selects on val
.venv\Scripts\python 06_evaluate_dispatcher.py   # test split, once
.venv\Scripts\python 07_stream_analysis.py       # S02 + S06 at 5 fps
```

Every step skips work whose output exists; `--force` redoes it (02, 03). `--smoke` writes to
`data/smoke/`, `models/finetuned/smoke/` and `results/<step>_smoke/`.

## Splits (config.py)

train S01 S03 S04, val S05, test S02, stream-only S06. Every 5th frame. Labels are pseudo-labels
from frozen yolov8l (road classes, conf 0.25); a frame is correct for a model if its box recall
vs yolov8l is >= 0.80 (IoU 0.5).

## Deviations from the spec

- The frames folder is `data/images/<split>/` (not `data/frames/`): ultralytics finds labels by
  swapping `/images/` for `/labels/` in the path.
- Fine-tuning uses AdamW with lr0 1e-4: with ultralytics' default `optimizer="auto"` the lr0 flag is ignored.

## Results

Frame-correct rate (box recall vs yolov8l >= 0.80), test split S02:

| model | original | fine-tuned |
|---|---|---|
| n | 0.199 | 0.518 |
| s | 0.557 | 0.780 |
| m | 0.721 | 0.867 |

On val (S05) n and s improve (0.12 to 0.42, 0.44 to 0.70) but m gets worse (0.68 to 0.58). Details: `results/REPORT_A.md`.

Dispatcher on the test split (head: inverse-frequency weights, underestimation penalty 8):

| policy | alpha_sys | GFLOPs per frame |
|---|---|---|
| dispatcher (incl. 3.2 extractor) | 0.851 | 74.5 |
| always n / s / m / l | 0.518 / 0.780 / 0.867 / 1.000 | 8.7 / 28.6 / 78.9 / 165.1 |
| oracle (ideal label) | 1.000 | 32.1 |

The dispatcher does not beat the static baselines: always-s gets 0.78 at 28.6 GFLOPs, always-m gets 0.867 at 78.9.
Train labels are saturated (88% n on train vs 42% on val), and the head routes by camera rather than by frame.
Stream numbers: `results/07_stream/summary.md`.

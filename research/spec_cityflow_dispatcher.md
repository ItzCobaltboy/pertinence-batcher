# Spec: `cityflow-dispatcher/` (fresh build, runs on my Windows laptop)

For Claude Code running on my laptop (Windows, NVIDIA RTX 5070 Ti, native Python). You write the code,
set up the environment, run each step and record the results. Read this whole file before writing anything.

## 0. Goal

Find out whether fine-tuning the small YOLOv8 models on CityFlow CCTV frames makes them good enough to
replace the big one often, and if so, whether a dispatcher can route frames between them in a balanced way
and what that routing looks like on video streams.

Chain: download CityFlow -> build a train/val/test frame set -> fine-tune yolov8 n, s, m -> compare with the
originals -> **stop and report (Part A)**. Only after I say go: build the dispatcher -> evaluate it -> analyse
it on video streams (Part B).

Why this exists: the PERTINENCE dispatcher heads trained on COCO collapsed on CityFlow (about 90 to 94 percent
of frames sent to yolov8l). Cause is domain shift (COCO photos vs fixed traffic cameras). This is a clean
restart of that work. The old code is hard to follow, which is why it is rewritten from scratch.

## 1. Rules

- **Hard stop (my standing rule):** never touch the C drive system areas, Program Files or the Windows folder.
  Everything lives inside this project folder on D:. The venv is `cityflow-dispatcher/.venv`. No system-wide
  installs, no PATH edits. If something needs a system-wide install (Python, CUDA toolkit, driver), stop and
  tell me exactly what to do by hand.
- Do not edit `Journel/` or `CLAUDE.md`. I do the journal. Record results under
  `cityflow-dispatcher/results/` instead (see section 7).
- `archive/` is old code. Read the few files listed in section 8 for reference. Do not import from it and do
  not copy it wholesale. Rewrite cleanly.
- **No fake testing.** No CPU stand-ins, no mocked models, no random tensors posing as results. Every check
  runs the real models on the GPU. Each entry point has `--smoke` (one camera, about 200 frames) that runs the
  real pipeline small; run it before every full run.
- **Flag, do not guess.** If a fact is unknown (dataset layout, class mapping, a number that looks off), write
  it in the step's summary under "Flags" and stop to ask me when it blocks the next step.

### 1b. Code style

Plain readable Python that I can follow without puzzling over syntax. One idea per function, short functions,
descriptive names. Prefer a simple loop to a clever one-liner. Comprehensions only when trivial. No decorators,
metaclasses, inheritance trees, `**kwargs` pass-through or framework-style abstractions. Comments explain why,
not what. Each file starts with 2 to 4 comment lines saying what it does. Not verbose either: no essay
docstrings, no layers that only forward calls. Scripts print progress and say where outputs landed.

## 2. Assumptions I made (change them if wrong, they are all flags or constants)

1. **Labels for fine-tuning are pseudo-labels from frozen yolov8l** (`--labels pseudo`, default). Reasons:
   the dispatcher's definition of "correct" is already box recall against yolov8l; CityFlow's official boxes,
   as far as I remember, only cover vehicles that appear in several cameras, so many real vehicles are
   unlabelled and would teach the small models to suppress them. `--labels gt` switches to official boxes
   (vehicles only, you must ask me how to map classes first). Because yolov8l is the teacher, **only n, s, m
   are fine-tuned**; l stays frozen as teacher and as the correctness reference.
2. As a sanity check against human labels, also compute recall on official GT vehicle boxes for the test
   split, ignoring extra detections (GT is incomplete, so precision against it is meaningless).
3. Splits are by scenario, never mixed: train S01, S03, S04; val S05; test S02. Optional stream-only
   scenario S06 (no GT needed). All in `config.py`.
4. Road classes (COCO ids): person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7. Keep the 80-class head,
   labels contain only these. Detection conf 0.25, NMS IoU as ultralytics default, imgsz 640.
5. A frame is **correct** for a model if its box recall vs yolov8l (same class, IoU >= 0.5, greedy by
   confidence) is >= 0.80. Frames where yolov8l finds nothing count as correct for every model.
   **Ideal label** of a frame = cheapest model that is correct (n < s < m < l).
6. Frame sampling: every 5th frame per camera, for all splits. Set in `config.py`.
7. "Balanced" dispatcher (Part B) = on val, no model receives more than 70 percent of frames and each model
   at least 5 percent, while keeping alpha_sys (fraction of frames served correctly) as high as possible.
   If the ideal labels are themselves skewed beyond that, report it and stop; thresholds are then the lever.

## 3. Layout

```
cityflow-dispatcher/
  README.md                 how to run each step in order, plus the final results table
  setup_env.py              creates .venv, picks the torch build, installs, runs preflight
  requirements.txt          what you chose; requirements.lock.txt = pip freeze of the working env
  config.py                 every constant in one place (paths, scenarios, thresholds, classes, stride)
  common/
    paths.py                where things live
    videos.py               open a camera video, iterate sampled frames
    boxes.py                IoU, greedy matching, box recall
    detect.py               run a YOLO model on frames, return road-class boxes
    labels.py               correct/ideal label logic
  00_preflight.py
  01_download.py
  02_make_dataset.py
  03_finetune.py
  04_compare.py
  05_build_dispatcher.py    (Part B)
  06_evaluate_dispatcher.py (Part B)
  07_stream_analysis.py     (Part B)
  data/                     gitignored: videos, frames, labels, embeddings
  models/                   gitignored: weights; fine-tuned .pt files
  results/                  small CSVs, plots and summaries stay tracked
```

Task-based entry points, shared logic in `common/`. A script does one task, reads its inputs from disk
(never from another script's memory) and writes its outputs to disk, so any step can be rerun alone. Steps are
resumable: skip work whose output already exists, `--force` redoes it. Add gitignore lines for `data/`,
`models/`, `.venv/`.

## 4. Part A

### 4.1 setup_env.py and 00_preflight.py
- Find a Python 3.10 to 3.12 (`py -0` on Windows). None found: stop and tell me.
- Create `.venv` inside the folder. Read the driver and GPU from `nvidia-smi`. The 5070 Ti is Blackwell
  (compute capability 12.0), which needs a torch build for CUDA 12.8 or newer (index
  `https://download.pytorch.org/whl/cu128` or newer, whichever the driver supports). Install it, then **prove
  it works** with a real CUDA matmul and by checking `torch.cuda.get_arch_list()` contains `sm_120`. If not,
  stop and report; do not fall back to CPU.
- Install ultralytics (latest), opencv-python, numpy, pandas, matplotlib, pyyaml, gdown, tqdm.
- Preflight prints and saves `results/00_preflight/preflight.json`: GPU, VRAM, driver, torch, CUDA, ultralytics,
  free disk on D:, one forward of each yolov8 model (n, s, m, l; weights auto-download into `models/`).

### 4.2 01_download.py
- Download CityFlowV2 from the Drive link I used before. The file id and folder layout are in
  `archive/cityflow-routing/download.py` and `dataset.py` (reference only). Resume if partial.
- Extract inside `data/cityflow/`. Check free space first (I have plenty, but print the number).
- Print and save per scenario and camera: video path, fps, resolution, frame count, whether a GT file exists.
  Verify every video decodes. Codec trouble: flag it.
- If Drive blocks the download (quota), stop and tell me; do not hunt for mirrors.

### 4.3 02_make_dataset.py
- For each split's scenarios and cameras, take every 5th frame, save as JPEG (quality 95) to
  `data/frames/<split>/<scenario>_<camera>_<frame>.jpg`.
- Run frozen yolov8l (conf 0.25, road classes) on every frame, write YOLO-format label files, plus
  `data/manifest.csv` (split, scenario, camera, frame_idx, time_sec, path, n_boxes_l).
- Write the ultralytics dataset yaml (80 names, train/val/test paths).
- If GT files exist: write `data/gt_test_boxes.csv` for the test split and save overlays of 6 test frames
  (GT boxes drawn on the image) to `results/02_dataset/gt_check/` so I can see how complete GT really is.
- Summary: frames and boxes per split and camera, class histogram of pseudo-labels, empty-frame fraction.
- Cameras are never shared between splits (they cannot be, scenarios are disjoint). Say so in the summary.

### 4.4 03_finetune.py (`--model n|s|m|all`, default all)
- Start from stock COCO weights, train on the train split, validate on val each epoch.
- Defaults: 10 epochs, imgsz 640, lr0 1e-4 with the ultralytics defaults otherwise, fixed seed, batch size
  tried at 16 and halved on CUDA OOM (laptop VRAM, check preflight). Windows dataloader workers: start at 2,
  use 0 if it hangs. Epochs, lr, batch are CLI flags.
- Save best weights to `models/finetuned/yolov8{n,s,m}_cityflow.pt`, the training curve CSV and the time per
  epoch to `results/03_finetune/`.

### 4.5 04_compare.py
For each of n, s, m: original vs fine-tuned, on the **test split** (and val, for reference):
1. box recall vs yolov8l, box precision vs yolov8l;
2. mAP50 and mAP50-95 against the pseudo-labels (ultralytics val);
3. **frame-correct rate** (fraction of frames with recall >= 0.80), the number the dispatcher depends on;
4. recall on official GT vehicle boxes (assumption 2), if GT exists;
5. per-camera table of 1 and 3, so I can see whether gains are uniform or from a few cameras.

Then the number that decides everything: the **ideal-label split** (fraction of test frames whose cheapest
correct model is n, s, m, l) for the original pool vs the fine-tuned pool. Save both CSVs and a bar chart.

Report plainly if fine-tuning did not help, and by how much it did if it did. Do not massage thresholds to make
it look good; if you think a threshold is the problem, say so and show the numbers.

### 4.6 Stop and report (`results/REPORT_A.md`)
Table original vs fine-tuned (metrics above), the ideal-label split for both pools, time taken per step, disk
used, `requirements.lock.txt` versions, seeds, and a Flags section (GT completeness, anything odd). Then
**stop. Do not start Part B.** I decide from this report.

## 5. Part B (write and run only after I say go)

### 5.1 05_build_dispatcher.py
- Extractor: frozen original yolov8n backbone, 448-d feature embedding per frame, the same definition as the
  old `archive/yolo-dispatcher` extractor (read it for the exact pooling and preprocessing; keep it identical
  so numbers compare with the earlier routing). Cache embeddings per camera in `data/embeddings/`.
- Per-frame label = ideal label under the fine-tuned pool.
- Head: one linear layer 448 -> 4, class-weighted cross-entropy plus an extra penalty for underestimating
  (sending a frame to a model smaller than the ideal one). Sweep the underestimation penalty (1, 2, 4, 8, 16)
  and the class weighting (none, inverse frequency, square root of inverse frequency). No NSGA-II unless I ask.
- Select on **val** using assumption 7 (balanced). Save every config's val metrics to a table, the chosen one
  to `models/dispatcher/`, and a plot of alpha_sys vs average cost (GFLOPs per frame at 640 including the
  extractor, from `model.info()`), with the balanced points marked.

### 5.2 06_evaluate_dispatcher.py
On the **test split**, once, with the chosen head: alpha_sys, average GFLOPs and saving vs always-l, routing
accuracy, underestimate and overestimate rates, confusion matrix, routing split vs ideal split, baselines
(always n, s, m, l and oracle), per-camera numbers.

### 5.3 07_stream_analysis.py
Run the test scenario's videos (and S06 if present) in time order at 5 frames per second through the
dispatcher. Per camera and per scenario:
- routing distribution, overall and as a rolling one-minute share plot;
- **stickiness**: run lengths of consecutive identical decisions (mean, median, p90, histogram, in frames and
  seconds), switches per minute, transition matrix, compared against the same numbers for the ideal labels;
- what a scheduler would see: per-model arrival rate per second and burstiness (index of dispersion over
  1 s windows);
- effect of smoothing: majority vote over windows of 3, 5, 10 frames, with the change in switch rate and in
  alpha_sys.
Save the per-frame decision traces (camera, time_sec, frame_idx, predicted model, ideal model, probabilities)
to `results/07_stream/traces/*.csv.gz`; these feed the scheduler later.

## 6. Time and failure handling
- Print an estimate before each long step (frames x per-frame time measured in the smoke run).
- Laptop GPU thermals: if per-frame times drift up a lot during a long run, note it in the summary.
- Out of memory: reduce batch size, never reduce image size or dataset silently.
- Anything that changes a number I will quote (a threshold, a class list, a split) goes into the summary's
  Flags section with the before and after.

## 7. Results format
`results/<step>/summary.md`: what ran (command line), inputs, outputs, the key numbers in a small table, Flags.
CSVs and PNGs next to it. `README.md` at the top has the exact command for each step in order.

## 8. Reference-only files in `archive/`
- `archive/cityflow-routing/download.py`, `dataset.py`: Drive id, folder layout, scenario/camera naming.
- `archive/cityflow-dispatcher/matching.py`, `label.py`: the box-recall matching and the 0.80 label rule.
- `archive/yolo-dispatcher/`: extractor (448-d), head shape, model cost numbers.
- `archive/dispatcher_analysis/`: the underestimation-penalty loss idea.
Known from earlier work: COCO-trained heads sent S03 93.5 percent and S04 90.2 percent of frames to yolov8l.

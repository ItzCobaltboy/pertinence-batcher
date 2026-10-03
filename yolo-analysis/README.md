# yolo-analysis

Multi-label class-recall benchmark for a YOLOv8 detection model pool on full COCO, both
val2017 and train2017. Prep work for redefining "accuracy" in the PERTINENCE dispatcher once
the model pool moves from CIFAR-100 single-label classifiers to object detectors -- see repo
root `CLAUDE.md` and `Journel/Week4.md`. **Data-gathering only**: this produces one CSV per
split, no dispatcher/labeling code lives here.

train2017 runs the identical pipeline as val2017 -- included so the dispatcher's eventual
labeling step has train-side ground truth to train the routing head on, not just the held-out
val numbers this was originally built to decide a correctness definition from. **train2017
here is a stratified 20,000-image subset, not the full 118,287-image split** -- ran out of
disk budget on the target server for the full ~18GB download, see `build_train_subset.py` and
`Journel/Week4.md`'s `[PIVOT]` entry.

Self-contained -- meant to be copied to a server (own `requirements.txt`, no imports from the
rest of the repo) and run there with a single command.

## Batch timing sweep

`batch_sweep/` is a separate tool in this folder: it times YOLOv8 n/s/m/l at several batch sizes
(eager and torch_tensorrt FP32/FP16) and writes measured T_i(b) profiles for `scheduler-sim/`.
See `batch_sweep/README.md`.

## Why multi-label, not top-1

The existing CIFAR-100 pipeline (`cifar-100/`) uses single-label top-1 correctness: each
image has exactly one ground-truth class, a model is either right or wrong. COCO images have
multiple ground-truth object classes per image, so that definition doesn't apply as-is. This
first pass drops bounding boxes and scores it as multi-label classification: did each model
correctly identify the *set* of classes present in the image, not where they are.

## Layout

```
yolo-analysis/
  config.py             every path + hyperparameter (splits, model pool, batch size,
                         thresholds) -- all per-split paths are functions of `split`
  dataset_setup.py      downloads + unpacks val2017's full COCO images + instances_<split>.json
                         for both splits (the annotations zip covers both)
  build_train_subset.py samples a stratified 20,000-image train2017 subset (proportional to
                         category frequency, not the full 118,287-image split -- disk budget)
                         and downloads just those images individually, in parallel
  coco_gt.py             parses ground-truth per-image class sets straight from the json,
                         restricted to images actually present on disk for that split
  inference.py          batched per-model inference, caches raw detections (boxes + classes
                         + confidences) to results/<split>/raw_predictions/<model>.json
  score.py              collapses cached raw predictions to a deduplicated class set per
                         image, computes recall + exact_match against ground truth
  run_benchmark.py      SINGLE ENTRY POINT -- runs the full pipeline once per split in
                         config.SPLITS (val2017, train2017), writes each split's final CSV
  plot_results.py       per split: 4x2 grid (recall histogram + exact_match True/False
                         counts per model), per-image exact_match pivot, and the 16-way
                         True/False permutation counts across the model pool -- run after
                         run_benchmark.py
  dataset/
    val2017/               raw COCO val2017 images, full split (5000)
    train2017/              raw COCO train2017 images, stratified subset only (20,000)
    train_subset_image_ids.txt   exact sampled image ids, one per line -- reproducible subset
    annotations/            instances_val2017.json + instances_train2017.json (full split's
                             annotations, used to build the subset and as ground truth)
  models/                YOLOv8 checkpoints (auto-downloaded by ultralytics on first load)
  results/
    val2017/
      raw_predictions/            <model>.json raw detection cache, one per model
      coco_class_recall_benchmark.csv       final flat output, one row per (model, image)
      per_image_exact_match.csv             one row per image, exact_match per model
      exact_match_permutation_counts.csv    16-way True/False combo counts across models
      recall_exact_match_distributions.png  4x2 plot grid
    train2017/             same five outputs, train-split ground truth
```

## How to run

```
pip install -r requirements.txt
python run_benchmark.py
python plot_results.py   # after both splits' CSVs exist
```

Downloads COCO val2017 (full split, ~1GB images) and train2017 (stratified 20,000-image
subset, ~3.1GB) on first run if not already present under `dataset/` (the shared annotations
zip, ~250MB for the instances jsons this pipeline uses, downloaded once). Runs YOLOv8n/s/m/l
over every image in each split/subset, batched (`config.INFERENCE_BATCH_SIZE`, default 512)
rather than looped image-by-image. Each (split, model) pair's raw detections are cached to
`results/<split>/raw_predictions/` before scoring, so a later change to the correctness
definition (exact-match, confidence-thresholded, IoU-aware, whatever) can be re-scored
straight from the cache without re-running inference on all 4 models x 2 splits again -- just
edit/re-run `score.py`'s logic, skip `run_benchmark.py`'s inference step (it already no-ops
when a model's cache file exists).

train2017's subset is built by `build_train_subset.py` (called automatically from
`run_benchmark.py`, or run standalone: `python build_train_subset.py`): samples image ids
proportional to each of COCO's 80 categories' frequency in the full train2017 split (not a
blind uniform sample over images, which would badly underrepresent rare categories), writes
the exact sampled ids to `dataset/train_subset_image_ids.txt` for reproducibility, then
downloads only those 20,000 images individually and in parallel (32 workers -- a sequential
loop of 20,000 individual HTTP requests is latency-bound and takes hours, parallelized it
takes minutes) rather than the full `train2017.zip`.

## Output schema

`results/<split>/coco_class_recall_benchmark.csv`, one row per (model, image_id):

| column | meaning |
|---|---|
| `image_id` | COCO image id |
| `model` | checkpoint name, e.g. `yolov8n.pt` |
| `num_gt_classes` | \|ground-truth class set\| |
| `num_pred_classes` | \|deduplicated predicted class set\| |
| `recall` | \|predicted ∩ true\| / \|true\| (1.0 if an image has zero gt classes) |
| `exact_match` | predicted_set == true_set (boolean) |
| `gt_classes` | pipe-separated COCO category **names** (not ids), sorted |
| `pred_classes` | pipe-separated COCO category **names** (not ids), sorted |

Class names (not numeric ids) were chosen for `gt_classes`/`pred_classes` so per-image
disagreement is readable without cross-referencing a category-id table downstream.

`plot_results.py` derives two more CSVs per split from the above: `per_image_exact_match.csv`
(one row per image, one boolean column per model) and
`exact_match_permutation_counts.csv` (one row per True/False combination across the model
pool -- 16 rows for 4 models -- with the count of images landing in that exact combination,
sorted descending; combinations with zero images are still listed with count 0).

## Explicitly out of scope for this pass

- No bounding-box / IoU scoring -- raw boxes are cached for later, not used in `recall`/
  `exact_match` here.
- No dispatcher/labeling code. The correctness definition (plain recall vs. exact-match vs.
  some threshold) gets decided from this data's actual per-model, per-image spread, not
  before seeing it.

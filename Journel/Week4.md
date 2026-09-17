# Week 4

## Meeting notes & tasks

**Tasks**:
1. Benchmark the YOLO model family (nano, small, and up) on COCO for object detection —
   cost, latency, and accuracy per model.
2. Benchmark the same models across a range of batch sizes — latency/throughput tradeoff,
   power draw if feasible.
3. Train a PERTINENCE dispatcher stack for the YOLO pool itself (not reusing the CIFAR-100
   dispatcher — a new stack over YOLO models).
4. If feasible, source open video streams and run the pipeline against real stream data,
   not just static COCO images.
5. Design goals not yet hardcoded — overarching targets are minimize cost, minimize
   latency, maximize throughput, maximize accuracy. Exact formalization (hard SLA vs. soft
   target, objective weighting, etc.) still open.

---

## [MEETING] Redo well received — pitched a new direction, greenlit

Meeting over video call. The redone presentation (Fig. 9(c)/9(d) reproduction, structured
walkthrough with the terminology slide, EDA, results, confusion matrices) landed well —
positive feedback on the work itself, no further rework requested on the CIFAR-100 track.

Asked for a new direction for the project. Pitched one of my own rather than waiting on an
assigned topic: **a scheduler for PERTINENCE**. Advisor liked the pitch and asked me to work
in that direction — this is now the project's next phase, not a side-quest.

## [DECISION] The pitch — from routing to scheduling

PERTINENCE as it stands solves *which* model an input should go to, evaluated per-image in
isolation — it says nothing about *when* that work actually executes once you have multiple
concurrent streams competing for the same GPU. Framed the new problem as:

Multiple video streams arrive concurrently. The dispatcher routes each incoming frame to a
queue for its assigned model, same as before — but the models are heterogeneous in cost:
bigger/more-accurate models are slower and more expensive to run, cheaper models are fast
but less accurate. Given a latency target the system has to meet, design an optimal
scheduler — static or dynamic — that decides how and when to actually run each model's
queue, to minimize per-frame turnaround time and runtime cost while meeting the latency
target, maximizing throughput and accuracy along the way. Batching is explicitly in scope:
grouping same-model requests before inference should recover a real throughput win (this
lines up with the batch-size sweep finding from `Journel/Week2.md`'s Exp2 — ~5x free
per-image speedup from bs=1 to bs=4 on resnet18), but the right batch size trades off
against queue backlog and per-frame latency, so it isn't a free knob.

This turns the problem from a pure multi-objective search (NSGA-II over a static routing
policy) into a scheduling/queueing problem layered on top of routing — closer to the kind of
policy Roveri's EEN paper (`113111259774 Roveri.pdf`, already in the project's reference set)
analyzes for early-exit networks, but for a model-selection dispatcher instead of a
single-network exit ladder. Worth revisiting that paper's reactive/rate-based/model-based
policy comparison once the scheduler design gets concrete — it's solving a structurally
adjacent problem (mean response time, loss ratio, and accuracy tradeoffs under queueing).

## [SETUP] Scope for the first pass — YOLO pool on COCO, benchmarked before any scheduler code

Advisor's guidance was pointers, not a fixed spec — the actual work plan below is mine.

**Model pool**: switching from image classification (CIFAR-10/100) to object detection —
the YOLO family (nano, small, and larger variants) on COCO. Object detection is a more
realistic stand-in for a real video-stream workload than single-label classification, and
YOLO's own size variants are a natural cost/accuracy ladder, similar in spirit to the
ResNet/ShuffleNet/MobileNet pools used so far.

**Plan**:
1. Benchmark each YOLO variant standalone on COCO — accuracy (mAP), latency, and
   compute cost (FLOPs), the same three axes every prior pool in this project has been
   measured on.
2. Sweep batch sizes per model — latency/throughput curve, and power draw if the
   measurement setup allows it (the A100/local GPU tooling from the quantization
   experiments in `Journel/Week1-2.md` should mostly carry over).
3. Train a PERTINENCE dispatcher over the YOLO pool — new stack, not a reuse of the
   CIFAR-100 dispatcher, since the task (detection) and the model family are both new.
4. Stretch goal: source open video streams and run the pipeline against real streamed
   frames instead of only static COCO images, to get closer to the actual target scenario
   (concurrent live streams, not a fixed dataset).

Formal objective definitions (hard latency SLA vs. a soft target to minimize, how
throughput/cost/accuracy get weighted against each other) are explicitly not settled yet —
noted as open in this week's Context above, to be hardcoded once the YOLO benchmarking
numbers exist to design against.

---

## [DECISION] Formal problem definition — the scheduling problem, stated properly

**Given.** A model pool where each model M_j has latency L_j, cost C_j, and an associated
queue Q_j; a batch-size set B available to each queue. The PERTINENCE dispatcher routes
each arriving image to a queue exactly as in the base system — routing itself is untouched
here, jobs arrive and are allocated to their respective model's queue by the existing
dispatcher.

**Definitions.**
- *Accuracy*: the fraction of images correctly classified by the model they were routed
  to. Unchanged from PERTINENCE's own alpha_sys — the scheduler changes when a routed job
  runs, not where it's routed.
- *I*: total number of images processed.
- *B_n*: total number of batches actually run (model invocations) — one call of any size
  counts as one, regardless of how many images it holds.
- *B_o*: the batch count under the no-batching baseline, i.e. batch size fixed at 1
  everywhere, so B_o = I by construction (a constant, not a variable — this is the thing
  B_n is trying to beat, not something B_n equals in general).
- *T_i*: turnaround time of image i, T_i = T_output(i) - T_arrival(i).
- *T-bar*: average turnaround time across all processed images, T-bar = (1/I) * sum_{i=1}^{I} T_i.

**Objectives.**
- Maximize accuracy.
- Minimize B_n / B_o — the batch-compression ratio: how much batching reduced actual
  model invocations relative to the unbatched baseline. 1 = no benefit from batching,
  smaller = more aggressive batching.
- Minimize T-bar, the average turnaround time across all processed images.

This is the formal spec behind the scheduler pitch from this week's meeting — the
multi-objective shape mirrors the existing NSGA-II routing search (maximize accuracy,
minimize cost), but adds the batching-efficiency and turnaround-time axes that only exist
once concurrent streams and queueing are in the picture.

---

## [SETUP] Detection breaks the old accuracy definition — running the pool first before redefining it

Before any dispatcher code for the YOLO pool, hit a real blocker: the old accuracy definition
(`model_j(x) is correct`, a single bool per image) was built for single-label classification and
doesn't survive contact with object detection. One image, multiple ground-truth objects, a model
can be right about some and wrong about others in the same frame — there's no longer a clean
"correct" without picking a convention.

**Compromise on scope**: dropping bounding boxes entirely, at least for this first pass. Keeping
COCO's multi-object nature (unlike single-label ImageNet-style classification) but scoring it as
multi-label classification instead of detection — did the model identify the right *set of
classes* present in the image, not where they are. Simpler to define and compute than mAP/IoU
matching, and defers the detection-specific machinery until it's clear it's actually needed.

**Open question flagged before committing**: multi-label recall (`|predicted ∩ true| /
|true|`) might be too smooth a metric to give PERTINENCE anything to route on. The whole
mechanism depends on models genuinely disagreeing on which images they can handle — cheap
vs. expensive models need to actually clash, not just cluster within a few points of each
other. A soft per-class recall score, averaged over an image with several objects, could
easily wash out the kind of per-image variance that made the CIFAR-100 pool's harder/ambiguous
images route differently across the pool. Rather than guess, decided to get the real
distribution first and look at it before picking a correctness definition (exact-match set
comparison vs. thresholded recall vs. plain recall as a continuous score).

**Plan**: run YOLOv8n/s/m/l over the *full* COCO val2017 set (5000 images, no subsampling —
plenty of compute headroom for this), batched per model to actually use the available VRAM
instead of looping image-by-image. Per model, per image: predicted class set, ground-truth
class set (from `instances_val2017.json`), recall, and exact-match bool — plus keep the raw
boxes/confidences cached even though unused for this metric, so the scoring definition can be
revisited later without re-running inference. Output is one flat CSV across all
(model, image) pairs, same shape as the CIFAR-100 ground-truth CSVs' `<model>_correct` columns
but detection-model-shaped.

Once this lands, the actual correctness/labeling definition for the YOLO pool gets decided
from the real per-model, per-image spread, not from a guess.

## [RESULT] YOLO/COCO benchmark built, ready to run on the A100

Built the full benchmark described above under a new top-level `yolo-analysis/` folder,
following the same layout convention as `cifar-100/` (own `dataset/`, `results/`, config
module, entry-point scripts at the root) — but fully self-contained (own
`requirements.txt`, no imports from the rest of the repo) since this folder gets copied to
the server to actually run.

Single entry point, `run_benchmark.py`: downloads COCO val2017 + `instances_val2017.json`
if not already present (`dataset_setup.py`), loads per-image ground-truth class sets
straight from the annotations json (`coco_gt.py`, no pycocotools dependency), runs each of
YOLOv8n/s/m/l batched over all 5000 images via `ultralytics`' own `model.predict(...,
batch=N, stream=True)` (`inference.py`, default batch 64, config-driven), caches every
model's raw detections (boxes + classes + confidences, not just the deduplicated class set)
to `results/raw_predictions/<model>.json` so the correctness definition can be revisited
later without re-running inference, then scores each model's cache against ground truth as
multi-label classification (`score.py`) and writes one flat CSV
(`results/coco_class_recall_benchmark.csv`, columns: `image_id, model, num_gt_classes,
num_pred_classes, recall, exact_match, gt_classes, pred_classes`) across all (model, image)
pairs. `gt_classes`/`pred_classes` are pipe-separated category *names*, not ids — easier to
eyeball per-image model disagreement downstream.

Deliberately not run yet — this is prep work only, to be executed on the university A100.
Verified the code by syntax-checking every module (`py_compile`), nothing more; no
downloads, no GPU, no dataset touched locally. Next step once it's actually run there: look
at the real recall/exact-match spread across the four models before deciding which
correctness definition feeds the YOLO pool's dispatcher labeling — see the open question
above.

## [RESULT] val2017 benchmark actually run — added agreement/disagreement analysis, extended to train2017

The val2017 pass above has since actually been run (all 5000 images, all four YOLOv8
variants), producing real `results/coco_class_recall_benchmark.csv`. Built `plot_results.py`
on top of it: a 4x2 plot grid (recall histogram + exact_match True/False bar counts, one row
per model) plus two derived CSVs — `per_image_exact_match.csv` (one row per image, one
boolean exact_match column per model, pivoted straight from the long-format benchmark CSV)
and `exact_match_permutation_counts.csv` (every True/False combination of exact_match across
the four models — 16 rows for a 4-model pool, count of images per exact combination,
zero-count combinations included, sorted descending). This is the actual cross-model
agreement structure the open question above was waiting on — whether the four variants
genuinely clash on different images (good, gives PERTINENCE something to route on) or mostly
move together (recall would be too smooth a signal).

Also extended the whole pipeline to run on **train2017** as well, not just val2017 —
needed since the eventual YOLO-pool dispatcher will need train-side ground truth to train its
routing head on, same as the CIFAR-100 track's `TRAIN_GROUND_TRUTH_CSV`. Refactored
`config.py` so every split-dependent path (`images_dir`, `annotations_json`,
`raw_predictions_dir`, `output_csv`, `results_dir`) is a function of `split` instead of a
fixed constant, with `config.SPLITS = ["val2017", "train2017"]` driving both
`run_benchmark.py` and `plot_results.py` in a loop — one shared pipeline over both splits
rather than a duplicated train-specific copy. `dataset_setup.py` downloads each split's image
zip separately but the annotations zip (which contains both `instances_val2017.json` and
`instances_train2017.json`) only once. Migrated the already-existing val2017 results
(`coco_class_recall_benchmark.csv`, the raw-prediction JSON caches, the derived CSVs/plot)
from the old flat `results/` layout into `results/val2017/` to match the new per-split
structure — untracked files, plain move, nothing lost. train2017 itself (118,287 images) not
run yet; same "prep work / A100" status as the original val2017 pass.

## [PIVOT] Full train2017 doesn't fit the target server's disk budget — switched to a stratified subset

Ran out of storage on the server attempting the full train2017 image download (118,287
images, ~18GB). Rather than shrink the whole plan, switched `config.SPLITS`'s train2017 entry
from "the full official split" to "a stratified subset of it," capped at 20,000 images
(~3.1GB at COCO's ~156KB/image train2017 average) — comfortably inside a 5GB pull budget on a
storage-constrained target server, with margin left for the shared annotations file (~250MB)
already needed for both splits.

**Sampling**: proportional-to-frequency per category, not uniform-random over images —
COCO's 80 categories are heavily skewed (e.g. "person" appears in tens of thousands of
train2017 images, "toaster"/"hair drier" in a few dozen), so a blind uniform sample would
likely miss or badly underrepresent the rare classes entirely. Built
`yolo-analysis/build_train_subset.py`: reads `instances_train2017.json` only (already on disk
from the shared annotations download, no train images touched yet), computes each category's
image-set, and processes categories rarest-first so a rare category's quota is filled before
a more common, overlapping category's larger quota can crowd out the images it needs (a
30-min-of-thought bug I caught before it shipped: process commonest-first and multi-label
overlap silently starves the categories the stratification was supposed to protect). Each
category gets `max(1, round(frequency * target_size))` images; quotas usually undershoot
20,000 slightly due to overlap and independent rounding, topped up with a random fill from
whatever's left. The exact sampled image_id list is written to
`yolo-analysis/dataset/train_subset_image_ids.txt` (one id per line, sorted) so the subset is
reproducible and documented rather than an ephemeral in-memory draw redone differently on
every run — `build_train_subset.py` reuses that file on a re-run instead of resampling.

Images are then fetched individually from
`http://images.cocodataset.org/train2017/<image_id:012d>.jpg` rather than the full
`train2017.zip`, straight into `dataset/train2017/` — same directory shape the rest of the
pipeline already expects for a split, so `run_benchmark.py`/`coco_gt.py` needed only a small
change: `coco_gt.load_ground_truth(split)` now filters ground truth down to images actually
present on disk (`os.listdir(images_dir)`) rather than every image_id in the full annotations
json — without that filter, `score.py` would have treated every train2017 image outside the
subset as "no model was ever run on this, count it as zero predictions" instead of correctly
excluding it from scoring entirely. `run_benchmark.py`/`dataset_setup.py` route val2017
through the existing full-zip path and train2017 through `build_train_subset.py` based on
whether a split has an entry in `config.COCO_IMAGES_URL`.

Downloaded the annotations zip and the full 20,000-image subset on this machine directly
(plenty of local disk here, 3.1GB is nothing) — 3.5GB total pulled this session (3.1GB
images + 459MB annotations, the latter bundling captions/keypoints jsons alongside the
instances jsons actually used), comfortably under the 5GB budget. First attempt at the
downloader did this sequentially (one `urllib.request.urlretrieve()` call per image) and was
far too slow -- ~11 images in several minutes, which would've meant tens of hours for 20,000
individual per-image HTTP requests, since the bottleneck is round-trip latency, not
bandwidth. Rewrote `_download_one`/`ensure_train_subset` to fan out over a
`ThreadPoolExecutor` (32 workers) instead -- finished the full subset in a few minutes once
parallelized. Verified after the fact: 20,000/20,000 files present, no zero-byte downloads, a
199-image size sample averaging ~164KB (matches COCO's typical JPEG size), 0 reported
download failures.

Per this week's guidance, **not** running the actual YOLOv8n/s/m/l inference pass over the
subset yet, same "prep work, benchmark happens on the target server" split as val2017
originally had. Also bumped `config.INFERENCE_BATCH_SIZE` from 256 to 512 for the eventual
run, matching the batch size settled on for this stage of the pipeline.

---

## [DECISION] Correctness definition for the YOLO pool — recall >= 0.80, not exact-match

Ran the full benchmark on the A100: val2017 (5000 images) and the 20,000-image train2017
subset, all four YOLOv8 variants. With real numbers in hand, picked the actual correctness
definition the open question from earlier this week was waiting on.

**Exact-match** (`pred_classes == gt_classes`) turned out too harsh for training data.
Per-model exact-match rates climb monotonically with size as expected (nano 38.7% → large
51.6% on val2017, confirming the pipeline itself is sound), and the cross-model agreement
structure from `plot_results.py`'s permutation-count analysis shows real per-image
disagreement — 35.4% of val2017 and 40.8% of the train2017 subset land somewhere other than
"all four models agree." That's a healthy amount of clash for a dispatcher to route on, and
the single biggest disagreement pattern (nano alone missing an image that small/medium/large
all get exactly right) is exactly the signal PERTINENCE needs. But the tails are lopsided:
only 23.6% of train images have all four models exactly correct, versus 35.6% with all four
exactly wrong — the labeling step would route most images to expensive models by necessity,
not because they're genuinely hard, just because "exactly right on every object in a busy
image" is an unreasonably strict bar.

**Plain recall thresholded loose (>= 0.65)** swung too far the other way. All-four-correct
jumped to 68.0%, all-four-wrong dropped to 4.1%, disagreement zone collapsed to 15.9% — most
of the clash signal disappeared, since a 0.65 bar is loose enough that even nano clears it on
most images. This is the failure mode flagged as a risk before ever running the benchmark:
too soft a metric gives the models nothing real to disagree about.

**Recall >= 0.80** is the sweet spot, checked against both extremes rather than picked by
feel:

| threshold | all-correct | all-wrong | disagreement zone |
|---|---|---|---|
| exact-match | 23.6% | 35.6% | 40.8% |
| recall >= 0.65 | 68.0% | 4.1% | 15.9% |
| recall >= 0.80 | 44.7% | 15.6% | 39.7% |

0.80 preserves almost exactly as much disagreement-zone signal as exact-match (39.7% vs.
40.8%) while fixing the lopsided tails — 44.7%/15.6% instead of exact-match's 23.6%/35.6%.
The disagreement *shape* is also consistent across both exact-match and the 0.80 threshold
(same "nano fails alone, rest agree" pattern dominates either way) — the underlying
per-image difficulty signal isn't changing, only where the pass/fail line sits on top of it.

**Guiding principle** (generalizes beyond this one decision): PERTINENCE only earns its keep
in the disagreement zone, where models genuinely differ on whether an image is easy or hard.
A correctness definition should be picked to preserve that zone's size and keep the tails
sensible — not to make the accuracy numbers look nicer, and not to maximize disagreement for
its own sake. 0.80 satisfies that on real data, not a guess.

**Still open**: plain recall vs. F1 as the underlying score before thresholding — recall
alone doesn't penalize a model padding its predictions with extra wrong classes, since false
positives are free once recall clears 0.80. Worth a quick side-by-side before this becomes
the label the dispatcher actually trains on.

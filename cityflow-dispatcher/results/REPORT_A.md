# Part A report: does fine-tuning n, s, m on CityFlow help?

Short answer: yes for n and s (large gains on both held-out scenarios), mixed for m (big gain on the test
scenario, a loss on val). The ideal-label split moves a lot towards the small models on both. Part B was
started automatically after this report, as requested.

Setup: train S01 S03 S04 (6,924 frames), val S05 (14,775 frames, 19 cameras), test S02 (1,622 frames, 4
cameras). Every 5th frame. Pseudo-labels from stock yolov8l (conf 0.25, road classes). Fine-tuned n, s, m
for 10 epochs, AdamW lr0 1e-4, batch 16, seed 0. A frame is correct for a model if its box recall vs
yolov8l is >= 0.80 (IoU 0.5). Numbers below are from `results/04_compare/`.

## Original vs fine-tuned, test split (S02)

| model | variant | recall vs l | precision vs l | mAP50 | mAP50-95 | frame-correct | GT recall |
|---|---|---|---|---|---|---|---|
| n | original | 0.600 | 0.721 | 0.374 | 0.268 | 0.199 | 0.751 |
| n | fine-tuned | 0.780 | 0.628 | 0.434 | 0.318 | 0.518 | 0.804 |
| s | original | 0.789 | 0.763 | 0.560 | 0.412 | 0.557 | 0.754 |
| s | fine-tuned | 0.866 | 0.669 | 0.496 | 0.379 | 0.780 | 0.777 |
| m | original | 0.853 | 0.824 | 0.650 | 0.540 | 0.721 | 0.722 |
| m | fine-tuned | 0.900 | 0.710 | 0.582 | 0.473 | 0.867 | 0.757 |
| l (stock) | original | | | | | | 0.770 |

## Original vs fine-tuned, val split (S05, for reference)

| model | variant | recall vs l | precision vs l | mAP50 | mAP50-95 | frame-correct |
|---|---|---|---|---|---|---|
| n | original | 0.573 | 0.783 | 0.336 | 0.249 | 0.122 |
| n | fine-tuned | 0.744 | 0.705 | 0.323 | 0.227 | 0.420 |
| s | original | 0.765 | 0.809 | 0.465 | 0.375 | 0.441 |
| s | fine-tuned | 0.842 | 0.694 | 0.388 | 0.297 | 0.697 |
| m | original | 0.843 | 0.855 | 0.611 | 0.525 | 0.683 |
| m | fine-tuned | 0.810 | 0.810 | 0.451 | 0.372 | 0.583 |

## Ideal-label split (cheapest correct model), fraction of frames

| split | pool | n | s | m | l |
|---|---|---|---|---|---|
| test | original | 0.199 | 0.376 | 0.226 | 0.200 |
| test | fine-tuned | 0.518 | 0.310 | 0.112 | 0.060 |
| val | original | 0.122 | 0.324 | 0.280 | 0.274 |
| val | fine-tuned | 0.420 | 0.321 | 0.054 | 0.205 |

The fine-tuned pool is far less skewed to l than the old COCO routing was (90 to 94 percent of frames went to
l). On test, over half the frames can be served by n.

## Reading the numbers plainly

- Recall vs l and frame-correct rise for n and s on both held-out scenarios. The gain is not just a few
  cameras on val: n improved on all 19 val cameras and s on 17 of 19 (mean frame-correct +0.30 and +0.25).
- Precision vs l falls for every fine-tuned model (0.72 to 0.63 for n on test). They produce more boxes than
  l would. This is the price of the recall gain and is invisible to the frame-correct rate, which ignores extras.
- mAP50 and mAP50-95 against the pseudo-labels go DOWN for s and m on both splits (and n on val), even while
  recall goes up. The recall gain comes with more false boxes; mAP punishes that, the 0.80-recall rule does not.
  If the dispatcher's notion of "correct" should also penalise extra boxes, the threshold definition is the
  problem, not the models. I have not changed it.
- m is the weak spot: on test it improves a lot (frame-correct 0.72 to 0.87) but on val it gets worse
  (0.68 to 0.58; better on only 6 of 19 val cameras). With only 10 epochs, 3 train scenarios and
  pseudo-labels, m may be overfitting to train-scenario appearance. I did not tune it.
- On test, per camera: c006, c008 gain most (n 0.10 to 0.55 and 0.01 to 0.23). c007 and c009 gain for
  n and s, but m is slightly worse on those two (0.855 to 0.835, 0.841 to 0.803). Gains are not uniform.
- Recall on official GT vehicle boxes (class-agnostic, extras ignored) goes up for all three on test
  (n 0.751 to 0.804, s 0.754 to 0.777, m 0.722 to 0.757), and stock l scores only 0.770. So the
  improvement is not just copying l's pseudo-labels.

## Time, disk, versions

| step | time |
|---|---|
| 01 download + extract | not timed (run in the background) |
| 02 dataset (23,321 frames, yolov8l labelling) | roughly 1 h (not exactly timed) |
| 03 fine-tune n / s / m | 71.8 / 63.7 / 67.1 min (7.2 / 6.4 / 6.7 min per epoch) |
| 04 compare | about 4 h (val mAP passes are CPU-bound, about 17 min per model-variant) |

Disk: data 51 GB (zip, extracted videos, frames), models 0.6 GB, results 8 MB, .venv 4 GB.
Versions: torch 2.11.0+cu128, ultralytics 8.4.184, opencv 5.0.0.93 (full list in `requirements.lock.txt`).
Seed 0 for fine-tuning and the head. GPU: RTX 5070 Ti Laptop (12.8 GB), batch 16 never hit OOM. GPU temperature was about 74 C when I sampled it.
Epoch times for n varied between 2:23 and 3:49 for the train pass; I did not separate thermal throttling from my
Part B smoke tests that shared the GPU for a while, so the per-epoch times above are not clean.

## Flags

- **GT is incomplete as expected.** On the test frames: 4,198 official GT boxes vs 16,070 from yolov8l.
  Overlays in `results/02_dataset/gt_check/`. GT recall is only a sanity check.
- **Train-label caveat for Part B.** Dispatcher labels on the train split come from models that were
  fine-tuned on those very frames, so train will look easier than val and test. This is the same trap as
  the CIFAR-100 labels. I followed the spec (train the head on train labels) and report the label split per split.
- **Deviations from the spec.** Frames are in `data/images/` not `data/frames/` (ultralytics label lookup).
  Fine-tuning uses AdamW (ultralytics default "auto" would ignore lr0). Costs use `get_flops` on the fused
  model because `model.info()` returned None in this ultralytics version (values match: 8.7 / 28.6 / 78.9 / 165.1).
- **Test scenario is small:** 4 cameras, 1,622 frames, so test numbers have wide error bars.
- **Val and test mAP/recall are measured against pseudo-labels from l, not human labels.**
- **S02 and S05 are both in CityFlow's "validation" folder** (S05 has 19 cameras, S02 has 4); the spec's
  val and test split names do not match CityFlow's own folder names.

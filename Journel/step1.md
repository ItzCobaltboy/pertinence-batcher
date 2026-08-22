# Day 2

Task : Train a labeller
- Ditched using Quantization at all, we are moving on with resnet 18, 23, 50, and 152 sizes original in fp32

### [RESULT] Step 1A — Dispatcher label generation complete

**Script**: `code/Dispatcher/labeler.py`
**Output**: `results/dispatcher_labels.csv` (9469 train images)

Label assignment: cheapest FP32 model (RN18→RN34→RN50→RN152) that correctly
classifies each image. Fallback to label 3 if none get it right.

| Label | Model | Count | % |
|-------|-------|-------|---|
| 0 | ResNet18  | 7363 | 77.8% |
| 1 | ResNet34  |  740 |  7.8% |
| 2 | ResNet50  |  530 |  5.6% |
| 3 | ResNet152 |  836 |  8.8% |

**Observation**: Heavily skewed toward label 0 — consistent with ImageNette
being an easy 10-class subset (paper saw ~68% majority on CIFAR-10, we get
77.8%). Label-3 bucket (8.8%) conflates two cases: images only RN152 can
handle vs. images no model gets right — need to split this in EDA.

**Implication**: naive FC will predict 0 always and hit 77.8% "accuracy"
while being useless. Sample weighting (INS/ISNS/ENS) is mandatory.

### [RESULT] Step 1B — EDA on dispatcher labels

**Script**: `code/Dispatcher/eda.py`
**Input**: `results/dispatcher_labels.csv` (9469 train images)

#### Label distribution
| Label | Model | Count | % |
|-------|-------|-------|---|
| 0 | ResNet18  | 7363 | 77.8% |
| 1 | ResNet34  |  740 |  7.8% |
| 2 | ResNet50  |  530 |  5.6% |
| 3 | ResNet152 |  836 |  8.8% |

Label-3 split: 307 (36.7%) are genuinely hard (RN152 correct),
529 (63.3%) are noise — no model gets them right.
**Effective trainable set: 8940 images (94.4% of train set).**

#### Correctness combinations (2^4 = 16 patterns)
All 16 patterns observed. Key split:
- **Monotonic** (bigger ≥ smaller per image): 90.9% of data
- **Non-monotonic** (bigger model fails where smaller succeeds): 9.1% of data

Notable non-monotonic patterns:
- `1 0 x x` (364 images, 3.8%): RN18 correct, RN34 blind spot
- `1 0 1 1` (249 images, 2.6%): RN34 specifically fails these
- `1 1 1 0` (40 images, 0.4%): RN152 worst of all four
- `x x 1 0` (166 images, 1.8%): RN152 worse than RN50

#### Per-class findings
- **Cassette player**: hardest class — RN18: 41.7%, RN152: 62.0% only
- **Church**: non-monotonic accuracy — RN50 (51.6%) worse than RN34 (65.0%)
- **Tench / golf ball**: easiest — RN18 handles 91–93%

#### Cumulative coverage ceiling
| Up to model | Coverage |
|-------------|----------|
| ResNet18    | 77.8%    |
| ResNet34    | 85.6%    |
| ResNet50    | 91.2%    |
| ResNet152   | 94.4%    |

Maximum achievable accuracy with this pool: **94.4%**.

### [DECISION] Filter noise images from dispatcher training

The 529 `0000` images (no model correct) carry no learnable routing signal
and will push the FC layer toward spurious label-3 predictions.

**Decision**: filter `dispatcher_labels.csv` to rows where at least one model
is correct (`resnet152_correct == 1 OR resnet50_correct == 1 OR ...`) before
training the FC layer. Retained: 8940 images.

### [DECISION] Revise overestimation cost assumption

Paper treats overestimation (routing to larger model than needed) as
accuracy-neutral, only wasting compute. EDA disproves this: 9.1% of images
have non-monotonic correctness — overestimating to RN34 on `1 0 x x` images
actively loses accuracy. Penalty matrix must assign non-zero cost to
overestimation errors, not just underestimation.

### [RESULT] Step 1C — Dispatcher FC training (ISNS, 30 epochs)

**Script**: `code/Dispatcher/trainer.py`
**Checkpoint**: `results/checkpoints/dispatcher_fc.pt`

- Architecture: ResNet18 backbone (frozen) + Linear(512→4) head — 2,052 trainable params
- Weight scheme: ISNS (inverse square root of class count) — counters 82.4% label-0 imbalance
- Penalty matrix: hand-tuned asymmetric 4×4
  - Underestimation (pred smaller model than needed): penalty 2.0–4.0×
  - Overestimation (pred larger model than needed): penalty 0.5×
- Best training loss: 0.4828 @ epoch 26
- Final training loss: 0.4925 @ epoch 30
- Final training acc: ~60.3% (on ISNS-rebalanced sampler — not raw distribution)
- Converged around ep 26, stable for last 4 epochs

**Next**: Step 1D — evaluate on val set, confusion matrix, accuracy-FLOPs plot.

### [RESULT] Step 1D — Dispatcher FC evaluation (hand-tuned, ISNS)

**Script**: `code/Dispatcher/evaluator.py`
**Outputs**: `results/eval/confusion_matrix.png`, `results/eval/dispatcher_predictions.csv`

Overall routing accuracy: **81.4%** (7273/8940 images)

| Routing outcome | Count | % |
|----------------|-------|---|
| Correct        | 7273  | 81.4% |
| Underestimated | 1017  | 11.4% ← accuracy risk |
| Overestimated  |  650  |  7.3% ← FLOPs waste |

Per-class recall:
| Label | Model    | N    | Correct | Underest. | Overest. | Recall |
|-------|----------|------|---------|-----------|----------|--------|
| 0 | ResNet18  | 7363 | 94.1%   | 0%        | 5.9%     | 94.1% |
| 1 | ResNet34  |  740 | 12.3%   | 69.2%     | 18.5%    | 12.3% |
| 2 | ResNet50  |  530 | 20.6%   | 64.0%     | 15.5%    | 20.6% |
| 3 | ResNet152 |  307 | 45.9%   | 54.1%     | 0%       | 45.9% |

**Finding**: FC collapses toward label-0 despite ISNS weighting. Minority class recall
(RN34: 12.3%, RN50: 20.6%) is too low for a useful dispatcher. ISNS alone is not
sufficient for an 82:8:6:3 imbalance ratio.

**Root cause**: hand-tuned penalty matrix + fixed weighting scheme cannot jointly
optimise the accuracy-vs-FLOPs tradeoff across all 4 classes. This is exactly the
problem NSGA-II solves — evolving the penalty matrix values and weighting scheme
together to trace the full Pareto front of dispatcher configurations.

**Next**: implement NSGA-II to search penalty matrix + weighting scheme jointly.
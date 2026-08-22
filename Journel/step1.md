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
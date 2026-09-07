# Week 0

## Meeting notes & tasks

New direction: PERTINENCE assumes single-image inference, but real deployment needs
batching with mixed-complexity images.

Tasks: build a hybrid model pool (ResNet variants + quantized versions), hand-build a
dispatcher to learn the mechanics, and train a labeller for it once the pool is locked.

## Log

### [SETUP] Repo initialized
- Mono-repo: journal/, code/, ResnetModels/, dataset/, results/
- Target: ImageNet-pretrained ResNet18/34/50/152, evaluated on ImageNette (10-class subset)
- Checkpoints stay local (later Drive/Colab); metrics/code go to git

### [DECISION] Dataset: CIFAR-10 → CIFAR-100 → ImageNet
- Started CIFAR-10, then CIFAR-100, for fast iteration
- Settled on ImageNet-pretrained ResNets evaluated on ImageNette
- Why: CIFAR's 32×32 images compress FLOPs differences between ResNet variants — spread
  too small for a meaningful Pareto study. ImageNet/ImageNette (224×224) gives realistic
  FLOPs separation, more representative of edge deployment

### [DECISION] FLOPs as the initial compute metric
- Chose FLOPs over latency/energy: hardware-agnostic, easy to compute via `thop`
- Later found (see RESULT below) FLOPs doesn't distinguish quantized variants of the same
  architecture, since quantization doesn't change op count

### [SETUP] ImageNette dataset
- Used ImageNette (fast.ai's 10-class ImageNet subset, 320px, ~1.5GB) instead of full
  ImageNet (~144GB) — same pretrained weights work, much faster to download/iterate
- Bug found + fixed: ImageNette folder names are ImageNet synset IDs; ImageFolder assigns
  labels 0-9 by default, but the pretrained ResNets output 1000-class ImageNet logits.
  Added `IMAGENETTE_LABEL_MAP` to remap folder indices to the correct ImageNet class
  indices. Root cause of an initial ~9% ("random guess") accuracy bug — fixed, ResNet18
  FP32 now reads ~78% Top-1 correctly.

### [SETUP] metrics.py created
- `model_metrics(model, dataset_path, device, pth_path)` → FLOPs, Top-1 accuracy, model
  size (MB, from the .pth on disk), and latency (ms, GPU-timed with warmup +
  `cuda.synchronize()`) over the ImageNette val set

### [RESULT] Step 0 — torchao weight-only quantization pool metrics

| Model     | Precision | FLOPs (G) | Size (MB) | Accuracy (%) |
|-----------|-----------|-----------|-----------|--------------|
| ResNet18  | float32   | 1.824     | 44.67     | 78.14        |
| ResNet18  | int8      | 1.824     | 43.21     | 78.34        |
| ResNet18  | float8    | 1.824     | 43.21     | 77.99        |
| ResNet34  | float32   | 3.679     | 83.28     | 81.35        |
| ResNet34  | int8      | 3.679     | 81.83     | 81.32        |
| ResNet34  | float8    | 3.679     | 81.83     | 81.58        |
| ResNet50  | float32   | 4.134     | 97.79     | 86.32        |
| ResNet50  | int8      | 4.134     | 91.94     | 86.19        |
| ResNet50  | float8    | 4.134     | 91.94     | 86.32        |
| ResNet152 | float32   | 11.604    | 230.48    | 90.60        |
| ResNet152 | int8      | 11.604    | 224.64    | 90.62        |
| ResNet152 | float8    | 11.604    | 224.64    | 90.96        |

### [DEAD-END] torchao weight-only quantization (Int8WeightOnlyConfig, Float8WeightOnlyConfig)
- Size reduction only 2-6% vs. the theoretical 4x expected from INT8
- Accuracy delta <0.4% across all variants — indistinguishable
- Root cause: these configs are weight-only — weights are stored in low precision on
  disk, but the actual matmul still runs in original (FP32) precision at runtime. A
  memory-bandwidth optimization (useful for LLM serving), not a compute optimization —
  wrong tool for CNN inference on ResNets

### [DEAD-END] torchao dynamic activation quantization (Int8DynamicActivationInt8WeightConfig, Float8DynamicActivationInt4WeightConfig)
- These are the correct configs for compute-bound models (quantize weights AND
  activations, real low-precision matmul) — theoretically the right tool this time
- Result: still no meaningful accuracy/size/latency differentiation for ResNets
- Hypothesis: torchao's dynamic quant kernels are optimized for transformer-shaped
  (large, regular) matmuls common in LLMs; ResNet conv layers don't benefit the same way

### [PIVOT] Dropped torchao quantization entirely, moving to ONNX + TensorRT
- Plan: export FP32 ResNets to ONNX → static INT8 PTQ via `onnxruntime.quantization`
  (calibrated on the ImageNette val subset) → benchmark via ONNX Runtime with
  `TensorrtExecutionProvider` for genuine GPU INT8 kernels
- Rationale: TensorRT is NVIDIA's own inference optimizer, has a mature INT8 quantization
  path, more likely to show real compute/latency differentiation than torchao did

### [DEAD-END] ONNX + CUDAExecutionProvider INT8 quantization
- Static INT8 PTQ via `onnxruntime.quantization`, evaluated through CUDAExecutionProvider
- Latency reduction only ~0.4ms — negligible, not meaningful pool diversity
- TensorRT EP not usable (`nvinfer_10.dll` missing, full TRT SDK not installed)
- Verdict: quantization does not create useful Pareto operating points for ResNet CNN
  inference regardless of backend (torchao or ONNX)
- Decision: use FP32 ResNet18/34/50/152 as the 4-model dispatcher pool — **ditching
  quantization**

### [RESULT] Step 0 — ONNX + CUDAExecutionProvider metrics

| Model     | Precision | FLOPs (G) | Latency (ms) | Accuracy (%) |
|-----------|-----------|-----------|--------------|--------------|
| ResNet18  | fp32      | 1.824     | 3.577        | 78.14        |
| ResNet18  | int8      | 1.824     | 3.152        | 78.70        |
| ResNet34  | fp32      | 3.679     | 3.197        | 81.30        |
| ResNet34  | int8      | 3.679     | 5.234        | 81.32        |
| ResNet50  | fp32      | 4.134     | 3.755        | 86.29        |
| ResNet50  | int8      | 4.134     | 5.423        | 84.97        |
| ResNet152 | fp32      | 11.604    | 8.736        | 90.62        |
| ResNet152 | int8      | 11.604    | 14.976       | 91.01        |

### [DEAD-END] ONNX INT8 quantization via CUDAExecutionProvider
- INT8 is slower than FP32 for 3/4 models — QDQ wrapper nodes add overhead that outweighs
  INT8 compute savings without TensorRT kernel fusion
- TensorRT EP still not usable (`nvinfer_10.dll` missing)
- FP32 model sizes are broken in the CSV (weights not embedded in the ONNX file) — not a
  blocker since latency/accuracy are valid
- Verdict: quantized variants don't improve the Pareto front, confirmed across torchao
  (weight-only + dynamic) and ONNX + CUDA backends

### [DECISION] Final pool: FP32 ResNet18/34/50/152
- 4 architecturally diverse models with meaningful latency spread (3.2–8.7ms) and
  accuracy spread (78–91%) — sufficient for dispatcher training
- Moving to dispatcher implementation
- Ditched quantization entirely for the pool — moving on with ResNet18/34/50/152,
  original sizes, FP32

### [RESULT] Step 1A — Dispatcher label generation complete
**Script**: `code/Dispatcher/labeler.py`
**Output**: `results/dispatcher_labels.csv` (9469 train images)

Label assignment: cheapest FP32 model (RN18→RN34→RN50→RN152) that correctly classifies
each image. Fallback to label 3 if none get it right.

| Label | Model | Count | % |
|-------|-------|-------|---|
| 0 | ResNet18  | 7363 | 77.8% |
| 1 | ResNet34  |  740 |  7.8% |
| 2 | ResNet50  |  530 |  5.6% |
| 3 | ResNet152 |  836 |  8.8% |

**Observation**: heavily skewed toward label 0 — consistent with ImageNette being an easy
10-class subset (the paper saw ~68% majority on CIFAR-10, we get 77.8%). Label-3 (8.8%)
conflates two cases — images only RN152 can handle vs. images no model gets right — needs
splitting in EDA.

**Implication**: a naive FC head will predict 0 always and hit 77.8% "accuracy" while
being useless. Sample weighting (INS/ISNS/ENS) is mandatory.

---

### [RESULT] Step 1B — EDA on dispatcher labels
**Script**: `code/Dispatcher/eda.py`
**Input**: `results/dispatcher_labels.csv` (9469 train images)

Label distribution — same table as above. Label-3 split: 307 (36.7%) genuinely hard
(RN152 correct), 529 (63.3%) noise — no model gets them right. **Effective trainable
set: 8940 images (94.4% of train set).**

Correctness combinations (2^4 = 16 patterns), all 16 observed. Key split: monotonic
(bigger model ≥ smaller model per image) 90.9% of data, non-monotonic (bigger model
fails where smaller succeeds) 9.1%. Notable non-monotonic patterns: `1 0 x x` (RN18
correct, RN34 blind spot — 364 images, 3.8%), `1 0 1 1` (RN34 specifically fails these —
249 images, 2.6%), `1 1 1 0` (RN152 worst of all four — 40 images, 0.4%), `x x 1 0`
(RN152 worse than RN50 — 166 images, 1.8%).

Per-class: cassette player hardest (RN18 41.7%, RN152 only 62.0%); church is
non-monotonic (RN50 51.6% worse than RN34 65.0%); tench/golf ball easiest (RN18 handles
91–93%).

Cumulative coverage ceiling — up to RN18: 77.8%, up to RN34: 85.6%, up to RN50: 91.2%, up
to RN152: 94.4%. **Max achievable accuracy with this pool: 94.4%.**

---

### [DECISION] Filter noise images from dispatcher training
The 529 `0000` images (no model correct) carry no learnable routing signal and would
push the FC layer toward spurious label-3 predictions. **Decision**: filter
`dispatcher_labels.csv` to rows where at least one model is correct before training.
Retained: 8940 images.

---

### [DECISION] Revise the overestimation cost assumption
The paper treats overestimation (routing to a larger model than needed) as
accuracy-neutral — only wasted compute. EDA disproves this: 9.1% of images are
non-monotonic, so overestimating to RN34 on `1 0 x x` images actively loses accuracy.
Penalty matrix must assign non-zero cost to overestimation, not just underestimation.

---

### [RESULT] Step 1C — Dispatcher FC training (ISNS, 30 epochs)
**Script**: `code/Dispatcher/trainer.py`
**Checkpoint**: `results/checkpoints/dispatcher_fc.pt`

Architecture: ResNet18 backbone (frozen) + Linear(512→4) head, 2,052 trainable params.
Weighting: ISNS (inverse square root of class count), countering the 82.4% label-0
imbalance. Penalty matrix: hand-tuned asymmetric 4×4 — underestimation 2.0–4.0×,
overestimation 0.5×. Best training loss 0.4828 @ epoch 26, final 0.4925 @ epoch 30,
final training accuracy ~60.3% (on the ISNS-rebalanced sampler, not raw distribution).
Converged around epoch 26, stable for the last 4.

**Next**: Step 1D — evaluate on val set, confusion matrix, accuracy-FLOPs plot.

---

### [RESULT] Step 1D — Dispatcher FC evaluation (hand-tuned, ISNS)
**Script**: `code/Dispatcher/evaluator.py`
**Outputs**: `results/eval/confusion_matrix.png`, `results/eval/dispatcher_predictions.csv`

Overall routing accuracy: **81.4%** (7273/8940). Correct 81.4%, underestimated 11.4%
(accuracy risk), overestimated 7.3% (FLOPs waste).

Per-class recall: RN18 94.1% (N=7363), RN34 12.3% (N=740), RN50 20.6% (N=530), RN152
45.9% (N=307).

**Finding**: the FC head collapses toward label-0 despite ISNS weighting — minority
recall (RN34 12.3%, RN50 20.6%) is too low to be useful. ISNS alone can't handle an
82:8:6:3 imbalance.

**Root cause**: a hand-tuned penalty matrix + fixed weighting scheme can't jointly
optimize the accuracy-vs-FLOPs tradeoff across 4 classes. Exactly the problem NSGA-II
solves — evolve the penalty matrix and weighting scheme together to trace the full
Pareto front of dispatcher configurations.

**Next**: implement NSGA-II to search penalty matrix + weighting scheme jointly.

---

### [DECISION] Fix the NSGA-II fitness objective before the first full run
Initial fitness used `obj1 = 1 - overall_accuracy`, which doesn't penalize per-class
imbalance — a chromosome nailing the 82%-majority RN18 class while ignoring
RN34/RN50/RN152 scores well despite being a useless dispatcher (same collapse as the
hand-tuned run). Caught before burning compute. **Fix**: `obj1` →
`underestimation_rate = mean(pred_label < true_label)`, directly targeting the
accuracy-risk failure mode; `obj2 = avg_flops_G` still captures the compute tradeoff.
Killed and restarted the run after 30 min on the wrong objective.

---

### [RESULT] Step 2 — NSGA-II search (50 pop, 50 gen, 30 FC epochs/individual)
**Script**: `code/Dispatcher/nsga2.py`
**Outputs**: `results/nsga2/pareto_front.csv`, `results/nsga2/checkpoint_gen*.npz`

Chromosome: 13 floats — 12 penalty matrix values `[0,5]` + weighting exponent α `[0,1]`
(`weight_i ∝ 1/count_i^α`; α=0 uniform, α≈0.5 ISNS-like, α=1 INS-like). Objectives (both
minimised): `obj1 = underestimation_rate`, `obj2 = avg_flops_G`.

Key optimization: backbone frozen → all 8940 embeddings precomputed once (26s), each
individual only trains/evals `Linear(512→4)` on in-memory tensors (~3-5s/eval). Standard
NSGA-II ops: non-dominated sort, crowding distance, SBX crossover (η=20), polynomial
mutation (η=25), binary tournament. Memory safety: explicit `del` +
`torch.cuda.empty_cache()` after every eval (2500+ total) to avoid CUDA fragmentation.
Runtime: ~3h05m unattended, zero crashes, checkpoints every 5 generations.

**Final Pareto front: 50 non-dominated individuals**, full spread across the tradeoff:

| Config | Underestimation rate | Avg FLOPs | α |
|--------|----------------------|-----------|---|
| Safest (max accuracy-safety) | 0.04% | 5.21G | 0.83 |
| Balanced middle | ~2-5% | 2.5-3G | ~0.5 |
| Cheapest (min compute) | 17.5% | 1.834G | 0.03 |

Compare to the hand-tuned baseline: 81.4% accuracy, **11.4%** underestimation, fixed at
one operating point. NSGA-II instead produces a full dial — any point on the front is a
valid, intentional accuracy/compute tradeoff rather than an accidental collapse.

**Next**: Step 1D-v2 — pick 2-3 representative Pareto points, run full `evaluator.py`
(confusion matrix, per-class recall) on each, compare against the hand-tuned baseline.
Meet 2 with the advisors happened right after this run — action items and everything
that follows from them are logged in `Journel/Week1.md`.

## Open threads (as of end of this file)
- Confirm onnx / onnxruntime-gpu / TensorRT installation status
- Pareto front evaluation needs per-model accuracy, not just underestimation rate —
  raised at Meet 2, addressed in `Journel/Week1.md`
- TensorRT quantization redo, per Meet 2 — see `Journel/Week1.md` (Torch-TensorRT pivot
  on the 5070ti), calibration fix + cross-GPU benchmarking follows in `Journel/Week2.md`

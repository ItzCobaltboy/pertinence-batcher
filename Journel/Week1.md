# Week 1

## Meeting notes & tasks

**Action items:**
- ~~[REDO] TensorRT quantization properly — fix the nvinfer EP setup, rule out an fp32
  typecasting misconfiguration from the previous attempt~~ __Pivoted to Torch-TensorRT on
  the 5070ti below; compiled fine, but the FP16 win turned out to be a compilation-only
  effect, and INT8/FP8 stayed dead due to the modelopt calibration gap (fix follows in
  Week2.md)__
- ~~[TODO] Pareto front evaluation — add per-model accuracy; current results only have
  underestimation rate~~ __See Step 2D-v2 below__
- ~~[TODO] Implement the loss-weighting scheme on the FC layer loss.~~ __See the
  class-imbalance fix below__
- [EXPLORE] Model compression (pruning, KD, etc.) — check if it creates useful
  Pareto-optimal points

**Tasks for the week**: evaluate Run #1's Pareto front properly (per-model accuracy, not
just underestimation rate), fix the loss-weighting implementation, and redo TensorRT
quantization on real hardware. Step 2D-v2 below (addressing the eval gap) then exposed
that raw routing accuracy is itself a majority-class proxy, not a real quality signal —
fixing the dispatcher's objective became the week's main thread on top of the original
tasks.

## Log

### [RESULT] Step 2D-v2 — Full Pareto-front evaluation (train + val, precision + recall)
**Code**: `code/dispatcher_analysis/` — retrains all 50 Pareto configs, predicts on
train (8940 images) and held-out val (3701 images), saves per-config summaries +
confusion matrices.

**Finding: routing accuracy is spoofed by class imbalance, not a meaningful quality
signal for picking a config.** `correlation(accuracy, recall_resnet18) = 0.997` on val —
accuracy is essentially just "how well did this config find the RN18 images," since RN18
is 82% of the dataset.

Top-3 configs by accuracy (val), all essentially RN18-only dispatchers:

| individual | accuracy | recall_RN18 | recall_RN34 | recall_RN50 | recall_RN152 |
|-----------|----------|-------------|-------------|-------------|--------------|
| 49 | 83.1% | 99.1% | 3.7%  | 11.7% | 0% |
| 8  | 83.0% | 99.3% | 0%    | 11.7% | 0% |
| 28 | 82.6% | 97.9% | 5.4%  | 17.8% | 0% |

Every top-5-by-accuracy config has **0% RN152 recall** — never once correctly routes to
the rarest, most expensive model. Meanwhile the worst-accuracy configs do the real
minority-class work, e.g. individual 6 (45.0% accuracy) gets 35.0% RN152 recall — far
better than any "high accuracy" config, at the direct cost of RN18 dominance. RN152
precision stays low (12–36%) even for the better minority-recall configs — those configs
aren't cleanly "better," they're trading majority dominance for imperfect minority
coverage.

**Conclusion**: raw routing accuracy shouldn't be used to pick or judge a Pareto config —
it's a proxy for "how RN18-biased is this config," not "how good is this dispatcher."
Config selection and any future NSGA-II objective should weight per-class recall/
precision (ideally macro-averaged) instead of, or alongside, accuracy.

## [DECISION] Class-imbalance bug found — sampler used instead of loss weighting

This project's own methodology doc says class imbalance should be handled with
INS/ISNS/ENS sample weighting **on the FC layer loss**. What was actually implemented
everywhere (`Dispatcher/trainer.py`, `Dispatcher/nsga2.py`, `dispatcher_analysis/
config_utils.py`) was a `WeightedRandomSampler` — biases *which images get drawn*
(oversampling minority classes with replacement), not *how much each image's loss
counts*. Different technique, same `1/count^α` formula. Likely cause of the Step 2D-v2
train→val collapse above (e.g. one config: 82% train RN152 recall → 39% val) —
oversampling-with-replacement repeats the same ~300 RN152 images redundantly across
epochs, a memorization risk; loss-weighting doesn't repeat images, just scales their
gradient.

**Fix**: implement only INS (not ISNS/ENS) as true loss weighting — multiply each
wrong-prediction's loss by `class_weight[true_label]`, plain shuffled `DataLoader`, no
oversampling. `code/Dispatcher/` rebuilt clean from scratch (old version preserved in
git history).

## [PIVOT] NSGA-II mechanics rebuilt on pymoo

Hand-rolled GA operators (selection, SBX crossover, polynomial mutation, non-dominated
sorting) replaced with `pymoo` (`NSGA2`, `SBX`, `PM`) for correctness confidence. Only
the domain logic — chromosome → penalty matrix → penalized loss → FC training →
objectives — is still hand-written.

Before launching, ground truth was independently re-verified from scratch: a fresh
4-model forward pass over all 9469 raw train images, diffed against the cached
`data/train_ground_truth.csv` — **0 mismatches**, identical label distribution
`[7363, 740, 530, 307]`.

## [DECISION] Fitness objective switched back to raw accuracy

Changed `obj1` from `underestimation_rate` to `1 - accuracy` (exact match against the
cheapest-correct label), to follow the paper's own approach more literally, despite
`underestimation_rate`'s known-good property of not symmetrically punishing
overestimation. Known risk flagged going in: this `accuracy` definition penalizes
overestimation exactly as hard as underestimation, so a config can't get credit for
"safe but wasteful" routing.

## [RESULT] NSGA-II run — accuracy objective, INS loss-weighted, pymoo

**Script**: `code/Dispatcher/main.py` (`run_nsga2()`)
**Config**: population=50, generations=50, FC_EPOCHS=50, SBX (η=20, p=0.9), polynomial
mutation (η=25), binary tournament, INS class weights `[0.0819, 0.8151, 1.1381,
1.9648]` (fixed, not searched).
**Runtime**: 20,054s (~5h34m) unattended, zero crashes, checkpoints every 5 gens.

**Final Pareto front: 15 individuals** (vs. 50 for the underestimation_rate run) —
accuracy 82.70%–84.84%, avg FLOPs 1.846G–2.012G. Much narrower band on both axes than the
earlier run. This run also saves the **actual trained FC weights** per individual
(`results/nsga2/models/individual_*.npz`), not just chromosomes — fixes the gap that
forced Step 2D-v2 to retrain from scratch.

**Local-minima check**: wanted to confirm the narrow band wasn't just the GA getting stuck
rather than genuinely exploring. Checked empirically instead of guessing — across all 1043
individuals evaluated during the run, `correlation(avg_flops_G, accuracy) = -0.699`. Top
10% by FLOPs averaged only 62.76% accuracy; bottom 10% by FLOPs averaged 83.20%. The
search DID explore the expensive region extensively — it's genuinely worse under this
accuracy definition, not a GA exploration failure. Root cause: the metric itself
structurally punishes overestimation (routing an easy image to a bigger model is scored
"wrong" even though classification would be correct), so spending more FLOPs has no way
to pay off under this objective. A photo of the PERTINENCE paper's CIFAR-100 CNN results
(shown mid-run) shows a similarly narrow accuracy band (76.9–77.3%, ~0.4pp) with the real
differentiation on the compute-savings axis instead — so this shape isn't necessarily
wrong, just a property of using raw accuracy as an objective, matching the paper's own
result shape.

## [RESULT] Step 1E-eval — Pareto front evaluated on train + held-out val

**Code**: `code/Dispatcher/src/evaluate.py` (+ `metrics.py`, both new). Loads the 15
saved `individual_*.npz` weights directly and predicts — no retraining, unlike Step
2D-v2. Val ground truth + its cached embeddings copied over from `dispatcher_analysis/`
(same frozen ResNet18 backbone, same labeling logic — byte-reusable). Outputs in
`results/eval/`, kept separate from `dispatcher_analysis`'s Step 2D-v2 output (different
Pareto front, different objective — don't merge).

**Generalization held up** — no repeat of the Step 2D-v2 sampler-collapse. Worst
single-class drop: individual 10's RN152 recall, 86.6% train → 49.6% val (a real drop,
but nowhere near the old 82%→39% collapse). Overall accuracy train→val gaps stayed
within ~2-4pp for all 15 configs. Tentatively: the loss-weighting fix (no more
oversampling-with-replacement) looks like it helped generalization, though this isn't a
controlled side-by-side (objective changed too).

**But the majority-class-bias finding from Step 2D-v2 reproduces here, even harder**, on
val: `correlation(accuracy, recall_resnet18) = 0.998` (vs. 0.997 in Step 2D-v2);
`correlation(accuracy, macro_recall) = -0.903` — **negative**: the configs raw accuracy
ranks *best* are the ones doing worst across the full class spectrum.

Highest-accuracy config (individual 9, 83.22% val accuracy) has macro_recall 0.324, 0%
RN34 recall, 3.25% RN152 recall — essentially an RN18-only dispatcher. Best macro-recall
config (individual 0, macro_recall 0.431) sits at the *bottom* of the accuracy ranking,
68.50% val accuracy. Same pattern as Step 2D-v2, now with harder numbers, because this
run's objective directly optimizes for accuracy rather than just being scored by it
after the fact.

**Conclusion**: confirms the pre-run concern about switching to raw accuracy — it
doesn't just fail to reward good minority-class routing, it actively selects against it.
Per-class recall/precision (ideally macro-averaged) remains the right way to judge or
pick a config, not accuracy, regardless of which objective the search itself optimizes.

---

## [SETUP] TensorRT via Torch-TensorRT (not ONNX Runtime)

Revisited TensorRT (Meet 2's `[REDO]` action item, above) on the 5070ti laptop GPU.
ONNX Runtime's TensorRT execution provider was a dead end again — onnxruntime 1.29.0's
provider DLL is hard-pinned to `nvinfer_10.dll`, but the available `tensorrt`/
`tensorrt-cu13` pip packages ship TensorRT 11.x (`nvinfer_11.dll`) — ABI mismatch,
unfixable without an old TensorRT release.

Switched to **Torch-TensorRT** instead — compiles directly from the live PyTorch model
(`ir='dynamo'`), no ONNX export and no onnxruntime provider involved. Installed
`torch-tensorrt==2.13.0` + `tensorrt-cu13==11.2.1.2` + `nvidia-modelopt` (small
pure-Python deps: psutil, dllist). ResNet18 compiled and ran successfully on the first
real test.

Rebuilt `code/model_analysis/` to match `code/dispatcher_analysis/`'s structure:
`main.py` (root) + `src/` (constants, data_loader, model_utils, trt_compiler, benchmark,
run_benchmark, eda). Compiles + caches one Torch-TensorRT engine per (model, precision)
to `model_cache/*.pt2`, benchmarks accuracy/latency/size over the full ImageNette val
set, saves to `results/torch_tensorrt_benchmark.csv`. `eda.py` here is separate from —
and doesn't touch — `code/dispatcher/eda.py`.

Removed the now-unused torchao float8/int8 checkpoints (1.4GB) and the ONNX export
folder (561MB, not needed — Torch-TensorRT skips ONNX entirely). Kept the fp32 `.pth`
checkpoints in `ResnetModels/`.

**Gotcha**: engines compiled for a static batch-size-1 input shape reject any other
input shape outright — first benchmark run crashed mid-way because the val accuracy
loader used batch_size=32. Fixed by setting the val loader to batch_size=1 to match the
compiled shape (see `constants.py`). A future improvement would be compiling with a
dynamic shape range (min/opt/max) instead — relevant for the batching extension where
sub-batch sizes vary.

## [RESULT] Torch-TensorRT benchmark (5070ti) — FP32 vs FP16 vs INT8 vs FP8

**Script**: `code/model_analysis/main.py`
**Output**: `results/torch_tensorrt_benchmark.csv`, `results/eda/*.png`

| Model | FP32 latency | FP16 latency | INT8 latency | FP8 latency | Speedup (FP32→FP16) |
|-------|-------------|-------------|-------------|------------|----------------------|
| resnet18  | 1.779ms | 0.878ms | 0.878ms | 0.877ms | ~2.0x |
| resnet34  | 4.309ms | 1.715ms | 1.722ms | 1.731ms | ~2.5x |
| resnet50  | 3.554ms | 1.540ms | 1.559ms | 1.543ms | ~2.3x |
| resnet152 | 10.698ms | 4.186ms | 4.161ms | 4.129ms | ~2.6x |

Accuracy: FP16 identical to FP32 for 3/4 models (resnet34: −0.03pp, noise). Model size:
FP16/INT8/FP8 are **byte-identical** for every model (e.g. resnet18: 60.27MB across all
three).

**INT8 and FP8 are dead ends via this path, same as every prior quantization attempt**:
size, latency, and accuracy for INT8/FP8 are indistinguishable from FP16 across all 4
models. `enabled_precisions={torch.int8}` / `{torch.float8_e4m3fn}` alone doesn't engage
real low-precision kernels — TensorRT's builder silently falls back to FP16 without an
explicit calibration step (real per-layer scale factors, e.g. via `modelopt`'s quantize
workflow before compiling). Matches the `modelopt` import warnings seen on every
compile, which persisted even after installing the package — the calibration gap this
raises gets fixed in `Journel/Week2.md`.

**FP16 looked like a genuine win at first**: ~2.0–2.6x latency reduction across the whole
pool, essentially zero accuracy cost. First apparent positive quantization/precision
result in this project (torchao weight-only, torchao dynamic, and ONNX+CUDA INT8 were
all dead ends — see `Journel/Week0.md`). Turned out to need a second look — see below.

**Next**: decide whether to pursue real INT8/FP8 via explicit `modelopt` calibration, or
accept FP16 as the practical precision win and move to the batching extension.

## [DECISION] Correction — the "FP16 win" above is a compilation win, not a precision win

Follow-up question: is the FP32→FP16 speedup from actual FP16/Tensor-Core execution, or
just from TensorRT's graph compilation itself (kernel fusion, no eager/Python dispatch
overhead) regardless of precision? The FP32 rows above were **eager, uncompiled
PyTorch** — never run through Torch-TensorRT — so the two effects (compile vs precision)
were never actually isolated.

Isolated test (resnet18, batch=1, each variant in its own process):

| Variant | Latency |
|---|---|
| Eager FP32 (uncompiled) | 1.833–1.912ms |
| **TensorRT-compiled FP32** | **0.891ms** |
| TensorRT-compiled FP16 | 0.886ms |

Compiled FP32 and compiled FP16 are statistically identical. **The ~2–2.6x speedup
above is almost entirely TensorRT compilation itself** — fusing Conv→BN→ReLU chains,
cutting Python/eager dispatch overhead — not FP16 precision or Tensor Core throughput.
This also better explains why INT8/FP8 measured identical to FP16: at batch=1 on models
this size, the workload isn't compute-bound enough for precision to matter at all — the
bottleneck compilation removes is overhead, not raw matmul throughput.

**Correction**: relabel the earlier result "TensorRT compilation win, precision-
independent" rather than "FP16 win." Whether real INT8/FP16 throughput differentiation
exists at all is still unverified — would need a compute-bound setup (larger batch size)
to actually test it, since batch=1 can't distinguish precision effects from overhead
effects. Root cause for INT8/FP8 specifically turned out to be the missing `modelopt`
calibration step — picked up and fixed in `Journel/Week2.md`, along with real cross-GPU
(5070ti + A100) benchmarking.

---

## [DECISION] The "raw accuracy" objective above still wasn't what the paper does — fixed to real alpha_sys

Read the actual PERTINENCE paper text this session (paper.pdf, 19 pages — previously
only one photographed page had been seen). Its real accuracy objective (Eq. 3) is:

    alpha_sys = fraction of images where the DISPATCHED model itself
                classifies that image correctly

Not exact-match against the argmin-cheapest-correct "ideal" label — what this project
built and ran for ~5.5h twice now. Under the paper's real metric, routing an easy image
to a bigger-but-still-correct model costs **zero** accuracy, only FLOPs (obj2). Under
what was built, that same routing was scored exactly as "wrong" as an actual
misclassification — exactly why the previous run's search converged to a narrow band and
structurally punished overestimation (`correlation(avg_flops_G, accuracy) = -0.699`
above).

Quantified the gap before touching code: on one random chromosome, the same trained FC
head's predictions scored 62.79% under exact-match vs. **95.04%** under real alpha_sys
(`correctness_matrix[image, predicted_model].mean()`, using the `<model>_correct`
columns already sitting in the ground truth CSV — no new computation needed, just the
right lookup).

**Fix**: `code/Dispatcher/src/fitness.py` (+ `dispatcher_problem.py`,
`nsga2_search.py`) now compute the real alpha_sys per individual. Renamed
"accuracy"/"accuracy_loss" to `alpha_sys`/`alpha_sys_loss` everywhere in the Dispatcher
codebase (`save_results.py`'s CSV column included) so the two metrics can't get silently
conflated again.

Also reduced `FC_EPOCHS` 50→30 and `GENERATIONS` 30 for faster turnaround (~2h vs.
~5.6h) — the earlier run's hyperparameters weren't in question, just its objective, so
no reason to keep paying the longer runtime while iterating.

## [RESULT] NSGA-II run — alpha_sys objective, INS loss-weighted, pymoo (Run #3)

**Config**: population=50, generations=30, FC_EPOCHS=30, same SBX/PM/INS setup as Run
#2. **Runtime**: ~2h30m.

**Final Pareto front: 50 individuals** — `alpha_sys` 83.36%–95.96%, avg FLOPs
1.92G–5.39G. Back to a full wide dial, much closer to the underestimation_rate run's
spread (Run #1: 50 individuals, 0.04–17.5% underestimation / 1.83–5.21G) than to Run
#2's narrow 15-individual band (82.70–84.84% / 1.846–2.012G). Confirms the narrow band
in Run #2 was specifically an artifact of the wrong accuracy definition, not something
inherent to this problem.

## [DECISION] Second bug caught: eval pipeline still used the old exact-match metric

After Run #3 finished, the existing `code/Dispatcher/src/evaluate.py` (built earlier
this session) was run against it automatically — but that eval code was never updated
when `fitness.py` was fixed. It still computed "accuracy" as exact-match against
`ideal_label`, not alpha_sys. Result: train alpha_sys (from the search log, 86–96%) and
"val accuracy" (from the stale eval, 36–79%) looked like a severe generalization
collapse — they were actually two different metrics being compared to each other, not a
real train→val gap. Caught before this got written up as a finding — would have been a
wrong conclusion in this log.

**Also flagged**: the eval pipeline had accumulated real folder-boundary violations —
`code/Dispatcher/data/val_ground_truth.csv` and `embeddings_cache/val_embeddings.npz`
had been copied in from `dispatcher_analysis/` to build that eval code, duplicating the
same CSVs across two "separate" pipelines (on top of `train_ground_truth.csv`, already
duplicated this way before this session).

## [PIVOT] Folder responsibilities split cleanly: Dispatcher = search only, dispatcher_analysis = eval only

**`code/Dispatcher/`**: stripped back to *only* the NSGA-II search. Deleted
`evaluate.py`/`metrics.py`, removed all val-split paths/functions from
`constants.py`/`embeddings.py`, deleted `results/eval/` (the wrong-metric output) and
`results/archive_exact_match_accuracy_2026-09-03/` (Run #2's now-superseded raw output).
`main.py` runs `run_nsga2()` only.

**`code/dispatcher_analysis/`**: emptied completely and rebuilt from scratch as the
standalone evaluation pipeline (old code — Step 2D-v2's sampler/alpha-based retraining
pipeline — preserved in git history). New pipeline: `cache_predictions.py` (loads each
Pareto individual's already-saved FC weights from `model_cache/`, predicts on train +
val, no retraining) → `summarize.py` (alpha_sys + avg_flops_G per individual per split,
via the same correctness-matrix lookup as the fixed `fitness.py` — explicitly not
exact-match, named `alpha_sys` not `accuracy` throughout to avoid repeating this exact
confusion). Populated from Run #3's outputs: `pareto_front.csv`, both ground-truth CSVs,
both embeddings caches, and all 50 `individual_*.npz` weight files — copied over once,
so no retraining or recomputation was needed to stand this pipeline back up.

Scope deliberately stopped after caching predictions + summaries — no confusion matrices
/ per-class recall-precision / plots yet, that's next.

## [RESULT] Step 1E-eval v2 — alpha_sys on train + held-out val (Run #3's front)

With the metric bug actually fixed: val `alpha_sys` sits in **87.5%–92.6%**, close to
Run #3's train-time search range (83.4–95.2%) — no generalization collapse, and no more
of the misleading 36–79% spread the metric-mismatch bug had produced. Notably,
individual 40 — the best-looking config under the broken exact-match eval (78.9%) — is
now the *worst* under real alpha_sys (87.5%). Confirms the fix changes which configs
actually look good, not just the headline numbers.

Caveat carried over from Run #2: recomputed train alpha_sys (via the saved weights)
differs from the search's own live number by up to 8.2 percentage points for one
individual (mean ~2.2pp) — `save_models.py` retrains once more after the search with
fresh init/shuffle, not bit-identical to what pymoo actually saw. Known, documented, not
newly introduced.

**Still open**: macro-recall / per-class breakdown for this alpha_sys front hasn't been
computed yet (scope stopped at predictions + summaries this round) — needed before
trusting any single "best" config, per the standing finding that overall accuracy
metrics (now alpha_sys too, potentially) can still be majority-class-biased even when
correctly defined.

---

## [DECISION] Stopped trusting any copied model cache — retrain from chromosome every run

The stale-checkpoint discovery above (Run #2's `checkpoint_gen035-050.npz` sitting
alongside Run #3's `gen005-030.npz` in the same folder, only distinguishable by file
mtime) raised the same doubt about `model_cache/individual_*.npz` — those were copied
over from `code/Dispatcher/results/nsga2/models/`, so if they'd been similarly stale
there'd be no code-level signal, only a silent wrong result. Checked by mtime: genuinely
written 20:33–20:34, right after Run #3's `checkpoint_gen030.npz` (20:33:12) — not
stale. But copied caches cost nothing to distrust and everything to get wrong silently,
so: **never trust a copied weight file here again.**

**Fix**: `dispatcher_analysis` now retrains every one of the 50 individuals directly from
`pareto_front.csv`'s chromosome columns on every run (`build_models.py`, using exact
copies of `code/Dispatcher`'s `penalty_matrix.py`/`class_weights.py`/`loss.py`/training
loop — same hyperparameters, `FC_EPOCHS=30`). Costs ~2-3 minutes for all 50 individuals
(retraining an FC head is ~3-5s), trivial next to the ~2.5h search itself. Also deleted
the stale `checkpoint_gen035-050.npz` files from `code/Dispatcher/` so they can't
confuse anyone again.

## [RESULT] Per-class recall/precision, Pareto scatter, confusion matrices — and the majority-bias question resolved

Extended the pipeline: `metrics.py` (confusion matrix / recall / precision against
`ideal_label`, not alpha_sys — a different, complementary question) folded into
`summarize.py`'s output; `plots.py` adds a Pareto scatter (`1 - alpha_sys` vs.
`avg_flops_G`, train + val, non-dominated points recomputed and highlighted
independently per split) and confusion-matrix heatmaps for three representative
individuals (cheapest / middle / highest-alpha_sys) on val.

**Scatter confirms a real front on both splits**: 14/50 non-dominated on train, 13/50 on
val — a genuine staircase, not degenerate. The two fronts mostly agree but aren't
identical (a handful of train-optimal individuals are dominated once evaluated on val) —
expected, since NSGA-II only ever saw train objectives.

**Resolves the standing open question** (had raw accuracy metrics, from Run #1/#2, been
majority-class-biased in a way that's actually about `alpha_sys` too, or was that an
artifact of the earlier wrong metrics?) — checked directly on Run #3's val summary:

| correlation with system-accuracy metric | Run #1/#2 (wrong metrics) | Run #3 (correct alpha_sys) |
|---|---|---|
| vs. `recall_resnet18` | +0.997 / +0.998 | **-0.576** |
| vs. `macro_recall` | -0.903 | **+0.561** |
| vs. `avg_flops_G` | -0.699 | **+0.786** |

Complete reversal. Under the wrong metrics, "accuracy" was essentially a proxy for
RN18-dominance and spending more FLOPs actively hurt the score. Under real `alpha_sys`,
higher scores now correlate *positively* with macro-recall and *negatively* with RN18
recall, and more FLOPs now genuinely buys more `alpha_sys` rather than costing it. The
majority-class-bias finding that shaped a large part of this session's earlier analysis
(Step 2D-v2, Step 1E-eval v1) was, to a significant degree, an artifact of measuring the
wrong thing — not a property of the dispatcher problem itself. Confusion matrices
confirm the mechanism visually: cheaper configs (e.g. individual 38, 2.18G) route almost
everything to RN18; pricier ones (individual 15, 3.79G) spread more into RN50/RN152 but
also over-route a lot of RN18-sufficient images there — the FLOPs-waste side of
overestimation, now correctly *not* showing up as an accuracy penalty.

## Notes
- Prepare a Google Doc report before the next meet
- Advisor's GitHub: ga-ananth

# Day 2

## Context
Task: train a labeller. Ditched quantization entirely — moving on with ResNet18/34/50/152,
original sizes, FP32.

## Log

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

---

### [MEETING] Meet 2 — Prof Gayathri / Traiola
**Action items:**
- [REDO] TensorRT quantization properly — fix the nvinfer EP setup, rule out an fp32
  typecasting misconfiguration from the previous attempt
- ~~[TODO] Pareto front evaluation — add per-model accuracy; current results only have
  underestimation rate~~ __See Step 2D-v2 below__
- ~~[TODO] Implement the loss-weighting scheme on the FC layer loss.~~
- [EXPLORE] Model compression (pruning, KD, etc.) — check if it creates useful
  Pareto-optimal points

---

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

---

# Day 3 (2026-09-03)

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

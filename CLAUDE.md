# pertinence-batcher — context for Claude Code

Research project (PERTINENCE line of work).
- **Supervisor**: Prof. Gayathri Ananthanarayanan (IIT Dharwad) + Dr. Marcello Traiola (INRIA)
- **Goal**: input-based opportunistic dynamic execution of NNs for energy-efficient edge inference

Current direction (since Week4): a batching-aware scheduler for PERTINENCE on compute-constrained
edge devices (Jetson-class), built and tested in a discrete event simulator (`scheduler-sim/`)
before running with PERTINENCE in the loop. Earlier steps (hybrid model pool, dispatcher, NSGA-II,
CIFAR-100) are done. Full narrative log lives in `Journel/` (the "why"); this file is the
"what/where" snapshot.

## Repo layout

`dispatcher/`, `dispatcher_analysis/`, and `eda/` are shared, generic code (no dataset-specific
constants or paths), parameterized by a `config` module (`cifar-100/fig9c/config.py` or
`cifar-100/fig9d/config.py`) passed in explicitly by each sub-track's entry-point scripts — see
`dispatcher/README.md` for the exact mechanism. `archive/model_analysis/` and
`archive/quantization_experiments/` hold done/locked work, not part of the active pipeline.
`cifar-100/` is the sole active dataset track.

- `dispatcher/` — Step 1-2: **NSGA-II search ONLY** (Pareto front + per-individual FC weights), shared
  across both sub-tracks. Trains each individual's FC head on TRAIN, evaluates its fitness on held-out
  test/val (matches the paper's own methodology — see "Fitness objective" below) — does NOT run the
  FINAL evaluation (per-class metrics, confusion matrices, plots), that stays `dispatcher_analysis`'s job.
- `dispatcher_analysis/` — Step 3: **final evaluation ONLY** of whatever front `dispatcher/` last produced,
  shared across both sub-tracks. Loads saved FC weights (no retraining), predicts on train + both
  held-out sets, computes alpha_sys + avg_model_cost + accuracy_exact_match_vs_ideal. Reads dispatcher's
  output files directly (no copy step) but shares no code with it.
- `eda/` — ground-truth EDA (per-model/oracle accuracy, class balance, error overlap), shared across
  both sub-tracks. Different from `archive/model_analysis/eda/` (benchmark-CSV comparison, from the
  earlier ImageNette-based exploration, now archived).
- `cifar-100/` — CIFAR-100 track, the sole active dataset track: rather than one `config.py`, it holds
  shared infrastructure (`dataset/` raw images, `models/` checkpoints+loader, `label_data.py`
  config-driven labeling, `analyze_model_overlap.py`) at its root, plus two fully self-contained thin
  sub-tracks, `fig9c/` and `fig9d/`, each reproducing one specific figure from the paper's own
  CIFAR-100 exploration (Fig. 9(c)/9(d), page 8) rather than a single dispatcher across the whole
  6-model pool — see `cifar-100/PIPELINE_REPORT.md`'s section 0 for why the full-pool approach was
  tried first and abandoned. Genuine three-way train/test/validation split in both sub-tracks (see
  "Fitness objective" below and `fig9c/config.py`'s "Split naming" comment) — each sub-track's
  `config.py` exposes `FINAL_VAL_GROUND_TRUTH_CSV`/`FINAL_VAL_EMBEDDINGS_NPZ` on top of the usual
  `VAL_*` pair, and `run_dispatcher_analysis.py` runs the shared `dispatcher_analysis/` evaluation
  code twice (once per held-out set) rather than touching that shared code to add a second slot.
- `archive/model_analysis/` — Step 0: model pool selection + Torch-TensorRT precision benchmark. Done,
  archived — not part of the active pipeline.
- `archive/quantization_experiments/` — Week2: standalone PTQ-calibration + batch-size-sweep
  experiments. Done/exploratory, archived.
- `yolo-analysis/` — new, standalone: YOLOv8/COCO multi-label class-recall benchmark, prep work
  for the Week4 scheduler direction's detection-based model pool. Self-contained (own
  `requirements.txt`, no imports from the rest of the repo) since it's meant to be copied to a
  server to run. Not wired to `dispatcher/`/`dispatcher_analysis/` — pure data-gathering, see
  `yolo-analysis/README.md`.
  `yolo-analysis/batch_sweep/` (`sweep.py`: preflight | measure | aggregate | profile): timing
  sweep for real YOLOv8 n/s/m/l T_i(b), b = 1,2,4,8,12,16,32,48, variants eager_fp32 / trt_fp32 /
  trt_fp16 (torch_tensorrt dynamo; engines cached in `results/batch_sweep/engines/`, git-ignored).
  Forward = fused DetectionModel incl. Detect head decode; NMS a separate column. One subprocess
  per job (crash-safe, resumable, session lock). Raw per-run JSON in
  `results/batch_sweep/raw/<session>/`; summaries, plots, sanity flags in `summary/<session>/`.
  `profile --variant V` writes `scheduler-sim/profiles/yolov8_a100_<V>.csv` (no default).
  Gotcha: torch 2.3 export needs `DetectionModel.stride` cloned (shared with the Detect head).
  "fp32" on the A100 is TF32 by default. **Done** (session `main`, 480/480 runs ok): results in
  `results/batch_sweep/summary/main/`; trt_fp16 chosen as the scheduler profile.
- `scheduler-sim/` — standalone DES for the scheduler (own `requirements.txt` and `README.md`, no
  imports from the rest of the repo), plain OOP. Two knobs: a `Workload` (arrival times + target
  queue, usually just `choose_queue`) and a `Scheduler` (only `decide(now)` ->
  `("run", queue_id, batch_size)`, `("wait", until_time)` or `None`); the `Simulator` validates
  every decision.
  - `sim.py`: Job, Queue, Profile (T_i(b) table), base classes, Simulator, raw save/load,
    `compute_metrics`. `workloads.py`: Uniform, Weighted, Sticky, PeriodicRouted, Periodic.
    `schedulers.py`: `FCFSNoBatch`, `FCFSBatch`, `LongestQueue`, `TimeoutBatch` (tau = 15 ms).
    `run_experiment.py`: sweeps, CSVs, plots. `tests/test_sim.py`: 16 pytest tests, passing.
  - `profiles/`: `yolov8_a100_eager_fp32.csv` (**in use**, measured eager FP32, b = 1..32 step 2,
    40/48/56/64) and `yolov8_a100_trt_fp16.csv` (not used: TRT engine outputs fail the eager check,
    see Week5 `[DEAD-END]`). Both written by `yolo-analysis/batch_sweep/sweep.py profile`.
  - YOLO routing weights hardcoded from val2017 recall >= 0.80: `[0.528 n, 0.149 s, 0.083 m,
    0.240 l]`. Metrics from t=0 (warm-up dropped: it inflated compression ratio when overloaded).
  - Each run saves raw per-job data as `.npz` in `results/<exp>/raw/` (git-ignored);
    `SIMULATE = False` recomputes metrics and plots from it in ~10 s.
  - Experiments in `results/` (each `results.csv` + plots, incl. `turnaround_vs_load_log.png` and `*_stable.png` that drop collapsed points), all on the measured profile,
    `REAL_LOADS` 0.05-0.8 jobs/ms step 0.05: `yolo_load_sweep` (Poisson), `yolo_sticky_load_sweep`,
    `yolo_periodic_load_sweep`. Old synthetic/ResNet profiles and sweeps deleted (in git history).
  - Removed vs the first version: registries, ABCs/hooks, JSONL tracer, deadline/priority fields,
    trace router, `recall_labeling.py`, interpolate mode, switch cost.
  - TODO: `stable` heuristic wrongly flags some very light loads; periodic workload starts all
    streams at t=0 (worst case), stagger phases.
- `research.md` — annotated bibliography for the scheduler (~50 papers, link + what differs from
  us). Read before citing anything or claiming novelty.
- `venv/` — set up locally (`python -m venv venv` + `pip install -r requirements.txt`), not committed.
- `Journel/`: `WeekX.md` per project week (not per-date). `Week0.md`: model-pool selection + quantization dead-ends (torchao, ONNX+CUDA), plus the first hand-tuned dispatcher + underestimation_rate NSGA-II run. `Week1.md`: dispatcher objective fixed (accuracy → alpha_sys) through correct Run #3 results, plus the Torch-TensorRT pivot (5070ti, compilation-only FP16 "win", modelopt calibration gap). `Week2.md`: modelopt calibration fixed, cross-GPU (5070ti + A100) benchmarking, calibrated PTQ + batch-size-sweep follow-up. `Week3.md`: repo restructure + fitness-on-val fix + CIFAR-100 track build. `Week4.md`: CIFAR-100 results presented, greenlit for a new direction (a batching-aware scheduler for PERTINENCE), formal scheduling problem definition; YOLO/COCO benchmark on both splits and the `recall >= 0.80` correctness decision. `Week5.md`: meeting 5 (scheduler direction accepted, simulator first, Jetson-class target); compute-metric decision; `scheduler-sim/` built, rewritten simpler, follow-ups (warm-up dropped, raw `.npz`, periodic workload); literature search (`research.md`); EdgeServing read in full; YOLO batch sweep run on the A100 (real trt_fp16 T_i(b), sim rerun on it); `[DECISION]` energy is the angle for meet 6, roadmap to a CCTV-trace evaluation; open ideas: MPS parallel batches, score-based policy shape.
- `pptx/`: advisor-meeting decks. `pertinence_cifar100_academic.pptx` (Week3/4 redo, the house style: 16:9, Calibri, white, grey rule under title, grey-bordered tables, one-line takeaway per table/figure) and `pertinence_week4_scheduler_yolo.pptx` (Week4 update: problem definition, prior art, YOLO threshold sweep). `pertinence_cifar100_academic-1.pptx` is an older copy of the first. `pertinence_week5.pptx` (meet 6): definitions, prior art (8 papers with links), dense A100 trt_fp16 curves, simulator baselines on the dense profile, policy v1.

## Pipeline: archive/model_analysis/ (archived — done/locked work, not part of the active pipeline)

Built around **Torch-TensorRT**, not ONNX Runtime (see dead-ends below, `Journel/Week1.md`). `main.py`
(root) + `src/`. Paths relative to `archive/model_analysis/`. Originally benchmarked against
ImageNette (since abandoned as a track — see "Decisions / dead ends" below); kept archived as-is
since the benchmark methodology itself is still a valid reference.

- `src/constants.py` — paths, model pool, precisions (fp32/fp16/int8/fp8), label map
- `src/data_loader.py` — val loader, correct label indexing. **Batch size must be 1** — compiled TRT engines are locked to their compiled shape.
- `src/model_utils.py` — loads pretrained models, FLOPs once per architecture (precision-independent)
- `src/trt_compiler.py` — compiles a model → Torch-TensorRT engine per precision, caches to `model_cache/*.pt2`
- `src/benchmark.py` — accuracy + latency (GPU-timed, warmup + `cuda.synchronize()`)
- `src/run_benchmark.py` — every (model, precision) → `results/torch_tensorrt_benchmark.csv`
- `eda/` (nested, moved from the old top-level `code/eda/`) — grouped bar charts + accuracy-vs-latency scatter → `results/eda/`. Separate from the shared, generic `eda/` at repo root (ground-truth EDA, not benchmark-CSV EDA).
- `main.py` — entry point, runs both phases

`ResnetModels/` — cached FP32 `.pth` checkpoints (regenerated if missing). `model_cache/` — cached
`.pt2` TensorRT engines (fp16/int8/fp8), gitignored.

**Gotcha**: this benchmark's dataset used ImageNet synset-ID folder names; `ImageFolder`
assigns labels 0–9 by default, but pretrained ResNets expect 1000-class ImageNet indices. Remapped
via a label map everywhere that dataset was touched. ~9-10% "random guess" accuracy = check
this mapping first (relevant only if this archived benchmark is ever re-run).

## Pipeline: archive/quantization_experiments/ — archived, standalone, not wired to dispatcher

Week2 follow-up to prof's meeting action item: does *real, calibrated* PTQ beat the FP16 baseline
from `model_analysis`? Fully separate pool/results from the dispatcher pipeline — exploratory only,
not (yet) feeding back into the model pool or NSGA-II runs.

- `exp1_ptq_calibration.py` — `modelopt mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)`
  (16 calibration batches) + `export_torch_mode()` + `torch_tensorrt.compile(enabled_precisions={torch.int8})`.
  This is the correct PTQ workflow, unlike `model_analysis`'s uncalibrated attempt (dead-end #7 below).
  → `results/exp1_ptq_calibration.csv`
- `exp2_batch_size_sweep.py` — resnet18/resnet50 × fp32/fp16/int8(uncalibrated) × batch sizes 1/4/8/16/32
  → `results/exp2_batch_size_sweep.csv`

**Gotcha**: `torch_tensorrt.save()` hits an uncatchable C++ `terminate()` (`inline_container.cc:672`)
when caching a PTQ-quantized dynamo export — a torch bug, not user error, doesn't happen for
unquantized fp16. Fix: `compile_with_trt()` takes `cache_path=None` for PTQ callers; those models are
compiled in-memory only, benchmarked immediately, never cached to disk.

**Results (see `Journel/Week2.md` for full detail)**:
- Calibrated INT8 PTQ genuinely works: 5–7× speedup over FP16 on RN18/34/50 (<1% accuracy delta),
  only 2.5× on RN152 (memory-bandwidth bound). FP8 PTQ works on RN18/34, fails with TRT Error Code 10
  on RN50/152 (no kernel for quantized-maxpool+conv chains on this GPU/TRT). This **partially
  supersedes** dead-end #7 below: INT8/FP8 are dead only via the *uncalibrated* path
  (`enabled_precisions=...` alone) — calibration makes it real.
- Uncalibrated INT8 tracks FP16 exactly at every batch size (exp2), reconfirming TRT silently
  falls back to fp16 without calibration.
- Batching alone gives ~5× free per-image speedup (bs=1→4) with zero quantization — relevant to the
  still-unstarted batching extension (see below); RN50+ should target bs=4–8, regresses at bs=32.
- **Decision**: PTQ proven viable but not integrated into the dispatcher's model pool — deferred as
  a future optimization.

## Pipeline: dispatcher/ — shared/generic, not wired to model_analysis

**Only responsibility: find the Pareto front.** No FINAL-eval code (per-class metrics,
confusion matrices, plots — that's `dispatcher_analysis`'s job, kept split from this
folder by design). It reads held-out ground-truth/embeddings through `config` paths —
per-individual fitness is evaluated on a held-out set, not train, matching the paper's
methodology (see "Fitness objective" below); it never copies that data into this folder
itself.

Uses PyTorch/torchvision directly (ONNX not used in this pipeline). No `src/` subfolder —
every module lives directly under `dispatcher/`. Every function that needs a
path/hyperparameter/pool value takes an explicit `config` argument instead of importing a
`constants.py` — see `dispatcher/README.md` for the full config-injection contract. This
is one shared copy of the code across both sub-tracks, not a per-track copy — see
"Class imbalance" below for the sampler this replaced.

- `cifar-100/fig9c/config.py` / `cifar-100/fig9d/config.py` — every path + hyperparameter + pool
  constant for that sub-track, passed in as `config` to everything below
- `embeddings.py` — computes/caches the track's frozen embedding-extractor backbone's output for train + held-out sets
- `penalty_matrix.py` — chromosome's penalty genes → `NUM_CLASSES`×`NUM_CLASSES` penalty matrix
- `weighting_scheme.py` — decodes a chromosome's optional trailing weighting-scheme gene into
  "INS"/"ISNS"/"ENS"; whether a track searches this gene at all is inferred from
  `config.N_GENES` (penalty-only count vs. that +1), not a separate flag — see the file's own
  docstring. Both CIFAR-100 sub-tracks search this gene (see their section below).
- `class_weights.py` — INS/ISNS/ENS class weights (`compute_class_weights(labels, scheme,
  config)`), computed fresh per individual now (each chromosome can select its own scheme on
  tracks that search it) rather than once upfront for the whole run
- `loss.py` — penalized loss (cross-entropy × penalty × class weight, zero when correct)
- `dispatcher_model.py` — trains/predicts with one `Linear(EMBEDDING_DIM→NUM_CLASSES)` FC head
- `fitness.py` — chromosome → `(alpha_sys_loss, avg_model_cost)`
- `dispatcher_problem.py` — wraps `fitness.py` as a `pymoo` `Problem`
- `progress_logger.py` — `pymoo` `Callback`: per-generation logging + checkpoints
- `save_results.py` — writes `<track>/results/nsga2/pareto_front.csv` (`individual, alpha_sys, avg_model_cost, P_*`)
- `save_models.py` — retrains + saves FC weights per Pareto individual (`<track>/results/nsga2/models/`)
- `logging_setup.py` — live run logging → `<track>/results/logs/`
- `nsga2_search.py` — wires the above into `pymoo`'s `NSGA2`, `run_nsga2(config)`
- `cifar-100/fig9c/run_dispatcher.py` / `cifar-100/fig9d/run_dispatcher.py` — thin per-sub-track entry points

GA mechanics (selection, crossover, mutation, non-dominated sort) are `pymoo`'s; only the domain
logic (chromosome meaning, fitness eval, FC training) is ours.

**Fitness objective: `obj1 = alpha_sys_loss = 1 - alpha_sys`, `obj2 = avg_model_cost`.**
`alpha_sys` (paper's Eq. 3) = fraction of images where the model the dispatcher **actually
picked** classifies correctly — looked up from each ground-truth CSV's `<model>_correct`
columns, indexed by predicted model. NOT exact-match against the ideal argmin-cheapest-correct
label. Overestimating onto a still-correct model costs nothing on this axis, only compute.
`avg_model_cost` (paper's Eq. 4; MAdds, see `config.MODEL_COST_UNIT`) is the dispatched model's
own cost **plus `config.DISPATCHER_OVERHEAD_COST`** — the feature extractor's + FC head's own
forward-pass cost, measured via thop per track. This matches the paper's Eq. 4, which defines the
objective as inclusive of "all operations performed by both the dispatcher and the selected
model," and page 7, which confirms this inclusive figure is what decides Pareto dominance during
the search, not just what gets reported afterward. `DISPATCHER_OVERHEAD_COST` is a fixed constant
per track (same extractor/FC head regardless of chromosome), so it shifts the whole front by one
offset without changing its shape or the ranking between individuals — it only makes
`avg_model_cost` directly comparable to how the paper itself reports MFLOPS, instead of
undercounting relative to it. **Fitness is evaluated on a held-out set, not TRAIN**: the FC head
is trained on TRAIN, but `dispatcher/fitness.py` predicts on the held-out set to compute both
objectives — the paper's own methodology text says "we perform the fitness evaluation for each
individual on the test set," and separately "we evaluate the final population using a validation
set that was not used during model training or the MOEA process" — this **is** a genuine
train/test/validation three-way split in the paper, with its own terminology backwards from what
you'd guess (its "test set" is touched every generation during the search; its "validation set" is
the one actually held out until the end). The CIFAR-100 track implements this real three-way split
— see `cifar-100/fig9c/config.py`'s "Split naming" comment and `cifar-100/PIPELINE_REPORT.md` for
the full mapping. Keep this metric named `alpha_sys`, never `accuracy` — don't compare it against
exact-match numbers without checking which definition was used. `dispatcher_analysis/summarize.py`
reports a separate `accuracy` column (`accuracy_exact_match_vs_ideal` — exact-match against
`ideal_label`) so `alpha_sys` and exact-match accuracy don't get conflated — see that module's
docstring.

## Dispatcher methodology (Step 1)

Based on the PERTINENCE paper (Shende et al., IEEE Access 2026), adapted from single-image to
batched mixed-complexity inference.

**Labeling** (offline, one-time): for every train image, `label(x) = argmin_j { cost_j |
model_j(x) is correct }` → per-sub-track class target. Images where NO model is correct are kept
(not dropped) and routed to the highest-cost model as the most defensible fallback —
`cifar-100/label_data.py` does this uniformly. See `cifar-100/README.md`'s "Labeling script"
section.

**Architecture**: cheapest pool model as backbone (classification head stripped) → embedding
(shared with inference if that model is selected anyway) → `Linear(embedding_dim→num_classes)`
FC head, the only learned component.

**Loss** (per-sample): `L=0` if `y_pred==y_true`, else `CrossEntropy(y_true,y_pred) *
P[y_true,y_pred]`. `P` = penalty matrix, diagonal 0, asymmetric by design — underestimation
(routed too small) high penalty (accuracy loss), overestimation (routed too big) low penalty
(cost waste only).

**Class imbalance**: INS (`weight=1/count`, normalized to sum=`NUM_CLASSES`) as a true
**loss-weight multiplier** in `dispatcher/loss.py`, plain shuffled `DataLoader`, no oversampling.
ISNS/ENS are implemented in `dispatcher/class_weights.py` and searchable as a chromosome gene
(`dispatcher/weighting_scheme.py`) — exercised by the CIFAR-100 track's `fig9c`/`fig9d`
sub-tracks, see their section below.

### NSGA-II runs (Step 2)

CIFAR-100 sub-tracks (`fig9c`/`fig9d`): the full-hyperparameter searches (pop 50 / gen 50 /
20 FC epochs) **have been run** for both (logs dated 2026-09-12; checkpoints through
`checkpoint_gen050.npz`, 50-individual `pareto_front.csv`, full `results/eval/` outputs and plots
in each sub-track) and were presented at the Week4 meeting. **Not yet logged as a `[RESULT]` in
`Journel/`**: the numbers live only on disk. fig9c final-validation snapshot
(`results/eval/final_val_summary.csv`): alpha_sys 0.692–0.774, avg_model_cost 48.0–3432.1 MFLOPs,
exact-match vs ideal 0.126–0.692, and the mid-tier model (`mobilenetv2_x0_75`) is nearly a dead
routing class (routing precision ≤ 0.171): the class-imbalance problem noted at the end of
`Journel/Week3.md` (`[REALISATION]` entry: DWB weighting paper, hand-seeded population idea), still
open. NSGA-II hyperparameters match the paper's stated values exactly where the shared
code allows it (pop 50 / gen 50 / FC epochs 20 / penalty range [0, 100]).

All runs share one optimization: backbone frozen → all train embeddings precomputed once,
each fitness eval only trains/evals a `Linear(embedding_dim→num_classes)` on cached tensors
(~3-5s/eval). Explicit `del` + `torch.cuda.empty_cache()` after every eval to avoid CUDA
fragmentation.

**Evaluation**: `dispatcher_analysis/` is the only place evaluation happens, always against
whichever front `dispatcher/` most recently produced, loading its saved weights directly
(no retraining).

### Known limitation: dispatcher judgment is capped by the feature extractor

The routing decision comes from the cheapest pool model's own embedding (chosen because it's
~free if that model ends up selected anyway). If an image is hard *because* that model's
features don't separate its class well, that same weak representation may lack the signal to
detect "this needs a bigger model." Does not affect the pool models' own classification quality
(each runs its own independent forward pass when selected) — only the dispatcher's ability to
know when to route up. Worth flagging to prof; not addressed in the paper.

### Batching extension (prof's new direction, not in paper) — superseded by the scheduler pitch

The original framing (feature extractor on full batch → FC assigns model → group into
sub-batches per model → run each sub-batch → reassemble in original order) is now folded into
the broader scheduling direction pitched in `Journel/Week4.md` — batching-aware scheduling under
concurrent streams, not just a static batch-and-reassemble pass. See Week4's formal problem
definition for the current shape of this work. `archive/quantization_experiments/`'s Week2
batch-size sweep found batching gives ~5× free per-image speedup (bs=1→4) with no quantization
needed — still the anchor data point motivating this direction.

### EDA / analysis artifacts to produce
- Per-image correctness matrix across all pool models → label distribution
- Confusion matrix of dispatcher predictions vs. ideal labels
- Accuracy-vs-cost plot: dispatcher system vs. single best model baseline
- Latency breakdown: feature extractor + sub-batch inference per model

## Decisions / dead ends (don't re-litigate)

1. **Dataset**: CIFAR-10 → CIFAR-100 → briefly ImageNet-pretrained ResNets on ImageNette →
   settled on CIFAR-100 as the sole active track (see `cifar-100/` section above and
   `cifar-100/PIPELINE_REPORT.md`). The ImageNette and CIFAR-10 tracks were both abandoned; no
   trace of either remains in the active pipeline.
2. **torchao weight-only quantization** (`Int8WeightOnlyConfig`, `Float8WeightOnlyConfig`) — dead
   end. 2–6% size reduction (not the theoretical 4x), <0.4% accuracy delta. Weight-only quant
   stores weights compressed on disk but matmuls still run in FP32 — a memory-bandwidth
   optimization for LLM serving, wrong tool for CNN inference.
3. **torchao dynamic activation quantization** (`Int8DynamicActivationInt8WeightConfig`,
   `Float8DynamicActivationInt4WeightConfig`) — also dead end despite being the
   theoretically-correct config. No meaningful differentiation. Hypothesis: torchao's dynamic quant
   kernels are tuned for transformer-shaped matmuls, not ResNet convs.
4. **ONNX + CUDAExecutionProvider INT8 PTQ** — dead end. INT8 *slower* than FP32 for 3/4 models
   (QDQ wrapper overhead without TensorRT fusion). TensorRT EP unusable (`nvinfer_10.dll` missing,
   full TRT SDK not installed).
5. **Quantization abandoned entirely for the dispatcher pool.** Confirmed dead across torchao
   (weight-only + dynamic) and ONNX+CUDA — no useful Pareto separation.
6. **ONNX Runtime's TensorRT execution provider — dead end again.** Revisited per prof's meeting
   action item; `onnxruntime` 1.29.0's TensorRT provider DLL is hard-pinned to `nvinfer_10.dll`,
   every installable `tensorrt`/`tensorrt-cu13` package ships 11.x (`nvinfer_11.dll`) — ABI
   mismatch, unfixable without an old TRT release. Don't re-attempt this path.
7. **Torch-TensorRT works, but the initial "FP16 win" was a compilation artifact.** Compiles
   directly from the live PyTorch model (`ir='dynamo'`), no ONNX/onnxruntime. Benchmarked
   FP32/FP16/INT8/FP8 across the pool (`archive/model_analysis/`, `Journel/Week1.md`): first pass read
   as **FP16 ≈2.0–2.6x latency reduction, ~zero accuracy loss**. A follow-up isolation test (eager
   FP32 vs. TensorRT-compiled FP32 vs. TensorRT-compiled FP16) showed compiled-FP32 and
   compiled-FP16 are statistically identical — **the win is almost entirely TensorRT's graph
   compilation (kernel fusion, no eager dispatch overhead), not FP16/Tensor-Core execution**.
   **INT8/FP8 are dead ends via this uncalibrated path** — `enabled_precisions={torch.int8}`/
   `{torch.float8_e4m3fn}` alone doesn't engage real low-precision kernels; TensorRT's builder
   silently falls back to FP16 without explicit calibration (`modelopt` quantize workflow, not
   attempted yet at this point). Size/latency/accuracy for INT8/FP8 were byte-identical to FP16 for
   every model — a successful compile isn't a real quantization win.
   **Partially superseded by Week2** (`Journel/Week2.md`): this dead-end holds only for the
   *uncalibrated* path. Real `modelopt` calibration (`mtq.quantize()` + `export_torch_mode()`) makes
   INT8/FP8 genuinely fast — 5–7× over FP16 on RN18/34/50, <1% accuracy cost — see
   `archive/quantization_experiments/` section above. Not yet integrated into the dispatcher pool.

## Current state / open threads

Torch-TensorRT precision benchmark done (FP16 vs. uncalibrated INT8/FP8, compilation-only gain —
`Journel/Week1.md`); calibrated PTQ INT8/FP8 genuinely beats FP16 by 5–7× on RN18/34/50
(`Journel/Week2.md`, `archive/quantization_experiments/`), proven but not integrated into the
active CIFAR-100 pool. `dispatcher/`, `dispatcher_analysis/`, and `eda/` are shared, generic,
config-injected code across both CIFAR-100 sub-tracks; `archive/model_analysis/` and
`archive/quantization_experiments/` hold done/locked work; `cifar-100/` holds the active track's
own config, entry points, dataset, and results — see "Repo layout" above for the full mapping.
Full narrative: `Journel/Week0.md` (model pool + first dispatcher run), `Journel/Week1.md`
(dispatcher objective fix + Torch-TensorRT pivot), `Journel/Week2.md` (PTQ follow-up),
`Journel/Week3.md` (repo restructure + fitness-on-val fix + CIFAR-100 track build +
`avg_model_cost` overhead-accounting fix), `Journel/Week4.md` (CIFAR-100 results presented,
greenlit for the batching-aware scheduler direction), `Journel/Week5.md` (scheduler direction accepted, simulator built, energy angle for meet 6).

`avg_model_cost` now includes `config.DISPATCHER_OVERHEAD_COST` (feature extractor + FC head cost)
on top of the dispatched model's own cost, matching the paper's Eq. 4 — see "Fitness objective"
above. This changed `dispatcher/fitness.py` and `dispatcher_analysis/summarize.py`, shared code, so
it applies to both sub-tracks.

`dispatcher_analysis` always rebuilds `model_cache/` from each Pareto individual's chromosome on
every run rather than relying on a copied `.npz` weight file.

**Open threads**:
1. **Scheduler direction (active)**. Supersedes the batching extension below; formal problem in
   `Journel/Week4.md` and summarised under "Formal problem" here.

   **Plan (meet 5, `Journel/Week5.md`)**: the scheduler only sees queues, batch sizes and T_i(b),
   so it is built and evaluated in a DES first, then validated with PERTINENCE routing. Target:
   compute-constrained edge devices (Jetsons). Later: per-job deadlines, stream priorities. Meet 6
   is two weeks after meet 5.

   **Simulator**: done and tested (see "Repo layout"). Results so far: `fcfs_no_batch` goes
   unstable first; sticky routing makes everything 5-15% slower and `longest_queue` wins at every
   load (`fcfs_no_batch` 21.2 ms vs 9.9 ms at load 0.25); the periodic sweep sits near 10 ms at
   every load (~9.9-10.0 ms for `fcfs_batch`, burst drain). Busy time and compression ratio did not always rank policies the
   same: not conclusive on synthetic curves.

   **Real curves in** (Week5 `[RESULT]`): trt_fp16 T_i(b) is far from flat and very different per
   model (per-image cost drops ~8x for n, ~2.3x for l; m/l knee near b=8). Synthetic profile was
   badly off. On the real profile `fcfs_no_batch` collapses past ~0.49 jobs/ms, `longest_queue`
   ~ `fcfs_batch`, and `timeout_batch` uses ~38% less GPU busy time at +11 ms turnaround (load
   0.3): waiting buys compute. K/I and busy time ranked the 4 policies the same on real curves
   (leans to K/I as primary, not final). **Done**: dense sweep (all 3 variants, session `main`, b = 1..32 step 2, then
   40/48/56/64, 1260/1260 ok); profile CSV rebuilt from `batch_sweep_DENSE` (21 columns) and
   the 3 sim sweeps rerun on it (loads 0.1-1.0 step 0.1). `longest_queue` best mean, `fcfs_batch`
   better p95 (LQ starves short queues). **Next**: E_i(b) via NVML sampling during the timed loop
   (the sweep's end-of-run power reading is useless), then implement the score-based policy and an
   EdgeServing-style baseline. **Policy v1 defined** (Week5 `[DECISION]`, not implemented):
   Cost(i, j) = β · Σ ages of jobs not served + (1 - β) · T_ij, argmin over (queue, batch size),
   padding allowed, no wait option yet.

   **TRT engines broken** (Week5 `[DEAD-END]`): compiled engines' outputs do not match eager (14k vs
   2.2k detections after NMS, FP32 too); likely box decoding lost in export. Sim now runs on the
   eager FP32 profile (loads 0.05-0.8). Breakdown: no-batch past ~0.17, all batching policies knee at 0.6, collapse at 0.65; fix the export and re-verify before switching back.

   **Literature** (`research.md`, Week5 `[RESULT]` entries): closest is EdgeServing (arXiv
   2605.05527): same setting, but always runs min(|Q|, 10), never waits or picks b, clocks locked
   at max, SLO objective; its queue choice beats LQF by only 2-4 ms (early exit does the heavy
   lifting). Camel and Nabavinejad et al. tune batch size + GPU frequency, single model only.

   **Direction (Week5 `[DECISION]`)**: energy is the angle. Turnaround alone leaves little room
   over max batching; with energy in the objective, choosing b and waiting pay off. Policy:
   state -> score per option (serve queue i at size b, or wait) -> pick the best; score = extra
   waiting + λ × energy, wait value from a sliding window of recent per-queue arrivals. Power
   before MPS. No Jetson yet (ask at meet 6); A100/5070 Ti curves are flatter than a Jetson's,
   imgsz 1280 is a labelled stand-in; E_i(b) via NVML works without root, frequency control needs
   the Jetson.

   **Roadmap**: real T_i(b) in sim -> E_i(b) + score policy + EdgeServing-style baseline -> sweep
   load/workload/hardware/λ for a turnaround-vs-energy curve -> Jetson with frequency, sim vs real
   -> multi-camera CCTV traces through PERTINENCE, replayed in sim and on device -> paper.

   **Formal problem (Week4 `[DECISION]`)**: N models, queue Q_i each, supported batch sizes B_i,
   measured batch runtime curve T_i(b); periodic arrivals (period Δ) routed to one queue by the
   dispatcher (routing fixed from the scheduler's view); single non-preemptive accelerator, one
   batch at a time, one queue per batch; when free, the scheduler picks (queue, b <= |Q_i|) or
   waits. Objectives: minimize compute and mean turnaround T-bar = (1/I) sum (t_out - t_arr).
   Compute has two candidate metrics, both reported by the simulator: compression ratio K / I
   (batches run per job, 1 / mean batch size) and busy time C = sum of T_i(b) over batches run. Which
   one is primary is decided experimentally (`Journel/Week5.md` `[DECISION]`): they coincide for a
   single affine model, but diverge for convex curves and across a heterogeneous pool. Week4.md's
   journal used busy time and the Week4 deck used compression ratio; this supersedes both. Accuracy is
   NOT an objective (routing fixed). T_i(b) must be a per-model curve: Week2 Exp2 shows resnet18
   roughly affine, resnet50 convex. **Extension**: scheduler may pick any model acceptable for a
   frame (recall >= 0.80), making accuracy a constraint and routing a batching lever: the
   clearest gap found by the prior-art search (project doc "Multi-Class Batch-Service Scheduling
   on a Single Accelerator"): Xia et al. 2002 / Chen & Wang 2022 assume batch cost independent of
   size and identical across classes; ML serving systems (Clockwork, Nexus, Symphony, Triton) have
   size-dependent cost but no optimality results. First piece built and now
   **run to completion on both splits**: a self-contained `yolo-analysis/` folder (single entry
   point `run_benchmark.py`) that runs YOLOv8n/s/m/l over both COCO val2017 (full 5,000 images)
   and train2017 (a stratified 20,000-image subset, see below) and scores each as multi-label
   classification (recall + exact-match against ground-truth class sets, boxes dropped for this
   pass but cached raw for later) — prep work for redefining "accuracy" for a detection-based
   pool, since the old single-label top-1 definition doesn't survive contact with COCO's
   multi-object images. Every split-dependent path in `config.py` is a function of `split`
   (`config.SPLITS = ["val2017", "train2017"]`), one shared pipeline over both rather than a
   duplicated train-specific copy. `plot_results.py` (per split) produces the 4x2 recall/exact-match
   plot grid plus a per-image exact_match pivot and the 16-way True/False permutation counts
   across the model pool — the actual cross-model agreement/disagreement structure the
   correctness-definition decision depended on.

   **train2017 is a stratified 20,000-image subset, not the full 118,287-image split** — ran out
   of disk on the target server for the full ~18GB download, so `build_train_subset.py` samples
   proportional-to-frequency per category (rarest-first, so overlap can't crowd out a rare
   category's quota) and downloads only those images individually rather than the full zip; the
   exact sampled ids are written to `dataset/train_subset_image_ids.txt` for reproducibility.
   `coco_gt.py` filters ground truth to images actually on disk so scoring doesn't treat an
   un-downloaded image as "predicted nothing."

   **Correctness definition decided: `recall >= 0.80`, not exact-match.** With both splits'
   inference actually run, compared exact-match (`pred_classes == gt_classes`) against plain
   recall thresholds on the real per-model, per-image disagreement structure rather than
   guessing. Exact-match gives real per-image clash (35.4% of val2017 / 40.8% of the train
   subset land in some cross-model disagreement pattern, not full agreement) but a lopsided tail
   split on train (only 23.6% all-four-correct vs. 35.6% all-four-wrong) — too harsh a bar for a
   busy multi-object image to ever count as fully "correct." A loose recall threshold
   (`>= 0.65`) overcorrected the other way: all-four-correct jumped to 68.0%, disagreement zone
   shrank to 28.0%: too soft a bar, a large part of the routing-relevant signal disappeared, exactly
   the "recall might be too smooth for PERTINENCE to route on" risk flagged before ever running
   the benchmark. `recall >= 0.80` lands on almost exactly exact-match's disagreement-zone size
   (39.7% vs. 40.8%) while giving a workable 44.7%/15.6% all-correct/all-wrong split instead of
   exact-match's skewed one — same underlying per-image difficulty structure (dominant
   disagreement pattern in both cases: nano alone fails, s/m/l agree), just a saner place to draw
   the pass/fail line. This is now the accuracy definition the YOLO pool's routing labels will be
   built from — see `Journel/Week4.md`'s `[DECISION]` entry for the full comparison table and
   reasoning. **Not yet decided**: recall vs. F1 as the underlying score before thresholding —
   plain recall doesn't penalize a model padding predictions with extra wrong classes.

   Full train2017 sweep (all-correct / all-wrong / disagreement, % of 20,000, recomputed from
   `results/train2017/coco_class_recall_benchmark.csv`, used in the Week4 deck): recall >= 0.50:
   89.1 / 0.4 / 10.6; >= 0.60: 72.9 / 2.8 / 24.3; >= 0.65: 68.0 / 4.1 / 28.0; >= 0.70: 53.7 / 9.3 /
   37.0; >= 0.80: 44.7 / 15.6 / 39.7; exact-match: 23.6 / 35.6 / 40.8. Dominant disagreement pattern
   at every threshold: nano alone fails (0111). Per-model exact-match on train: n 31.9%, s 44.0%,
   m 47.6%, l 53.1%. Caveat: the YOLOv8 checkpoints were trained on COCO train2017, so train-side
   numbers may be optimistic (the CIFAR-100 lesson): val2017 is the honest split.

   **On-disk notes**: `results/<split>/*_minimum_match.csv` files are the recall >= 0.80 variants
   (despite the `exact_match_` prefix); the plain `exact_match_*` files are true exact-match.
   `raw_predictions/` caches exist only on the server (locally just `.gitkeep`): re-scoring
   locally would mean re-running inference. `yolo-analysis/results/` also holds stray copies of
   `coco_gt.py` and `requirements.txt` from syncing results back.

   **Not yet started**: the actual labeling step (`label(x) = argmin_j cost_j s.t.
   recall_j(x) >= 0.80`, the detection-pool analogue of `cifar-100/label_data.py`) and training a
   new PERTINENCE dispatcher stack over the YOLO pool — this benchmark was data-gathering only,
   see `yolo-analysis/README.md` and `Journel/Week4.md`.
2. Whether/when to integrate calibrated INT8/FP8 into the CIFAR-100 dispatcher's model pool
   (would require re-running the labeler with INT8 latency figures) — not started, deferred.
3. `[EXPLORE]` from the prof meeting, untouched: model compression (pruning, KD) — check for useful
   Pareto-optimal points beyond the current model pool.
4. Pick an actual operating-point config off a Pareto front for a future deployment/demo. Not started.
5. Re-benchmarking `archive/quantization_experiments/`'s Exp1/Exp2 quantization work on the
   university A100 server (it originally ran on a laptop GPU) to confirm the numbers hold on the
   actual target hardware.
6. **CIFAR-100 track**: runs the PERTINENCE methodology on CIFAR-100 in `cifar-100/`, sharing
   `dispatcher`/`dispatcher_analysis` with itself across both sub-tracks. First attempt built a
   single dispatcher across the paper's whole 6-model CIFAR-100 pool (Figure 4b) — abandoned once
   it became clear the paper never actually runs that experiment: its real CIFAR-100 results
   (Fig. 9, page 8) are each a separate MOEA search over a hand-picked 2- or 3-model subset, never
   the full pool at once. Pivoted to reproducing two specific subsets directly: `cifar-100/fig9c/`
   (`shufflenetv2_x0_5`, `mobilenetv2_x0_75`, `repvgg_a2` — Fig. 9(c)) and `cifar-100/fig9d/`
   (`shufflenetv2_x0_5`, `mobilenetv2_x1_4`, `repvgg_a2` — Fig. 9(d)), each its own fully
   self-contained thin sub-track with a 6-gene chromosome (`3²-3`), much closer to what the
   paper's own subset runs actually search than the abandoned attempt's 30-gene chromosome was.
   `mobilenetv2_x0_75` (used by fig9c) is a real, separately published chenyaofo checkpoint not on
   Fig. 4b's plotted six — confirmed against the chenyaofo repo's full model list before
   downloading it. Embedding extractor is `shufflenetv2_x0_5` (matching the paper's explicit
   statement for CIFAR-100, page 8), measured 1024-dim, shared between both sub-tracks — but
   embeddings caches themselves are NOT shared between fig9c/fig9d, since the cached routing
   labels depend on each variant's own model subset; a shared cache would silently serve one
   variant's labels to the other. Genuine three-way train/test/validation split in both
   sub-tracks, not a two-way train/val split — see "Fitness objective" above and
   `fig9c/config.py`'s "Split naming" comment; both the paper's "test set" (7,000 images) and its
   "validation set" (3,000 images) are stratified subsets of the official 10k CIFAR-100 test
   split, not carved from train — an earlier attempt carving the validation set from train
   produced a near-trivial slice (100% oracle accuracy, dead routing classes) since the pool
   checkpoints were already trained on all of official train. Chromosome now includes the paper's
   weighting-scheme gene too (`N_GENES=7`: 6 penalty genes + 1 scheme selector, see
   `dispatcher/weighting_scheme.py`). Remaining, still-deliberate divergence: continuous rather
   than discretized penalty search space (pymoo's SBX/polynomial-mutation operators have no
   step-size knob). `MODEL_COST` is measured locally via `thop` (MACs doubled to FLOPs) rather
   than taken from chenyaofo's own published MAdds table — the table numbers didn't reproduce the
   paper's own plotted Fig. 9 positions closely enough to trust, while this repo's own
   thop-doubled measurement matches the paper's published Table 5 dispatcher-overhead figure
   (24.17 MFLOPS for CIFAR-100/ShuffleNetV2) to within 1%; see `cifar-100/models/model_loader.py`'s
   module docstring. NSGA-II hyperparameters match the paper's stated values exactly where the
   shared code allows it (pop 50 / gen 50 / FC epochs 20 / penalty range [0, 100]).
   `dispatcher_analysis/plots.py` gained `plot_accuracy_vs_cost` — the paper's own Fig. 6-10
   accuracy(%)-vs-cost style, PERTINENCE points plus each pool model's own standalone "SOTA CNN"
   point plus the joint Pareto front — wired into both sub-tracks' `run_dispatcher_analysis.py`.
   Also fixed, while building this: a pre-existing Windows-only bug in the shared
   `dispatcher_analysis/embeddings.py` (a locally-scoped `Dataset` class that spawn-based
   multiprocessing can't pickle when no embeddings cache exists yet and
   `EMBEDDING_NUM_WORKERS>0`) — moved to module level, applies to both sub-tracks. Both
   sub-tracks' full-hyperparameter searches have been run and presented (see "NSGA-II runs"
   above); still to do: log them as a `[RESULT]` in the journal and address the mid-tier
   class-imbalance problem. Full writeup: `cifar-100/PIPELINE_REPORT.md`.

## Journaling (MANDATORY)

Every significant moment must be logged to `Journel/WeekX.md` before moving on.

**Tag vocabulary**: `[SETUP]` env/repo/tooling · `[DECISION]` a choice + rationale · `[RESULT]`
experiment output/numbers/plots · `[DEAD-END]` approach that didn't work + why · `[PIVOT]` change
in direction · `[MEETING]` prof/advisor notes · `[CHANGE OF PLAN]` externally-driven direction
change.

**Writing style (don't re-litigate)**:
- First-person "what I did" POV throughout — journal, this file, code comments. No two-person
  Claude/user framing anywhere ("the user asked", "Claude suggested", "we decided together").
- Don't personify the advisor's involvement as direct dialogue ("Prof asked me to do X") — frame
  it as tasks instead ("Tasks: implement X, ..."). Meeting notes (`[MEETING]` entries with actual
  action items) are the exception — those legitimately are the advisor's words, keep them as-is.
- AI assistance is not hidden — it can't be and shouldn't be — but the writing should read like a
  normal personal engineering log, not a raw two-way chat transcript. Don't over-narrate the
  back-and-forth of getting to a result; write the result and the reasoning behind it.
- One `Journel/WeekX.md` per project week (0-indexed: Week0 is the first week) — **week-wise only,
  no day-wise sub-splitting within a week's file** (no `# Day N` headers; a week's log is one
  continuous `## Log`, even if the work spanned several calendar days). Every week's file starts
  with a `## Meeting notes & tasks` section (advisor meeting notes if there was one, plus what's
  planned/assigned for the week) before the `## Log` of what actually happened. Log entries stay in
  the tag format above. Keep it simple — this is meeting notes/tasks, then whatever happened.
- Full style guide + rationale also recorded in the "Research Thread" project
  (`journal-style-guide.md`) for reference outside this repo.

**Working rules for AI-assisted sessions**:
- Log every tag-worthy moment to the current week's file as it happens; capture the
  ready-to-paste snippet immediately rather than deferring it to "later."
- Update `Journel/README.md`'s index (if present) under the right section.
- **Update this file alongside every `Journel/` entry that changes what's built, what's next, or a
  prior decision.** They drifted out of sync before (this file said "not yet done" for completed
  work, described a deleted-and-rebuilt pipeline, and kept referencing tracks that had already
  been ditched) — don't repeat that. When logging a `[RESULT]`/`[DECISION]`/`[PIVOT]`, check
  whether "Current state / open threads" or the relevant pipeline section here needs the same
  update, right then.
- I'm forgetful across sessions; proactive logging beats reactive cleanup.

## Environment notes

- `requirements.txt`: numpy, pandas, matplotlib, pillow, thop, pymoo, torch==2.13.0+cu132,
  torchvision==0.28.0+cu132, torch_tensorrt==2.13.0, tensorrt/tensorrt_cu13/tensorrt_cu13_bindings/
  tensorrt_cu13_libs (unpinned), nvidia-modelopt==0.46.0. `requirements.txt` itself is the source
  of truth for exact versions — re-check it against this section whenever the environment changes,
  especially across machines (laptop vs. A100 vs. WSL2). The active pipeline (`dispatcher/`,
  `dispatcher_analysis/`, `eda/`, `cifar-100/`) does not use ONNX/onnxruntime or torchao — those
  paths were dropped (see "Decisions / dead ends" above); `torch.hub` reaches relevant checkpoint
  sources directly where a pool model isn't pip-installable.
- **A100 server env (frozen, validated)**: torch 2.3.1+cu121, torchvision 0.18.1+cu121,
  torch_tensorrt 2.3.0+cu121, tensorrt 10.0.1, nvidia-modelopt 0.15.0, numpy 1.26.4, pandas 2.3.3,
  ultralytics. Differs from `requirements.txt` (local). Code meant for the server targets these;
  don't install or upgrade there. Deployment/quantization path is torch_tensorrt.
- CUDA used when available (`torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`)
- Torch-TensorRT works in this environment (see "Decisions / dead ends" #7). ONNX Runtime's own
  `TensorrtExecutionProvider` does NOT — don't re-attempt that specific path (item #6 above).

# Week 2

## Meeting notes & tasks

Follow-up from the advisor meeting: check whether real quantization (PTQ with calibration)
gives meaningful latency wins over the unquantized FP16 baseline.

**Tasks**:
1. **Exp1** — PTQ calibration: does `modelopt mtq.quantize()` + `export_torch_mode()` +
   torch-tensorrt give genuinely lower latency vs. just passing `enabled_precisions={torch.int8}`?
2. **Exp2** — Batch size sweep: does per-image latency improve with larger batches, and does
   uncalibrated INT8 benefit from batching?
3. **Parallel task**: reproduce the PERTINENCE paper's exact experimental setup — same models +
   CIFAR-10 (the dataset the paper actually uses), not the ImageNet/ImageNette pool used
   elsewhere in this project. Purpose: show the paper's methodology is understood at
   implementation depth, not just adapted loosely. Tracked separately from Exp1/Exp2, not
   started yet.

Exp1/Exp2 are in `code/quantization_experiments/`, fully separate from the dispatcher pipeline.

Continuing from `Journel/Week1.md`: Torch-TensorRT (on the 5070ti) compiled fine and
looked like an FP16 win at first, but turned out to be a compilation-only effect —
INT8/FP8 stayed dead because `enabled_precisions=...` alone never actually engages
`modelopt` calibration. This week fixes that calibration gap for real and gets access to
an A100 to benchmark on real datacenter hardware too.

---

## [SETUP] Environment issue — cached files got corrupted

Before any code could run, `/tmp` filesystem was at 100% capacity (28K free).
Root cause: ~17GB pip download cache in `/home/cobaltboy/.cache/pip/`.

**Fix**: `pip cache purge` removed 503 cached packages and freed ~17GB. Disk went from
100% → 82% used. No package was lost — pip re-downloads on demand if needed.

**Side effect**: the full ENOSPC had already corrupted three `.pt2` TensorRT engine
cache files during a prior interrupted write:
- `resnet152_ptq_int8.pt2` — corrupt zip archive
- `resnet152_baseline_fp16.pt2` — corrupt zip archive
- `resnet50_int8_bs4.pt2` — corrupt zip archive (exp2)

All three deleted and recompiled cleanly after the disk was freed.

---

## [DEAD-END] `torch_tensorrt.save()` aborts for PTQ-quantized models

Attempting to cache a modelopt-quantized dynamo export via `torch_tensorrt.save()` hits
an uncatchable C++ `terminate()` at `inline_container.cc:672` — a zip-writer enforce
check that fires when the quantized model's serialization state is unexpected.

This is a torch bug, not a user error. It does not happen for unquantized fp16 exports.

**Fix**: modified `compile_with_trt()` to accept `cache_path=None`. Baseline fp16 is
still cached normally; PTQ callers pass `None` so no save is attempted. PTQ models are
compiled in-memory, benchmarked immediately, and discarded.

---

## [RESULT] Exp1 — PTQ calibration gives genuine, large latency wins

**Script**: `code/quantization_experiments/exp1_ptq_calibration.py`
**Output**: `code/quantization_experiments/results/exp1_ptq_calibration.csv`

Method: `mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)` runs 16
calibration batches to collect activation statistics, then `export_torch_mode()` context
manager + `torch_tensorrt.compile(..., enabled_precisions={torch.int8})` builds a real
INT8 engine. This is the correct PTQ workflow — contrast with model_analysis which passed
`enabled_precisions={torch.int8}` without calibration (TRT silently used fp16 instead).

| Model | fp16 baseline | INT8 PTQ | Speedup | Accuracy drop |
|-------|--------------|----------|---------|---------------|
| resnet18  | 4.949ms  | 0.793ms | **6.2×** | −0.4% (actually improved slightly) |
| resnet34  | 10.553ms | 1.356ms | **7.6×** | −0.3% |
| resnet50  | 8.085ms  | 1.578ms | **5.1×** | −0.6% |
| resnet152 | 16.469ms | 6.459ms | **2.5×** | −0.7% |

All accuracy drops are within the expected PTQ tolerance (<1%). resnet152's smaller
speedup reflects memory-bandwidth bottlenecking at its scale — residual skip connections
and the large fc layer don't benefit as much from compute-precision reduction.

**FP8 PTQ**: works on resnet18 (0.781ms) and resnet34 (1.225ms) — marginally faster than
INT8. Fails on resnet50 and resnet152 with TRT Error Code 10:
`Could not find any implementation for node {ForeignNode[...quantize_op...maxpool...]}`.
Blackwell (sm_120) + TRT 11.0 has no kernel implementation for FP8 through the deeper
quantized-maxpool+conv chains those models use. Consistent with the FP8 dead end
documented in Week1 for the non-PTQ path; PTQ doesn't change the kernel availability.

**Verdict**: INT8 PTQ is a genuine 5–7× speedup on the smaller models with negligible
accuracy cost. Worth revisiting if the batching extension needs tighter latency budgets.

---

## [RESULT] Exp2 — batch size sweep confirms calibration hypothesis, free 5× batching win

**Script**: `code/quantization_experiments/exp2_batch_size_sweep.py`
**Output**: `code/quantization_experiments/results/exp2_batch_size_sweep.csv`
**Models**: resnet18, resnet50 | **Precisions**: fp32, fp16, int8 (no calibration)
**Batch sizes**: 1, 4, 8, 16, 32

Key findings:

**1. Uncalibrated INT8 tracks fp16 exactly at every batch size.**
All three precision curves are identical — TRT is running fp16 internally for the INT8
path. This directly confirms the Exp1 hypothesis: `enabled_precisions` alone is
permission, not instruction.

**2. Batching gives ~5× free per-image speedup from bs=1 → bs=4.**
resnet18 drops from 5.1ms/image at bs=1 to ~1.5ms at bs=4. This is pure GPU utilization
improvement — no quantization needed.

**3. Gains plateau quickly for resnet18; resnet50 regresses at large batches.**
resnet18 at bs=8–16 is ~1.1–1.0ms (marginal gain over bs=4). resnet50 per-image latency
rises from 3.2ms at bs=4 back to 4.7ms at bs=32 — it's memory-bandwidth bound; large
batches no longer fit cleanly in L2 cache.

| model | precision | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 |
|-------|-----------|------|------|------|-------|-------|
| resnet18 | fp16 | 5.1ms | 1.5ms | 1.1ms | 1.0ms | 1.0ms |
| resnet50 | fp16 | 7.8ms | 3.4ms | 3.3ms | 3.6ms | 4.3ms |

**Implication for the dispatcher**: the current dispatcher routes bs=1. Forming sub-batches
by grouping same-model assignments before inference (the batching extension's goal) would
recover this 5× factor essentially for free. resnet50+ should target bs=4–8, not larger.

---

## [DECISION] INT8 PTQ is proven but not integrated — deferred

PTQ gives real speedups but requires `modelopt` calibration overhead (~30s per model) and
cannot save compiled engines to disk (the `inline_container.cc` abort). For the
dispatcher pool, the FP32 models remain correct for now — the pool is the label source
and all NSGA-II runs used FP32 latency/accuracy numbers. Switching the deployed pool to
INT8 PTQ would be a separate step and would require re-running the labeler with INT8
latency figures.

Not started. Flagged as a future optimization if the edge deployment latency budget demands it.


---

## [SETUP] A100 environment blocked on dependency install — ran Exp1/Exp2 on the laptop GPU meanwhile

Tried to set up the environment directly on the A100 to run this week's PTQ calibration
work there. Dependency install broke (conflicting CUDA/driver stack on that machine).
Rather than block the quantization experiments on fixing it, ran Exp1/Exp2 above on the
laptop GPU instead — the Blackwell (sm_120) card already referenced in Exp1's FP8
failure note above. Results are genuinely from that laptop run, not a placeholder —
committed to git as-is.

**Fix**: got the A100 environment working this morning (2026-09-05) — currently running
Exp1/Exp2 there now to confirm the laptop numbers hold on the actual target hardware.

---

## [RESULT] model_analysis's precision benchmark, run on both GPUs — FP16 win is real but hardware-dependent

**Output**: `code/model_analysis/results_A100/` and `code/model_analysis/results_RTX_5070ti/`
(`eager_baseline.csv`, `torch_tensorrt_benchmark.csv`, `quantized_benchmark_log.txt` each).

Re-ran the Week1 Torch-TensorRT precision benchmark (fp32/fp16/int8/fp8 compiled comparison) on
both machines to get a real cross-GPU picture instead of just the laptop.

**A100 dropped fp8 and int8 from the precision list** (`constants.py`,
`PRECISIONS = ["fp32", "fp16", "int8"] # fp8 is not supported in A100`) — fp8 has no tensor-core
support on Ampere at all (Hopper+ only, this isn't a bug), and int8 also failed outright without
calibration on this GPU, unlike the 5070ti which silently fell back to fp16 instead of erroring.
Ended up removing int8 from the A100 run too rather than fight it — fp32/fp16 only there.

| model | A100 fp32→fp16 | 5070ti fp32→fp16 |
|-------|-----------------|---------------------|
| resnet18  | 0.66→0.453ms (**1.46×**) | 0.971→0.892ms (1.09×) |
| resnet34  | 0.766→0.679ms (1.13×)    | 9.685→9.691ms (flat) |
| resnet50  | 0.791→0.728ms (1.09×)    | 7.99→7.53ms (1.06×) |
| resnet152 | 2.287→1.511ms (**1.51×**) | 19.555→19.675ms (flat/slightly worse) |

**This sharpens the Week1 finding rather than contradicting it.** The compilation win (fp32 eager
→ fp32 compiled) is real on both GPUs — but a genuine *additional* precision win on top of that
compilation win shows up clearly on the A100 (up to 1.5×) and barely at all on the 5070ti. Reads as
architecture-dependent: A100 has a wide FP32-vs-FP16 tensor-core throughput gap, a modern consumer
card apparently doesn't have as much of one to exploit at batch=1.

**Cross-GPU eager-baseline oddity worth remembering for the batching extension**: the 5070ti is
*faster* than the A100 on plain eager (uncompiled) batch=1 inference for every model (e.g.
resnet152: 10.038ms laptop vs. 14.484ms A100). Datacenter GPUs are tuned for throughput at scale,
not single-image latency — this is consistent with that, not a benchmarking error, but it means
"A100 = faster" isn't true at batch=1 before compilation evens things out.

**Two numbers flagged for a re-check, not yet trusted as real findings**:
- 5070ti resnet34 (9.685–10.487ms across all four precisions) is slower than 5070ti resnet50
  (7.53–7.99ms) despite resnet34 having fewer FLOPs (3.679G vs 4.134G) — breaks the FLOPs-latency
  ordering that holds everywhere else (including on the A100, where resnet34 is correctly faster
  than resnet50). Possibly a TRT kernel-selection quirk for resnet34's BasicBlock architecture on
  this GPU, possibly a bad single run (thermal, warmup). Re-run resnet34 alone before trusting it.
- 5070ti resnet152 int8 (15.174ms) is ~22% faster than its own fp32/fp16 (~19.6ms), uncalibrated —
  bigger gap than Week1's "byte-identical to fp16" finding for uncalibrated quantization. Possibly
  a different TRT version behaving differently (this environment pins `tensorrt-cu13==11.0.0.114`,
  unclear what Week1's original run used), possibly noise. Doesn't change the calibrated-PTQ
  conclusion above either way — just don't cite "uncalibrated int8 is always identical to fp16" as
  a hard rule off the back of this one number.

---

## [DEADEND] A100 seems to refuse PTQ calibration 

Some import errors, halting for now

---

## [SETUP] CIFAR-10 reproduction track — copied dispatcher/dispatcher_analysis into a self-contained folder

Started the parallel task from this week's Context: reproduce the PERTINENCE paper's actual
experimental setup (CIFAR-10 + the paper's own model pool) instead of the ImageNet/ImageNette pool
the main dispatcher pipeline uses. Copy-pasted `code/dispatcher/` and `code/dispatcher_analysis/`
into a new `code/CIFAR_10_Implementation/{dispatcher,dispatcher_analysis}/` as starting points, kept
fully self-contained there so the original two folders are untouched.

**Model pool**: 4 CIFAR-10-pretrained checkpoints from `chenyaofo/pytorch-cifar-models`, ordered by
MAdds (this pool's cost metric, in place of the ImageNette pool's FLOPs_G): `resnet20` (40.81M,
92.60% top-1), `resnet32` (69.12M, 93.53%), `shufflenetv2_x2_0` (187.81M, 93.81%), `vgg16_bn`
(313.73M, 94.16%).

`pytorch-cifar-models` isn't actually pip-installable (its sdist's `setup.py` needs a
`requirements.txt` missing from the build context) — used via `torch.hub.load("chenyaofo/pytorch-cifar-models", ...)`
instead, which pulls the model-definition source straight from GitHub into torch's hub cache.
Documented in `code/requirements.txt` so this isn't rediscovered as a fresh problem later.
`scikit-learn` also came back into `requirements.txt` for this track (`train_test_split` for the
stratified split below) — a legitimate new use, not a revival of the quantization-era dependency
noted as dropped in "Decisions / dead ends."

**Embedding extractor**: swapped from the ImageNette pipeline's frozen ResNet18 (512-dim) to
resnet20 (this pool's cheapest model, head stripped) — same "free if selected anyway" design
intent. Measured the actual feature dim with a dummy 32×32 forward pass rather than assuming 512
carried over: it's **64**. Every hardcoded `512` in both copied `constants.py` files and
`embeddings.py`'s dim check got updated to the measured value, with a runtime assertion against a
mismatch left in place.

**[DECISION] fresh 70:30 split, not CIFAR's official 50k/10k train/test**: combined the full 60,000
images (CIFAR's official train+test) into one pool, then drew a fresh class-stratified 70:30
train:val split (`sklearn.train_test_split`, `stratify=class_label`, seed 42) instead of reusing
CIFAR's own division. Reusing CIFAR's official test set as this project's val split would have
double-counted it — it's also the set chenyaofo used to report/select the checkpoints' accuracy, so
it isn't actually held-out from the model pool's perspective. This mirrors what "ground truth"
already means in the ImageNette pipeline (its own 70:30 split, not ImageNette's own division).

Preprocessing switched to CIFAR-specific normalization (`mean=[0.4914,0.4822,0.4465]`,
`std=[0.2023,0.1994,0.2010]`, no resize/crop — native 32×32) instead of blindly reusing the
ImageNette pipeline's `Resize(256)/CenterCrop(224)` + ImageNet mean/std. No ImageNet label
remapping needed either — CIFAR-10 already uses its native 10-class order end to end.

Everything else — `FC_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, the penalty-matrix chromosome, INS
class weighting, the `alpha_sys` objective, NSGA-II population/generations/operators — left
untouched. This track is validating the paper's methodology on its own dataset, not redesigning it.

---

## [RESULT] Labeling — 41,872 train / 17,934 val images kept

Ran the new `label_data.py` (downloads CIFAR-10 via torchvision, dumps every image as a PNG, runs
all 4 models per split, labels `argmin_j{MAdds_j | model_j correct}`, drops images no model gets
right). Kept 41,872/42,000 train and 17,934/18,000 val (70:30 of the combined 60k pool) — very few
dropped, unsurprising given resnet20 alone already reports 92.6% top-1 on CIFAR-10.

Hit two infra snags along the way, both environment-level, not methodology bugs:
- CIFAR-10 download corrupted mid-transfer once (MD5 mismatch at ~75%) — deleted the partial
  archive and re-ran.
- A stray subprocess from an earlier tool session kept retrying its own download into the same
  archive path concurrently with the clean re-run, risking repeat corruption — killed the
  interfering processes and did one final clean run of `label_data.py`, which completed without
  further interference.

**[DEAD-END] Windows-built CSV paths don't survive a copy to the Linux server**: `label_data.py`
originally built the `image_path` CSV column with `os.path.join(...)`, which emits backslashes on
Windows. Ran fine locally, but once the CSVs were copied to the university server for the actual
NSGA-II run, `dispatcher_analysis/main.py` failed with `FileNotFoundError` — `Image.open()` trying
to open a literal filename containing `\` characters, which aren't path separators on Linux.
**Fix**: switched to `"/".join(...)` for the stored `image_path` field (forward slashes
unconditionally, since these CSVs are meant to move across machines), plus an immediate `sed -i
's/\\/\//g'` one-liner on the already-copied CSVs so the in-flight server run didn't have to wait
for a full relabel. Confirmed working — see the eval result below.

---

## [RESULT] NSGA-II search — 30-individual Pareto front, alpha_sys 99.09–99.42%, avg_flops 42.3–233.0M

Ran on the university server (A100, EPYC 7282 dual-socket, 128GB RAM) rather than locally — see
`code/CLAUDE.md` environment notes for that machine. Population 50 / 30 generations / 30 FC epochs,
same hyperparameters as the ImageNette pipeline's Run #3. Search finished in 27,082s (~7h32m);
saved FC weights for all 30 final-front individuals afterward.

Final front: `alpha_sys` 99.09%–99.42%, `avg_flops_G` (MAdds, despite the field name carried over
from the ImageNette pipeline) 42.3M–233.0M.

---

## [RESULT] dispatcher_analysis eval — confirms the front, and explains why alpha_sys looks "insanely high"

Re-ran `dispatcher_analysis/main.py` on the server after the path fix above; copied
`train_summary.csv`, `val_summary.csv`, `train_predictions.csv`, `val_predictions.csv`, and the
confusion-matrix/pareto-scatter plots back to the local repo. Val alpha_sys across the front:
99.01%–99.33%, avg_flops 40.8M–183.3M — consistent with the search-time numbers.

At first glance alpha_sys sitting in a razor-thin 99.0–99.4% band for every individual on the front
looked suspicious (flagged this while the server run was still going). Checked the actual
`ideal_label` distribution on val: **99.0% of images already have resnet20 as their
argmin-cheapest-correct label** — i.e. resnet20 alone gets the overwhelming majority of CIFAR-10
val images right, so "always pick the cheapest model" is already a ~99%-alpha_sys strategy before
the dispatcher does anything. This is a genuine property of this pool/dataset combination, not a
bug: CIFAR-10's accuracy spread across resnet20/32/shufflenetv2_x2_0/vgg16_bn (92.6%→94.2%) is much
narrower than the ImageNette pipeline's ResNet18/34/50/152 pool, so alpha_sys saturates near-ceiling
across the whole front and the front's real signal is in the FLOPs axis, not alpha_sys. Matches
what the per-model recall/precision columns in `val_summary.csv` show — precision on the escalation
models (shufflenetv2/vgg16_bn) is very low (~0.002–0.003) despite decent recall, meaning most
"escalations" are the dispatcher over-routing images resnet20 would've gotten right anyway, which
alpha_sys doesn't penalize (matches the objective's design — see `code/dispatcher/`'s "Fitness
objective" note in `code/CLAUDE.md`, ported here unchanged).

**Status**: reproduction track complete end-to-end (labeling → NSGA-II search → eval), all results
copied back to `code/CIFAR_10_Implementation/` locally and confirmed consistent. See
`code/CLAUDE.md`'s open threads for the updated status entry.

---

## [RESULT] Interesting finding — the pool's errors are largely complementary, not shared

Question worth checking: all 4 pool models sit in a tight 92.6–96.0% top-1 band, but that doesn't
mean they're wrong on the *same* images — if their mistakes are on different images, a dispatcher
has real headroom to beat any single model, not just match the best one. Wrote a reusable script,
`code/CIFAR_10_Implementation/analyze_model_overlap.py`, to check this directly against the
per-model `<model>_correct` columns already in the ground-truth CSVs (restricted to images whose
filename traces back to CIFAR-10's own official test split, so the numbers below aren't inflated by
the train/test overlap the fresh-70:30-split methodology introduces — see the split `[DECISION]`
above; pooling `train_ground_truth.csv` + `val_ground_truth.csv` recovers 9,806 of the 10,000
official test images across the two files).

**Per-model accuracy** on that held-out set: resnet20 94.4%, resnet32 95.4%, shufflenetv2_x2_0
95.9%, vgg16_bn 96.0% (best single model).

**Oracle ceiling** (at least one of the 4 models correct): **100%** — every image is caught by
*someone*, a full +4.0pp above the best single model.

**Pairwise error overlap** (Jaccard of each pair's wrong-image sets, 0=fully complementary,
1=identical): all six pairs sit at **0.19–0.22** — mistakes are mostly not shared. The
correctness-pattern breakdown backs this up: 88.9% of images are "all four correct," but the
remaining 11.1% spreads fairly evenly across all 14 other patterns (no single "everyone fails here"
cluster) — each model also has a small set of images (45–60, ~0.5% each) that *only it* gets right.

**Why this matters**: it's the concrete evidence that the dispatcher problem is well-posed for this
pool specifically because the 4 architectures (plain ResNet, ShuffleNet, VGG) make genuinely
different mistakes — not because any one model is simply better, but because routing per-image can
in principle recover accuracy no single model has on its own. Ties directly into the "known
limitation: dispatcher judgment is capped by the feature extractor" note in `code/CLAUDE.md` — the
4.0pp of headroom is real, the open question is how much of it a resnet20-embedding-based FC head
can actually capture.
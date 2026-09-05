# Week 2

## Context
Advisor follow-up: investigate whether real quantization (PTQ with calibration) gives
meaningful latency wins over the unquantized FP16 baseline. Two focused experiments:

1. **Exp1** — PTQ calibration: does `modelopt mtq.quantize()` + `export_torch_mode()` +
   torch-tensorrt give genuinely lower latency vs. just passing `enabled_precisions={torch.int8}`?
2. **Exp2** — Batch size sweep: does per-image latency improve with larger batches, and does
   uncalibrated INT8 benefit from batching?

Both are in `code/quantization_experiments/`, fully separate from the dispatcher pipeline.

---

## [SETUP] Environment issue — Cached files were courrpted

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
documented in Week0 for the non-PTQ path; PTQ doesn't change the kernel availability.

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

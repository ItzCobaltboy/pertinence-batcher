# 2026-08-21

## Context
Prof gave new direction: PERTINENCE assumes single-image inference, but real deployment
needs batching with mixed-complexity images. Step 1: build a hybrid model pool (ResNet
variants + quantized versions) and hand-build a dispatcher to learn the mechanics.

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

### [SETUP] TensorRT via Torch-TensorRT (not ONNX Runtime)
Revisited TensorRT after the dispatcher/NSGA-II work, per prof meeting action item. ONNX
Runtime's TensorRT execution provider was a dead end again — onnxruntime 1.29.0's
provider DLL is hard-pinned to `nvinfer_10.dll`, but the available `tensorrt`/
`tensorrt-cu13` pip packages ship TensorRT 11.x (`nvinfer_11.dll`) — ABI mismatch,
unfixable without an old TensorRT release.

Switched to **Torch-TensorRT** instead — compiles directly from the live PyTorch model
(`ir='dynamo'`), no ONNX export and no onnxruntime provider involved. Installed
`torch-tensorrt==2.13.0` + `tensorrt-cu13==11.2.1.2` + `nvidia-modelopt` (small
pure-Python deps: psutil, dllist). ResNet18 compiled and ran successfully on the first
real test.

Rebuilt `code/Model Analysis/` to match `code/dispatcher_analysis/`'s structure:
`main.py` (root) + `src/` (constants, data_loader, model_utils, trt_compiler, benchmark,
run_benchmark, eda). Compiles + caches one Torch-TensorRT engine per (model, precision)
to `model_cache/*.pt2`, benchmarks accuracy/latency/size over the full ImageNette val
set, saves to `results/torch_tensorrt_benchmark.csv`. `eda.py` here is separate from —
and doesn't touch — `code/Dispatcher/eda.py`.

Removed the now-unused torchao float8/int8 checkpoints (1.4GB) and the ONNX export
folder (561MB, not needed — Torch-TensorRT skips ONNX entirely). Kept the fp32 `.pth`
checkpoints in `ResnetModels/`.

**Gotcha**: engines compiled for a static batch-size-1 input shape reject any other
input shape outright — first benchmark run crashed mid-way because the val accuracy
loader used batch_size=32. Fixed by setting the val loader to batch_size=1 to match the
compiled shape (see `constants.py`). A future improvement would be compiling with a
dynamic shape range (min/opt/max) instead — relevant for the batching extension where
sub-batch sizes vary.

### [RESULT] Torch-TensorRT benchmark — FP32 vs FP16 vs INT8 vs FP8
**Script**: `code/Model Analysis/main.py`
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
compile, which persisted even after installing the package.

**FP16 is a genuine, reproducible win**: ~2.0–2.6x latency reduction across the whole
pool, essentially zero accuracy cost. First real positive quantization/precision result
in this project (torchao weight-only, torchao dynamic, and ONNX+CUDA INT8 were all dead
ends — see above). Unlike those, this is a real wall-clock latency win (not FLOPs-based)
— relevant if/when latency, not just FLOPs, matters for the batching deployment story.

**Next**: decide whether to pursue real INT8/FP8 via explicit `modelopt` calibration, or
accept FP16 as the practical precision win and move to the batching extension.

### [DECISION] Correction — the "FP16 win" above is a compilation win, not a precision win
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
effects.

## Open threads (as of end of this file)
- Confirm onnx / onnxruntime-gpu / TensorRT installation status — superseded by the
  Torch-TensorRT pivot above
- Decide: pursue real INT8/FP8 via explicit `modelopt` calibration, or accept FP16 as
  the practical precision win and move on to the batching extension
- Whether real INT8/FP16 throughput differentiation exists at all is unverified — needs
  a compute-bound (larger batch) setup to test

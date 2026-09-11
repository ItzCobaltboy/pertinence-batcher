# model_analysis

Step 0 of the pipeline: picks the model pool and benchmarks precision options.

**Does**: loads pretrained ResNet18/34/50/152, benchmarks accuracy/latency/size on
ImageNette both raw (eager PyTorch fp32) and across precisions compiled through
Torch-TensorRT (FP32/FP16/INT8/FP8).
**Does not**: any dispatcher logic (routing, labeling, training) — that starts in
`../../dispatcher/`. No plotting/analysis either — that's `eda/` (nested here, not a
sibling — see its own README), which reads this pipeline's CSVs.

Archived 2026-09 (`../../Journel/Week3.md`) — done/locked work, not part of the active
pipeline. Moved here unmodified from the old `code/model_analysis/`.

## How to run

```
cd archive/model_analysis
python main.py
```

Runs both phases in order:
1. `src/run_benchmark.py: run_eager_baseline()` — 4 pool models, raw PyTorch eager-mode
   fp32, no TRT compilation. Writes `results/eager_baseline.csv`.
2. `src/run_benchmark.py: run_benchmark()` — 4 models x 4 precisions (fp32/fp16/int8/fp8),
   all compiled through Torch-TensorRT (fp32 included, to isolate compilation effects
   from precision effects). Writes `results/torch_tensorrt_benchmark.csv`. Compiles and
   caches engines to `model_cache/*.pt2` on first run (slow); subsequent runs reuse the
   cache.

## Inputs

- ImageNette dataset at `../../imagenette/dataset/` (see its own `main.py` `download()` if missing).
- No upstream pipeline output required — this is the first stage.

## Outputs

- `ResnetModels/` — cached FP32 `.pth` checkpoints (regenerated if missing).
- `model_cache/*.pt2` — cached Torch-TensorRT engines per (model, precision), gitignored.
- `results/eager_baseline.csv` — accuracy/latency/size per model, raw PyTorch fp32.
- `results/torch_tensorrt_benchmark.csv` — accuracy/latency/size per (model, precision),
  all TRT-compiled.

## Why

Model pool selection, quantization dead ends (torchao, ONNX INT8), and the
Torch-TensorRT FP16 result are narrated in `../../Journel/Week1.md` (model pool selection
itself is `../../Journel/Week0.md`).

"""
model_analysis — Torch-TensorRT benchmark, single entry point.

Benchmarking only. EDA/plots on the resulting CSVs live in ../eda/, a
separate pipeline that reads these results but shares no code.

  PHASE 1 (src/run_benchmark.py: run_eager_baseline())
    For every pool model (resnet18/34/50/152), raw PyTorch eager-mode fp32
    (no TRT compilation): measure accuracy/latency/size over the ImageNette
    val set, save to results/eager_baseline.csv.

  PHASE 2 (src/run_benchmark.py: run_benchmark())
    For every pool model x every precision (fp32, fp16, int8, fp8): compile
    (or load cached) Torch-TensorRT engine, measure accuracy/latency/size,
    save to results/torch_tensorrt_benchmark.csv.

Folder layout:
  main.py                   <- you are here, run this file
  src/
    constants.py              paths + settings, shared by every module
    data_loader.py              ImageNette val loader with correct labels
    model_utils.py                loads regular pretrained models, FLOPs
    trt_compiler.py                 compiles/caches Torch-TensorRT engines
    run_benchmark.py                  phase 1 + phase 2 drivers
    benchmark.py                        accuracy/latency measurement

  ResnetModels/            cached FP32 .pth checkpoints (regular models)
  model_cache/               cached Torch-TensorRT engines (fp16, int8, fp8)
  results/                     eager_baseline.csv + torch_tensorrt_benchmark.csv
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

from run_benchmark import run_eager_baseline, run_benchmark


if __name__ == "__main__":
    run_eager_baseline()
    run_benchmark()
    print("\nDone.")

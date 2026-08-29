"""
Model Analysis — Torch-TensorRT benchmark, single entry point.

  PHASE 1 (src/run_benchmark.py)
    For every pool model (resnet18/34/50/152) and every precision
    (fp32, fp16, int8): load the regular model or compile+cache a
    Torch-TensorRT engine, measure accuracy/latency/size over the
    ImageNette val set, save results to results/torch_tensorrt_benchmark.csv.

  PHASE 2 (src/eda.py)
    Analyze that CSV: accuracy/latency/size grouped bar charts per
    precision, and an accuracy-vs-latency scatter. Plots saved to
    results/eda/. This EDA is separate from, and does not touch,
    code/Dispatcher/eda.py.

Folder layout:
  main.py                   <- you are here, run this file
  src/
    constants.py              paths + settings, shared by every module
    data_loader.py              ImageNette val loader with correct labels
    model_utils.py                loads regular pretrained models, FLOPs
    trt_compiler.py                 compiles/caches Torch-TensorRT engines
    run_benchmark.py                  phase 1 driver
    benchmark.py                        accuracy/latency measurement
    eda.py                                phase 2 driver

  ResnetModels/            cached FP32 .pth checkpoints (regular models)
  model_cache/               cached Torch-TensorRT engines (fp16, int8)
  results/                     benchmark_results.csv + eda/ plots
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

from run_benchmark import run_benchmark
from eda import run_eda


if __name__ == "__main__":
    run_benchmark()
    run_eda()
    print("\nDone.")

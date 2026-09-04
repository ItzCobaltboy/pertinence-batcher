"""
Paths for the eda pipeline. Reads model_analysis's benchmark CSVs (not
copied — read live from ../model_analysis/results/) and writes its own
plots to ./results/.
"""

import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)          # "eda/"
MODEL_ANALYSIS_RESULTS_DIR = os.path.join(PROJECT_ROOT, "..", "model_analysis", "results")

EAGER_BASELINE_CSV = os.path.join(MODEL_ANALYSIS_RESULTS_DIR, "eager_baseline.csv")
BENCHMARK_RESULTS_CSV = os.path.join(MODEL_ANALYSIS_RESULTS_DIR, "torch_tensorrt_benchmark.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
PRECISIONS = ["fp32", "fp16", "int8", "fp8"]

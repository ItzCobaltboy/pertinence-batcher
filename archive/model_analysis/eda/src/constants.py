"""
Paths for the eda pipeline. Reads model_analysis's benchmark CSVs (not
copied — read live from ../results/) and writes its own plots to
./results/.

Archived 2026-09 alongside model_analysis/ (Journel/Week3.md): this folder
used to be a sibling of model_analysis/ (code/eda/ next to
code/model_analysis/) and is now nested one level inside it instead
(archive/model_analysis/eda/) — MODEL_ANALYSIS_RESULTS_DIR repointed
accordingly, otherwise unmodified.
"""

import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)          # "archive/model_analysis/eda/"
MODEL_ANALYSIS_RESULTS_DIR = os.path.join(PROJECT_ROOT, "..", "results")

EAGER_BASELINE_CSV = os.path.join(MODEL_ANALYSIS_RESULTS_DIR, "eager_baseline.csv")
BENCHMARK_RESULTS_CSV = os.path.join(MODEL_ANALYSIS_RESULTS_DIR, "torch_tensorrt_benchmark.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
PRECISIONS = ["fp32", "fp16", "int8", "fp8"]

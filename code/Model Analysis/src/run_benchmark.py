"""
Phase 1: for every model in the pool and every precision (fp32, fp16, int8),
load or compile the model, measure accuracy/latency/size, and save one row
per (model, precision) to results/torch_tensorrt_benchmark.csv.
"""

import os
import torch
import pandas as pd

import constants as c
from data_loader import get_val_loader
from model_utils import load_regular_model, get_regular_model_size_mb, compute_flops
from trt_compiler import load_or_compile_model, get_engine_size_mb
from benchmark import measure_accuracy, measure_latency


def run_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_loader = get_val_loader()

    results_rows = []

    for model_name in c.MODEL_NAMES:
        print(f"\n=== {model_name} ===")

        regular_model = load_regular_model(model_name, device)
        flops = compute_flops(regular_model, device)

        for precision in c.PRECISIONS:
            print(f"\n-- {model_name} / {precision} --")

            if precision == "fp32":
                model_to_test = regular_model
                size_mb = get_regular_model_size_mb(model_name)
            else:
                model_to_test = load_or_compile_model(model_name, precision, regular_model, device)
                size_mb = get_engine_size_mb(model_name, precision)

            accuracy = measure_accuracy(model_to_test, val_loader, device)
            latency_ms = measure_latency(model_to_test, device)

            print(f"  accuracy={accuracy}%  latency={latency_ms}ms  size={size_mb}MB")

            results_rows.append({
                "model": model_name,
                "precision": precision,
                "flops": flops,
                "flops_G": round(flops / 1e9, 3),
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "model_size_mb": size_mb,
            })

            del model_to_test
            torch.cuda.empty_cache()

        del regular_model
        torch.cuda.empty_cache()

    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(c.BENCHMARK_RESULTS_CSV, index=False)
    print(f"\nSaved -> {c.BENCHMARK_RESULTS_CSV}")

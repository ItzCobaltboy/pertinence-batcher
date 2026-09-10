"""
Experiment 2: Batch size sweep — does quantization latency benefit emerge
at larger batch sizes?

model_analysis was locked to batch=1 (TRT engines compiled to fixed shape).
This experiment compiles separate TRT engines for each batch size, runs
fp32 vs fp16 vs int8 (no-calib, same as model_analysis) and measures
latency per-image at each batch size.

Batch sizes: 1, 4, 8, 16, 32
Models: resnet18, resnet50 (bracketing cheapest/most-expensive pool models)
Precisions: fp32, fp16, int8

Results saved to results/exp2_batch_size_sweep.csv
"""

import os
import sys
import torch
import torch_tensorrt
import torchvision.models as tvm
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC)

import constants as c
from benchmark import measure_accuracy, measure_latency
from data_loader import get_val_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZES = [1, 4, 8, 16, 32]
MODELS = ["resnet18", "resnet50"]
PRECISIONS = {
    "fp32": {torch.float32},
    "fp16": {torch.float16},
    "int8": {torch.int8},
}

PRECISION_DTYPES = {
    "fp32": {torch.float32},
    "fp16": {torch.float16},
    "int8": {torch.int8},
}


def load_model(name):
    builders = {
        "resnet18": tvm.resnet18,
        "resnet50": tvm.resnet50,
    }
    return builders[name](weights="DEFAULT").to(DEVICE).eval()


def get_or_compile(model_name, precision, batch_size, base_model):
    cache = os.path.join(
        c.MODEL_CACHE_DIR,
        f"{model_name}_{precision}_bs{batch_size}.pt2"
    )
    if os.path.exists(cache):
        print(f"    loading cached: {cache}")
        return torch_tensorrt.load(cache).module()

    print(f"    compiling {model_name} {precision} bs={batch_size}...")
    dummy = torch.randn(batch_size, 3, 224, 224, device=DEVICE)
    compiled = torch_tensorrt.compile(
        base_model,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=[batch_size, 3, 224, 224],
                                     dtype=torch.float32)],
        enabled_precisions=PRECISION_DTYPES[precision],
    )
    os.makedirs(c.MODEL_CACHE_DIR, exist_ok=True)
    torch_tensorrt.save(compiled, cache, inputs=[dummy])
    print(f"    saved -> {cache}")
    return compiled


def run():
    rows = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        for precision in PRECISIONS:
            print(f"\n  -- {precision} --")
            base_model = load_model(model_name)

            for bs in BATCH_SIZES:
                trt_model = get_or_compile(model_name, precision, bs, base_model)

                # Accuracy only at bs=1 (fixed val loader)
                if bs == 1:
                    val_loader = get_val_loader(batch_size=1)
                    acc = measure_accuracy(trt_model, val_loader, DEVICE)
                else:
                    acc = None  # skip full val sweep for larger batches

                lat_total = measure_latency(trt_model, DEVICE, batch_size=bs)
                lat_per_image = round(lat_total / bs, 4)

                print(f"    bs={bs:2d}  latency_total={lat_total}ms  "
                      f"per_image={lat_per_image}ms"
                      + (f"  acc={acc}%" if acc is not None else ""))

                rows.append({
                    "model": model_name,
                    "precision": precision,
                    "batch_size": bs,
                    "latency_ms_total": lat_total,
                    "latency_ms_per_image": lat_per_image,
                    "accuracy": acc,
                })

                del trt_model
                torch.cuda.empty_cache()

            del base_model
            torch.cuda.empty_cache()

    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out = os.path.join(c.RESULTS_DIR, "exp2_batch_size_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run()

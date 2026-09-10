"""Resume exp1 for resnet152 only — other models already have results in CSV."""
import os
import sys
import torch
import torch_tensorrt
import torchvision.models as tvm
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC)

import constants as c
from data_loader import get_val_loader, get_train_loader
from benchmark import measure_accuracy, measure_latency

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode
    MODELOPT_OK = True
except ImportError as e:
    print(f"modelopt import failed: {e}")
    MODELOPT_OK = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MODEL_NAME = "resnet152"


def calibrate(model, calib_loader):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= c.CALIB_NUM_BATCHES:
                break
            model(images.to(DEVICE))


def compile_trt(model, precision_set):
    return torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=[BATCH_SIZE, 3, 224, 224],
                                     dtype=torch.float32)],
        enabled_precisions=precision_set,
    )


def run():
    if not MODELOPT_OK:
        print("Skipping: modelopt not available")
        return

    val_loader = get_val_loader(batch_size=BATCH_SIZE)
    calib_loader = get_train_loader(batch_size=c.CALIB_BATCH_SIZE)
    dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)

    rows = []

    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME}")
    print(f"{'='*60}")

    # ── Baseline: unquantized fp16 TRT ────────────────────────────────
    print("  [baseline] fp16 TRT (no calibration)...")
    model = tvm.resnet152(weights="DEFAULT").to(DEVICE).eval()
    cache = os.path.join(c.MODEL_CACHE_DIR, f"{MODEL_NAME}_baseline_fp16.pt2")
    if os.path.exists(cache):
        trt_model = torch_tensorrt.load(cache).module()
    else:
        trt_model = compile_trt(model, {torch.float16})
        # Save baseline only (not quantized models — they abort on save)
        os.makedirs(c.MODEL_CACHE_DIR, exist_ok=True)
        torch_tensorrt.save(trt_model, cache, inputs=[dummy])
    acc = measure_accuracy(trt_model, val_loader, DEVICE)
    lat = measure_latency(trt_model, DEVICE, batch_size=BATCH_SIZE)
    print(f"    accuracy={acc}%  latency={lat}ms")
    rows.append({"model": MODEL_NAME, "method": "trt_fp16_no_calib",
                 "accuracy": acc, "latency_ms": lat})
    del trt_model, model
    torch.cuda.empty_cache()

    # ── INT8 PTQ with calibration ─────────────────────────────────────
    print("  [ptq] int8 with calibration...")
    model = tvm.resnet152(weights="DEFAULT").to(DEVICE).eval()
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG,
                 forward_loop=lambda m: calibrate(m, calib_loader))
    print("    calibration done, compiling...")
    with export_torch_mode():
        trt_model = compile_trt(model, {torch.int8})
    acc = measure_accuracy(trt_model, val_loader, DEVICE)
    lat = measure_latency(trt_model, DEVICE, batch_size=BATCH_SIZE)
    print(f"    accuracy={acc}%  latency={lat}ms")
    rows.append({"model": MODEL_NAME, "method": "trt_int8_ptq_calib",
                 "accuracy": acc, "latency_ms": lat})
    del trt_model, model
    torch.cuda.empty_cache()

    # ── FP8 PTQ (expected to FAIL on TRT 11.0 / sm_120) ──────────────
    print("  [ptq] fp8 with calibration...")
    try:
        model = tvm.resnet152(weights="DEFAULT").to(DEVICE).eval()
        mtq.quantize(model, mtq.FP8_DEFAULT_CFG,
                     forward_loop=lambda m: calibrate(m, calib_loader))
        print("    calibration done, compiling...")
        with export_torch_mode():
            trt_model = compile_trt(model, {torch.float8_e4m3fn})
        acc = measure_accuracy(trt_model, val_loader, DEVICE)
        lat = measure_latency(trt_model, DEVICE, batch_size=BATCH_SIZE)
        print(f"    accuracy={acc}%  latency={lat}ms")
        rows.append({"model": MODEL_NAME, "method": "trt_fp8_ptq_calib",
                     "accuracy": acc, "latency_ms": lat})
        del trt_model
    except Exception as e:
        print(f"    FAILED: {e}")
        rows.append({"model": MODEL_NAME, "method": "trt_fp8_ptq_calib",
                     "accuracy": "FAILED", "latency_ms": "FAILED"})
    finally:
        del model
        torch.cuda.empty_cache()

    # Append to existing CSV
    csv_path = os.path.join(c.RESULTS_DIR, "exp1_ptq_calibration.csv")
    existing = pd.read_csv(csv_path)
    new_rows = pd.DataFrame(rows)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"\nAppended resnet152 results -> {csv_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    run()

"""
Experiment 1: PTQ calibration via modelopt + torch-tensorrt.

The model_analysis benchmark used enabled_precisions={torch.int8} alone,
which tells TRT that int8 is *allowed* but provides no activation scale
information — so TRT falls back to fp16 for most layers.

This experiment does it properly:
  1. Load FP32 model
  2. Run modelopt MTQ quantization (INT8 PTQ) with a calibration dataloader
     so every layer gets real activation scales
  3. Compile the quantized model through torch-tensorrt
  4. Benchmark accuracy + latency at batch=1 vs the unquantized TRT fp16 baseline

Results saved to results/exp1_ptq_calibration.csv
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


def load_model(name):
    builders = {
        "resnet18": tvm.resnet18,
        "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50,
        "resnet152": tvm.resnet152,
    }
    model = builders[name](weights="DEFAULT").to(DEVICE).eval()
    return model


def calibrate(model, calib_loader):
    """Run forward passes on calibration data so MTQ can compute scales."""
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= c.CALIB_NUM_BATCHES:
                break
            model(images.to(DEVICE))


def compile_with_trt(model, precision_set, cache_path, dummy_input):
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=[BATCH_SIZE, 3, 224, 224],
                                     dtype=torch.float32)],
        enabled_precisions=precision_set,
    )
    if cache_path is not None:
        os.makedirs(c.MODEL_CACHE_DIR, exist_ok=True)
        torch_tensorrt.save(compiled, cache_path, inputs=[dummy_input])
    return compiled


def run():
    if not MODELOPT_OK:
        print("Skipping: modelopt not available")
        return

    val_loader = get_val_loader(batch_size=BATCH_SIZE)
    calib_loader = get_train_loader(batch_size=c.CALIB_BATCH_SIZE)
    dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)

    rows = []

    for model_name in c.MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # ── Baseline: unquantized fp16 TRT (same as model_analysis) ──────
        # Save partial results after each model so a crash doesn't lose everything
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(c.RESULTS_DIR, "exp1_ptq_calibration.csv"), index=False)
        print("  [baseline] fp16 TRT (no calibration)...")
        model = load_model(model_name)
        cache = os.path.join(c.MODEL_CACHE_DIR, f"{model_name}_baseline_fp16.pt2")
        if os.path.exists(cache):
            trt_model = torch_tensorrt.load(cache).module()
        else:
            trt_model = compile_with_trt(model, {torch.float16}, cache, dummy)
        acc = measure_accuracy(trt_model, val_loader, DEVICE)
        lat = measure_latency(trt_model, DEVICE, batch_size=BATCH_SIZE)
        print(f"    accuracy={acc}%  latency={lat}ms")
        rows.append({"model": model_name, "method": "trt_fp16_no_calib",
                     "accuracy": acc, "latency_ms": lat})
        del trt_model, model
        torch.cuda.empty_cache()

        # ── INT8 PTQ with calibration ─────────────────────────────────────
        print("  [ptq] int8 with calibration...")
        model = load_model(model_name)
        quant_config = mtq.INT8_DEFAULT_CFG
        mtq.quantize(model, quant_config, forward_loop=lambda m: calibrate(m, calib_loader))
        print("    calibration done, compiling...")

        with export_torch_mode():
            trt_model = compile_with_trt(model, {torch.int8}, None, dummy)

        acc = measure_accuracy(trt_model, val_loader, DEVICE)
        lat = measure_latency(trt_model, DEVICE, batch_size=BATCH_SIZE)
        print(f"    accuracy={acc}%  latency={lat}ms")
        rows.append({"model": model_name, "method": "trt_int8_ptq_calib",
                     "accuracy": acc, "latency_ms": lat})
        del trt_model, model
        torch.cuda.empty_cache()

        # ── FP8 PTQ with calibration ──────────────────────────────────────
        # FP8 PTQ via modelopt generates quantize ops that TRT 11.0 cannot
        # implement on sm_120 (Blackwell) — no kernel tactic found for
        # quantized maxpool/conv combos. Caught and logged rather than crash.
        print("  [ptq] fp8 with calibration...")
        try:
            model = load_model(model_name)
            quant_config = mtq.FP8_DEFAULT_CFG
            mtq.quantize(model, quant_config, forward_loop=lambda m: calibrate(m, calib_loader))
            print("    calibration done, compiling...")

            with export_torch_mode():
                trt_model = compile_with_trt(model, {torch.float8_e4m3fn}, None, dummy)

            acc = measure_accuracy(trt_model, val_loader, DEVICE)
            lat = measure_latency(trt_model, DEVICE, batch_size=BATCH_SIZE)
            print(f"    accuracy={acc}%  latency={lat}ms")
            rows.append({"model": model_name, "method": "trt_fp8_ptq_calib",
                         "accuracy": acc, "latency_ms": lat})
            del trt_model
        except Exception as e:
            print(f"    FAILED: {e}")
            rows.append({"model": model_name, "method": "trt_fp8_ptq_calib",
                         "accuracy": "FAILED", "latency_ms": "FAILED"})
        finally:
            del model
            torch.cuda.empty_cache()

    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out = os.path.join(c.RESULTS_DIR, "exp1_ptq_calibration.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run()

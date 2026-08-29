"""
Compiles a regular PyTorch model into a Torch-TensorRT engine at a given
precision, and caches the compiled engine to disk so we don't recompile
every time we rerun the pipeline.
"""

import os
import torch
import torch_tensorrt

import constants as c

_PRECISION_DTYPES = {
    "fp16": {torch.float16},
    "int8": {torch.int8},
    "fp8":  {torch.float8_e4m3fn},
}


def _engine_cache_path(model_name, precision):
    return os.path.join(c.MODEL_CACHE_DIR, f"{model_name}_{precision}.pt2")


def compile_model(regular_model, precision, device):
    """Compiles a regular PyTorch model into a Torch-TensorRT engine."""
    compile_input = torch_tensorrt.Input(
        shape=c.COMPILE_INPUT_SHAPE,
        dtype=torch.float32,
    )
    compiled = torch_tensorrt.compile(
        regular_model,
        ir="dynamo",
        inputs=[compile_input],
        enabled_precisions=_PRECISION_DTYPES[precision],
    )
    return compiled


def load_or_compile_model(model_name, precision, regular_model, device):
    """
    If a cached engine already exists on disk for this model+precision,
    loads it. Otherwise compiles a fresh one and saves it for next time.
    """
    cache_path = _engine_cache_path(model_name, precision)

    if os.path.exists(cache_path):
        print(f"  loading cached engine: {cache_path}")
        return torch_tensorrt.load(cache_path).module()

    print(f"  compiling {model_name} at {precision}...")
    compiled = compile_model(regular_model, precision, device)

    os.makedirs(c.MODEL_CACHE_DIR, exist_ok=True)
    dummy_input = torch.randn(c.COMPILE_INPUT_SHAPE, device=device)
    torch_tensorrt.save(compiled, cache_path, inputs=[dummy_input])
    print(f"  saved -> {cache_path}")

    return compiled


def get_engine_size_mb(model_name, precision):
    cache_path = _engine_cache_path(model_name, precision)
    size_bytes = os.path.getsize(cache_path)
    return round(size_bytes / (1024 * 1024), 2)

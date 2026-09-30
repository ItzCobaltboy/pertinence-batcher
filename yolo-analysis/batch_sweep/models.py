"""
Model loading, the exact "forward" being timed, TRT compile/save/load and NMS.

Loading follows yolo-analysis/inference.py: YOLO(<checkpoint in models/ if
present, else the bare name>). ultralytics' predict() wraps the network in
AutoBackend with fuse=True (Conv+BN folded), fp32, eval, so I do the same.

WHAT IS TIMED ("forward"): ForwardOnly below. One call of the ultralytics
DetectionModel on a (B, 3, 640, 640) float32 batch: backbone + neck + the
full Detect head, including DFL and box decoding, returning the decoded
(B, 84, 8400) tensor that NMS consumes. NMS is NOT part of forward; it is
timed separately as the forward+NMS column. All three variants go through
this same wrapper, so the definition is identical across variants. If the
TRT build leaves some ops in PyTorch (fallback subgraphs), those still run
inside the timed call; subgraph_report() records exactly which ones.
"""

import os
import time

import settings


def resolve_weights(model_name):
    for candidate in settings.weights_candidates(model_name):
        if os.path.exists(candidate):
            return candidate
    return model_name  # ultralytics downloads a .pt / builds a .yaml


def sha256_of(path):
    import hashlib
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_only_class():
    import torch

    class ForwardOnly(torch.nn.Module):
        """DetectionModel in eval mode returns (decoded, raw_head_outputs); keep
        only the decoded tensor. Nothing else changes."""

        def __init__(self, detection_model):
            super().__init__()
            self.net = detection_model

        def forward(self, x):
            out = self.net(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

    return ForwardOnly


def load_eager(model_name, device):
    """Returns (ForwardOnly module on device, info dict)."""
    import torch
    from ultralytics import YOLO

    weights = resolve_weights(model_name)
    t0 = time.perf_counter()
    yolo = YOLO(weights)
    net = yolo.model
    net.fuse(verbose=False)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    # DetectionModel.stride and its Detect head's .stride are the same tensor
    # object; torch 2.3's export (first step of the TRT dynamo path) refuses
    # duplicate constant references. The model-level copy is metadata only
    # (forward never reads it), so give it its own copy. Compute is unchanged.
    if isinstance(getattr(net, "stride", None), torch.Tensor):
        net.stride = net.stride.clone()
    module = forward_only_class()(net).to(device).eval()
    info = {
        "weights": os.path.abspath(weights) if os.path.exists(weights) else weights,
        "weights_sha256": sha256_of(weights),
        "fused": True,
        "dtype": "float32",
        "load_s": time.perf_counter() - t0,
        "n_params": int(sum(p.numel() for p in net.parameters())),
    }
    return module, info


def prime_head(module, example):
    """Run one eager forward at the target shape. The Detect head builds its
    anchors on the first call for a new shape; doing it here means export
    sees a head that no longer mutates itself."""
    import torch
    with torch.no_grad():
        module(example)


def subgraph_report(compiled):
    """Which parts of the compiled graph run in TensorRT and which fell back
    to PyTorch. torch_tensorrt's dynamo partitioner names TRT blocks
    _run_on_acc_<n> and PyTorch blocks _run_on_gpu_<n>."""
    report = {"trt_subgraphs": 0, "torch_subgraphs": 0, "torch_fallback_ops": []}
    try:
        children = list(compiled.named_children())
    except Exception as e:  # pragma: no cover
        report["error"] = repr(e)
        return report
    for name, child in children:
        if "_run_on_acc" in name:
            report["trt_subgraphs"] += 1
        elif "_run_on_gpu" in name:
            report["torch_subgraphs"] += 1
            graph = getattr(child, "graph", None)
            if graph is not None:
                for node in graph.nodes:
                    if node.op in ("call_function", "call_method", "call_module"):
                        target = str(node.target)
                        if "getitem" not in target:
                            report["torch_fallback_ops"].append(target)
    report["torch_fallback_ops"] = sorted(set(report["torch_fallback_ops"]))
    report["full_trt"] = report["torch_subgraphs"] == 0 and report["trt_subgraphs"] > 0
    return report


def trt_precisions(variant):
    import torch
    if variant == "trt_fp32":
        return {torch.float32}
    if variant == "trt_fp16":
        return {torch.float16}
    raise ValueError(f"not a TRT variant: {variant}")


def compile_trt(module, example, variant):
    """torch_tensorrt 2.3 dynamo path, fixed input shape. Returns
    (compiled GraphModule, compile seconds)."""
    import torch
    import torch_tensorrt

    t0 = time.perf_counter()
    compiled = torch_tensorrt.compile(
        module,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=list(example.shape), dtype=torch.float32)],
        enabled_precisions=trt_precisions(variant),
    )
    torch.cuda.synchronize()
    return compiled, time.perf_counter() - t0


def save_trt(compiled, path, example):
    import torch_tensorrt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch_tensorrt.save(compiled, tmp, output_format="exported_program", inputs=[example])
    os.replace(tmp, path)


def load_trt(path):
    import torch_tensorrt  # registers the TRT runtime ops before deserializing
    t0 = time.perf_counter()
    module = torch_tensorrt.load(path).module()
    return module, time.perf_counter() - t0


def nms_function():
    """ultralytics moved NMS from utils.ops to utils.nms in newer releases."""
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression


def run_nms(nms, prediction):
    return nms(prediction, conf_thres=settings.CONF_THRESHOLD, iou_thres=settings.IOU_THRESHOLD,
               max_det=settings.MAX_DET)

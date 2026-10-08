"""
Everything that needs torch/ultralytics: loading YOLOv8 weights, the frozen
yolov8n backbone embedder, and a conv-level MAC counter for cost measurement.
"""

import os

import torch
import torch.nn as nn


def load_detection_model(model_name, search_dirs, download_dir):
    """Returns the fused, eval-mode, fp32 ultralytics DetectionModel for
    `model_name` ("yolov8n", ...). Looks for <model_name>.pt in search_dirs
    first; otherwise ultralytics downloads it into download_dir (passing a
    full path makes it download there instead of the current directory)."""
    from ultralytics import YOLO

    file_name = f"{model_name}.pt"
    path = next((os.path.join(d, file_name) for d in search_dirs
                 if os.path.exists(os.path.join(d, file_name))), None)
    if path is None:
        os.makedirs(download_dir, exist_ok=True)
        path = os.path.join(download_dir, file_name)

    model = YOLO(path).model.float().eval()
    model.fuse(verbose=False)   # conv+bn folded, what ultralytics predict() runs
    for param in model.parameters():
        param.requires_grad = False
    return model


class BackboneEmbedder(nn.Module):
    """Runs a YOLOv8 backbone (layers 0..max(tap_layers), a plain chain in
    every yolov8 yaml) and returns the global-average-pooled outputs of
    tap_layers, concatenated: one fixed-size vector per image."""

    def __init__(self, detection_model, tap_layers):
        super().__init__()
        last = max(tap_layers)
        layers = list(detection_model.model[: last + 1])
        for i, layer in enumerate(layers):
            if getattr(layer, "f", -1) != -1:
                raise ValueError(f"layer {i} ({type(layer).__name__}) is not sequential (f={layer.f}); "
                                 f"the backbone embedder assumes a plain chain up to layer {last}")
        self.layers = nn.ModuleList(layers)
        self.tap_layers = set(tap_layers)

    def forward(self, x):
        pooled = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.tap_layers:
                pooled.append(x.mean(dim=(2, 3)))
        return torch.cat(pooled, dim=1)


def build_backbone_embedder(model_name, tap_layers, expected_dim, search_dirs, download_dir, img_size=640):
    detection_model = load_detection_model(model_name, search_dirs, download_dir)
    embedder = BackboneEmbedder(detection_model, tap_layers).eval()
    with torch.no_grad():
        dim = embedder(torch.zeros(1, 3, img_size, img_size)).shape[1]
    assert dim == expected_dim, f"{model_name} backbone embedding dim is {dim}, config says {expected_dim}"
    return embedder


def count_macs(module, input_shape):
    """Multiply-accumulates of one forward pass, counted on Conv2d and Linear
    layers only (bias excluded), via forward hooks. This is what thop counts
    for a fused YOLOv8 (no BatchNorm left; SiLU, upsample, concat and
    maxpool count as zero), without needing thop installed."""
    total = [0]

    def conv_hook(m, inputs, output):
        kh, kw = m.kernel_size
        total[0] += output.numel() * (m.in_channels // m.groups) * kh * kw

    def linear_hook(m, inputs, output):
        total[0] += output.numel() * m.in_features

    handles = []
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            module(torch.zeros(*input_shape))
    finally:
        for h in handles:
            h.remove()
    return total[0]

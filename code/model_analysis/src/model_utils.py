"""
Loads the regular (uncompiled) pretrained ResNet pool models, and computes
their FLOPs. FLOPs are an architecture property, not a precision property,
so we only ever compute them once per model from the plain FP32 version.
"""

import os
import torch
import torchvision.models as tvm
from thop import profile

import constants as c

_MODEL_BUILDERS = {
    "resnet18":  tvm.resnet18,
    "resnet34":  tvm.resnet34,
    "resnet50":  tvm.resnet50,
    "resnet152": tvm.resnet152,
}


def load_regular_model(model_name, device):
    """Loads a pretrained torchvision ResNet in FP32, in eval mode."""
    builder = _MODEL_BUILDERS[model_name]
    model = builder(weights="DEFAULT")
    model = model.to(device)
    model.eval()
    return model


def get_regular_model_size_mb(model_name):
    """
    Reads the on-disk size of the cached FP32 checkpoint. If it doesn't
    exist yet, saves the current pretrained weights there first.
    """
    pth_path = os.path.join(c.RESNET_MODELS_DIR, f"{model_name}_float32.pth")

    if not os.path.exists(pth_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_regular_model(model_name, device)
        os.makedirs(c.RESNET_MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), pth_path)

    size_bytes = os.path.getsize(pth_path)
    return round(size_bytes / (1024 * 1024), 2)


def compute_flops(model, device):
    """FLOPs for one forward pass at the standard 224x224 input size."""
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return int(flops)

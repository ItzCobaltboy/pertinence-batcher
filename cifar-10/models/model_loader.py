"""
Loads the 4 CIFAR-10-pretrained models (chenyaofo/pytorch-cifar-models) from
the local checkpoints in this folder.

NOTE on packaging: `pytorch-cifar-models` is NOT a real PyPI package (pip
install fails - its setup.py looks for a requirements.txt that doesn't ship
in the sdist) and `pip install git+https://github.com/chenyaofo/pytorch-cifar-models.git`
also fails to build for the same reason. The upstream-supported way to use it
is `torch.hub.load(...)`, which pulls the repo's *source* (not the sdist) into
torch's hub cache and imports the model-definition modules directly - no pip
package involved. That's what this loader does, with `pretrained=False`
(architecture only) since we load our own local checkpoints below rather than
the hub's own weight URLs.

MAdds pool order (ascending, from chenyaofo/pytorch-cifar-models model zoo table):
  shufflenetv2_x0_5 (10.90M) < resnet20 (40.81M) < resnet32 (69.12M) < vgg11_bn (153.29M)
"""

import os
import torch

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

_HUB_REPO = "chenyaofo/pytorch-cifar-models"

# name -> (torch.hub entrypoint, local checkpoint filename)
_MODEL_SPECS = {
    "shufflenetv2_x0_5": ("cifar10_shufflenetv2_x0_5", "cifar10_shufflenetv2_x0_5-1308b4e9.pt"),
    "resnet20":          ("cifar10_resnet20",          "cifar10_resnet20-4118986f.pt"),
    "resnet32":          ("cifar10_resnet32",          "cifar10_resnet32-ef93fc4d.pt"),
    "vgg11_bn":          ("cifar10_vgg11_bn",          "cifar10_vgg11_bn-eaeebf42.pt"),
}


def load_model(name, device=None, trust_repo=True):
    """Instantiates the named architecture (untrained) via torch.hub and loads
    the matching local checkpoint's state_dict. Returns the model in eval mode."""
    if name not in _MODEL_SPECS:
        raise ValueError(f"Unknown CIFAR model '{name}', expected one of {list(_MODEL_SPECS)}")

    entrypoint, checkpoint_filename = _MODEL_SPECS[name]
    model = torch.hub.load(_HUB_REPO, entrypoint, pretrained=False, trust_repo=trust_repo)

    checkpoint_path = os.path.join(_MODELS_DIR, checkpoint_filename)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def strip_classifier_head(model, name):
    """Returns a feature-extractor version of `model` with its final FC layer
    removed, plus the extractor's output dim, measured via a real forward
    pass over a dummy 32x32 CIFAR-sized input.

    A global average pool is always inserted before the final flatten,
    regardless of architecture: ResNet exposes its avgpool as a stored
    submodule (so it's included by truncating at children()[:-1] alone,
    making this a no-op), but ShuffleNetV2 does its global average pooling
    inline in forward() (`x = x.mean([2, 3])`), not as a stored submodule —
    without an explicit pool here, the truncated extractor leaves the
    feature map spatial and flattening it multiplies the true channel count
    by the spatial size.

    Only used for the embedding extractor (shufflenetv2_x0_5 - the cheapest
    pool model, per the dispatcher methodology). resnet20/resnet32/
    shufflenetv2_x0_5 all expose a final `.fc` Linear the same way in
    pytorch_cifar_models; vgg11_bn's `.classifier` is a multi-layer MLP and
    is NOT supported here since it's never used as the embedding extractor."""
    import torch
    import torch.nn as nn

    if name not in ("resnet20", "resnet32", "shufflenetv2_x0_5"):
        raise ValueError(f"strip_classifier_head does not support '{name}' "
                          f"(only used for the shufflenetv2_x0_5 embedding extractor)")

    modules = list(model.children())[:-1]  # drop .fc

    class _Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)

    extractor = nn.Sequential(*modules, nn.AdaptiveAvgPool2d(1), _Flatten())

    extractor.eval()
    with torch.no_grad():
        measured_dim = extractor(torch.zeros(1, 3, 32, 32)).shape[1]

    return extractor, measured_dim


MODEL_NAMES_BY_MADDS = ["shufflenetv2_x0_5", "resnet20", "resnet32", "vgg11_bn"]
MADDS_M = [10.90, 40.81, 69.12, 153.29]

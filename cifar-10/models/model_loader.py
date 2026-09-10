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

MAdds pool order (ascending, from models/model_params.txt):
  resnet20 (40.81M) < resnet32 (69.12M) < shufflenetv2_x2_0 (187.81M) < vgg16_bn (313.73M)
"""

import os
import torch

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

_HUB_REPO = "chenyaofo/pytorch-cifar-models"

# name -> (torch.hub entrypoint, local checkpoint filename)
_MODEL_SPECS = {
    "resnet20":          ("cifar10_resnet20",          "cifar10_resnet20-4118986f.pt"),
    "resnet32":          ("cifar10_resnet32",          "cifar10_resnet32-ef93fc4d.pt"),
    "shufflenetv2_x2_0": ("cifar10_shufflenetv2_x2_0", "cifar10_shufflenetv2_x2_0-ec31611c.pt"),
    "vgg16_bn":          ("cifar10_vgg16_bn",          "cifar10_vgg16_bn-6ee7ea24.pt"),
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
    removed, plus the extractor's output dim (measured, not assumed).

    Only used for the embedding extractor (resnet20 - the cheapest pool
    model, per the dispatcher methodology). resnet20/resnet32/
    shufflenetv2_x2_0 all expose a final `.fc` Linear the same way in
    pytorch_cifar_models; vgg16_bn's `.classifier` is a multi-layer MLP and
    is NOT supported here since it's never used as the embedding extractor."""
    import torch.nn as nn

    if name not in ("resnet20", "resnet32", "shufflenetv2_x2_0"):
        raise ValueError(f"strip_classifier_head does not support '{name}' "
                          f"(only used for the resnet20 embedding extractor)")

    in_features = model.fc.in_features
    modules = list(model.children())[:-1]  # drop .fc

    class _Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)

    extractor = nn.Sequential(*modules, _Flatten())
    return extractor, in_features


MODEL_NAMES_BY_MADDS = ["resnet20", "resnet32", "shufflenetv2_x2_0", "vgg16_bn"]
MADDS_M = [40.81, 69.12, 187.81, 313.73]

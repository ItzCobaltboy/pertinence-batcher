"""
Loads CIFAR-100-pretrained models (chenyaofo/pytorch-cifar-models) from the
local checkpoints in this folder. Shared across every CIFAR-100 dispatcher
variant (fig9c/, fig9d/, ...) — which of these get used as a pool for any
given variant is that variant's own config.py's problem, not this file's.

Same packaging situation as the CIFAR-10 track's loader: `pytorch-cifar-models`
isn't a real pip package, so this goes through `torch.hub.load(...)` for the
architecture definitions (source pulled into torch's hub cache, not a wheel)
with `pretrained=False`, then loads our own locally-downloaded checkpoint
files rather than the hub's weight URLs.

Seven models total, all sourced from chenyaofo/pytorch-cifar-models'
published checkpoints. Cost figures below are NOT chenyaofo's own
model-zoo table numbers — an earlier version of this file used those
directly, but they don't line up with what the paper's own Fig. 9 plots
show for the same architectures closely enough to trust blindly (checked
by eye against the plotted PERTINENCE/SOTA points; the gap is bigger than
any reasonable measurement-tool drift). Instead, MFLOPS_M_BY_NAME below is
measured directly with THOP on this repo's own loaded checkpoints (one
forward pass, 32x32 input, MACs count doubled to FLOPs — see class
docstring in fitness.py for why doubling is the right convention: the
paper states it uses THOP "to obtain MACs... and the resulting FLOPS,"
implying FLOPS = 2x MACs, and this exactly reproduces the paper's own
published dispatcher-overhead figure, Table 5's 24.17 MFLOPS for
CIFAR-100/ShuffleNetV2, to within 1%):

  shufflenetv2_x0_5 (24.07M) < mobilenetv2_x0_5 (63.51M) <
  shufflenetv2_x1_0 (94.69M) < mobilenetv2_x0_75 (130.21M) <
  mobilenetv2_x1_4 (358.99M) < repvgg_a1 (1715.70M) < repvgg_a2 (3718.22M)

Whenever a variant's config.py lists MODEL_NAMES, that list needs to stay
MFLOPS-ascending (per this ordering, not the earlier chenyaofo-table one)
for label_data.py's argmin-cheapest-correct routing logic to mean what
it's supposed to — for fig9c/fig9d specifically this doesn't change
anything, since both variants' three models keep the same relative order
under either measurement.

Six of these are the paper's Figure 4b pool (page 5) — what an earlier
version of this track built a single 6-way dispatcher against. That attempt
got dropped: the paper's actual CIFAR-100 experiments (Fig. 9, page 8) never
dispatch across all six at once, they run a separate MOEA search per hand-
picked 2- or 3-model subset. `mobilenetv2_x0_75` is the seventh model here
specifically because Fig. 9(a)/9(c)'s subsets use it, and it isn't one of
the six plotted on Fig. 4b — confirmed as a real, distinct, separately
published checkpoint (not a typo for mobilenetv2_x0_5) by cross-checking the
chenyaofo repo's full CIFAR-100 model list before downloading it.
"""

import os
import torch

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

_HUB_REPO = "chenyaofo/pytorch-cifar-models"

# name -> (torch.hub entrypoint, local checkpoint filename)
_MODEL_SPECS = {
    "shufflenetv2_x0_5": ("cifar100_shufflenetv2_x0_5", "cifar100_shufflenetv2_x0_5-1977720f.pt"),
    "mobilenetv2_x0_5":  ("cifar100_mobilenetv2_x0_5",  "cifar100_mobilenetv2_x0_5-9f915757.pt"),
    "mobilenetv2_x0_75": ("cifar100_mobilenetv2_x0_75", "cifar100_mobilenetv2_x0_75-d7891e60.pt"),
    "shufflenetv2_x1_0": ("cifar100_shufflenetv2_x1_0", "cifar100_shufflenetv2_x1_0-9ae22beb.pt"),
    "mobilenetv2_x1_4":  ("cifar100_mobilenetv2_x1_4",  "cifar100_mobilenetv2_x1_4-8a269f5e.pt"),
    "repvgg_a1":         ("cifar100_repvgg_a1",         "cifar100_repvgg_a1-c06b21a7.pt"),
    "repvgg_a2":         ("cifar100_repvgg_a2",         "cifar100_repvgg_a2-8e71b1f8.pt"),
}

# Locally measured FLOPs (millions) per model — thop MACs on a real forward
# pass at 32x32, doubled (see module docstring for why doubling and why
# this repo's own measurement rather than chenyaofo's table). Each
# variant's config.py builds its own MODEL_COST list by looking these up
# for whichever subset it dispatches to — this dict is the single source
# of truth so the number never gets retyped or drifts between variants.
MFLOPS_M_BY_NAME = {
    "shufflenetv2_x0_5": 24.0732,
    "mobilenetv2_x0_5":  63.5146,
    "mobilenetv2_x0_75": 130.2139,
    "shufflenetv2_x1_0": 94.6931,
    "mobilenetv2_x1_4":  358.9893,
    "repvgg_a1":         1715.7043,
    "repvgg_a2":         3718.2195,
}


def load_model(name, device=None, trust_repo=True):
    """Instantiates the named architecture (untrained) via torch.hub and loads
    the matching local checkpoint's state_dict. Returns the model in eval mode."""
    if name not in _MODEL_SPECS:
        raise ValueError(f"Unknown CIFAR-100 model '{name}', expected one of {list(_MODEL_SPECS)}")

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

    A global average pool is always inserted before the final flatten: like
    the CIFAR-10 track's shufflenetv2_x0_5, this architecture does its global
    average pooling inline in forward() (`x = x.mean([2, 3])`) rather than as
    a stored submodule, so truncating at children()[:-1] alone would leave
    the feature map spatial and multiply the true channel count by the
    spatial size once flattened.

    Only used for shufflenetv2_x0_5, the embedding extractor per the
    dispatcher methodology (cheapest pool model, matching the paper's own
    statement that it uses this exact model as the CIFAR-100 feature
    extractor) — true for every variant here, since the paper never swaps
    extractors between Fig. 9's subsets, only which models get dispatched
    to. Not implemented for the other six pool models since none of them
    are ever used as an embedding extractor in this repo."""
    import torch.nn as nn

    if name != "shufflenetv2_x0_5":
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

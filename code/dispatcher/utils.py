import os
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMAGENETTE_LABEL_MAP = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}

# Ordered cheapest → most expensive by FLOPs.
# label index = dispatcher class target (0 = route to ResNet18, etc.)
POOL_MODELS = [
    {"name": "resnet18",  "flops_G": 1.824,  "label": 0},
    {"name": "resnet34",  "flops_G": 3.679,  "label": 1},
    {"name": "resnet50",  "flops_G": 4.134,  "label": 2},
    {"name": "resnet152", "flops_G": 11.604, "label": 3},
]

LABEL_TO_MODEL = {m["label"]: m["name"] for m in POOL_MODELS}

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_pool_models(device: torch.device) -> dict:
    """Pulls all 4 pretrained ResNets from torchvision, sets eval mode."""
    pool = {
        "resnet18":  models.resnet18(weights="DEFAULT"),
        "resnet34":  models.resnet34(weights="DEFAULT"),
        "resnet50":  models.resnet50(weights="DEFAULT"),
        "resnet152": models.resnet152(weights="DEFAULT"),
    }
    for name, model in pool.items():
        model.to(device).eval()
        print(f"  Loaded {name}")
    return pool


def get_imagenette_loader(dataset_path: str, split: str, batch_size: int = 64) -> tuple:
    """
    Returns (DataLoader, list[image_path]) for the given split ('train' or 'val').
    Labels remapped from ImageFolder 0-9 → ImageNet 1000-class indices.
    shuffle=False to keep paths and outputs in sync.
    """
    root = os.path.join(dataset_path, "imagenette2-320", split)
    dataset = ImageFolder(root=root, transform=IMAGENET_TRANSFORM)

    synset_to_imagenet = {v: IMAGENETTE_LABEL_MAP[k] for k, v in dataset.class_to_idx.items()}
    dataset.targets = [synset_to_imagenet[t] for t in dataset.targets]
    dataset.samples = [(s, synset_to_imagenet[t]) for s, t in dataset.samples]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    paths = [s for s, _ in dataset.samples]
    return loader, paths

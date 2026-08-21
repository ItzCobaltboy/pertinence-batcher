import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from thop import profile
import os


# ImageNette is a 10-class subset of ImageNet, but the folder names are ImageNet
# synset IDs, and pretrained ResNets output 1000-class ImageNet logits.
# So we need to map each synset to its actual ImageNet class index,
# otherwise accuracy eval compares against 0-9 instead of real ImageNet labels.
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


def get_val_loader(dataset_path: str, batch_size: int = 32) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(
        root=os.path.join(dataset_path, "imagenette2-320/val"),
        transform=transform
    )

    # ImageFolder assigns labels 0-9 alphabetically by folder name.
    # Remap those to the actual ImageNet class indices so model predictions
    # are comparable against ground truth.
    synset_to_imagenet = {v: IMAGENETTE_LABEL_MAP[k]
                          for k, v in dataset.class_to_idx.items()}

    dataset.targets = [synset_to_imagenet[t] for t in dataset.targets]
    dataset.samples = [(s, synset_to_imagenet[t]) for s, t in dataset.samples]

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


import time

def model_metrics(model: torch.nn.Module, dataset_path: str, device: torch.device, pth_path: str, batch_size: int = 32) -> dict:
    """
    Computes FLOPs, Top-1 accuracy, model size, and inference latency for a given model over the ImageNette val set.

    Args:
        model:        PyTorch model (already loaded, weights applied)
        dataset_path: Path to dataset root (expects imagenette2-320/val/ inside)
        device:       torch.device to run eval on
        pth_path:     Path to the .pth file, used to compute model size on disk
        batch_size:   DataLoader batch size

    Returns:
        dict with keys: 'flops', 'accuracy', 'model_size_mb', 'latency_ms'
    """
    model = model.to(device)
    model.eval()

    # FLOPs are computed on a single dummy image — architecture-level metric,
    # independent of the dataset
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    # warm up the GPU before timing — first few runs are slow due to CUDA
    # kernel compilation and memory allocation, not representative of steady-state
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # synchronize before and after to make sure we're timing actual execution,
    # not just kernel launches (CUDA is async by default)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy_input)
    torch.cuda.synchronize()
    latency_ms = round((time.time() - start) / 100 * 1000, 3)

    # size straight from disk — reflects actual storage cost of the precision
    model_size_mb = round(os.path.getsize(pth_path) / (1024 * 1024), 2)

    val_loader = get_val_loader(dataset_path, batch_size)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return {
        "flops":         int(flops),
        "accuracy":      round(accuracy, 4),
        "model_size_mb": model_size_mb,
        "latency_ms":    latency_ms
    }
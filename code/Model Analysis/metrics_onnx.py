import os
import time
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Same ImageNette synset -> ImageNet class index mapping as metrics.py.
# Needed because ImageFolder assigns 0-9 by folder order, but ONNX-exported
# ResNets still output 1000-class ImageNet logits.
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


def get_val_data(dataset_path: str, batch_size: int = 32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=os.path.join(dataset_path, "imagenette2-320/val"), transform=transform)

    synset_to_imagenet = {v: IMAGENETTE_LABEL_MAP[k]
                          for k, v in dataset.class_to_idx.items()}
    dataset.targets = [synset_to_imagenet[t] for t in dataset.targets]
    dataset.samples = [(s, synset_to_imagenet[t]) for s, t in dataset.samples]

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_providers(use_tensorrt: bool = False):
    """
    Picks the ONNX Runtime execution provider. Falls back gracefully if
    TensorRT isn't installed — CUDA EP still gives GPU execution, just
    without TensorRT's extra INT8 kernel optimizations.
    """
    available = ort.get_available_providers()

    if use_tensorrt and "TensorrtExecutionProvider" in available:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def model_metrics_onnx(onnx_path: str, dataset_path: str, use_tensorrt: bool = False, batch_size: int = 32) -> dict:
    """
    Computes accuracy, model size, and latency for an ONNX model over the ImageNette val set.

    Note: FLOPs aren't recomputed here since they don't change between the PyTorch
    and ONNX versions of the same architecture — reuse the value from metrics.py
    if you need it in the same table.

    Args:
        onnx_path:    path to the .onnx model file
        dataset_path: path to dataset root (expects imagenette2-320/val/ inside)
        use_tensorrt: try TensorrtExecutionProvider first if available
        batch_size:   batch size for eval and latency timing

    Returns:
        dict with keys: 'accuracy', 'model_size_mb', 'latency_ms', 'provider'
    """
    providers = get_providers(use_tensorrt)
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    actual_provider = session.get_providers()[0]

    # --- latency ---
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # warm up — first few runs include provider/kernel initialization overhead
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    start = time.time()
    for _ in range(100):
        session.run(None, {input_name: dummy_input})
    latency_ms = round((time.time() - start) / 100 * 1000, 3)

    # --- accuracy ---
    val_loader = get_val_data(dataset_path, batch_size)
    correct = 0
    total = 0

    for images, labels in val_loader:
        images_np = images.numpy().astype(np.float32)
        outputs = session.run(None, {input_name: images_np})[0]
        predicted = np.argmax(outputs, axis=1)
        correct += (predicted == labels.numpy()).sum()
        total += labels.size(0)

    accuracy = round(100.0 * correct / total, 4)

    # --- size ---
    model_size_mb = round(os.path.getsize(onnx_path) / (1024 * 1024), 2)

    return {
        "accuracy": accuracy,
        "model_size_mb": model_size_mb,
        "latency_ms": latency_ms,
        "provider": actual_provider
    }
import torch
import time


def measure_accuracy(model, val_loader, device):
    try:
        model.eval()
    except (NotImplementedError, AttributeError):
        pass
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return round(100 * correct / total, 4)


def measure_latency(model, device, batch_size=1, warmup=10, timed=100):
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    try:
        model.eval()
    except (NotImplementedError, AttributeError):
        pass
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed):
            model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return round(elapsed / timed * 1000, 3)

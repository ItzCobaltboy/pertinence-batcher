"""
Measures accuracy and latency for a given (already loaded/compiled) model.
"""

import time
import torch

import constants as c


def measure_accuracy(model, val_loader, device):
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct_count += 1
                total_count += 1

    accuracy = 100.0 * correct_count / total_count
    return round(accuracy, 4)


def measure_latency(model, device):
    dummy_input = torch.randn(c.COMPILE_INPUT_SHAPE, device=device)

    # warm up — first few runs pay for CUDA kernel compilation/allocation,
    # not representative of steady-state speed
    with torch.no_grad():
        for _ in range(c.LATENCY_WARMUP_RUNS):
            model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(c.LATENCY_TIMED_RUNS):
            model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()

    total_seconds = end_time - start_time
    latency_ms = (total_seconds / c.LATENCY_TIMED_RUNS) * 1000
    return round(latency_ms, 3)

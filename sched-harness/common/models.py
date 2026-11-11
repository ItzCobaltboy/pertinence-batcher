# Load the ResNet classifiers (torchvision, random weights, FP16 eval) and run one batch through the serving pipeline:
# stack + H2D + convert, forward. Every stage is CUDA-synchronised so its time is real. No post-processing.
import time

import torch

import config


def load_net(label):
    """torchvision ResNet with random weights (timing and power do not depend on the values), FP16 eval on the GPU."""
    import torchvision
    name = config.MODELS[label]
    try:
        net = getattr(torchvision.models, name)(weights=None)
    except TypeError:                                   # torchvision before 0.13
        net = getattr(torchvision.models, name)(pretrained=False)
    net = net.half().eval().cuda()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_nets():
    """All of them stay resident for the whole run. Returns {label: net}."""
    return {label: load_net(label) for label in config.QUEUES}


def run_batch(net, buffer, arrays):
    """Returns (h2d_ms, forward_ms, 0.0, 0); the last two keep the log columns of the old detector pipeline."""
    t0 = time.perf_counter()
    x = buffer.to_gpu(arrays)
    t1 = time.perf_counter()
    with torch.no_grad():
        net(x)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, 0.0, 0


def warm_up(nets, buffers, pool):
    """Run every model at every warmup batch size twice so the first real batch is not a cold one."""
    for label in config.QUEUES:
        for b in config.WARMUP_BATCH_SIZES:
            for _ in range(2):
                run_batch(nets[label], buffers[label], pool[:b])
    torch.cuda.synchronize()


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()

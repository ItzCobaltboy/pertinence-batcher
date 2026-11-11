# The ready input pool and the per-model pinned batch buffer.
# A job is "one input for model i": it points at a pool array. Nothing camera or decode specific.
import torch

import config
from common import cifar


def build_pool():
    """INPUT_POOL_SIZE ready uint8 IMGSZ x IMGSZ x 3 arrays: random CIFAR-10 test images (config.CIFAR_NPZ), upscaled
    once, BGR order. Built once per run. Returns (pool, source text)."""
    return cifar.load_pool(config.CIFAR_NPZ, config.IMGSZ, config.INPUT_POOL_SIZE)


class BatchBuffer:
    """One pinned uint8 buffer per model (size B_MAX). to_gpu stacks the referenced arrays into it,
    copies to the GPU and converts BGR to RGB, FP16, 0..1."""

    def __init__(self):
        self.host = torch.empty((config.B_MAX, config.IMGSZ, config.IMGSZ, 3), dtype=torch.uint8).pin_memory()
        self.host_np = self.host.numpy()

    def to_gpu(self, arrays):
        for i, array in enumerate(arrays):
            self.host_np[i] = array
        gpu = self.host[:len(arrays)].cuda(non_blocking=True)
        x = gpu.permute(0, 3, 1, 2).flip(1).contiguous()     # NHWC BGR -> NCHW RGB
        x = x.half().div_(255.0)
        torch.cuda.synchronize()
        return x

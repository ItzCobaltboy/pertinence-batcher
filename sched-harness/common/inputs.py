# The ready input pool and the per-model pinned batch buffer.
# A job is "one input for model i": it points at a pool array. Nothing camera or decode specific.
import glob
import os

import cv2
import numpy as np
import torch

import config


def letterbox(img, size):
    """Resize keeping aspect ratio, pad with grey 114 to size x size (copied from jetson-power bench.py)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img
    return canvas


def read_images(paths):
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError("cannot read image %s" % path)
        images.append(letterbox(img, config.IMGSZ))
    return images


def build_pool():
    """64 ready uint8 640x640x3 BGR arrays, built once. Returns (pool, source text).
    Content only matters for NMS time (box count), so the source is recorded in run.json."""
    paths = sorted(glob.glob(os.path.join(config.INPUTS_DIR, "*.jpg")) + glob.glob(os.path.join(config.INPUTS_DIR, "*.png")))
    if paths:
        images = read_images(paths[:config.INPUT_POOL_SIZE])
        source = "%s (%d images, cycled to %d)" % (config.INPUTS_DIR, len(images), config.INPUT_POOL_SIZE)
    else:
        import ultralytics
        assets = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
        images = read_images([os.path.join(assets, "bus.jpg"), os.path.join(assets, "zidane.jpg")])
        source = "ultralytics samples bus.jpg and zidane.jpg (cycled to %d)" % config.INPUT_POOL_SIZE
        print("*** NOTICE: %s is empty or missing; the pool is just the two ultralytics sample images ***" % config.INPUTS_DIR)
    pool = [images[i % len(images)] for i in range(config.INPUT_POOL_SIZE)]
    return pool, source


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

# Load a YOLOv8 network, build a batch from real images, and time the forward pass.
# Only the network is timed: FP16, input already on the GPU, no decoding, no NMS.
import os
import time

import cv2
import numpy as np
import torch

import config


def load_net(letter):
    """Stock COCO weights -> fused FP16 eval network on the GPU."""
    from ultralytics import YOLO
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, config.MODELS[letter])
    net = YOLO(path).model      # downloads into models/ when missing
    net = net.float().fuse()    # fold BN into conv, as ultralytics does for inference
    net = net.half().eval().cuda()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def letterbox(img, size):
    """Resize keeping aspect ratio, pad with grey 114 to size x size."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img
    return canvas


def sample_images():
    import ultralytics
    assets = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    images = []
    for name in ["bus.jpg", "zidane.jpg"]:
        img = cv2.imread(os.path.join(assets, name))
        if img is None:
            raise RuntimeError("cannot read %s in %s" % (name, assets))
        images.append(letterbox(img, config.IMGSZ))
    return images


def make_batch(batch, images):
    """Cycle through the real images to fill the batch; BGR->RGB, 0..1, FP16, on GPU."""
    chosen = [images[i % len(images)] for i in range(batch)]
    x = np.stack(chosen)[..., ::-1]                       # BGR -> RGB
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)
    return (x.float() / 255.0).half().cuda()


def forward(net, x):
    with torch.no_grad():
        return net(x)


def warmup(net, x, iters):
    for _ in range(iters):
        forward(net, x)
    torch.cuda.synchronize()


def timed_run(net, x, min_seconds, min_iters, chunk_iters):
    """Run forwards back to back for at least min_seconds and min_iters.
    Syncs every chunk_iters iterations only to check the clock, so the GPU stays busy.
    Returns (latencies_ms list, wall_start, wall_end) with times from time.monotonic()."""
    latencies = []
    torch.cuda.synchronize()
    t_start = time.monotonic()
    while True:
        events = []
        for _ in range(chunk_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward(net, x)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        t_end = time.monotonic()
        latencies.extend(s.elapsed_time(e) for s, e in events)
        if t_end - t_start >= min_seconds and len(latencies) >= min_iters:
            return latencies, t_start, t_end


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()

# The dispatcher's feature extractor: frozen ORIGINAL yolov8n backbone (layers 0-9), global average
# pooled at layers 4, 6, 9 and concatenated -> 448-d. Same definition and preprocessing as
# archive/yolo-dispatcher so numbers compare with the earlier routing.
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO

import config
from common import paths

TAP_LAYERS = (4, 6, 9)
EMBEDDING_DIM = 448


def letterbox(image, size=config.IMG_SIZE, fill=114):
    """PIL image -> float tensor [3, size, size] in 0..1: aspect-preserving resize, grey padding."""
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill,) * 3)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class Embedder(nn.Module):
    def __init__(self, detection_model):
        super().__init__()
        self.layers = nn.ModuleList(list(detection_model.model[: max(TAP_LAYERS) + 1]))

    def forward(self, x):
        pooled = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in TAP_LAYERS:
                pooled.append(x.mean(dim=(2, 3)))
        return torch.cat(pooled, dim=1)


def load_embedder():
    model = YOLO(paths.weights_path("n")).model.float().eval()
    model.fuse(verbose=False)
    for p in model.parameters():
        p.requires_grad = False
    return Embedder(model).cuda().eval()


def embed_pil(embedder, pil_images):
    """List of PIL images -> numpy [n, 448]."""
    batch = torch.stack([letterbox(im) for im in pil_images]).cuda()
    with torch.no_grad():
        return embedder(batch).float().cpu().numpy()


def embed_files(embedder, image_paths, batch=32):
    out = []
    for start in range(0, len(image_paths), batch):
        images = [Image.open(p) for p in image_paths[start:start + batch]]
        out.append(embed_pil(embedder, images))
    return np.concatenate(out)


def extractor_gflops(embedder):
    """FLOPs (2 x MACs) of the extractor for one 640x640 frame, in GFLOPs."""
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE).cuda()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        embedder(x)
    return counter.get_total_flops() / 1e9

"""
YOLO-style letterbox as a picklable transform (module-level class, so
DataLoader workers can receive it under any multiprocessing start method).
Aspect-preserving resize so the long side is `size`, centred on a size x size
canvas filled with grey 114, RGB scaled to [0, 1] (what ultralytics feeds
YOLOv8, minus its rect-mode minimal padding, which would break batching).
"""

import numpy as np
from PIL import Image


class Letterbox:
    def __init__(self, size=640, fill=114):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        import torch

        image = image.convert("RGB")
        w, h = image.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

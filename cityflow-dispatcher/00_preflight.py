# Checks the environment: GPU, VRAM, driver, library versions, free disk on D:,
# and one real forward pass of each yolov8 model (weights download into models/).
import json
import os
import shutil
import subprocess

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

import config
from common import paths


def driver_version():
    out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Not falling back to CPU.")
    os.makedirs(paths.MODELS, exist_ok=True)
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "driver": driver_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "ultralytics": ultralytics.__version__,
        "free_disk_gb_D": round(shutil.disk_usage(paths.ROOT).free / 1e9, 1),
    }

    # one real forward per model on a noise-free real-size image (zeros is fine for a shape/runs check,
    # the real accuracy checks happen on real frames in later steps)
    image = np.full((540, 960, 3), 114, dtype=np.uint8)
    info["forward_ok"] = {}
    for letter in config.MODEL_LETTERS:
        model = YOLO(paths.weights_path(letter))
        result = model.predict(image, imgsz=config.IMG_SIZE, device=0, verbose=False)
        info["forward_ok"][letter] = len(result) == 1
        print(f"yolov8{letter}: forward ok")

    out_dir = paths.result_dir("00_preflight")
    with open(os.path.join(out_dir, "preflight.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print("saved", out_dir)


if __name__ == "__main__":
    main()

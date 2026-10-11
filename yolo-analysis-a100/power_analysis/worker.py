# One worker = one (model, batch) combo, run in its own process by power_sweep.py.
#
# The power sampler (nvidia-smi) runs in the parent for the whole sweep. This
# worker never touches it: it only writes down time.time() at the start and end
# of every phase. Later, aggregate.py uses those times to cut the power trace.
#
# Phases:
#   load the model + one warmup forward     (not measured)
#   idle   : model loaded, GPU doing nothing
#   run k  : forward the batch FORWARDS_PER_PHASE times, however long that takes
#   buffer : sleep between run phases
#
# Usage (power_sweep.py does this): python worker.py '<job as json>'

import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def sync(device):
    # wait until the GPU has finished everything we gave it
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def find_weights(model_name, models_dir):
    path = os.path.join(models_dir, model_name)
    if os.path.exists(path):
        return path
    return model_name  # not found locally: ultralytics downloads it


def load_model(model_name, models_dir, device):
    yolo = YOLO(find_weights(model_name, models_dir))
    net = yolo.model
    net.fuse(verbose=False)  # fold batchnorm into conv, like ultralytics predict() does
    net = net.to(device).eval()
    return net


def load_batch(images_dir, batch_size, img_size, seed, device):
    # pick batch_size real COCO images and stack them into one (B, 3, H, W) tensor
    files = []
    for name in sorted(os.listdir(images_dir)):
        if name.lower().endswith(".jpg"):
            files.append(name)
    rng = random.Random(seed)
    if batch_size <= len(files):
        picked = rng.sample(files, batch_size)
    else:
        picked = []
        for _ in range(batch_size):
            picked.append(rng.choice(files))

    images = []
    for name in picked:
        img = cv2.imread(os.path.join(images_dir, name))  # BGR, H x W x 3
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, ::-1]                              # BGR -> RGB
        img = img.transpose(2, 0, 1)                       # H x W x 3 -> 3 x H x W
        images.append(img)
    batch = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(batch).to(device)


def add_mark(marks, label):
    # remember when something happened (same clock as the power samples)
    marks.append([label, time.time()])


def main():
    job = json.loads(sys.argv[1])
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    marks = []

    # 1. load the model and the batch, then one warmup forward
    #    (the first forward builds anchors and picks cuDNN kernels, we don't want that in run 0)
    load_start = time.time()
    model = load_model(job["model"], job["models_dir"], device)
    batch = load_batch(job["images_dir"], job["batch"], job["img_size"], job["seed"], device)
    with torch.no_grad():
        model(batch)
    sync(device)
    load_s = time.time() - load_start

    # 2. idle with the model loaded
    add_mark(marks, "idle_start")
    time.sleep(job["idle_s"])
    add_mark(marks, "idle_end")

    # 3. run phases
    phases = []
    with torch.no_grad():
        for k in range(job["num_run_phases"]):
            add_mark(marks, f"run{k}_start")
            start = time.time()
            forwards = job["forwards_per_phase"]
            for i in range(forwards):
                model(batch)
            sync(device)  # the phase ends when the GPU is really done
            end = time.time()
            add_mark(marks, f"run{k}_end")
            phases.append({"phase": k, "start": start, "end": end,
                           "runtime_s": end - start, "forwards": forwards})
            time.sleep(job["buffer_s"])

    # 4. save everything for aggregate.py
    result = {
        "model": job["model"],
        "batch": job["batch"],
        "device": device,
        "load_s": load_s,
        "marks": marks,
        "phases": phases,
    }
    with open(job["out"], "w") as f:
        json.dump(result, f, indent=1)
    print(f"done: {job['model']} batch {job['batch']}, load {load_s:.1f} s, "
          f"{len(phases)} run phases, {phases[0]['forwards']} forwards in phase 0")


if __name__ == "__main__":
    main()

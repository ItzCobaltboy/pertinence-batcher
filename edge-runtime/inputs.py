# Where the frames come from.
#
#   tensor : a pool of pre-made 3x640x640 float tensors already on the GPU, cycled
#            (built from the frame pool if there is one, else uniform noise). Pure
#            GPU behaviour, no preprocessing.
#   frames : a pool of real JPEGs held in RAM as encoded bytes. Each job decodes
#            its JPEG (PIL), letterboxes it with yolo-dispatcher's IMAGE_TRANSFORM
#            and moves it to the GPU. Where that happens is runtime.py's --prep flag.
#
# prepare(frame_ref) -> (payload, decode_ms, prep_ms). frame_ref is taken modulo
# the pool size.
#
#   python inputs.py export-frames --cityflow-data ../cityflow-routing/data --scenario S02
#   python inputs.py export-frames --from-dir ../yolo-analysis-a100/dataset/val2017

import argparse
import glob
import io
import os
import time

import config


def jpeg_paths(frames_dir):
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))


class FakeInputs:
    def __init__(self, prep_ms=0.0):
        self.prep_ms = prep_ms

    def prepare(self, frame_ref):
        if self.prep_ms:
            time.sleep(self.prep_ms / 1000.0)
        return None, 0.0, self.prep_ms

    def sample_batch(self, b):
        return None


class FrameInputs:
    def __init__(self, frames_dir, device, limit=config.FRAME_POOL_SIZE):
        from models import yolo_dispatcher
        import torch
        self.torch = torch
        self.device = device
        self.transform = yolo_dispatcher().IMAGE_TRANSFORM
        paths = jpeg_paths(frames_dir)[:limit]
        if not paths:
            raise FileNotFoundError("no *.jpg in %s (run: python inputs.py export-frames ...)" % frames_dir)
        self.blobs = []
        for p in paths:
            with open(p, "rb") as f:
                self.blobs.append(f.read())

    def prepare(self, frame_ref):
        from PIL import Image
        t0 = time.perf_counter()
        image = Image.open(io.BytesIO(self.blobs[frame_ref % len(self.blobs)]))
        image.load()
        t1 = time.perf_counter()
        x = self.transform(image).to(self.device)
        t2 = time.perf_counter()
        return x, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    def sample_batch(self, b):
        return self.torch.stack([self.prepare(i)[0] for i in range(b)])


class TensorInputs:
    def __init__(self, frames_dir, device, size=config.TENSOR_POOL_SIZE, log=print):
        import torch
        self.torch = torch
        if jpeg_paths(frames_dir):
            frames = FrameInputs(frames_dir, device, limit=size)
            self.pool = [frames.prepare(i)[0] for i in range(min(size, len(frames.blobs)))]
            self.source = "frames"
        else:
            log("WARNING: no frames in %s, tensor pool is uniform noise (NMS time not representative)" % frames_dir)
            gen = torch.Generator().manual_seed(0)
            self.pool = [torch.rand(3, config.IMG_SIZE, config.IMG_SIZE, generator=gen).to(device)
                         for _ in range(size)]
            self.source = "noise"

    def prepare(self, frame_ref):
        return self.pool[frame_ref % len(self.pool)], 0.0, 0.0

    def sample_batch(self, b):
        return self.torch.stack([self.pool[i % len(self.pool)] for i in range(b)])


def make_inputs(mode, device, frames_dir=config.FRAMES_DIR):
    if mode == "tensor":
        return TensorInputs(frames_dir, device)
    if mode == "frames":
        return FrameInputs(frames_dir, device)
    raise ValueError("unknown input mode %r" % mode)


# ---------------------------------------------------------------------------
# export-frames (laptop): N real frames as JPEGs, copied to the device later
# ---------------------------------------------------------------------------
def export_from_cityflow(data_dir, scenarios, n, out_dir, quality):
    import cv2
    import sys
    sys.path.insert(0, config.CITYFLOW_ROUTING_DIR)
    import dataset
    root = dataset.extracted_dir(data_dir)
    cams = dataset.find_cameras(root, scenarios)
    if not cams:
        raise FileNotFoundError("no CityFlow videos for %s under %s (run cityflow-routing/extract.py)" % (scenarios, root))
    per_cam = max(1, n // len(cams))
    written = 0
    for cam in cams:
        cap = cv2.VideoCapture(cam["video_path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for k in range(per_cam):
            if written >= n:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * total / per_cam))
            ok, frame = cap.read()
            if not ok:
                continue
            path = os.path.join(out_dir, "%s_%s_%05d.jpg" % (cam["scenario"], cam["camera"], k))
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            written += 1
        cap.release()
    return written


def export_from_dir(src_dir, n, out_dir, quality):
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    step = max(1, len(paths) // n)
    written = 0
    for p in paths[::step][:n]:
        Image.open(p).convert("RGB").save(os.path.join(out_dir, os.path.basename(p)), quality=quality)
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the frame pool used by --input frames.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export-frames")
    p.add_argument("--out", default=config.FRAMES_DIR)
    p.add_argument("--n", type=int, default=config.FRAME_POOL_SIZE)
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--cityflow-data", default=os.path.join(config.CITYFLOW_ROUTING_DIR, "data"))
    p.add_argument("--scenario", default="S02", help="comma-separated CityFlow scenarios")
    p.add_argument("--from-dir", help="take JPEGs from this folder instead of CityFlow videos")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.from_dir:
        n = export_from_dir(args.from_dir, args.n, args.out, args.quality)
    else:
        n = export_from_cityflow(args.cityflow_data, args.scenario.split(","), args.n, args.out, args.quality)
    size = sum(os.path.getsize(p) for p in jpeg_paths(args.out)) / 2 ** 20
    print("%d frames -> %s (%.1f MB)" % (n, args.out, size))


if __name__ == "__main__":
    main()

"""
Routes every frame of every selected CityFlow camera through the PERTINENCE
dispatcher: the frozen yolov8n backbone embedder runs once per frame (batched),
then EVERY Pareto FC head is applied to that embedding, so one pass gives the
routing of every operating point. The dispatcher runs on every frame (D = 1).

Preprocessing is yolo-dispatcher's own, imported, not copied: OpenCV's BGR
frame -> RGB -> PIL image -> config.IMAGE_TRANSFORM (Letterbox(640), the
transform the training embeddings went through after PIL's .convert("RGB"))
-> config.build_feature_extractor() (fused yolov8n, GAP at layers 4/6/9, 448-d).

Resumable per camera: a camera whose routing file exists is skipped.
Frames are never written to disk.

Run: python route.py --scenarios S01 --run-name s01
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import dataset
import heads as heads_lib

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_DISPATCHER_DIR = os.path.join(HERE, "..", "yolo-dispatcher")
sys.path.insert(0, YOLO_DISPATCHER_DIR)
import config as yd_config  # noqa: E402  (yolo-dispatcher/config.py)

PREPROCESSING = ("cv2 BGR frame -> cv2.cvtColor(BGR2RGB) -> PIL.Image.fromarray -> "
                 "yolo-dispatcher config.IMAGE_TRANSFORM (Letterbox 640: PIL bilinear, "
                 "centred, pad 114, /255, CHW float32, no mean/std) -> "
                 "config.build_feature_extractor() (fused yolov8n, GAP layers 4/6/9)")


def preprocess_bgr(frame_bgr):
    """One OpenCV frame -> the same (3, 640, 640) tensor yolo-dispatcher's
    training embeddings were computed from."""
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return yd_config.IMAGE_TRANSFORM(Image.fromarray(rgb))


def short_names(model_names):
    """yolov8n -> n, ..."""
    return [name[-1] for name in model_names]


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, message):
        line = f"{datetime.datetime.now():%H:%M:%S}  {message}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def environment_info(device):
    import cv2
    import torch
    import ultralytics
    import PIL

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - folder copied without .git
            return "unknown"

    status = git("status", "--porcelain")
    return {
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "host": platform.node(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": status not in ("", "unknown"),
    }


def write_atomic(path, write_fn):
    """Calls write_fn(tmp_path), then renames tmp onto path, so a crash never
    leaves a half-written file that resume would treat as finished."""
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def save_manifest(manifest, path):
    manifest["updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
    write_atomic(path, write)


def save_embeddings(path, embeddings, frame_idx):
    """float16 [n_frames, 448] plus the frame indices, for re-applying any
    future FC head without decoding video again."""
    def write(tmp):
        with open(tmp, "wb") as f:
            np.savez(f, embeddings=embeddings.astype(np.float16), frame_idx=frame_idx)
    write_atomic(path, write)


def embed_batch(extractor, batch, device):
    import torch

    with torch.no_grad():
        return extractor(torch.stack(batch).to(device)).float().cpu().numpy()


def routing_table(cam, fps, frame_idx, offset, results, model_names):
    """One row per frame: identity and time columns, then per point
    model_p<id> and prob_<n|s|m|l>_p<id>."""
    letters = short_names(model_names)
    video_time = (frame_idx - 1) / fps
    columns = {
        "scenario": cam["scenario"],
        "camera": cam["camera"],
        "frame_idx": frame_idx,
        "fps": fps,
        "video_time_s": np.round(video_time, 4),
        "synced_time_s": np.round(video_time + offset, 4),
    }
    for point_id, (probs, chosen) in results.items():
        columns[f"model_p{point_id}"] = np.array(model_names)[chosen]
        for j, letter in enumerate(letters):
            columns[f"prob_{letter}_p{point_id}"] = np.round(probs[:, j], 4)
    return pd.DataFrame(columns)


def route_camera(cam, extractor, heads, device, batch_size, root, raw_dir, log, progress):
    """Embeds and routes every frame of one camera, writes its two raw files.
    Returns the camera's manifest entry."""
    offsets = dataset.read_offsets(root, cam["scenario"])
    if cam["camera"] not in offsets:
        raise ValueError(f"{cam['camera']} missing from cam_timestamp/{cam['scenario']}.txt")
    offset = offsets[cam["camera"]]
    listed = dataset.read_frame_counts(root, cam["scenario"]).get(cam["camera"])

    capture, fps, header_count = dataset.open_video(cam["video_path"])
    expected = listed or header_count
    log(f"{cam['scenario']}/{cam['camera']}: fps {fps:g}, offset {offset:g} s, "
        f"~{expected} frames ({cam['split']})")

    started = time.time()
    last_print = started
    embeddings, frame_ids, batch = [], [], []
    for frame_idx, frame in dataset.iter_frames(capture):
        batch.append(preprocess_bgr(frame))
        frame_ids.append(frame_idx)
        if len(batch) == batch_size:
            embeddings.append(embed_batch(extractor, batch, device))
            progress["done"] += len(batch)
            batch = []
            if time.time() - last_print > 10:
                last_print = time.time()
                rate = progress["done"] / (time.time() - progress["start"])
                eta_min = (progress["total"] - progress["done"]) / rate / 60
                log(f"  {cam['camera']}: {len(frame_ids)}/{expected} frames | "
                    f"{rate:.1f} frames/s | run ETA {eta_min:.1f} min")
    if batch:
        embeddings.append(embed_batch(extractor, batch, device))
        progress["done"] += len(batch)
    capture.release()

    if not frame_ids:
        raise RuntimeError(f"no frames decoded from {cam['video_path']}")
    embeddings = np.concatenate(embeddings)
    frame_idx = np.array(frame_ids, dtype=np.int32)
    results = heads_lib.apply_heads(embeddings, heads)   # float32 embeddings, like training

    stem = f"{cam['scenario']}_{cam['camera']}"
    emb_path = os.path.join(raw_dir, f"embeddings_{stem}.npz")
    save_embeddings(emb_path, embeddings, frame_idx)
    table = routing_table(cam, fps, frame_idx, offset, results, yd_config.MODEL_NAMES)
    routing_path = os.path.join(raw_dir, f"routing_{stem}.csv.gz")
    write_atomic(routing_path, lambda tmp: table.to_csv(tmp, index=False, compression="gzip"))

    seconds = time.time() - started
    if expected and len(frame_ids) != expected:
        log(f"  WARNING: decoded {len(frame_ids)} frames, cam_framenum/header says {expected}")
    log(f"  {cam['camera']} done: {len(frame_ids)} frames in {seconds:.0f} s "
        f"({len(frame_ids) / seconds:.1f} frames/s)")
    return {
        "split": cam["split"], "scenario": cam["scenario"], "camera": cam["camera"],
        "video_path": os.path.abspath(cam["video_path"]), "fps": fps, "offset_s": offset,
        "frames_decoded": len(frame_ids), "frames_header": header_count,
        "frames_cam_framenum": listed, "seconds": round(seconds, 1),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=["S01"])
    parser.add_argument("--run-name", default="run1")
    parser.add_argument("--yolo-results", default=os.path.join(YOLO_DISPATCHER_DIR, "results"),
                        help="yolo-dispatcher's results/ (nsga2/ and eval/)")
    parser.add_argument("--data-dir", default=dataset.DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)

    import torch

    run_dir = os.path.join(args.results_dir, args.run_name)
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, "route.log"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = dataset.extracted_dir(args.data_dir)
    cameras = dataset.find_cameras(root, args.scenarios)
    if not cameras:
        sys.exit(f"No cameras for {args.scenarios} under {root}: run extract.py first")

    points, heads = heads_lib.load_pareto(args.yolo_results)
    head_ids = [{"point_id": int(r.point_id), "weights_path": r.weights_path, "sha256": r.weights_sha256}
                for r in points.itertuples()]

    manifest_path = os.path.join(raw_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if [h["sha256"] for h in manifest["heads"]] != [h["sha256"] for h in head_ids]:
            sys.exit(f"{run_dir} was routed with different Pareto heads; use a new --run-name")
        manifest["scenarios"] = sorted(set(manifest["scenarios"]) | set(args.scenarios))
    else:
        manifest = {
            "run_name": args.run_name,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": {"name": "AI City Challenge 2022 Track 1 (CityFlowV2, MTMC)",
                        "gdrive_file_id": dataset.DATASET_FILE_ID, "zip_name": dataset.ZIP_NAME,
                        "zip_size": dataset.ZIP_SIZE, "extracted_dir": os.path.abspath(root)},
            "scenarios": sorted(args.scenarios),
            "yolo_results": os.path.abspath(args.yolo_results),
            "model_names": list(yd_config.MODEL_NAMES),
            "heads": head_ids,
            "embedding_dim": yd_config.EMBEDDING_DIM,
            "preprocessing": PREPROCESSING,
            "dispatch_interval_frames": 1,
            "batch_size": args.batch,
            "cameras": {},
        }
    manifest["environment"] = environment_info(device)
    points.to_csv(os.path.join(raw_dir, "pareto_points.csv"), index=False)
    save_manifest(manifest, manifest_path)

    env = manifest["environment"]
    log(f"Run {args.run_name}: {len(cameras)} cameras in {args.scenarios}, {len(heads)} Pareto points, "
        f"device {env['gpu']}, torch {env['torch']}, ultralytics {env['ultralytics']}")

    todo = []
    for cam in cameras:
        stem = f"{cam['scenario']}_{cam['camera']}"
        if os.path.exists(os.path.join(raw_dir, f"routing_{stem}.csv.gz")):
            log(f"{cam['scenario']}/{cam['camera']}: already routed, skipping")
        else:
            todo.append(cam)
    if not todo:
        log("Nothing to do.")
        return

    total = 0
    for cam in todo:
        listed = dataset.read_frame_counts(root, cam["scenario"]).get(cam["camera"])
        total += listed or dataset.open_video(cam["video_path"])[2]
    progress = {"done": 0, "total": total, "start": time.time()}

    extractor = yd_config.build_feature_extractor(device).to(device).eval()
    for cam in todo:
        entry = route_camera(cam, extractor, heads, device, args.batch, root, raw_dir, log, progress)
        manifest["cameras"][f"{cam['scenario']}_{cam['camera']}"] = entry
        save_manifest(manifest, manifest_path)

    minutes = (time.time() - progress["start"]) / 60
    log(f"All done: {progress['done']} frames in {minutes:.1f} min. Raw data in {raw_dir}")


if __name__ == "__main__":
    main()

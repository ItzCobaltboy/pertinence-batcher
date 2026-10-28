# Opens a CityFlow camera video and yields every Nth frame with its 1-based frame index
# (CityFlow's own gt files count frames from 1).
import os

import cv2

import config
from common import paths


def find_cameras(scenarios):
    """Cameras of the given scenarios in the extracted dataset: list of dicts
    (split_folder, scenario, camera, video_path, gt_path or None), sorted."""
    root = paths.extracted_dir()
    cameras = []
    for folder in ["train", "validation", "test"]:
        for scenario in scenarios:
            scenario_dir = os.path.join(root, folder, scenario)
            if not os.path.isdir(scenario_dir):
                continue
            for camera in sorted(os.listdir(scenario_dir)):
                video = os.path.join(scenario_dir, camera, "vdo.avi")
                gt = os.path.join(scenario_dir, camera, "gt", "gt.txt")
                if os.path.exists(video):
                    cameras.append({"folder": folder, "scenario": scenario, "camera": camera,
                                    "video_path": video, "gt_path": gt if os.path.exists(gt) else None})
    return cameras


def video_info(video_path):
    """(fps, header frame count, width, height); raises if the video does not open."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    info = (capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    return info


def sampled_frames(video_path, stride=config.FRAME_STRIDE, max_frames=None):
    """Yields (frame_idx, bgr_frame) for frames 1, 1+stride, 1+2*stride, ... Stops after
    max_frames yielded frames if given."""
    capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    yielded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx - 1) % stride != 0:
            continue
        yield frame_idx, frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break
    capture.release()

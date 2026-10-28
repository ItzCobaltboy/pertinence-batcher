"""
CityFlowV2 (AI City Challenge 2022 Track 1) helpers: where the zip and the
extracted files live, which cameras a scenario has, their start offsets and
frame counts, and reading frames straight from vdo.avi.

Layout inside the zip (checked by listing the real zip):
    {train,validation,test}/S0x/c0xx/vdo.avi
    cam_timestamp/S0x.txt    lines "c002 1.640": camera start offset in seconds
    cam_framenum/S0x.txt     lines "c002 2110": frames per camera
    ReadMe.txt, list_cam.txt
All videos are 10 FPS except c015 (S03) at 8 FPS; FPS is read from each video.
"""

import os
import re

DATASET_FILE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"
ZIP_NAME = "AICity22_Track1_MTMC_Tracking.zip"
ZIP_SIZE = 16_912_529_116

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data")

# matches a camera video inside the zip or the extracted folder
VIDEO_PATTERN = re.compile(r"^(train|validation|test)/(S\d+)/(c\d+)/vdo\.avi$")


def zip_path(data_dir):
    return os.path.join(data_dir, ZIP_NAME)


def extracted_dir(data_dir):
    return os.path.join(data_dir, "AIC22_Track1")


def find_cameras(root, scenarios):
    """Cameras of the given scenarios under an extracted root, sorted.
    Returns a list of dicts: split, scenario, camera, video_path."""
    cameras = []
    for split in ["train", "validation", "test"]:
        for scenario in scenarios:
            scenario_dir = os.path.join(root, split, scenario)
            if not os.path.isdir(scenario_dir):
                continue
            for camera in sorted(os.listdir(scenario_dir)):
                video_path = os.path.join(scenario_dir, camera, "vdo.avi")
                if os.path.exists(video_path):
                    cameras.append({"split": split, "scenario": scenario,
                                    "camera": camera, "video_path": video_path})
    return cameras


def _read_camera_table(path, cast):
    """Parses 'c001 0' style lines (the files use CRLF) into {camera: value}."""
    table = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                table[parts[0]] = cast(parts[1])
    return table


def read_offsets(root, scenario):
    """{camera: start offset in seconds} from cam_timestamp/<scenario>.txt."""
    return _read_camera_table(os.path.join(root, "cam_timestamp", f"{scenario}.txt"), float)


def read_frame_counts(root, scenario):
    """{camera: frame count} from cam_framenum/<scenario>.txt (used for ETA and
    as a cross-check; the real count is whatever the decoder returns)."""
    path = os.path.join(root, "cam_framenum", f"{scenario}.txt")
    return _read_camera_table(path, int) if os.path.exists(path) else {}


def open_video(video_path):
    """Returns (capture, fps, frame_count_from_header)."""
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    header_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return capture, fps, header_count


def iter_frames(capture):
    """Yields (frame_idx, bgr_frame) until the decoder stops. frame_idx starts
    at 1, like CityFlow's own gt/det files."""
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        frame_idx += 1
        yield frame_idx, frame

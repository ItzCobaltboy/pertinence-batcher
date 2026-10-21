"""
CPU tests for cityflow-dispatcher, no YOLO weights needed.

- matching: box recall and cheapest-correct on hand-built cases.
- label.py end to end on a fake CityFlow layout (two scenarios, one train,
  one eval) with an injected fake embedder and fake detectors: dark frames
  are easy (yolov8n finds every reference box), bright frames need yolov8m.
- finetune.py end to end on that data, starting from a fake COCO front whose
  heads send everything to yolov8l: fine-tuning must move the easy frames
  down, keep alpha_sys, cut cost, and write results cityflow-routing can load.

Run: python -m pytest tests/ -q      (from cityflow-dispatcher/)
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cf_config as C  # noqa: E402
import finetune  # noqa: E402
import label  # noqa: E402
import matching  # noqa: E402

TRAIN, EVAL = "S98", "S99"
CAMERAS = {TRAIN: [("c981", 60), ("c982", 40)], EVAL: [("c991", 50)]}


# ---------------------------------------------------------------- matching

def det(boxes, cls, conf=None):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    return {"boxes": boxes, "cls": np.array(cls, dtype=int),
            "conf": np.array(conf if conf is not None else [0.9] * len(boxes), dtype=float)}


def test_recall_cases():
    ref = det([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]], [2, 2, 7, 0])
    assert matching.recall_vs_reference(ref, ref, 0.5) == 1.0
    half = det([[0, 0, 10, 10], [40, 40, 50, 50]], [2, 7])
    assert matching.recall_vs_reference(ref, half, 0.5) == 0.5
    wrong_class = det([[0, 0, 10, 10]], [7])
    assert matching.recall_vs_reference(ref, wrong_class, 0.5) == 0.0
    shifted = det([[6, 6, 16, 16]], [2])          # IoU 16/184 < 0.5
    assert matching.recall_vs_reference(ref, shifted, 0.5) == 0.0
    assert matching.recall_vs_reference(det([], []), half, 0.5) == 1.0   # nothing to miss
    # one prediction can only match one reference box
    two_refs = det([[0, 0, 10, 10], [1, 1, 11, 11]], [2, 2])
    assert matching.recall_vs_reference(two_refs, det([[0, 0, 10, 10]], [2]), 0.5) == 0.5


def test_cheapest_correct():
    assert matching.cheapest_correct([0.9, 0.95, 1, 1], 0.8) == 0
    assert matching.cheapest_correct([0.5, 0.79, 0.8, 1], 0.8) == 2
    assert matching.cheapest_correct([0.1, 0.1, 0.1, 0.5], 0.8) == 3


# ---------------------------------------------------------------- fakes

def is_bright(frame):
    return frame.mean() > 100


def write_fake_dataset(video_data_dir):
    """cityflow-routing's extracted layout: AIC22_Track1/<split>/<S>/<c>/vdo.avi.
    Frames alternate dark/bright in blocks of 10."""
    root = C.cf_dataset.extracted_dir(video_data_dir)
    for scenario, cams in CAMERAS.items():
        split = "train" if scenario == TRAIN else "validation"
        for camera, n_frames in cams:
            cam_dir = os.path.join(root, split, scenario, camera)
            os.makedirs(cam_dir, exist_ok=True)
            writer = cv2.VideoWriter(os.path.join(cam_dir, "vdo.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                     10.0, (160, 120))
            for i in range(n_frames):
                value = 200 if (i // 10) % 2 else 20
                writer.write(np.full((120, 160, 3), value, dtype=np.uint8))
            writer.release()


def fake_embed(frames):
    rng = np.random.default_rng(len(frames))
    emb = rng.normal(0, 0.1, (len(frames), C.yd.EMBEDDING_DIM)).astype(np.float32)
    emb[:, 0] = [1.0 if is_bright(f) else -1.0 for f in frames]
    return emb


REF = det([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]], [2, 2, 2, 2])


def fake_detectors():
    """yolov8l = REF. n finds all 4 boxes on dark frames, 1 on bright;
    s finds 4 / 2; m and l always 4. So ideal = n (dark), m (bright)."""
    def make(dark_k, bright_k):
        def detect(frames):
            out = []
            for f in frames:
                k = bright_k if is_bright(f) else dark_k
                out.append(det(REF["boxes"][:k], REF["cls"][:k]))
            return out
        return detect
    return {"yolov8n": make(4, 1), "yolov8s": make(4, 2), "yolov8m": make(4, 4), "yolov8l": make(4, 4)}


def write_fake_coco_results(coco_dir, n_points=3):
    """Pareto front + model_cache heads in yolo-dispatcher's format; every
    head sends every frame to yolov8l."""
    os.makedirs(os.path.join(coco_dir, "nsga2"), exist_ok=True)
    cache = os.path.join(coco_dir, "eval", "model_cache")
    os.makedirs(cache, exist_ok=True)
    rows = []
    for point in range(n_points):
        chromosome = np.append(np.full(12, 10.0 + point), 0.5)   # scheme gene 0.5 -> INS
        W = np.zeros((4, C.yd.EMBEDDING_DIM), dtype=np.float32)
        b = np.array([0, 0, 0, 3], dtype=np.float32)
        np.savez(os.path.join(cache, f"individual_{point}.npz"), W=W, b=b, chromosome=chromosome, scheme="INS")
        rows.append({"individual": point, "scheme": "INS", "scheme_gene": 0.5,
                     **{f"P_{i}": chromosome[i] for i in range(12)}})
    pd.DataFrame(rows).to_csv(os.path.join(coco_dir, "nsga2", "pareto_front.csv"), index=False)
    summary = pd.DataFrame({"individual": range(n_points), "alpha_sys": 0.9, "avg_model_cost": 170.0})
    for split in ("test", "final_val"):
        summary.to_csv(os.path.join(coco_dir, "eval", f"{split}_summary.csv"), index=False)


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "TRAIN_SCENARIOS", [TRAIN])
    monkeypatch.setattr(C, "EVAL_SCENARIOS", [EVAL])
    video = tmp_path / "videos"
    data = tmp_path / "data"
    write_fake_dataset(str(video))
    return {"video": str(video), "data": str(data), "tmp": tmp_path}


def run_label(ws, stride=2):
    label.main(["--scenarios", TRAIN, EVAL, "--video-data-dir", ws["video"], "--data-dir", ws["data"],
                "--stride", str(stride), "--batch", "7"], embed=fake_embed, detectors=fake_detectors())


# ---------------------------------------------------------------- label.py

def test_label_end_to_end(workspace):
    run_label(workspace)
    labels = os.path.join(workspace["data"], "labels")
    assert sorted(os.listdir(labels)) == ["S98_c981.csv.gz", "S98_c982.csv.gz", "S99_c991.csv.gz"]

    table = pd.read_csv(os.path.join(labels, "S98_c981.csv.gz"))
    assert list(table["frame_idx"]) == list(range(1, 61, 2))            # stride 2, 1-based
    assert set(table["role"]) == {"train"} and set(table["split"]) == {"train"}
    # frame i (0-based) is bright when (i // 10) is odd
    bright = ((table["frame_idx"] - 1) // 10) % 2 == 1
    assert (table.loc[bright, "ideal_model"] == "yolov8m").all()
    assert (table.loc[~bright, "ideal_model"] == "yolov8n").all()
    assert (table.loc[bright, "recall_n"] == 0.25).all() and (table["recall_l"] == 1.0).all()

    saved = np.load(os.path.join(workspace["data"], "embeddings", "S98_c981.npz"))
    assert saved["embeddings"].shape == (30, C.yd.EMBEDDING_DIM)
    assert saved["embeddings"].dtype == np.float32
    np.testing.assert_array_equal(saved["frame_idx"], table["frame_idx"].values)
    assert set(pd.read_csv(os.path.join(labels, "S99_c991.csv.gz"))["role"]) == {"eval"}


def test_label_resume_skips_done_cameras(workspace):
    run_label(workspace)
    path = os.path.join(workspace["data"], "labels", "S98_c981.csv.gz")
    before = os.path.getmtime(path)
    os.remove(os.path.join(workspace["data"], "labels", "S99_c991.csv.gz"))
    label.main(["--scenarios", TRAIN, EVAL, "--video-data-dir", workspace["video"],
                "--data-dir", workspace["data"], "--stride", "2"],
               embed=fake_embed, detectors=fake_detectors())
    assert os.path.getmtime(path) == before
    assert os.path.exists(os.path.join(workspace["data"], "labels", "S99_c991.csv.gz"))


def test_label_refuses_unknown_scenario(workspace):
    with pytest.raises(ValueError):
        label.main(["--scenarios", "S42", "--video-data-dir", workspace["video"], "--data-dir", workspace["data"]],
                   embed=fake_embed, detectors=fake_detectors())


# ---------------------------------------------------------------- finetune.py

def test_finetune_end_to_end(workspace):
    run_label(workspace)
    coco = str(workspace["tmp"] / "coco_results")
    write_fake_coco_results(coco)
    results = str(workspace["tmp"] / "results")
    finetune.main(["--coco-results", coco, "--data-dir", workspace["data"], "--results-dir", results,
                   "--epochs", "40", "--lr", "0.05", "--batch", "16"])

    eval_dir = os.path.join(results, "eval")
    coco_eval = pd.read_csv(os.path.join(eval_dir, "coco_heads_eval_summary.csv"))
    ft_eval = pd.read_csv(os.path.join(eval_dir, "final_val_summary.csv"))
    base = pd.read_csv(os.path.join(eval_dir, "baselines_eval.csv")).set_index("policy")

    # COCO heads: everything to yolov8l, always correct, most expensive
    assert (coco_eval["frac_yolov8l"] == 1.0).all() and (coco_eval["alpha_sys"] == 1.0).all()
    # fine-tuned: easy frames moved to yolov8n, still correct, cheaper
    assert (ft_eval["frac_yolov8n"] > 0.3).all()
    assert (ft_eval["alpha_sys"] == 1.0).all()
    assert (ft_eval["avg_model_cost"] < coco_eval["avg_model_cost"]).all()
    assert base.loc["oracle", "alpha_sys"] == 1.0
    assert base.loc["always_yolov8n", "alpha_sys"] < 1.0

    # cityflow-routing loads the fine-tuned heads unchanged
    points, heads = C.cf_heads.load_pareto(results)
    assert sorted(points["point_id"]) == [0, 1, 2]
    assert all(p.startswith(os.path.join(eval_dir, "model_cache")) for p in points["weights_path"])
    W, b = heads[0]
    assert W.shape == (4, C.yd.EMBEDDING_DIM)
    for name in ("summary.md", "label_distribution.csv", "finetune_manifest.json",
                 os.path.join("plots", "front_eval.png"), os.path.join("plots", "label_split.png")):
        assert os.path.exists(os.path.join(eval_dir, name)), name


def test_finetune_refuses_overlapping_data(workspace, monkeypatch):
    run_label(workspace)
    # relabel the eval camera as train: same scenario can't be in both roles
    path = os.path.join(workspace["data"], "labels", "S98_c982.csv.gz")
    table = pd.read_csv(path)
    table["role"] = "eval"
    table.to_csv(path, index=False, compression="gzip")
    coco = str(workspace["tmp"] / "coco_results")
    write_fake_coco_results(coco)
    with pytest.raises(SystemExit):
        finetune.main(["--coco-results", coco, "--data-dir", workspace["data"],
                       "--results-dir", str(workspace["tmp"] / "results"), "--epochs", "1"])


def test_preprocessing_matches_cityflow_routing():
    """label.preprocess_bgr must be the exact transform route.py uses."""
    sys.path.insert(0, C.CITYFLOW_ROUTING_DIR)
    try:
        import route
    finally:
        sys.path.remove(C.CITYFLOW_ROUTING_DIR)
    frame = np.random.default_rng(0).integers(0, 255, (270, 480, 3), dtype=np.uint8)
    np.testing.assert_array_equal(label.preprocess_bgr(frame).numpy(), route.preprocess_bgr(frame).numpy())

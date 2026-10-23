"""
CPU tests for cityflow-routing. A fake CityFlow scenario (2 cameras at 10 and
8 FPS, CRLF timestamp file) and a fake Pareto front (3 random Linear(448, 4)
heads, saved in yolo-dispatcher's exact formats) go through extract, route and
analyze end to end. The extractor is the real yolov8n backbone (yolov8n.pt is
downloaded into yolo-dispatcher/models/ if missing).

Run: python -m pytest tests/ -q     (from cityflow-routing/)
"""

import os
import sys
import types
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(REPO, "dispatcher"))   # for save_results.save_pareto_front

import analyze  # noqa: E402
import dataset  # noqa: E402
import extract  # noqa: E402
import heads  # noqa: E402
import metrics  # noqa: E402
import route  # noqa: E402

SCENARIO = "S99"
CAMERAS = [("c901", 10.0, 120, 0.0), ("c902", 8.0, 96, 1.5)]   # name, fps, frames, offset
N_POINTS = 3


# ---------------------------------------------------------------- fakes

def write_fake_dataset(root):
    """train/S99/c90x/vdo.avi + cam_timestamp/S99.txt (CRLF) + cam_framenum/S99.txt."""
    rng = np.random.default_rng(0)
    for name, fps, n_frames, _ in CAMERAS:
        cam_dir = os.path.join(root, "train", SCENARIO, name)
        os.makedirs(cam_dir, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(cam_dir, "vdo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
        for i in range(n_frames):
            frame = rng.integers(0, 60, (240, 320, 3), dtype=np.uint8)
            x = (5 * i) % 280   # a bright box moving across, so frames differ
            frame[80:160, x:x + 40] = (40, 200, 240)
            writer.write(frame)
        writer.release()
    for folder, values in [("cam_timestamp", [c[3] for c in CAMERAS]), ("cam_framenum", [c[2] for c in CAMERAS])]:
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        with open(os.path.join(root, folder, f"{SCENARIO}.txt"), "w", newline="") as f:
            f.write("".join(f"{c[0]} {v}\r\n" for c, v in zip(CAMERAS, values)))


def embedding_stats(root):
    """Mean and std of the real extractor's embeddings over some fake frames,
    so the fake heads can be centred and routing switches between frames
    (plain random heads put every frame on one model)."""
    import torch

    extractor = route.yd_config.build_feature_extractor("cpu").eval()
    capture, _, _ = dataset.open_video(os.path.join(root, "train", SCENARIO, CAMERAS[0][0], "vdo.avi"))
    batch = [route.preprocess_bgr(f) for i, f in dataset.iter_frames(capture) if i % 4 == 0]
    with torch.no_grad():
        emb = extractor(torch.stack(batch)).numpy()
    return emb.mean(axis=0), emb.std(axis=0) + 1e-6


def write_fake_yolo_results(results_dir, mean, std):
    """pareto_front.csv via the real dispatcher/save_results.py, weights npz
    with the same fields as dispatcher/save_models.py and
    dispatcher_analysis/build_models.py, and the two eval summaries."""
    from save_results import save_pareto_front

    rng = np.random.default_rng(1)
    cfg = types.SimpleNamespace(NUM_CLASSES=4, N_GENES=13, SCHEME_GENE_LOWER_BOUND=0.0, SCHEME_GENE_UPPER_BOUND=3.0,
                                NSGA2_DIR=os.path.join(results_dir, "nsga2"), MODEL_COST_UNIT="GFLOPs",
                                PARETO_FRONT_CSV=os.path.join(results_dir, "nsga2", "pareto_front.csv"))
    chromosomes = np.column_stack([rng.uniform(0, 100, (N_POINTS, 12)), rng.uniform(0, 3, N_POINTS)])
    objectives = np.column_stack([rng.uniform(0.2, 0.4, N_POINTS), rng.uniform(20, 160, N_POINTS)])
    save_pareto_front(chromosomes, objectives, types.SimpleNamespace(info=lambda *a: None), cfg)

    for sub in [("nsga2", "models"), ("eval", "model_cache")]:
        os.makedirs(os.path.join(results_dir, *sub), exist_ok=True)
    for i, chromosome in enumerate(chromosomes):
        R = rng.normal(0, 3 / np.sqrt(448), (4, 448))
        W = (R / std).astype(np.float32)                  # logits = R @ standardised embedding
        b = (-W @ mean).astype(np.float32)
        for sub in [("nsga2", "models"), ("eval", "model_cache")]:
            np.savez(os.path.join(results_dir, *sub, f"individual_{i}.npz"),
                     W=W, b=b, chromosome=chromosome, scheme="INS")
    for split, alphas in [("test", [0.70, 0.74, 0.72]), ("final_val", [0.69, 0.75, 0.73])]:
        pd.DataFrame({"individual": range(N_POINTS), "alpha_sys": alphas,
                      "accuracy_exact_match_vs_ideal": 0.3, "avg_model_cost": [30.0, 120.0, 80.0]}
                     ).to_csv(os.path.join(results_dir, "eval", f"{split}_summary.csv"), index=False)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Fake data + fake front, then one route.py run. Returns its paths."""
    base = tmp_path_factory.mktemp("cityflow")
    data_dir = str(base / "data")
    root = dataset.extracted_dir(data_dir)
    write_fake_dataset(root)
    yolo_results = str(base / "yolo_results")
    write_fake_yolo_results(yolo_results, *embedding_stats(root))
    args = ["--scenarios", SCENARIO, "--run-name", "test", "--data-dir", data_dir,
            "--yolo-results", yolo_results, "--results-dir", str(base / "results"), "--batch", "16"]
    route.main(args)
    return {"base": base, "args": args, "run_dir": str(base / "results" / "test"),
            "raw": str(base / "results" / "test" / "raw"), "yolo_results": yolo_results}


# ---------------------------------------------------------------- route

def test_route_writes_raw_files(run):
    for name, fps, n_frames, offset in CAMERAS:
        emb = np.load(os.path.join(run["raw"], f"embeddings_{SCENARIO}_{name}.npz"))
        assert emb["embeddings"].dtype == np.float16
        assert emb["embeddings"].shape == (n_frames, 448)
        assert list(emb["frame_idx"]) == list(range(1, n_frames + 1))

        table = pd.read_csv(os.path.join(run["raw"], f"routing_{SCENARIO}_{name}.csv.gz"))
        assert len(table) == n_frames
        assert table["fps"].iloc[0] == pytest.approx(fps)          # read from the video
        expected_video = (table["frame_idx"] - 1) / fps
        assert np.allclose(table["video_time_s"], expected_video, atol=1e-4)
        assert np.allclose(table["synced_time_s"], expected_video + offset, atol=1e-4)
        for p in range(N_POINTS):
            probs = table[[f"prob_{x}_p{p}" for x in "nsml"]].to_numpy()
            assert np.allclose(probs.sum(axis=1), 1, atol=2e-3)
            names = np.array(["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
            assert (table[f"model_p{p}"] == names[probs.argmax(axis=1)]).mean() > 0.99


def test_heads_reapply_to_saved_embeddings(run):
    """Saved float16 embeddings reproduce the routing (up to float16 rounding)."""
    _, loaded = heads.load_pareto(run["yolo_results"])
    emb = np.load(os.path.join(run["raw"], f"embeddings_{SCENARIO}_c901.npz"))["embeddings"].astype(np.float32)
    table = pd.read_csv(os.path.join(run["raw"], f"routing_{SCENARIO}_c901.csv.gz"))
    names = np.array(["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
    for point_id, (_, chosen) in heads.apply_heads(emb, loaded).items():
        assert (names[chosen] == table[f"model_p{point_id}"].to_numpy()).mean() > 0.98


def test_pareto_points_and_manifest(run):
    points = pd.read_csv(os.path.join(run["raw"], "pareto_points.csv"))
    assert list(points["point_id"]) == list(range(N_POINTS))
    assert all("model_cache" in p for p in points["weights_path"])
    assert list(points["final_val_alpha_sys"]) == [0.69, 0.75, 0.73]
    import json
    manifest = json.load(open(os.path.join(run["raw"], "manifest.json")))
    assert manifest["model_names"] == ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
    assert manifest["cameras"][f"{SCENARIO}_c902"]["frames_decoded"] == 96
    assert manifest["cameras"][f"{SCENARIO}_c902"]["fps"] == pytest.approx(8.0)
    assert manifest["dataset"]["gdrive_file_id"] == dataset.DATASET_FILE_ID
    assert "gpu" in manifest["environment"] and "torch" in manifest["environment"]


def test_resume_skips_finished_cameras(run):
    paths = [os.path.join(run["raw"], f"routing_{SCENARIO}_{c[0]}.csv.gz") for c in CAMERAS]
    before = [os.path.getmtime(p) for p in paths]
    route.main(run["args"])
    assert [os.path.getmtime(p) for p in paths] == before
    log = open(os.path.join(run["run_dir"], "route.log")).read()
    assert log.count("already routed, skipping") == len(CAMERAS)
    assert "Nothing to do." in log


# ---------------------------------------------------------------- analyze

def test_analyze_end_to_end(run):
    analyze.main(["--run", run["run_dir"]])
    out = os.path.join(run["run_dir"], "analysis")
    summary = pd.read_csv(os.path.join(out, "summary.csv"))
    assert len(summary) == N_POINTS                     # 3 points x 1 scenario
    assert np.allclose(summary[[f"share_{x}" for x in "nsml"]].sum(axis=1), 1, atol=1e-3)   # 4-decimal rounding
    md = open(os.path.join(out, "summary.md")).read()
    assert "Main point: **1**" in md                    # highest final_val alpha_sys (0.75)
    assert "above 90:10" in md
    point_dir = os.path.join(out, "point_1")
    for name, *_ in CAMERAS:
        assert os.path.exists(os.path.join(point_dir, f"timeline_{SCENARIO}_{name}.png"))
    for f in ["run_lengths.png", "run_lengths.csv", "split.png", f"timeline_{SCENARIO}.png",
              f"drift_{SCENARIO}.png", f"transitions_{SCENARIO}.png", f"arrival_rates_{SCENARIO}.png"]:
        assert os.path.exists(os.path.join(point_dir, f)), f
    rates = pd.read_csv(os.path.join(point_dir, f"arrival_rates_{SCENARIO}.csv"))
    assert rates["total"].sum() == sum(c[2] for c in CAMERAS)   # every frame counted once


# ---------------------------------------------------------------- metrics, by hand

def test_run_lengths_split_switches_by_hand():
    seq = [0, 0, 0, 1, 1, 3, 0, 0]   # n n n s s l n n
    assert metrics.run_lengths(seq) == [(0, 3), (1, 2), (3, 1), (0, 2)]
    assert metrics.switches(seq) == 3
    assert np.allclose(metrics.split(seq, 4), [5 / 8, 2 / 8, 0, 1 / 8])
    counts = metrics.transition_counts(seq, 4)
    assert counts[0, 0] == 3 and counts[0, 1] == 1 and counts[1, 1] == 1 and counts[1, 3] == 1 and counts[3, 0] == 1
    assert counts.sum() == 7
    assert np.allclose(metrics.row_normalize(counts)[0], [0.75, 0.25, 0, 0])


def test_scenario_summary_by_hand():
    seq = [0, 0, 0, 1, 1, 3, 0, 0]
    df = pd.DataFrame({"scenario": "S99", "camera": "c1", "frame_idx": range(1, 9), "fps": 10.0,
                       "video_time_s": np.arange(8) / 10, "synced_time_s": np.arange(8) / 10, "model": seq})
    row = analyze.scenario_rows(df, ["yolov8n", "yolov8s", "yolov8m", "yolov8l"], point_id=0)[0]
    assert row["share_n"] == 0.625 and row["share_s"] == 0.25 and row["share_l"] == 0.125
    assert row["mean_run_frames_n"] == 2.5 and row["mean_run_frames_s"] == 2.0 and row["mean_run_frames_l"] == 1.0
    assert np.isnan(row["mean_run_frames_m"])
    assert row["switches_per_min"] == pytest.approx(3 / (8 / 10 / 60))   # 22.5
    assert row["heaviest_model"] == "yolov8n" and not row["above_90_10"]


def test_merge_synced_orders_frames():
    a = pd.DataFrame({"camera": "c1", "frame_idx": [1, 2, 3], "synced_time_s": [0.0, 0.1, 0.2], "model": [0, 0, 0]})
    b = pd.DataFrame({"camera": "c2", "frame_idx": [1, 2, 3], "synced_time_s": [0.05, 0.15, 0.2], "model": [3, 3, 3]})
    merged = metrics.merge_synced(pd.concat([b, a]))
    assert list(zip(merged["camera"], merged["frame_idx"])) == [
        ("c1", 1), ("c2", 1), ("c1", 2), ("c2", 2), ("c1", 3), ("c2", 3)]   # tie at 0.2: camera order


def test_window_counts_and_max_share():
    times = [0.0, 0.5, 1.0, 1.2, 1.9, 2.5]
    models = [0, 1, 0, 0, 0, 1]
    counts = metrics.window_counts(times, models, 1.0, 2)
    assert counts.to_numpy().tolist() == [[1, 1], [3, 0], [0, 1]]
    # window 3 has 1 frame (< half the median of 2), so it is ignored; window 2 is all model 0
    assert metrics.max_window_share(times, models, 1.0, 2) == 1.0


# ---------------------------------------------------------------- preprocessing parity

def test_preprocessing_matches_yolo_dispatcher(tmp_path):
    """Same image through yolo-dispatcher's path (PIL open + convert RGB +
    IMAGE_TRANSFORM) and route.py's path (cv2 BGR decode + preprocess_bgr)
    gives the same tensor and the same embedding."""
    import torch
    from PIL import Image

    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)   # 16:9, like CityFlow
    image[:, :, 0] = np.linspace(0, 255, 1280, dtype=np.uint8)     # channel-dependent pattern
    png = str(tmp_path / "frame.png")                               # lossless: same pixels both ways
    cv2.imwrite(png, image)

    training_tensor = route.yd_config.IMAGE_TRANSFORM(Image.open(png).convert("RGB"))
    routing_tensor = route.preprocess_bgr(cv2.imread(png))
    assert torch.equal(training_tensor, routing_tensor)

    unconverted = route.yd_config.IMAGE_TRANSFORM(Image.fromarray(cv2.imread(png)))   # BGR fed as RGB
    assert not torch.equal(training_tensor, unconverted)            # the check has teeth

    extractor = route.yd_config.build_feature_extractor("cpu").eval()
    with torch.no_grad():
        a = extractor(training_tensor[None])
        b = extractor(routing_tensor[None])
    assert a.shape == (1, 448)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------- extract

def test_extract_only_chosen_scenario(tmp_path, capsys):
    """A fake zip with two scenarios: only S99's vdo.avi and timing files come
    out, gt files stay in, and a second run skips files already there."""
    src = tmp_path / "src"
    write_fake_dataset(str(src))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with zipfile.ZipFile(dataset.zip_path(str(data_dir)), "w") as zf:
        for folder, _, files in os.walk(src):
            for f in files:
                full = os.path.join(folder, f)
                zf.write(full, os.path.relpath(full, src).replace(os.sep, "/"))
        zf.writestr(f"train/{SCENARIO}/c901/gt/gt.txt", "1,1,0,0,10,10,1,-1,-1,-1\n")
        zf.writestr("validation/S98/c950/vdo.avi", b"not a real video")
        zf.writestr("cam_timestamp/S98.txt", "c950 0\r\n")
        zf.writestr("cam_framenum/S98.txt", "c950 1\r\n")
        zf.writestr("ReadMe.txt", "readme")

    extract.main(["--data-dir", str(data_dir), "--scenarios", SCENARIO])
    out = dataset.extracted_dir(str(data_dir))
    assert os.path.exists(os.path.join(out, "train", SCENARIO, "c901", "vdo.avi"))
    assert os.path.exists(os.path.join(out, "cam_timestamp", f"{SCENARIO}.txt"))
    assert os.path.exists(os.path.join(out, "ReadMe.txt"))
    assert not os.path.exists(os.path.join(out, "train", SCENARIO, "c901", "gt"))
    assert not os.path.exists(os.path.join(out, "validation"))
    assert dataset.read_offsets(out, SCENARIO) == {"c901": 0.0, "c902": 1.5}   # CRLF parsed

    capsys.readouterr()
    extract.main(["--data-dir", str(data_dir), "--scenarios", SCENARIO])
    assert "extracted" not in capsys.readouterr().out.replace("already there", "")

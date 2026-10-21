import gzip

import numpy as np
import pandas as pd

import traces
from common import Profile, capacity_fps

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
SPLIT = [0.528, 0.149, 0.083, 0.240]


def test_periodic_phases_are_staggered_and_periodic():
    df = traces.periodic(8, 10.0, 5.0, SPLIT, seed=1)
    firsts = df.groupby("camera")["t_arrival_s"].min()
    assert len(firsts) == 8
    assert firsts.nunique() == 8 and (firsts > 0).all() and (firsts < 0.1).all()
    for _, cam in df.groupby("camera"):
        gaps = np.diff(cam["t_arrival_s"].values)
        assert np.allclose(gaps, 0.1)
    assert (np.diff(df["t_arrival_s"].values) >= 0).all()
    assert list(df["job_id"]) == list(range(len(df)))


def test_split_within_tolerance():
    df = traces.periodic(8, 50.0, 60.0, SPLIT, seed=2)
    got = traces.split_of(df)
    assert np.allclose(got, SPLIT, atol=0.01)
    sticky = traces.periodic(8, 50.0, 60.0, SPLIT, sticky=0.9, seed=2)
    assert np.allclose(traces.split_of(sticky), SPLIT, atol=0.03)
    # sticky routing: far fewer switches between consecutive frames of a camera
    def switches(d):
        return sum((c["queue"].values[1:] != c["queue"].values[:-1]).sum() for _, c in d.groupby("camera"))
    assert switches(sticky) < 0.3 * switches(df)


def test_same_seed_same_trace():
    a = traces.poisson(30, 20, SPLIT, seed=5)
    b = traces.poisson(30, 20, SPLIT, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert abs(len(a) / 20.0 - 30) < 5


def test_load_fraction_and_fps_for_load():
    prof = Profile.from_table({m: {1: 5 * (i + 1), 16: 30 * (i + 1)} for i, m in enumerate(MODELS)})
    fps = traces.fps_for_load(0.5, prof, 8, SPLIT, b=16)
    df = traces.periodic(8, fps, 30.0, SPLIT, seed=3)
    assert abs(traces.load_fraction(df, prof, b=16) - 0.5) < 0.05
    cap = capacity_fps(prof, MODELS, SPLIT, 16)
    assert abs(fps * 8 / cap - 0.5) < 1e-9


def test_cityflow_replay_preserves_times_and_queues(tmp_path):
    raw = tmp_path / "run" / "raw"
    raw.mkdir(parents=True)
    rows = {
        "c006": [(1, 0.5, "yolov8m"), (2, 0.6, "yolov8l"), (3, 0.7, "yolov8m")],
        "c007": [(1, 0.55, "yolov8l"), (2, 0.65, "yolov8l")],
    }
    for cam, frames in rows.items():
        with gzip.open(str(raw / ("routing_S02_%s.csv.gz" % cam)), "wt") as f:
            f.write("scenario,camera,frame_idx,fps,video_time_s,synced_time_s,model_p80,prob_n_p80,model_p3\n")
            for idx, t, m in frames:
                f.write("S02,%s,%d,10.0,%.2f,%.2f,%s,0.1,yolov8n\n" % (cam, idx, t, t, m))
    df = traces.from_cityflow(str(tmp_path / "run"), "S02", 80)
    assert list(df["camera"]) == ["c006", "c007", "c006", "c007", "c006"]
    assert np.allclose(df["t_arrival_s"], [0.0, 0.05, 0.1, 0.15, 0.2])
    assert list(df["queue"]) == ["yolov8m", "yolov8l", "yolov8l", "yolov8l", "yolov8m"]
    assert len(traces.from_cityflow(str(tmp_path / "run"), "S02", 80, cameras=["c007"])) == 2
    assert len(traces.from_cityflow(str(tmp_path / "run"), "S02", 80, duration_s=0.12)) == 3
    path = str(tmp_path / "t.csv")
    meta = traces.write_trace(df, path, {"source": "cityflow"})
    back, meta2 = traces.read_trace(path)
    assert meta2["jobs"] == 5 and meta2["hash"] == meta["hash"]
    assert list(back["queue"]) == list(df["queue"])

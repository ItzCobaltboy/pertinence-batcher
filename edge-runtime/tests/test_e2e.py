import json
import os

import analyze
import config
import plan
import traces
from common import Profile

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
TABLE = {m: {1: 2 * (i + 1), 2: 3 * (i + 1), 4: 4 * (i + 1), 8: 6 * (i + 1), 16: 10 * (i + 1)}
         for i, m in enumerate(MODELS)}


def test_plan_expansion_and_budget(tmp_path):
    p = dict(plan.EXAMPLE, traces=[str(tmp_path / "a.csv"), str(tmp_path / "b.csv")], slos_ms=[50, 100],
             duration_s=30, cooldown_s=0)
    runs = plan.expand(p)
    # 6 single policies + 6 triton delays = 12 per (trace, slo)
    assert len(runs) == 2 * 2 * 12
    assert runs[0]["run_id"] == "fcfs_nobatch__a__slo50__tensor-arrival__r0"
    assert any(r["run_id"].startswith("triton-maxque10__") for r in runs)
    assert len({r["run_id"] for r in runs}) == len(runs)


def test_fake_plan_every_policy_then_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    dev = config.device_dir("fake")
    os.makedirs(dev)
    prof = Profile.from_table(TABLE)
    with open(os.path.join(dev, "profile.json"), "w") as f:
        json.dump(prof.data, f)
    tr = []
    for load in (0.3, 0.6):
        path = str(tmp_path / ("periodic_load%d.csv" % (load * 100)))
        fps = traces.fps_for_load(load, prof, 4, config.ROUTING_SPLIT)
        df = traces.periodic(4, fps, 2.5, config.ROUTING_SPLIT, seed=1)
        traces.write_trace(df, path, {"source": "periodic", "target_load": load,
                                      "load_fraction": traces.load_fraction(df, prof)})
        tr.append(path)
    policies = [{"name": n} for n in ["fcfs_nobatch", "fcfs_batch", "lqf", "edgeserving", "clockwork",
                                      "symphony", "cost_v1"]]
    policies.append({"name": "triton", "grid": {"max_queue_delay_ms": [0, 10]}})
    p = {"device_tag": "fake", "policies": policies, "traces": tr, "slos_ms": [60], "input": "tensor",
         "prep": "arrival", "duration_s": None, "warmup_s": 0.5, "repeats": 1, "cooldown_s": 0, "fake": True,
         "idle_s": 0.2, "power": "fake", "b_max": 16}
    done = plan.run_plan(p, "fake", log=lambda *a: None)
    assert len(done) == 2 * 9
    status = dict(done)
    # one-at-a-time serving cannot keep up with 60% of the max-batching capacity
    assert status.pop("fcfs_nobatch__periodic_load60__slo60__tensor-arrival__r0") == "collapsed"
    assert all(s == "ok" for s in status.values()), status
    for run_id, _ in done:
        run_dir = os.path.join(dev, "runs", run_id)
        for name in ["requests.csv", "batches.csv", "waits.csv", "decisions.csv", "power.csv.raw.gz", "run.json"]:
            assert os.path.exists(os.path.join(run_dir, name)), (run_id, name)
    # resumable: nothing left to do
    assert plan.run_plan(p, "fake", log=lambda *a: None) == []

    df = analyze.analyze(dev, timelines=True)
    assert len(df) == 18
    out = os.path.join(dev, "analysis")
    assert os.path.exists(os.path.join(out, "summary.csv")) and os.path.exists(os.path.join(out, "summary.md"))
    plots = os.listdir(os.path.join(out, "plots"))
    for prefix in ["cdf_", "p95_vs_load_", "energy_vs_load_", "pareto_", "batches_"]:
        assert any(x.startswith(prefix) for x in plots), prefix
    assert len(os.listdir(os.path.join(out, "plots", "timelines"))) == 18
    lqf = df[df["label"] == "lqf"]
    assert (lqf["miss_rate"] < 0.05).all()
    assert (df["energy_per_frame_j"] > 0).all()
    assert set(df["load"].round(1)) == {0.3, 0.6}

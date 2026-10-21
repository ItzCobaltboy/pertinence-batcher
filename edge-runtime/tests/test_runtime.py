import os
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest

from common import Profile
from inputs import FakeInputs
from models import FakeRunner
from policies import Drop, Promote, Serve, Wait
from runtime import InvalidDecision, Runtime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
PROFILE = Profile.from_table({m: {1: 2 * (i + 1), 2: 3 * (i + 1), 4: 4 * (i + 1), 8: 6 * (i + 1)}
                              for i, m in enumerate(MODELS)})


def make_trace(rows):
    """rows: (t, queue)"""
    return pd.DataFrame({"job_id": range(len(rows)), "camera": ["c"] * len(rows),
                         "t_arrival_s": [r[0] for r in rows], "queue": [r[1] for r in rows],
                         "slo_ms": [float("nan")] * len(rows), "frame_ref": range(len(rows))})


class Scripted:
    """Calls fn(state, n_calls) for every decision."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self.seen = []

    def decide(self, state):
        self.calls += 1
        self.seen.append({m: [j.job_id for j in q] for m, q in state.queues.items()})
        return self.fn(state, self.calls)


def runtime(trace_rows, fn, **kw):
    policy = Scripted(fn)
    rt = Runtime(MODELS, policy, PROFILE, FakeRunner(PROFILE), FakeInputs(), make_trace(trace_rows), 100.0,
                 b_max=8, log=lambda *a: None, **kw)
    return rt, policy


def serve_all(state, _):
    m = [m for m in state.models if state.queues[m]][0]
    return Serve(m, min(len(state.queues[m]), 8))


@pytest.mark.parametrize("bad", [
    lambda s: Serve("yolov8n", 5),                 # more than queued
    lambda s: Serve("yolovX", 1),                  # unknown queue
    lambda s: Serve("yolov8n", 1, [999]),          # job not queued
    lambda s: Wait(s.now - 1.0),                   # in the past
    lambda s: Promote("yolov8l", "yolov8n", 1),    # to a smaller model
    lambda s: Promote("yolov8n", "yolov8l", 3),    # more than queued
    lambda s: Drop("yolov8n", [999]),
    lambda s: "serve please",
])
def test_executor_rejects_invalid_decisions(bad):
    rt, _ = runtime([(0.0, "yolov8n")], lambda s, n: bad(s))
    with pytest.raises(InvalidDecision):
        rt.run()


def test_promote_moves_the_oldest_jobs_in_arrival_order():
    rows = [(0.000, "yolov8n"), (0.001, "yolov8l"), (0.002, "yolov8n"), (0.003, "yolov8n"), (0.004, "yolov8l")]

    def fn(state, n):
        if not hasattr(fn, "done"):
            if sum(len(q) for q in state.queues.values()) < 5:
                return Wait(state.now + 5.0)
            fn.done = True
            return Promote("yolov8n", "yolov8l", 2)
        return serve_all(state, n)

    rt, policy = runtime(rows, fn)
    assert rt.run() == "ok"
    first_after_promote = [s for s in policy.seen if s["yolov8n"] == [3]][0]
    assert first_after_promote["yolov8l"] == [0, 1, 2, 4]          # merged by arrival
    jobs = {j.job_id: j for j in rt.jobs}
    assert jobs[0].promoted_from == "yolov8n" and jobs[0].model_run == "yolov8l"
    assert jobs[3].model_run == "yolov8n" and jobs[3].promoted_from == ""
    assert len(rt.promotions) == 1 and rt.promotions[0]["k"] == 2


def test_wait_wakes_on_arrival():
    rows = [(0.0, "yolov8n"), (0.2, "yolov8s")]

    def fn(state, n):
        if n == 1:
            return Wait(state.now + 10.0)
        return serve_all(state, n)

    rt, _ = runtime(rows, fn)
    t0 = time.time()
    assert rt.run() == "ok"
    assert time.time() - t0 < 1.0
    waits = [w for w in rt.waits if w["kind"] == "policy"]
    assert waits[0]["woke"] == "arrival"
    assert 0.15 < waits[0]["t_end"] - waits[0]["t_start"] < 0.3


def test_drop_and_drain_and_lateness_logged(tmp_path):
    rows = [(0.0, "yolov8m"), (0.01, "yolov8m"), (0.02, "yolov8m")]

    def fn(state, n):
        q = state.queues["yolov8m"]
        if q and q[0].job_id == 0:
            return Drop("yolov8m", [0])
        return Serve("yolov8m", len(q))

    rt, _ = runtime(rows, fn)
    assert rt.run() == "ok"
    counts = rt.write_logs(str(tmp_path))
    assert counts["dropped"] == 1 and counts["unserved"] == 0
    req = pd.read_csv(str(tmp_path / "requests.csv"))
    assert (req["t_actual"] - req["t_target"]).abs().max() < 0.05
    assert req.loc[req.job_id == 0, "dropped"].item() == 1
    assert req.loc[req.job_id == 2, "t_end"].notna().item()


def test_collapse_guard():
    rows = [(0.001 * i, "yolov8l") for i in range(50)]
    rt, _ = runtime(rows, lambda s, n: Wait(s.now + 10.0), max_queued=10)
    assert rt.run() == "collapsed"


def test_sampler_dies_with_a_killed_run(tmp_path):
    prof = tmp_path / "profile.json"
    import json
    json.dump(PROFILE.data, open(str(prof), "w"))
    trace = tmp_path / "trace.csv"
    make_trace([(0.05 * i, "yolov8n") for i in range(400)]).to_csv(str(trace), index=False)
    pidfile = str(tmp_path / "smi.pid")
    env = dict(os.environ, EDGE_RUNTIME_NVIDIA_SMI="%s %s" % (sys.executable, os.path.join(HERE, "fake_nvidia_smi.py")),
               FAKE_SMI_PIDFILE=pidfile)
    for sig in (signal.SIGKILL, signal.SIGTERM):
        if os.path.exists(pidfile):
            os.remove(pidfile)
        proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "runtime.py"), "--fake", "--power", "nvidia-smi",
                                 "--policy", "lqf", "--trace", str(trace), "--profile", str(prof),
                                 "--out", str(tmp_path / ("run%d" % sig)), "--idle-s", "0"], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 15
        while not os.path.exists(pidfile) and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        pid = int(open(pidfile).read())
        os.kill(pid, 0)                                  # sampler is running
        proc.send_signal(sig)
        proc.wait(timeout=10)
        deadline = time.time() + 5
        gone = False
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
                with open("/proc/%d/stat" % pid) as f:   # a zombie counts as gone
                    if f.read().split()[2] == "Z":
                        gone = True
                        break
            except (OSError, IOError):
                gone = True
                break
            time.sleep(0.05)
        assert gone, "sampler survived %s" % sig

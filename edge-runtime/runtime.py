# One run = one trace replayed in real time against one policy.
#
#   python runtime.py --policy lqf --trace data/traces/periodic_load50.csv --slo-ms 100 \
#       --profile results/<dev>/profile.json --out results/<dev>/runs/<run_id>
#
# Threads: the main thread loads everything, starts the power sampler and the
# arrival thread, then runs the executor loop itself (it owns the GPU, one batch at
# a time). The arrival thread sleeps until each job's absolute arrival time,
# preprocesses it if --prep arrival, appends it to its model's queue and wakes the
# executor. Whenever the GPU is idle and a queue is non-empty, the executor asks the
# policy and carries out the decision (Serve / Wait / Promote / Drop), validating it.
#
# Clock: every timestamp is host time.time(), taken as t0_wall + (monotonic - t0_mono)
# so it lines up with the power samplers but never jumps.
#
# Output (in --out): requests.csv, batches.csv, waits.csv, decisions.csv,
# promotions.csv, power.csv(.raw/.meta.json), run.json (written last: its presence
# with a status marks the run complete).

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections import deque

import config
from common import Job, Profile, file_hash, git_commit, power_mode, save_json, versions
from policies import Drop, Promote, Serve, State, Wait, make_policy


class InvalidDecision(Exception):
    pass


class Clock:
    def __init__(self):
        self.t0_wall = time.time()
        self.t0_mono = time.monotonic()

    def __call__(self):
        return self.t0_wall + (time.monotonic() - self.t0_mono)


class Runtime:
    def __init__(self, models, policy, profile, runner, inputs, trace, slo_ms, prep="arrival",
                 b_max=config.B_MAX, profile_stat=None, drain_timeout_s=config.DRAIN_TIMEOUT_S,
                 max_queued=config.MAX_QUEUED, clock=None, log=print):
        self.models = list(models)
        self.policy = policy
        self.profile = profile
        self.runner = runner
        self.inputs = inputs
        self.trace = trace                      # DataFrame: job_id, camera, t_arrival_s, queue, slo_ms, frame_ref
        self.slo_ms = float(slo_ms)
        self.prep = prep
        self.b_max = int(b_max)
        self.profile_stat = profile_stat
        self.drain_timeout_s = drain_timeout_s
        self.max_queued = max_queued
        self.clock = clock or Clock()
        self.log = log

        self.cond = threading.Condition()
        self.queues = {m: deque() for m in self.models}
        self.jobs = []
        self.by_id = {}
        self.n_arrived = 0
        self.arrivals_done = False
        self.abort = None                       # reason string once the run must stop
        self.batches = []
        self.waits = []
        self.decisions = []
        self.promotions = []
        self.status = "ok"
        self.t_start = None
        self.t_arrivals_done = None
        self.t_end = None
        self.arrival_error = None

    # ------------------------------------------------------------------ arrivals
    def _arrival_loop(self):
        try:
            for row in self.trace.itertuples(index=False):
                target = self.t_start + float(row.t_arrival_s)
                delay = target - self.clock()
                if delay > 0:
                    time.sleep(delay)
                if self.abort:
                    break
                slo = self.slo_ms if row.slo_ms != row.slo_ms else float(row.slo_ms)   # NaN -> run SLO
                job = Job(int(row.job_id), str(row.camera), str(row.queue), target, slo, int(row.frame_ref))
                job.t_actual = self.clock()
                if self.prep == "arrival":
                    job.payload, job.decode_ms, job.prep_ms = self.inputs.prepare(job.frame_ref)
                with self.cond:
                    job.t_enqueued = self.clock()
                    self.queues[job.queue].append(job)
                    self.jobs.append(job)
                    self.by_id[job.job_id] = job
                    self.n_arrived += 1
                    if sum(len(q) for q in self.queues.values()) > self.max_queued and not self.abort:
                        self.abort = "collapsed"
                    self.cond.notify_all()
        except BaseException as e:                # surfaced by the executor
            self.arrival_error = e
            self.abort = "error"
        finally:
            with self.cond:
                self.arrivals_done = True
                self.t_arrivals_done = self.clock()
                self.cond.notify_all()

    # ------------------------------------------------------------------ executor
    def _validate(self, d, snapshot, now):
        if isinstance(d, Serve):
            if d.queue not in self.queues:
                raise InvalidDecision("Serve: unknown queue %r" % (d.queue,))
            n = len(snapshot[d.queue])
            if not (1 <= d.b <= min(n, self.b_max)):
                raise InvalidDecision("Serve(%s, %s): queue has %d jobs, b_max %d" % (d.queue, d.b, n, self.b_max))
            if d.job_ids is not None:
                ids = {j.job_id for j in snapshot[d.queue]}
                if len(d.job_ids) != d.b or len(set(d.job_ids)) != d.b or not set(d.job_ids) <= ids:
                    raise InvalidDecision("Serve(%s, %s): job_ids must be %d distinct queued jobs" % (d.queue, d.b, d.b))
        elif isinstance(d, Wait):
            if not d.until > now:
                raise InvalidDecision("Wait(until=%r) is not after now=%r" % (d.until, now))
        elif isinstance(d, Promote):
            if d.src not in self.queues or d.dst not in self.queues:
                raise InvalidDecision("Promote: unknown queue")
            if self.models.index(d.dst) <= self.models.index(d.src):
                raise InvalidDecision("Promote(%s -> %s): dst must be a larger model" % (d.src, d.dst))
            if not (1 <= d.k <= len(snapshot[d.src])):
                raise InvalidDecision("Promote: k=%s with %d jobs in %s" % (d.k, len(snapshot[d.src]), d.src))
        elif isinstance(d, Drop):
            if d.queue not in self.queues:
                raise InvalidDecision("Drop: unknown queue %r" % (d.queue,))
            ids = {j.job_id for j in snapshot[d.queue]}
            if not d.job_ids or not set(d.job_ids) <= ids:
                raise InvalidDecision("Drop: job_ids must be queued in %s" % d.queue)
        else:
            raise InvalidDecision("not a decision: %r" % (d,))

    def _take(self, queue, b, job_ids):
        q = self.queues[queue]
        if job_ids is None:
            return [q.popleft() for _ in range(b)]
        wanted = set(job_ids)
        taken = [j for j in q if j.job_id in wanted]
        self.queues[queue] = deque(j for j in q if j.job_id not in wanted)
        return taken

    def _serve(self, d, decide_ms):
        with self.cond:
            jobs = self._take(d.queue, d.b, d.job_ids)
        batch_id = len(self.batches)
        t_start = self.clock()
        prep_ms = 0.0
        if self.prep == "executor":
            for job in jobs:
                job.payload, job.decode_ms, job.prep_ms = self.inputs.prepare(job.frame_ref)
                prep_ms += job.decode_ms + job.prep_ms
        result = self.runner.run_batch(d.queue, [job.payload for job in jobs])
        t_end = self.clock()
        for job in jobs:
            job.payload = None                    # free the tensor
            job.model_run = d.queue
            job.batch_id = batch_id
            job.t_start = t_start
            job.t_end = t_end
        self.batches.append({
            "batch_id": batch_id, "model": d.queue, "b": len(jobs), "t_start": t_start, "t_end": t_end,
            "prep_ms": prep_ms, "assemble_ms": result.get("assemble_ms", 0.0),
            "forward_ms": result["forward_ms"], "nms_ms": result.get("nms_ms", 0.0), "decide_ms": decide_ms,
            "planned_ms": self.profile.t(d.queue, len(jobs), self.profile_stat)})
        hook = getattr(self.policy, "on_batch_done", None)
        if hook is not None:
            hook(d.queue, len(jobs), result["forward_ms"] + result.get("nms_ms", 0.0))

    def _promote(self, d):
        with self.cond:
            src = self.queues[d.src]
            moved = [src.popleft() for _ in range(d.k)]
            for job in moved:
                job.model_run = d.dst
                job.promoted_from = d.src
            merged = sorted(list(self.queues[d.dst]) + moved, key=lambda j: (j.t_arrival, j.job_id))
            self.queues[d.dst] = deque(merged)
        self.promotions.append({"t": self.clock(), "src": d.src, "dst": d.dst, "k": d.k,
                                "job_ids": " ".join(str(j.job_id) for j in moved)})

    def _drop(self, d):
        now = self.clock()
        with self.cond:
            wanted = set(d.job_ids)
            q = self.queues[d.queue]
            for job in q:
                if job.job_id in wanted:
                    job.dropped = True
                    job.t_drop = now
            self.queues[d.queue] = deque(j for j in q if j.job_id not in wanted)

    def _wait(self, until, seen, reason):
        t0 = self.clock()
        with self.cond:
            while self.n_arrived == seen and not self.abort:
                remaining = until - self.clock()
                if remaining <= 0:
                    break
                self.cond.wait(remaining)
            woke = "arrival" if self.n_arrived != seen else ("abort" if self.abort else "timeout")
        t1 = self.clock()
        self.waits.append({"t_start": t0, "t_end": t1, "kind": reason, "until": until, "woke": woke})

    def run(self):
        self.t_start = self.clock()
        arrivals = threading.Thread(target=self._arrival_loop, name="arrivals", daemon=True)
        arrivals.start()
        try:
            self._executor_loop()
        except BaseException:
            with self.cond:
                self.abort = self.abort or "error"     # stops the arrival thread
                self.cond.notify_all()
            raise
        finally:
            arrivals.join(timeout=5)
        if self.arrival_error is not None:
            raise self.arrival_error
        self.t_end = self.clock()
        return self.status

    def _executor_loop(self):
        while True:
            with self.cond:
                empty = not any(self.queues.values())
                if self.abort == "collapsed":
                    self.status = "collapsed"
                    return
                if self.abort == "error":
                    return
                if empty and self.arrivals_done:
                    return
                if self.arrivals_done and self.clock() > self.t_arrivals_done + self.drain_timeout_s:
                    self.status = "drain_timeout"
                    return
                seen = self.n_arrived
                if empty:
                    t0 = self.clock()
                    self.cond.wait()
                    self.waits.append({"t_start": t0, "t_end": self.clock(), "kind": "empty",
                                       "until": float("nan"), "woke": "arrival"})
                    continue
                now = self.clock()
                snapshot = {m: tuple(q) for m, q in self.queues.items()}
            state = State(now, snapshot, self.models, self.profile, self.b_max, self.profile_stat)
            t0 = time.perf_counter()
            decision = self.policy.decide(state)
            decide_ms = (time.perf_counter() - t0) * 1000.0
            self._validate(decision, snapshot, now)
            self.decisions.append({"t": now, "decision": type(decision).__name__,
                                   "queue": getattr(decision, "queue", getattr(decision, "src", "")),
                                   "b": getattr(decision, "b", getattr(decision, "k", 0)),
                                   "decide_ms": decide_ms,
                                   "queued": sum(len(q) for q in snapshot.values())})
            if isinstance(decision, Serve):
                self._serve(decision, decide_ms)
            elif isinstance(decision, Wait):
                self._wait(decision.until, seen, "policy")
            elif isinstance(decision, Promote):
                self._promote(decision)
            else:
                self._drop(decision)

    # ------------------------------------------------------------------ logs
    def write_logs(self, out_dir):
        import pandas as pd
        rows = []
        unserved = 0
        for job in self.jobs:
            if job.t_end is None and not job.dropped:
                unserved += 1
            rows.append({
                "job_id": job.job_id, "camera": job.camera, "queue": job.queue, "model_run": job.model_run,
                "promoted_from": job.promoted_from, "t_target": job.t_target, "t_actual": job.t_actual,
                "t_enqueued": job.t_enqueued, "t_start": job.t_start, "t_end": job.t_end,
                "deadline": job.deadline, "slo_ms": job.slo_ms, "batch_id": job.batch_id,
                "decode_ms": job.decode_ms, "prep_ms": job.prep_ms, "dropped": int(job.dropped),
                "t_drop": job.t_drop, "unserved": int(job.t_end is None and not job.dropped)})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "requests.csv"), index=False)
        for name, data, cols in [
                ("batches.csv", self.batches, ["batch_id", "model", "b", "t_start", "t_end", "prep_ms", "assemble_ms",
                                               "forward_ms", "nms_ms", "decide_ms", "planned_ms"]),
                ("waits.csv", self.waits, ["t_start", "t_end", "kind", "until", "woke"]),
                ("decisions.csv", self.decisions, ["t", "decision", "queue", "b", "decide_ms", "queued"]),
                ("promotions.csv", self.promotions, ["t", "src", "dst", "k", "job_ids"])]:
            pd.DataFrame(data, columns=cols).to_csv(os.path.join(out_dir, name), index=False)
        return {"jobs": len(self.jobs), "unserved": unserved, "batches": len(self.batches),
                "dropped": sum(1 for j in self.jobs if j.dropped), "promoted": sum(1 for j in self.jobs if j.promoted_from)}


# ---------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Replay one trace against one policy on real (or fake) models.")
    p.add_argument("--policy", required=True)
    p.add_argument("--params", default="{}", help="policy parameters as JSON")
    p.add_argument("--trace", required=True)
    p.add_argument("--slo-ms", type=float, default=100.0)
    p.add_argument("--profile", required=True, help="profile.json from calibrate.py")
    p.add_argument("--profile-stat", default=config.PROFILE_STAT, choices=["p50", "p95", "p99", "mean"])
    p.add_argument("--input", default="tensor", choices=["tensor", "frames"])
    p.add_argument("--prep", default="arrival", choices=["arrival", "executor"])
    p.add_argument("--frames-dir", default=config.FRAMES_DIR)
    p.add_argument("--no-nms", action="store_true")
    p.add_argument("--b-max", type=int, default=config.B_MAX)
    p.add_argument("--models", default=",".join(config.MODELS))
    p.add_argument("--duration", type=float, help="cut the trace to its first N seconds")
    p.add_argument("--warmup-s", type=float, default=config.WARMUP_S)
    p.add_argument("--idle-s", type=float, default=config.IDLE_LOADED_S)
    p.add_argument("--drain-timeout", type=float, default=config.DRAIN_TIMEOUT_S)
    p.add_argument("--max-queued", type=int, default=config.MAX_QUEUED)
    p.add_argument("--power", default="auto", choices=["auto", "nvidia-smi", "tegrastats", "fake", "none"])
    p.add_argument("--gpu-index", default="0")
    p.add_argument("--out", required=True)
    p.add_argument("--fake", action="store_true", help="fake models sleeping T(b) from the profile (no torch)")
    p.add_argument("--fake-jitter", type=float, default=0.0, help="fake models: relative jitter")
    p.add_argument("--fake-prep-ms", type=float, default=0.0, help="fake inputs: preprocessing time per frame")
    p.add_argument("--extra", default="{}", help="JSON merged into run.json (used by plan.py)")
    return p.parse_args(argv)


def _sigterm(signum, frame):
    raise SystemExit(128 + signum)   # unwinds through the finally blocks (sampler stop)


def main(argv=None):
    import power
    from traces import read_trace

    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _sigterm)
    os.makedirs(args.out, exist_ok=True)
    models = args.models.split(",")
    nms = not args.no_nms
    params = json.loads(args.params)
    profile = Profile.load(args.profile, stat=args.profile_stat, part="total" if nms else "forward")
    if args.b_max > max(profile.batch_sizes):
        print("b_max %d is past the largest profiled size; capping at %d" % (args.b_max, max(profile.batch_sizes)))
        args.b_max = max(profile.batch_sizes)
    policy = make_policy(args.policy, params)
    trace, trace_meta = read_trace(args.trace)
    if args.duration is not None:
        trace = trace[trace["t_arrival_s"] < args.duration]
    unknown = sorted(set(trace["queue"]) - set(models))
    if unknown:
        raise SystemExit("trace routes to models not in the pool: %s" % unknown)

    clock = Clock()
    marks = []

    def mark(label):
        marks.append([label, clock()])

    sampler = power.make_sampler("fake" if args.power == "fake" else args.power, args.gpu_index)
    power_path = os.path.join(args.out, "power.csv")
    sampler.start(power_path)
    try:
        mark("load_start")
        if args.fake:
            from inputs import FakeInputs
            from models import FakeRunner
            runner = FakeRunner(profile, jitter=args.fake_jitter)
            inputs = FakeInputs(args.fake_prep_ms)
        else:
            from inputs import make_inputs
            from models import Runner
            runner = Runner(models, nms=nms)
            inputs = make_inputs(args.input, runner.device, args.frames_dir)
            runner.warmup(inputs, profile.batch_sizes, args.b_max)
        mark("load_end")
        mark("idle_start")
        time.sleep(args.idle_s)
        mark("idle_end")

        rt = Runtime(models, policy, profile, runner, inputs, trace, args.slo_ms, prep=args.prep,
                     b_max=args.b_max, profile_stat=None, drain_timeout_s=args.drain_timeout,
                     max_queued=args.max_queued, clock=clock)
        mark("trace_start")
        status = rt.run()
        mark("trace_end")
        counts = rt.write_logs(args.out)
        time.sleep(sampler.tail_s)
    finally:
        sampler.stop()

    power_df = power.read(power_path)
    idle = [t for label, t in marks if label.startswith("idle_")]
    temp_start = None
    if "temp_c" in power_df and len(idle) == 2:
        inside = power_df[(power_df["t"] >= idle[0]) & (power_df["t"] <= idle[1])]
        if len(inside):
            temp_start = float(inside["temp_c"].mean())
    run = {
        "status": status, "policy": args.policy, "params": params, "trace": os.path.abspath(args.trace),
        "trace_hash": file_hash(args.trace), "trace_meta": trace_meta, "slo_ms": args.slo_ms,
        "input": "fake" if args.fake else args.input, "prep": args.prep, "nms": nms, "b_max": args.b_max,
        "models": models, "profile": os.path.abspath(args.profile), "profile_stat": args.profile_stat,
        "profile_device": profile.data.get("device_tag"), "warmup_s": args.warmup_s,
        "duration_s": float(trace["t_arrival_s"].max()) if len(trace) else 0.0,
        "t_trace_start": rt.t_start, "t_arrivals_done": rt.t_arrivals_done, "t_trace_end": rt.t_end,
        "marks": marks, "counts": counts, "temp_start_c": temp_start,
        "power_backend": sampler.backend, "power_meta": power.read_meta(power_path),
        "power_mode": power_mode(), "device": runner.info(), "versions": versions(),
        "git_commit": git_commit(), "argv": sys.argv[1:], "extrapolated_sizes": profile.extrapolated(),
    }
    run.update(json.loads(args.extra))
    save_json(os.path.join(args.out, "run.json"), run)
    print("%s: %s, %d jobs, %d batches, %d dropped, %d unserved" % (
        args.out, status, counts["jobs"], counts["batches"], counts["dropped"], counts["unserved"]))


if __name__ == "__main__":
    main()

# Arrival traces. A trace is a CSV, one row per frame, sorted by arrival:
#   job_id, camera, t_arrival_s (from trace start), queue (model name), slo_ms, frame_ref
# plus <trace>.meta.json (source, parameters, seed, split, load fraction if a
# profile was given). slo_ms is empty unless the trace sets it per camera; then
# the run's SLO applies. frame_ref is an index into the frame pool (taken modulo
# its size at run time). Same seed -> same trace, for every policy.
#
#   python traces.py periodic --load 0.5 --profile results/<dev>/profile.json --out data/traces/p50.csv
#   python traces.py periodic --cameras 8 --fps 10 --duration 130 --out ...
#   python traces.py poisson --rate 40 --duration 130 --out ...
#   python traces.py from-cityflow --run ../cityflow-routing/results/all --scenario S02 --point 80 --out ...
#   python traces.py suite --profile results/<dev>/profile.json      default synthetic set
#   python traces.py info data/traces/x.csv [--profile ...]

import argparse
import glob
import gzip
import os

import numpy as np
import pandas as pd

import config
from common import Profile, capacity_fps, file_hash, load_json, save_json

COLUMNS = ["job_id", "camera", "t_arrival_s", "queue", "slo_ms", "frame_ref"]


def _finish(rows):
    df = pd.DataFrame(rows, columns=["camera", "t_arrival_s", "queue", "slo_ms"])
    df = df.sort_values(["t_arrival_s", "camera"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "job_id", np.arange(len(df)))
    df["frame_ref"] = np.arange(len(df))
    return df[COLUMNS]


def periodic(cameras, fps, duration_s, split, models=config.MODELS, sticky=None, seed=config.TRACE_SEED):
    """C cameras at `fps` each, each with a random phase in [0, 1/fps). Each frame
    goes to a model drawn from `split`, or with sticky = p, stays on the camera's
    previous model with probability p and is redrawn from `split` otherwise (so the
    long-run split is still `split`)."""
    rng = np.random.default_rng(seed)
    split = np.asarray(split, dtype=float) / np.sum(split)
    period = 1.0 / fps
    rows = []
    for c in range(cameras):
        phase = rng.uniform(0, period)
        n = int(np.floor((duration_s - phase) / period)) + 1
        times = phase + period * np.arange(n)
        times = times[times < duration_s]
        draws = rng.choice(len(models), size=len(times), p=split)
        if sticky is not None:
            stay = rng.random(len(times)) < sticky
            for k in range(1, len(times)):
                if stay[k]:
                    draws[k] = draws[k - 1]
        for t, q in zip(times, draws):
            rows.append(("cam%02d" % c, float(t), models[q], np.nan))
    return _finish(rows)


def poisson(total_rate, duration_s, split, models=config.MODELS, seed=config.TRACE_SEED):
    """Independent Poisson arrivals per queue at total_rate * p_i frames/s."""
    rng = np.random.default_rng(seed)
    split = np.asarray(split, dtype=float) / np.sum(split)
    rows = []
    for model, p in zip(models, split):
        rate = total_rate * p
        if rate <= 0:
            continue
        t = rng.exponential(1.0 / rate)
        while t < duration_s:
            rows.append(("poisson", float(t), model, np.nan))
            t += rng.exponential(1.0 / rate)
    return _finish(rows)


def from_cityflow(run_dir, scenario, point, cameras=None, duration_s=None):
    """One camera per CityFlow camera, arrival = synced_time_s (shifted so the
    first frame is at 0), queue = the model point `point` routed the frame to."""
    raw_dir = os.path.join(run_dir, "raw")
    paths = sorted(glob.glob(os.path.join(raw_dir, "routing_%s_*.csv.gz" % scenario)))
    if not paths:
        raise FileNotFoundError("no routing_%s_*.csv.gz under %s" % (scenario, raw_dir))
    column = "model_p%s" % point
    frames = []
    for path in paths:
        with gzip.open(path, "rt") as f:
            head = f.readline().strip().split(",")
        if column not in head:
            raise KeyError("%s has no column %s (points: %s)" % (
                path, column, ", ".join(h[len("model_p"):] for h in head if h.startswith("model_p"))))
        df = pd.read_csv(path, usecols=["camera", "frame_idx", "synced_time_s", column])
        if cameras and df["camera"].iloc[0] not in cameras:
            continue
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["t_arrival_s"] = df["synced_time_s"] - df["synced_time_s"].min()
    if duration_s is not None:
        df = df[df["t_arrival_s"] < duration_s]
    rows = list(zip(df["camera"], df["t_arrival_s"].astype(float), df[column], [np.nan] * len(df)))
    return _finish(rows)


def split_of(df, models=config.MODELS):
    counts = df["queue"].value_counts()
    total = float(len(df))
    return [counts.get(m, 0) / total for m in models]


def load_fraction(df, profile, models=config.MODELS, b=config.B_MAX):
    """Offered load as a fraction of capacity (max batching, this trace's split)."""
    duration = max(float(df["t_arrival_s"].max()), 1e-9)
    rate = len(df) / duration
    return rate / capacity_fps(profile, models, split_of(df, models), b)


def fps_for_load(load, profile, cameras, split, models=config.MODELS, b=config.B_MAX):
    return load * capacity_fps(profile, models, split, b) / cameras


def write_trace(df, path, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    meta = dict(meta)
    meta.update({"jobs": int(len(df)), "duration_s": float(df["t_arrival_s"].max()) if len(df) else 0.0,
                 "cameras": int(df["camera"].nunique()), "split": split_of(df), "hash": file_hash(path)})
    save_json(path + ".meta.json", meta)
    print("%s: %d jobs, %d cameras, %.1f s, split %s%s" % (
        path, meta["jobs"], meta["cameras"], meta["duration_s"],
        " ".join("%.3f" % s for s in meta["split"]),
        ", load %.2f" % meta["load_fraction"] if meta.get("load_fraction") is not None else ""))
    return meta


def read_trace(path):
    df = pd.read_csv(path)
    meta_path = path + ".meta.json"
    meta = load_json(meta_path) if os.path.exists(meta_path) else {}
    return df, meta


def build_suite(profile, out_dir, loads=config.LOAD_FRACTIONS, duration_s=None, seed=config.TRACE_SEED,
                cameras=config.CAMERAS, b=config.B_MAX):
    """Default synthetic traces at each load: periodic (fixed split), periodic
    sticky, Poisson. FPS is scaled to hit the load; the camera count stays."""
    duration_s = duration_s or config.RUN_DURATION_S
    written = []
    for load in loads:
        tag = "%02d" % round(load * 100)
        fps = fps_for_load(load, profile, cameras, config.ROUTING_SPLIT, b=b)
        for kind in ["periodic", "sticky", "poisson"]:
            if kind == "poisson":
                df = poisson(fps * cameras, duration_s, config.ROUTING_SPLIT, seed=seed)
                params = {"rate": fps * cameras}
            else:
                sticky = config.STICKY_STAY if kind == "sticky" else None
                df = periodic(cameras, fps, duration_s, config.ROUTING_SPLIT, sticky=sticky, seed=seed)
                params = {"cameras": cameras, "fps": fps, "sticky": sticky}
            path = os.path.join(out_dir, "%s_load%s.csv" % (kind, tag))
            params.update({"source": kind, "seed": seed, "target_load": load,
                           "load_fraction": load_fraction(df, profile, b=b), "b_for_capacity": b})
            write_trace(df, path, params)
            written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--duration", type=float, default=config.RUN_DURATION_S)
        p.add_argument("--seed", type=int, default=config.TRACE_SEED)
        p.add_argument("--profile", help="profile.json; needed for --load, reports the load fraction")
        p.add_argument("--load", type=float, help="target fraction of capacity (needs --profile)")
        p.add_argument("--b-max", type=int, default=config.B_MAX)

    p = sub.add_parser("periodic")
    common(p)
    p.add_argument("--cameras", type=int, default=config.CAMERAS)
    p.add_argument("--fps", type=float, default=config.CAMERA_FPS)
    p.add_argument("--sticky", type=float, help="stay probability (Markov routing)")
    p = sub.add_parser("poisson")
    common(p)
    p.add_argument("--rate", type=float, help="total frames/s")
    p = sub.add_parser("from-cityflow")
    common(p)
    p.add_argument("--run", default=os.path.join(config.CITYFLOW_ROUTING_DIR, "results", "all"))
    p.add_argument("--scenario", default="S02")
    p.add_argument("--point", required=True, help="Pareto point id (model_p<id> column)")
    p.add_argument("--cameras", help="comma-separated camera ids to keep")
    p = sub.add_parser("suite")
    p.add_argument("--profile", required=True)
    p.add_argument("--out-dir", default=config.TRACES_DIR)
    p.add_argument("--duration", type=float, default=config.RUN_DURATION_S)
    p.add_argument("--loads", default=",".join(str(x) for x in config.LOAD_FRACTIONS))
    p.add_argument("--seed", type=int, default=config.TRACE_SEED)
    p.add_argument("--b-max", type=int, default=config.B_MAX)
    p = sub.add_parser("info")
    p.add_argument("trace")
    p.add_argument("--profile")
    p.add_argument("--b-max", type=int, default=config.B_MAX)
    args = parser.parse_args(argv)

    profile = Profile.load(args.profile) if getattr(args, "profile", None) else None
    if args.cmd == "suite":
        build_suite(profile, args.out_dir, [float(x) for x in args.loads.split(",")],
                    args.duration, args.seed, b=args.b_max)
        return
    if args.cmd == "info":
        df, meta = read_trace(args.trace)
        print(meta)
        print("split:", split_of(df))
        if profile:
            print("load fraction on this profile: %.3f" % load_fraction(df, profile, b=args.b_max))
        return
    if getattr(args, "load", None) is not None and profile is None:
        parser.error("--load needs --profile")

    meta = {"source": args.cmd, "seed": args.seed}
    if args.cmd == "periodic":
        fps = args.fps
        if args.load is not None:
            fps = fps_for_load(args.load, profile, args.cameras, config.ROUTING_SPLIT, b=args.b_max)
        df = periodic(args.cameras, fps, args.duration, config.ROUTING_SPLIT, sticky=args.sticky, seed=args.seed)
        meta.update({"cameras": args.cameras, "fps": fps, "sticky": args.sticky})
    elif args.cmd == "poisson":
        rate = args.rate
        if args.load is not None:
            rate = fps_for_load(args.load, profile, 1, config.ROUTING_SPLIT, b=args.b_max)
        if rate is None:
            parser.error("poisson needs --rate or --load")
        df = poisson(rate, args.duration, config.ROUTING_SPLIT, seed=args.seed)
        meta["rate"] = rate
    else:
        cams = args.cameras.split(",") if args.cameras else None
        df = from_cityflow(args.run, args.scenario, args.point, cams, args.duration)
        meta.update({"run": os.path.abspath(args.run), "scenario": args.scenario, "point": args.point})
    if args.load is not None:
        meta["target_load"] = args.load
    if profile is not None:
        meta["load_fraction"] = load_fraction(df, profile, b=args.b_max)
        meta["b_for_capacity"] = args.b_max
    write_trace(df, args.out, meta)


if __name__ == "__main__":
    main()

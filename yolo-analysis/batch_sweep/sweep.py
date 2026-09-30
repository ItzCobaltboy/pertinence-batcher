"""
SINGLE ENTRY POINT for the YOLOv8 batched-inference timing sweep.

    python sweep.py preflight              check GPU, versions, data, weights, disk
    python sweep.py measure                compile TRT engines, then all timed runs (resumable)
    python sweep.py aggregate              raw/ -> summary CSVs, plots, sanity flags (no GPU)
    python sweep.py profile --variant V    write scheduler-sim/profiles/yolov8_a100_<V>.csv

`measure` never runs model code itself. Every compile and every timed run
is its own subprocess (`sweep.py _worker ...`, see worker.py), so one crash
or OOM is recorded and the sweep moves on. Rerunning `measure` skips every
job that already has an ok record, so it is safe to kill and restart.

This file imports no torch; aggregate and profile run anywhere with pandas
and matplotlib.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import gpu_log
import settings
import store

SWEEP_PATH = os.path.abspath(__file__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=settings.REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=settings.REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("+dirty" if dirty else "") if head else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def spawn(job, log_path, timeout_s):
    """Run one worker. Returns (returncode or 'timeout', seconds)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, SWEEP_PATH, "_worker", json.dumps(job)]
    t0 = time.perf_counter()
    with open(log_path, "w") as log_file:
        try:
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                                  cwd=os.path.dirname(SWEEP_PATH), timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code, time.perf_counter() - t0


def tail(path, lines=40):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return ""


def record_written(base):
    for status in store.STATUSES:
        if os.path.exists(base + f"__{status}.json"):
            return status
    return None


def run_worker(job, timeout_s, log):
    """Spawn, then make sure exactly one record exists for this attempt."""
    base = job["base"]
    log_path = os.path.join(settings.LOG_DIR, job["session"], "jobs", os.path.basename(base) + ".log")
    code, seconds = spawn(job, log_path, timeout_s)
    status = record_written(base)
    if status is None:
        # The worker died without writing (abort, segfault, OOM kill, timeout).
        status = "failed"
        store.write_json_atomic(base + "__failed.json", {
            "kind": job["kind"], "session": job["session"], "variant": job["variant"],
            "model": settings.model_stem(job["model"]), "batch_size": job["batch_size"],
            "run_index": job.get("run_index"), "status": "failed",
            "error": f"worker exited with {code} before writing a record",
            "returncode": code, "log_tail": tail(log_path), "log_path": log_path,
            "finished_at": store.now_iso(), "gpu_end": gpu_log.snapshot(),
        })
    elif status == "failed":
        try:
            err = store.read_json(base + "__failed.json").get("error", "")
        except (OSError, ValueError):
            err = ""
        log(f"    error: {err[:300]}")
    return status, seconds


def probe(log):
    job = {"kind": "probe", "session": "probe", "variant": "eager_fp32", "model": "none",
           "batch_size": 0, "base": "probe"}
    cmd = [sys.executable, SWEEP_PATH, "_worker", json.dumps(job)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(SWEEP_PATH), timeout=600)
    for line in out.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    log("probe failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    raise SystemExit(1)


def save_full_gpu_report(session, label):
    path = os.path.join(settings.GPU_LOG_DIR, session, f"{label}_{store.timestamp()}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(gpu_log.full_report())
        f.write("\n\nsnapshot json:\n" + json.dumps(gpu_log.snapshot(), indent=1))
    return path


def warn_about_gpu(snap, log):
    if not snap.get("gpu"):
        log("GPU: nvidia-smi not available (fine for CPU tests)")
        return
    g = snap["gpu"]
    log(f"GPU: {g.get('name')} | SM {g.get('clocks.sm')}/{g.get('clocks.max.sm')} MHz | "
        f"{g.get('temperature.gpu')} C | mem {g.get('memory.used')}/{g.get('memory.total')} MiB | "
        f"util {g.get('utilization.gpu')}%")
    others = gpu_log.other_processes(snap)
    if others:
        log(f"WARNING: {len(others)} other compute process(es) on this GPU: "
            + ", ".join(f"{a['process_name']} ({a['used_memory_mib']} MiB)" for a in others))
        log("         timings will be flagged; consider waiting until the GPU is free")


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------
def acquire_session_lock(session):
    """Two `measure` processes on one session would run the same jobs twice.
    Hold an exclusive lock for the whole sweep; the OS drops it if we die."""
    import fcntl
    path = os.path.join(settings.LOG_DIR, session, "measure.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another `measure` is already running for session '{session}' ({path}). "
                         f"Attach to its tmux window instead of starting a second one.")
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def cmd_measure(args):
    session = args.session
    lock = acquire_session_lock(session)  # noqa: F841  (held until the process exits)
    log = store.Log(os.path.join(settings.LOG_DIR, session, "sweep.log"))
    log(f"=== measure, session '{session}' ===")
    log("command: " + " ".join(sys.argv))

    info = probe(log)
    cuda = info["cuda_available"] and not args.cpu
    device = args.device if cuda else "cpu"
    log(f"torch {info['versions']['torch']} | torch_tensorrt {info['versions'].get('torch_tensorrt')} | "
        f"ultralytics {info['versions'].get('ultralytics')} | device {device}")
    save_full_gpu_report(session, "sweep_start")
    warn_about_gpu(gpu_log.snapshot(), log)

    # The image pool is fixed per session, so reruns draw from the same images.
    manifest_path = store.pool_manifest_path(session)
    if os.path.exists(manifest_path):
        manifest = store.read_json(manifest_path)
        log(f"reusing image pool ({len(manifest['images'])} images) from {manifest_path}")
    else:
        from images import build_pool_manifest
        manifest = build_pool_manifest(args.splits, args.pool_per_split, args.seed, args.dataset_root)
        store.write_json_atomic(manifest_path, manifest)
        log(f"sampled image pool: {len(manifest['images'])} images from {args.splits} -> {manifest_path}")

    common = {"session": session, "img_size": args.imgsz, "dataset_root": manifest.get("dataset_root"),
              "device": device, "engine_dir": args.engine_dir, "seed": args.seed,
              "git_commit": git_commit(), "no_tf32": args.no_tf32}
    trt_variants = [v for v in args.variants if v in settings.TRT_VARIANTS]

    # ---- phase 1: TRT compiles (once per config, cached on disk) ----
    compile_jobs = [(v, m, b) for v in trt_variants for m in args.models for b in args.batch_sizes]
    if trt_variants and not cuda:
        log(f"no CUDA: skipping all {len(compile_jobs)} TRT configs")
        for v, m, b in compile_jobs:
            if not store.compile_attempts(session, v, m, b):
                store.write_json_atomic(store.compile_base(session, v, m, b) + "__skipped.json", {
                    "kind": "compile", "session": session, "variant": v, "model": settings.model_stem(m),
                    "batch_size": b, "status": "skipped", "error": "no CUDA device (TRT needs a GPU)",
                    "finished_at": store.now_iso()})
        compile_jobs = []
    elif trt_variants and not info.get("torch_tensorrt_importable"):
        log(f"torch_tensorrt does not import ({info.get('torch_tensorrt_error')}); TRT compiles will fail")

    todo = []
    for v, m, b in compile_jobs:
        if store.engine_ready(args.engine_dir, v, m, b, args.imgsz):
            continue
        if store.job_state(store.compile_attempts(session, v, m, b), args.max_attempts) == "gave_up":
            continue
        todo.append((v, m, b))
    log(f"phase 1: {len(todo)} TRT compiles to do ({len(compile_jobs) - len(todo)} cached or given up)")
    for i, (v, m, b) in enumerate(todo, 1):
        job = dict(common, kind="compile", variant=v, model=m, batch_size=b,
                   base=store.compile_base(session, v, m, b))
        log(f"[compile {i}/{len(todo)}] {v} {settings.model_stem(m)} bs={b}")
        status, seconds = run_worker(job, args.compile_timeout, log)
        log(f"    -> {status} in {seconds:.0f}s")

    # ---- phase 2: timed runs, round by round, shuffled within each round ----
    configs = [(v, m, b) for v in args.variants for m in args.models for b in args.batch_sizes]
    plan = []
    for r in range(args.runs):
        order = list(configs)
        np.random.default_rng([args.seed, r]).shuffle(order)
        for v, m, b in order:
            plan.append((r, v, m, b))

    todo = []
    skipped_no_engine = set()
    for r, v, m, b in plan:
        if v in settings.TRT_VARIANTS and not store.engine_ready(args.engine_dir, v, m, b, args.imgsz):
            skipped_no_engine.add((v, m, b))
            continue
        if store.job_state(store.run_attempts(session, v, m, b, r), args.max_attempts) == "todo":
            todo.append((r, v, m, b))
    if skipped_no_engine:
        log(f"phase 2: {len(skipped_no_engine)} TRT configs have no engine (compile failed or skipped), "
            f"their runs are not attempted")
    log(f"phase 2: {len(todo)} timed runs to do (of {len(plan)} planned)")

    t_start = time.perf_counter()
    for i, (r, v, m, b) in enumerate(todo, 1):
        job = dict(common, kind="run", variant=v, model=m, batch_size=b, run_index=r,
                   warmup=args.warmup, repeats=args.repeats, nms=not args.no_nms,
                   max_batches=args.max_batches, base=store.run_base(session, v, m, b, r))
        status, seconds = run_worker(job, args.run_timeout, log)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (len(todo) - i)
        extra = ""
        if status == "ok":
            s = store.read_json(job["base"] + "__ok.json")["summary"]["forward_ms"]
            extra = f" forward mean {s['mean']:.3f} ms (median {s['median']:.3f})"
        log(f"[run {i}/{len(todo)}] round {r} {v} {settings.model_stem(m)} bs={b}: {status} "
            f"in {seconds:.0f}s{extra} | eta {eta / 60:.0f} min")

    save_full_gpu_report(session, "sweep_end")
    warn_about_gpu(gpu_log.snapshot(), log)
    log("measure done; aggregating")
    import aggregate
    aggregate.aggregate(session, log=log)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def cmd_preflight(args):
    log = store.Log(os.path.join(settings.LOG_DIR, args.session, "preflight.log"))
    info = probe(log)
    log("versions: " + json.dumps(info["versions"]))
    log(f"cuda available: {info['cuda_available']} | torch_tensorrt importable: "
        f"{info.get('torch_tensorrt_importable')} {info.get('torch_tensorrt_error', '')}")
    if info.get("device"):
        d = info["device"]
        log(f"device: {d['gpu_name']} (cc {d['compute_capability']}), {d['gpu_total_memory_mib']} MiB | "
            f"tf32 matmul={d['matmul_allow_tf32']} cudnn={d['cudnn_allow_tf32']} | "
            f"cudnn.benchmark={d['cudnn_benchmark']}")
    warn_about_gpu(gpu_log.snapshot(), log)

    ok = True
    for split in args.splits:
        folder = settings.images_dir(split, args.dataset_root)
        n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        log(f"data: {folder}: {n} files" + ("" if n else "  <-- MISSING"))
        ok &= n > 0
    for m in args.models:
        found = next((c for c in settings.weights_candidates(m)[:-1] if os.path.exists(c)), None)
        log(f"weights {m}: {found or 'not found locally, ultralytics will download it on first load'}")

    for label, path in (("repo", settings.REPO_ROOT), ("engine dir", args.engine_dir)):
        probe_path = path
        while not os.path.exists(probe_path):
            probe_path = os.path.dirname(probe_path)
        free_gib = shutil.disk_usage(probe_path).free / 2**30
        log(f"disk free under {label} ({probe_path}): {free_gib:.1f} GiB")

    n_trt = len([v for v in args.variants if v in settings.TRT_VARIANTS])
    n_compiles = n_trt * len(args.models) * len(args.batch_sizes)
    n_runs = len(args.variants) * len(args.models) * len(args.batch_sizes) * args.runs
    log(f"plan: {n_compiles} TRT compiles, {n_runs} timed runs "
        f"({args.warmup} warmup + {args.repeats} timed batches each)")
    log("expected disk: engines ~4 GiB (fp32+fp16, all batch sizes), raw/ + logs < 100 MiB")
    log("preflight " + ("OK" if ok else "FOUND PROBLEMS"))
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# aggregate / profile
# ---------------------------------------------------------------------------
def cmd_aggregate(args):
    import aggregate
    aggregate.aggregate(args.session)


def cmd_profile(args):
    import aggregate
    for variant in args.variant:
        aggregate.write_profile(args.session, variant, args.profile_dir, args.tag, args.force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--session", default=settings.DEFAULT_SESSION,
                       help="name grouping one campaign of runs; resume works per session")

    def add_plan(p):
        p.add_argument("--models", nargs="+", default=settings.MODELS)
        p.add_argument("--batch-sizes", nargs="+", type=int, default=settings.BATCH_SIZES)
        p.add_argument("--variants", nargs="+", default=settings.VARIANTS, choices=settings.VARIANTS)
        p.add_argument("--runs", type=int, default=settings.RUNS)
        p.add_argument("--warmup", type=int, default=settings.WARMUP_BATCHES)
        p.add_argument("--repeats", type=int, default=settings.TIMED_BATCHES)
        p.add_argument("--splits", nargs="+", default=settings.SPLITS)
        p.add_argument("--dataset-root", default=None,
                       help="folder holding <split>/ image dirs (default: yolo-analysis/dataset)")
        p.add_argument("--engine-dir", default=settings.ENGINE_DIR)

    p = sub.add_parser("measure", help="compile + time everything (resumable)")
    add_common(p)
    add_plan(p)
    p.add_argument("--imgsz", type=int, default=settings.IMG_SIZE)
    p.add_argument("--pool-per-split", type=int, default=settings.POOL_PER_SPLIT)
    p.add_argument("--max-batches", type=int, default=settings.MAX_DISTINCT_BATCHES,
                   help="distinct pre-built batches per run, cycled through")
    p.add_argument("--seed", type=int, default=settings.SEED)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu", action="store_true", help="force CPU (tests); TRT variants are skipped")
    p.add_argument("--no-nms", action="store_true", help="skip the forward+NMS column")
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 (default: torch defaults, TF32 on)")
    p.add_argument("--max-attempts", type=int, default=settings.MAX_ATTEMPTS)
    p.add_argument("--compile-timeout", type=int, default=settings.COMPILE_TIMEOUT_S)
    p.add_argument("--run-timeout", type=int, default=settings.RUN_TIMEOUT_S)
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("preflight", help="check GPU, versions, data, weights, disk")
    add_common(p)
    add_plan(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("aggregate", help="raw/ -> summary CSVs, plots, sanity flags")
    add_common(p)
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("profile", help="write the scheduler-sim profile CSV for a variant")
    add_common(p)
    p.add_argument("--variant", nargs="+", required=True, choices=settings.VARIANTS,
                   help="no default on purpose: pick after looking at all three")
    p.add_argument("--profile-dir", default=settings.PROFILE_DIR)
    p.add_argument("--tag", default=settings.PROFILE_HARDWARE_TAG, help="hardware tag in the file name")
    p.add_argument("--force", action="store_true", help="overwrite an existing profile file")
    p.set_defaults(func=cmd_profile)

    return parser.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        import worker
        worker.main(argv[1])
        return
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

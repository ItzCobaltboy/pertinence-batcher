# T_i(b) on this device -> profile.json. Measured here, never copied from another GPU.
#
#   python calibrate.py                                   default grid, results/<device_tag>/profile.json
#   python calibrate.py --batch-sizes 1,2,4,8,16,32 --max-s 30
#
# Per (model, b): warmup forwards, then timed forwards on the real input pool, each
# with torch.cuda.synchronize() before and after, NMS timed separately. A combo
# stops after --iters forwards or --max-s seconds, whichever comes first, but never
# before --min-iters (a Jetson runs yolov8l at b = 16 in seconds, not milliseconds).
# Also: model load time, first-forward time, GPU memory per model and per batch,
# whether all four models fit together, and the model switch cost (n, l, n, l, ...
# at b = 1 against back-to-back runs of each).
#
# profile.json: {"models": {m: {"<b>": {"forward": stats, "nms": stats, "total": stats,
#   "iters": n, "peak_mb": ...}}}, "batch_sizes": [...], "load": ..., "switch": ...,
#   "device": ..., "device_tag": ..., "capacity_fps": ...}  stats = p50/p95/p99/mean/std_ms

import argparse
import os
import time

import numpy as np

import config
from common import Profile, capacity_fps, device_tag, power_mode, save_json, suggest_slos, versions


def stats(values):
    v = np.asarray(values, dtype=float)
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)),
            "p99_ms": float(np.percentile(v, 99)), "mean_ms": float(v.mean()), "std_ms": float(v.std()),
            "min_ms": float(v.min()), "max_ms": float(v.max())}


def is_oom(e):
    return "out of memory" in str(e).lower()


def measure_combo(runner, model, batch, warmup, iters, min_iters, max_s):
    torch = runner.torch
    if runner.cuda:
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t_end = time.time() + max(1.0, max_s / 5.0)
    for i in range(warmup):
        out, _ = runner.forward(model, batch)
        runner.nms(out)
        if i >= 1 and time.time() > t_end:
            break
    fwd, nms, total = [], [], []
    t_end = time.time() + max_s
    while len(fwd) < iters and (len(fwd) < min_iters or time.time() < t_end):
        out, f_ms = runner.forward(model, batch)
        n_ms = runner.nms(out)
        fwd.append(f_ms)
        nms.append(n_ms)
        total.append(f_ms + n_ms)
    entry = {"forward": stats(fwd), "nms": stats(nms), "total": stats(total), "iters": len(fwd)}
    if runner.cuda:
        entry["peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    return entry


def switch_cost(runner, inputs, a, b, iters):
    """Mean forward time at b = 1 when alternating a, b, a, b, ... minus back-to-back."""
    batch = inputs.sample_batch(1)
    for m in (a, b):
        for _ in range(3):
            runner.forward(m, batch)
    same = {a: [], b: []}
    for m in (a, b):
        for _ in range(iters):
            same[m].append(runner.forward(m, batch)[1])
    alt = {a: [], b: []}
    for i in range(2 * iters):
        m = a if i % 2 == 0 else b
        alt[m].append(runner.forward(m, batch)[1])
    out = {"pair": [a, b], "iters": iters}
    for m in (a, b):
        out[m] = {"same_ms_p50": float(np.median(same[m])), "alternating_ms_p50": float(np.median(alt[m])),
                  "penalty_ms_p50": float(np.median(alt[m]) - np.median(same[m])),
                  "penalty_ms_mean": float(np.mean(alt[m]) - np.mean(same[m]))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure T_i(b) on this device.")
    parser.add_argument("--models", default=",".join(config.MODELS))
    parser.add_argument("--batch-sizes", default=",".join(str(b) for b in config.BATCH_SIZES))
    parser.add_argument("--warmup", type=int, default=config.CALIB_WARMUP)
    parser.add_argument("--iters", type=int, default=config.CALIB_ITERS)
    parser.add_argument("--min-iters", type=int, default=10)
    parser.add_argument("--max-s", type=float, default=20.0, help="time cap per (model, b)")
    parser.add_argument("--switch-iters", type=int, default=config.SWITCH_ITERS)
    parser.add_argument("--frames-dir", default=config.FRAMES_DIR)
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument("--device-tag")
    parser.add_argument("--out", help="default results/<device_tag>/profile.json")
    args = parser.parse_args(argv)

    import torch
    from inputs import TensorInputs
    from models import Runner

    models = args.models.split(",")
    sizes = [int(b) for b in args.batch_sizes.split(",")]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device != "cpu" else None
    tag = args.device_tag or device_tag(gpu_name)
    out = args.out or os.path.join(config.device_dir(tag), "profile.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("device %s (%s), tag %s -> %s" % (device, gpu_name, tag, out))
    print("%d models x %d sizes, each capped at %.0f s (but at least %d forwards): ~%.0f min plus loading "
          "if no combo needs the floor" % (len(models), len(sizes), args.max_s, args.min_iters,
                                           len(models) * len(sizes) * args.max_s * 1.2 / 60))

    # load all four at once: do they fit?
    all_fit = True
    try:
        runner = Runner(models, nms=not args.no_nms, device=device)
    except RuntimeError as e:
        if not is_oom(e):
            raise
        all_fit = False
        print("FLAG: the %d models do not fit in GPU memory together (%s)" % (len(models), e))
        torch.cuda.empty_cache()
        runner = None
    inputs = TensorInputs(args.frames_dir, device)

    profile = {"models": {}, "batch_sizes": [], "load": {}, "oom": [], "all_models_fit": all_fit,
               "nms": not args.no_nms, "input_source": inputs.source, "device_tag": tag,
               "device": {"device": device, "gpu_name": gpu_name}, "power_mode": power_mode(),
               "versions": versions(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "settings": {"warmup": args.warmup, "iters": args.iters, "min_iters": args.min_iters,
                            "max_s": args.max_s}}
    if device != "cpu":
        props = torch.cuda.get_device_properties(0)
        profile["device"]["total_mb"] = props.total_memory / 2 ** 20

    measured_sizes = set(sizes)
    for m in models:
        r = runner or Runner([m], nms=not args.no_nms, device=device)
        batch1 = inputs.sample_batch(1)
        out_, first_ms = r.forward(m, batch1)          # the very first forward: cuDNN picks kernels
        profile["load"][m] = dict(r.load_info[m], first_forward_ms=first_ms)
        profile["models"][m] = {}
        for b in sizes:
            try:
                entry = measure_combo(r, m, inputs.sample_batch(b), args.warmup, args.iters,
                                      args.min_iters, args.max_s)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                print("  %s b=%d: out of memory, skipping larger sizes" % (m, b))
                profile["oom"].append([m, b])
                torch.cuda.empty_cache()
                measured_sizes &= set(x for x in sizes if x < b)
                break
            profile["models"][m][str(b)] = entry
            print("  %s b=%2d: forward p50 %8.2f ms  nms p50 %6.2f ms  (%d iters)" % (
                m, b, entry["forward"]["p50_ms"], entry["nms"]["p50_ms"], entry["iters"]))
        if runner is None:
            del r
            torch.cuda.empty_cache()
    profile["batch_sizes"] = sorted(measured_sizes)

    if runner is not None and len(models) >= 2:
        profile["switch"] = switch_cost(runner, inputs, models[0], models[-1], args.switch_iters)
        print("switch cost %s<->%s: %s" % (models[0], models[-1], {
            m: round(profile["switch"][m]["penalty_ms_p50"], 3) for m in profile["switch"]["pair"]}))
    if device != "cpu":
        profile["device"]["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20

    prof = Profile(profile)
    b_cap = max(b for b in profile["batch_sizes"] if b <= config.B_MAX)
    profile["capacity_fps"] = capacity_fps(prof, models, config.ROUTING_SPLIT, b_cap)
    profile["capacity_b"] = b_cap
    profile["suggested_slos_ms"] = suggest_slos(prof, models)
    save_json(out, profile)
    print("capacity at b=%d with the COCO split: %.1f frames/s" % (b_cap, profile["capacity_fps"]))
    slowest = max(prof.t(m, 1, "p95") for m in models)
    bad = [s for s in config.SLOS_MS if s < 1.5 * slowest]
    if bad:
        print("FLAG: SLOs %s ms are below 1.5x the slowest model at b=1 (%.1f ms p95); suggested: %s" % (
            bad, slowest, profile["suggested_slos_ms"]))
    print("wrote", out)


if __name__ == "__main__":
    main()

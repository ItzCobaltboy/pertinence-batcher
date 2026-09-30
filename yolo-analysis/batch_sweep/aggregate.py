"""
Aggregation: reads raw/ only (never touches the GPU, imports no torch), writes

    results/batch_sweep/summary/<session>/
        run_summary.csv        one row per successful run: mean/median/std/p95/min/max + GPU state
        aggregated.csv         one row per (variant, model, batch size): mean across runs of the
                               per-run means, std across runs, number of runs used, status
        failures.csv           every failed / skipped config with its last error
        sanity_flags.csv/.txt  the checks below
        plots/                 T(b) and T(b)/b per variant, plus the three-variant comparison

and, only when asked (`sweep.py profile --variant V`), the scheduler-sim
profile CSV scheduler-sim/profiles/yolov8_<tag>_<V>.csv.

Sanity flags:
  - T(b) not increasing in b
  - per-image cost T(b)/b not decreasing in b
  - std across runs above STD_FLAG_FRACTION of the mean
  - failed / skipped configs, or fewer runs than planned
  - a TRT engine with PyTorch fallback subgraphs (the timed forward is not all TensorRT)
  - interference on the shared GPU: other compute processes, SM clock sag, throttling, heat
"""

import os

import numpy as np
import pandas as pd

import gpu_log
import settings
import store

KEY = ["variant", "model", "batch_size"]


def _g(snap, field):
    if not snap or not snap.get("gpu"):
        return None
    return snap["gpu"].get(field)


# ---------------------------------------------------------------------------
# raw -> tables
# ---------------------------------------------------------------------------
def run_rows(session):
    rows = []
    for path in store.all_run_records(session):
        if store.status_of(path) != "ok":
            continue
        r = store.read_json(path)
        fwd = r["summary"]["forward_ms"]
        fnms = r["summary"].get("forward_nms_ms") or {}
        eng = r.get("engine") or {}
        sub = eng.get("subgraphs") or {}
        start, end = r.get("gpu_start"), r.get("gpu_end")
        sm_end, sm_max = _g(end, "clocks.sm"), _g(end, "clocks.max.sm")
        rows.append({
            "session": r["session"], "variant": r["variant"], "model": r["model"],
            "batch_size": r["batch_size"], "run_index": r["run_index"], "img_size": r.get("img_size"),
            "attempt_timestamp": r.get("attempt_timestamp"), "started_at": r.get("started_at"),
            "fwd_mean_ms": fwd["mean"], "fwd_median_ms": fwd["median"], "fwd_std_ms": fwd["std"],
            "fwd_p95_ms": fwd["p95"], "fwd_min_ms": fwd["min"], "fwd_max_ms": fwd["max"], "fwd_n": fwd["n"],
            "fwd_nms_mean_ms": fnms.get("mean"), "fwd_nms_median_ms": fnms.get("median"),
            "fwd_nms_p95_ms": fnms.get("p95"),
            "loop_mean_ms": r["summary"].get("loop_mean_ms"),
            "per_image_ms": fwd["mean"] / r["batch_size"],
            "peak_memory_mib": r.get("peak_memory_mib"),
            "device": (r.get("device_info") or {}).get("device"),
            "gpu_name": (r.get("device_info") or {}).get("gpu_name"),
            "driver": _g(start, "driver_version"),
            "sm_clock_start_mhz": _g(start, "clocks.sm"), "sm_clock_end_mhz": sm_end,
            "sm_clock_max_mhz": sm_max,
            "temp_start_c": _g(start, "temperature.gpu"), "temp_end_c": _g(end, "temperature.gpu"),
            "power_end_w": _g(end, "power.draw"),
            "mem_used_start_mib": _g(start, "memory.used"),
            "other_procs_start": len(r.get("other_gpu_processes_start") or []),
            "other_procs_end": len(r.get("other_gpu_processes_end") or []),
            "throttle_end": "|".join(gpu_log.throttle_names(end)),
            "engine_trt_subgraphs": sub.get("trt_subgraphs"),
            "engine_torch_subgraphs": sub.get("torch_subgraphs"),
            "engine_fallback_ops": "|".join(sub.get("torch_fallback_ops") or []),
            "engine_compile_s": eng.get("compile_s"), "engine_load_s": eng.get("load_s"),
            "tf32": (r.get("device_info") or {}).get("cudnn_allow_tf32"),
            "torch": (r.get("versions") or {}).get("torch"),
            "torch_tensorrt": (r.get("versions") or {}).get("torch_tensorrt"),
            "ultralytics": (r.get("versions") or {}).get("ultralytics"),
            "file": os.path.relpath(path, settings.OUT_DIR),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["interference"] = df.apply(interference_reason, axis=1)
        df = df.sort_values(KEY + ["run_index"]).reset_index(drop=True)
    return df


def interference_reason(row):
    reasons = []
    if (row["other_procs_start"] or 0) > 0 or (row["other_procs_end"] or 0) > 0:
        reasons.append("other GPU processes")
    if row["sm_clock_end_mhz"] and row["sm_clock_max_mhz"] and \
            row["sm_clock_end_mhz"] < settings.CLOCK_SAG_FRACTION * row["sm_clock_max_mhz"]:
        reasons.append(f"SM clock {row['sm_clock_end_mhz']:.0f}/{row['sm_clock_max_mhz']:.0f} MHz at end")
    if row["throttle_end"]:
        reasons.append("throttle " + row["throttle_end"])
    if row["temp_end_c"] and row["temp_end_c"] >= settings.HOT_TEMPERATURE_C:
        reasons.append(f"{row['temp_end_c']:.0f} C")
    return "; ".join(reasons)


def failure_rows(session):
    """Latest failed/skipped record per config, for configs with no ok run at all,
    plus individual failed runs of configs that otherwise have data."""
    rows = []
    for path in store.all_compile_records(session) + store.all_run_records(session):
        status = store.status_of(path)
        if status == "ok":
            continue
        r = store.read_json(path)
        rows.append({"variant": r.get("variant"), "model": r.get("model"), "batch_size": r.get("batch_size"),
                     "stage": r.get("kind"), "run_index": r.get("run_index"), "status": status,
                     "error": (r.get("error") or "").splitlines()[0][:500] if r.get("error") else "",
                     "finished_at": r.get("finished_at"),
                     "file": os.path.relpath(path, settings.OUT_DIR)})
    return pd.DataFrame(rows, columns=["variant", "model", "batch_size", "stage", "run_index", "status",
                                       "error", "finished_at", "file"])


def aggregate_table(runs, failures):
    keys = set()
    if len(runs):
        keys |= set(map(tuple, runs[KEY].values.tolist()))
    if len(failures):
        keys |= set(map(tuple, failures[KEY].dropna().values.tolist()))
    planned_runs = int(runs["run_index"].max()) + 1 if len(runs) else 0

    rows = []
    for variant, model, b in sorted(keys, key=lambda k: (k[0], k[1], int(k[2]))):
        b = int(b)
        g = runs[(runs.variant == variant) & (runs.model == model) & (runs.batch_size == b)] if len(runs) else runs
        f = failures[(failures.variant == variant) & (failures.model == model) & (failures.batch_size == b)] \
            if len(failures) else failures
        n = len(g)
        means = g["fwd_mean_ms"].to_numpy() if n else np.array([])
        row = {"variant": variant, "model": model, "batch_size": b, "n_runs": n}
        if n:
            mean = float(means.mean())
            std = float(means.std(ddof=1)) if n > 1 else 0.0
            row.update({
                "mean_ms": mean, "std_across_runs_ms": std, "cv_across_runs": std / mean if mean else None,
                "min_run_mean_ms": float(means.min()), "max_run_mean_ms": float(means.max()),
                "median_ms": float(g["fwd_median_ms"].median()), "p95_ms": float(g["fwd_p95_ms"].mean()),
                "mean_within_run_std_ms": float(g["fwd_std_ms"].mean()),
                "per_image_ms": mean / b,
                "fwd_nms_mean_ms": float(g["fwd_nms_mean_ms"].mean()) if g["fwd_nms_mean_ms"].notna().any() else None,
                "loop_mean_ms": float(g["loop_mean_ms"].mean()),
                "peak_memory_mib": float(g["peak_memory_mib"].max()) if g["peak_memory_mib"].notna().any() else None,
                "engine_compile_s": g["engine_compile_s"].dropna().iloc[0] if g["engine_compile_s"].notna().any() else None,
                "engine_torch_subgraphs": g["engine_torch_subgraphs"].dropna().iloc[0]
                if g["engine_torch_subgraphs"].notna().any() else None,
                "runs_with_interference": int((g["interference"] != "").sum()),
            })
            row["status"] = "ok" if n >= planned_runs else "partial"
        else:
            last = f.sort_values("finished_at").iloc[-1] if len(f) else None
            row["status"] = last["status"] if last is not None else "missing"
            row["error"] = last["error"] if last is not None else ""
        rows.append(row)
    return pd.DataFrame(rows)


def sanity_flags(agg):
    flags = []

    def add(v, m, b, flag, detail):
        flags.append({"variant": v, "model": m, "batch_size": b, "flag": flag, "detail": detail})

    if len(agg) == 0:
        return pd.DataFrame(columns=["variant", "model", "batch_size", "flag", "detail"])
    for (v, m), g in agg.groupby(["variant", "model"]):
        g = g.sort_values("batch_size")
        ok = g[g.n_runs > 0]
        for prev, cur in zip(ok.itertuples(), list(ok.itertuples())[1:]):
            if cur.mean_ms <= prev.mean_ms:
                add(v, m, cur.batch_size, "T(b) not increasing",
                    f"T({prev.batch_size})={prev.mean_ms:.3f} ms >= T({cur.batch_size})={cur.mean_ms:.3f} ms")
            if cur.per_image_ms >= prev.per_image_ms:
                add(v, m, cur.batch_size, "per-image cost not decreasing",
                    f"T/b at {prev.batch_size}={prev.per_image_ms:.4f} ms, at {cur.batch_size}={cur.per_image_ms:.4f} ms")
        for r in g.itertuples():
            if r.n_runs == 0:
                add(v, m, r.batch_size, f"config {r.status}", getattr(r, "error", "") or "")
                continue
            if r.status == "partial":
                add(v, m, r.batch_size, "fewer runs than planned", f"{r.n_runs} runs")
            if r.n_runs > 1 and r.cv_across_runs > settings.STD_FLAG_FRACTION:
                add(v, m, r.batch_size, f"std across runs > {settings.STD_FLAG_FRACTION:.0%} of mean",
                    f"std {r.std_across_runs_ms:.3f} ms = {r.cv_across_runs:.1%} of {r.mean_ms:.3f} ms")
            if r.engine_torch_subgraphs and r.engine_torch_subgraphs > 0:
                add(v, m, r.batch_size, "TRT engine has PyTorch fallback",
                    f"{int(r.engine_torch_subgraphs)} PyTorch subgraph(s), see run_summary engine_fallback_ops")
            if r.runs_with_interference:
                add(v, m, r.batch_size, "possible GPU interference",
                    f"{r.runs_with_interference} of {r.n_runs} runs, see run_summary 'interference'")
    return pd.DataFrame(flags)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
# Fixed categorical order (never cycled): one slot per model, one per variant.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
TEXT = "#3d3d3a"
GRID = "#e6e5df"


def _style(ax, batch_sizes, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.minorticks_off()
    ax.set_xlabel("batch size b", color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ac")
    ax.tick_params(colors=TEXT)


def _line(ax, g, y, yerr, color, marker, label):
    g = g[g.n_runs > 0].sort_values("batch_size")
    if len(g) == 0:
        return
    ax.errorbar(g.batch_size, g[y], yerr=g[yerr] if yerr else None, color=color, marker=marker,
                markersize=6, linewidth=2, capsize=3, label=label)
    last = g.iloc[-1]
    ax.annotate(label, (last.batch_size, last[y]), xytext=(6, 0), textcoords="offset points",
                color=TEXT, fontsize=8, va="center")


def plots(agg, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if len(agg) == 0:
        return []
    agg = agg.copy()
    agg["per_image_std_ms"] = agg["std_across_runs_ms"] / agg["batch_size"]
    batch_sizes = sorted(agg.batch_size.unique())
    model_order = [settings.model_stem(m) for m in settings.MODELS if settings.model_stem(m) in set(agg.model)]
    model_order += sorted(set(agg.model) - set(model_order))
    with_data = set(agg[agg.n_runs > 0].variant)
    variant_order = [v for v in settings.VARIANTS if v in with_data]
    written = []

    for variant in variant_order:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for i, model in enumerate(model_order):
            g = agg[(agg.variant == variant) & (agg.model == model)]
            c, mk = SERIES_COLORS[i % 4], MARKERS[i % 4]
            _line(axes[0], g, "mean_ms", "std_across_runs_ms", c, mk, model)
            _line(axes[1], g, "per_image_ms", "per_image_std_ms", c, mk, model)
        _style(axes[0], batch_sizes, "T(b), ms per batch")
        _style(axes[1], batch_sizes, "T(b) / b, ms per image")
        axes[0].set_title("Batch runtime T(b)", color=TEXT, loc="left")
        axes[1].set_title("Per-image cost T(b)/b", color=TEXT, loc="left")
        axes[0].legend(frameon=False)
        fig.suptitle(f"YOLOv8 forward-only runtime, {variant} (mean of run means, bars = std across runs)",
                     color=TEXT)
        fig.tight_layout()
        path = os.path.join(out_dir, f"T_of_b_{variant}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(2, len(model_order), figsize=(4.2 * len(model_order), 8), squeeze=False)
    for j, model in enumerate(model_order):
        for i, variant in enumerate(variant_order):
            g = agg[(agg.variant == variant) & (agg.model == model)]
            c, mk = SERIES_COLORS[i % 4], MARKERS[i % 4]
            _line(axes[0][j], g, "mean_ms", "std_across_runs_ms", c, mk, variant)
            _line(axes[1][j], g, "per_image_ms", "per_image_std_ms", c, mk, variant)
        _style(axes[0][j], batch_sizes, "T(b), ms")
        _style(axes[1][j], batch_sizes, "T(b) / b, ms per image")
        axes[0][j].set_title(model, color=TEXT, loc="left")
    axes[0][0].legend(frameon=False)
    fig.suptitle("eager_fp32 vs trt_fp32 vs trt_fp16, per model", color=TEXT)
    fig.tight_layout()
    path = os.path.join(out_dir, "variant_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def summary_dir(session):
    return os.path.join(settings.SUMMARY_DIR, session)


def aggregate(session, log=print):
    out = summary_dir(session)
    os.makedirs(out, exist_ok=True)
    runs = run_rows(session)
    failures = failure_rows(session)
    agg = aggregate_table(runs, failures)
    flags = sanity_flags(agg)

    runs.to_csv(os.path.join(out, "run_summary.csv"), index=False)
    agg.to_csv(os.path.join(out, "aggregated.csv"), index=False)
    failures.to_csv(os.path.join(out, "failures.csv"), index=False)
    flags.to_csv(os.path.join(out, "sanity_flags.csv"), index=False)
    written = plots(agg, os.path.join(out, "plots"))

    lines = [f"session {session}: {len(runs)} ok runs, {len(agg)} configs, "
             f"{int((agg.n_runs == 0).sum()) if len(agg) else 0} configs with no data, "
             f"{len(failures)} failed/skipped records, {len(flags)} sanity flags"]
    for r in flags.itertuples():
        lines.append(f"  [{r.flag}] {r.variant} {r.model} bs={r.batch_size}: {r.detail}")
    text = "\n".join(lines)
    with open(os.path.join(out, "sanity_flags.txt"), "w") as f:
        f.write(text + "\n")
    log(text)
    log(f"wrote {out}/ (run_summary.csv, aggregated.csv, failures.csv, sanity_flags.*, "
        f"{len(written)} plots)")
    return agg


def write_profile(session, variant, profile_dir, tag, force):
    """scheduler-sim Profile format: '#' comment lines, header model,<b1>,<b2>,...,
    one row per model in pool order, ms per batch, empty cell = no data (failed)."""
    runs = run_rows(session)
    agg = aggregate_table(runs, failure_rows(session))
    g = agg[agg.variant == variant] if len(agg) else agg
    if len(g) == 0 or (g.n_runs > 0).sum() == 0:
        raise SystemExit(f"no successful runs for variant {variant} in session {session}")

    path = os.path.join(profile_dir, f"yolov8_{tag}_{variant}.csv")
    if os.path.exists(path) and not force:
        raise SystemExit(f"{path} already exists; pass --force to overwrite it")

    batch_sizes = sorted(set(int(b) for b in g.batch_size))
    model_order = [settings.model_stem(m) for m in settings.MODELS if settings.model_stem(m) in set(g.model)]
    model_order += sorted(set(g.model) - set(model_order))
    first = runs[runs.variant == variant].iloc[0]
    n_runs = sorted(set(int(n) for n in g.n_runs if n > 0))

    header = [
        f"# MEASURED YOLOv8 batched-inference runtimes T_i(b) in ms, variant {variant}.",
        f"# Built by yolo-analysis/batch_sweep/sweep.py profile from session '{session}' raw files.",
        f"# GPU {first.gpu_name}, driver {first.driver}, torch {first.torch}, "
        f"torch_tensorrt {first.torch_tensorrt}, ultralytics {first.ultralytics}.",
        "# Value = mean across independent runs of each run's mean forward time (runs per cell: "
        f"{','.join(map(str, n_runs))}). Forward only: fused DetectionModel incl. Detect head decode,",
        f"# fp32 input (B,3,{first.img_size},{first.img_size}) already on {first.device}, NMS excluded. "
        "Empty cell = config failed / no data.",
    ]
    lines = list(header)
    lines.append(",".join(["model"] + [str(b) for b in batch_sizes]))
    for model in model_order:
        cells = []
        for b in batch_sizes:
            row = g[(g.model == model) & (g.batch_size == b)]
            cells.append(f"{row.iloc[0].mean_ms:.3f}" if len(row) and row.iloc[0].n_runs > 0 else "")
        if all(c == "" for c in cells):
            lines.insert(len(header), f"# {model}: every batch size failed, row left out")
            continue
        lines.append(",".join([model] + cells))

    os.makedirs(profile_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path

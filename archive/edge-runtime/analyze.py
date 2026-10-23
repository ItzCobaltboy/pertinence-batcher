# Metrics and plots from run logs. No torch: runs on the laptop.
#
#   python analyze.py results/<device_tag>              every run in runs/ -> analysis/
#   python analyze.py results/<device_tag> --timeline <run_id>   one run's timeline only
#
# Per run, jobs that arrived in the first warmup_s of the trace are left out.
# Latency = finish - arrival, arrival = when the camera produced the frame (trace time),
# so arrival lateness and preprocessing are inside it. A job "meets" its SLO if it
# finished by its deadline; dropped and never-served jobs count as misses.
#
# Outputs (analysis/): summary.csv (one row per run), summary.md, plots/*.png,
# plots/timelines/<run_id>.png.

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import power
from common import load_json

PCTS = [50, 95, 99]


def policy_label(run):
    params = run.get("params") or {}
    if not params:
        return run["policy"]
    short = {"max_queue_delay_ms": "d", "beta": "b", "lam": "l", "buffer_ms": "buf", "drop": "drop"}
    return "%s(%s)" % (run["policy"], ",".join("%s=%s" % (short.get(k, k), v) for k, v in sorted(params.items())))


def trace_kind(run):
    meta = run.get("trace_meta") or {}
    kind = meta.get("source", "trace")
    if kind == "periodic" and meta.get("sticky"):
        kind = "sticky"
    if kind == "from-cityflow":
        kind = "cityflow_%s" % meta.get("scenario", "")
    return kind


def load_of(run):
    meta = run.get("trace_meta") or {}
    for k in ("load_fraction", "target_load"):
        if meta.get(k) is not None:
            return float(meta[k])
    return float("nan")


def _stats(values, prefix):
    out = {}
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        for k in ["mean", "p50", "p95", "p99", "max"]:
            out["%s_%s" % (prefix, k)] = float("nan")
        return out
    out[prefix + "_mean"] = float(values.mean())
    for p in PCTS:
        out["%s_p%d" % (prefix, p)] = float(np.percentile(values, p))
    out[prefix + "_max"] = float(values.max())
    return out


def _intervals_energy(t, p, intervals, t0, t1):
    total = 0.0
    for a, b in intervals:
        a, b = max(a, t0), min(b, t1)
        if b > a:
            e = power.energy_j(t, p, a, b)
            if e == e:
                total += e
    return total


def load_run(run_dir):
    run = load_json(os.path.join(run_dir, "run.json"))
    req = pd.read_csv(os.path.join(run_dir, "requests.csv"))
    bat = pd.read_csv(os.path.join(run_dir, "batches.csv"))
    waits = pd.read_csv(os.path.join(run_dir, "waits.csv"))
    dec = pd.read_csv(os.path.join(run_dir, "decisions.csv"))
    pw = power.read(os.path.join(run_dir, "power.csv"))
    return run, req, bat, waits, dec, pw


def run_metrics(run_dir):
    run, req, bat, waits, dec, pw = load_run(run_dir)
    t_start = run["t_trace_start"]
    w0 = t_start + run.get("warmup_s", 0.0)
    w1 = run["t_trace_end"]
    jobs = req[req["t_target"] >= w0]
    served = jobs[jobs["t_end"].notna() & (jobs["dropped"] == 0)]
    met = served[served["t_end"] <= served["deadline"]]
    n = max(len(jobs), 1)
    window_arrivals = max(run["duration_s"] - run.get("warmup_s", 0.0), 1e-9)

    m = {"run_id": os.path.basename(os.path.normpath(run_dir)), "status": run["status"],
         "policy": run["policy"], "label": policy_label(run), "params": json.dumps(run.get("params") or {}),
         "trace": os.path.basename(run["trace"]), "trace_kind": trace_kind(run), "load": load_of(run),
         "slo_ms": run["slo_ms"], "input": run["input"], "prep": run["prep"], "nms": run.get("nms"),
         "jobs": int(len(jobs)), "served": int(len(served))}
    lat = (served["t_end"] - served["t_target"]) * 1000.0
    m.update(_stats(lat, "latency_ms"))
    m.update(_stats((served["t_start"] - served["t_enqueued"]) * 1000.0, "queue_ms"))
    m.update(_stats(served["decode_ms"] + served["prep_ms"], "prep_ms"))
    m.update(_stats((served["t_end"] - served["t_start"]) * 1000.0, "batch_ms"))
    for q, g in served.groupby("queue"):
        part = _stats((g["t_end"] - g["t_target"]) * 1000.0, "latency_%s_ms" % q)
        m["latency_%s_ms_mean" % q] = part["latency_%s_ms_mean" % q]
        m["latency_%s_ms_p95" % q] = part["latency_%s_ms_p95" % q]

    drain_start = run.get("t_arrivals_done") or w1
    m["drain_s"] = w1 - drain_start
    m["drain_jobs"] = int((served["t_end"] > drain_start).sum())

    m["miss_rate"] = 1.0 - len(met) / float(n)
    m["drop_rate"] = float(jobs["dropped"].sum()) / n
    m["unserved_rate"] = float(jobs["unserved"].sum()) / n
    m["late_rate"] = float((served["t_end"] > served["deadline"]).sum()) / n
    m["goodput_fps"] = len(met) / window_arrivals

    b = bat[bat["t_end"] >= w0]
    m["batches"] = int(len(b))
    m["mean_b"] = float(b["b"].mean()) if len(b) else float("nan")
    m["batches_per_s"] = len(b) / max(w1 - w0, 1e-9)
    m["b_hist"] = json.dumps({str(model): {str(int(k)): int(v) for k, v in g["b"].value_counts().sort_index().items()}
                              for model, g in b.groupby("model")})
    m["promotions"] = int((jobs["promoted_from"].fillna("") != "").sum())
    switches = (b["model"].values[1:] != b["model"].values[:-1]) if len(b) > 1 else np.array([])
    m["switches"] = int(switches.sum())
    if len(b) > 1:
        err = (b["forward_ms"] + b["nms_ms"] - b["planned_ms"]).values[1:]
        if switches.any() and (~switches).any():
            m["switch_penalty_ms"] = float(err[switches].mean() - err[~switches].mean())
        m["profile_error_ms_mean"] = float(err.mean())

    # hidden costs
    m.update(_stats((jobs["t_actual"] - jobs["t_target"]) * 1000.0, "arrival_late_ms"))
    m.update(_stats(dec["decide_ms"], "decide_ms"))

    # energy
    if len(pw) >= 2 and pw["power_w"].notna().any():
        t, p = pw["t"].values, pw["power_w"].values
        marks = dict((k, v) for k, v in run.get("marks", []))
        idle_w = power.mean_power(t, p, marks.get("idle_start", np.nan), marks.get("idle_end", np.nan)) \
            if "idle_start" in marks else float("nan")
        e_total = power.energy_j(t, p, w0, w1)
        busy = list(zip(b["t_start"], b["t_end"]))
        e_busy = _intervals_energy(t, p, busy, w0, w1)
        busy_s = sum(max(0.0, min(e, w1) - max(s, w0)) for s, e in busy)
        frames = max(len(served), 1)
        m.update({"idle_loaded_w": idle_w, "mean_power_w": e_total / (w1 - w0), "energy_j": e_total,
                  "energy_per_frame_j": e_total / frames,
                  "dyn_energy_per_frame_j": (e_total - idle_w * (w1 - w0)) / frames,
                  "busy_energy_j": e_busy, "idle_energy_j": e_total - e_busy,
                  "gpu_busy_frac": busy_s / (w1 - w0)})
        inside = pw[(pw["t"] >= w0) & (pw["t"] <= w1)]
        for col in ["temp_c", "sm_clock_mhz", "gpu_freq_mhz"]:
            if col in inside and inside[col].notna().any() and len(inside) > 10:
                head = inside[inside["t"] < w0 + 10][col].mean()
                tail = inside[inside["t"] > w1 - 10][col].mean()
                m[col + "_start"] = float(head)
                m[col + "_drift"] = float(tail - head)
        if "power_source" in pw:
            m["power_source"] = str(pw["power_source"].iloc[0])
    m["power_backend"] = run.get("power_backend")
    return m


# ---------------------------------------------------------------------- plots
def _groups(df, keys):
    return df.groupby(keys, dropna=False) if len(df) else []


def plot_cdfs(runs_dir, df, out_dir):
    for key, g in _groups(df, ["trace", "slo_ms", "input", "prep"]):
        plt.figure(figsize=(7, 4.5))
        for _, row in g.sort_values("label").iterrows():
            req = pd.read_csv(os.path.join(runs_dir, row["run_id"], "requests.csv"))
            run = load_json(os.path.join(runs_dir, row["run_id"], "run.json"))
            req = req[(req["t_target"] >= run["t_trace_start"] + run.get("warmup_s", 0)) & req["t_end"].notna()]
            lat = np.sort((req["t_end"] - req["t_target"]).values * 1000.0)
            if len(lat):
                plt.plot(lat, np.arange(1, len(lat) + 1) / len(lat), label=row["label"])
        plt.axvline(key[1], color="black", linestyle="--", linewidth=1, label="SLO")
        plt.xscale("log")
        plt.xlabel("latency, ms (served jobs)")
        plt.ylabel("CDF")
        plt.title("%s, SLO %g ms, %s/%s" % key)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cdf_%s_slo%g_%s_%s.png" % (os.path.splitext(key[0])[0], key[1], key[2], key[3])),
                    dpi=110)
        plt.close()


def plot_vs_load(df, out_dir, metric, ylabel, name):
    for key, g in _groups(df, ["trace_kind", "slo_ms", "input", "prep"]):
        if g["load"].nunique() < 2:
            continue
        plt.figure(figsize=(7, 4.5))
        for label, h in g.groupby("label"):
            h = h.groupby("load")[metric].mean().sort_index()
            plt.plot(h.index, h.values, marker="o", label=label)
        plt.xlabel("load (fraction of capacity)")
        plt.ylabel(ylabel)
        plt.title("%s, SLO %g ms, %s/%s" % key)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "%s_%s_slo%g_%s_%s.png" % (name, key[0], key[1], key[2], key[3])), dpi=110)
        plt.close()


def plot_pareto(df, out_dir):
    if "energy_per_frame_j" not in df:
        return
    for key, g in _groups(df, ["trace", "slo_ms", "input", "prep"]):
        g = g.dropna(subset=["energy_per_frame_j", "latency_ms_p95"])
        if not len(g):
            continue
        plt.figure(figsize=(7, 4.5))
        for policy, h in g.groupby("policy"):
            h = h.sort_values("energy_per_frame_j")
            style = "-o" if len(h) > 1 else "o"
            plt.plot(h["energy_per_frame_j"], h["latency_ms_p95"], style, label=policy)
        pts = g[["energy_per_frame_j", "latency_ms_p95"]].values
        front = [pt for pt in pts if not any((o <= pt).all() and (o < pt).any() for o in pts)]
        front = np.array(sorted(front, key=lambda x: x[0]))
        plt.plot(front[:, 0], front[:, 1], color="black", linewidth=1, linestyle=":", label="Pareto front")
        plt.xlabel("energy per finished frame, J")
        plt.ylabel("p95 latency, ms")
        plt.title("%s, SLO %g ms, %s/%s" % key)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pareto_%s_slo%g_%s_%s.png" % (os.path.splitext(key[0])[0], key[1], key[2], key[3])),
                    dpi=110)
        plt.close()


def plot_batch_hists(df, out_dir):
    for key, g in _groups(df, ["trace", "slo_ms", "input", "prep"]):
        g = g.sort_values("label")
        models = sorted({m for h in g["b_hist"] for m in json.loads(h)})
        if not models:
            continue
        fig, axes = plt.subplots(len(g), len(models), figsize=(3 * len(models), 1.8 * len(g) + 0.5),
                                 squeeze=False, sharex=True)
        for i, (_, row) in enumerate(g.iterrows()):
            hist = json.loads(row["b_hist"])
            for j, m in enumerate(models):
                ax = axes[i][j]
                h = hist.get(m, {})
                ax.bar([int(k) for k in h], list(h.values()), width=0.8)
                if i == 0:
                    ax.set_title(m, fontsize=9)
                if j == 0:
                    ax.set_ylabel(row["label"], fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("batch size")
        fig.suptitle("batch sizes: %s, SLO %g ms, %s/%s" % key, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "batches_%s_slo%g_%s_%s.png" % (os.path.splitext(key[0])[0], key[1], key[2], key[3])),
                    dpi=100)
        plt.close(fig)


def plot_timeline(run_dir, path, span_s=None):
    run, req, bat, waits, dec, pw = load_run(run_dir)
    t0 = run["t_trace_start"]
    t1 = run["t_trace_end"] if span_s is None else t0 + span_s
    models = run["models"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.5]})
    grid = np.linspace(t0, t1, 2000)
    for m in models:
        q = req[req["model_run"] == m]
        enq = np.sort(q["t_enqueued"].dropna().values)
        left = np.sort(np.concatenate([q["t_start"].dropna().values, q["t_drop"].dropna().values]))
        length = np.searchsorted(enq, grid) - np.searchsorted(left, grid)
        axes[0].plot(grid - t0, length, label=m, linewidth=1)
    axes[0].set_ylabel("queued jobs")
    axes[0].legend(fontsize=8, loc="upper right")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, m in enumerate(models):
        g = bat[(bat["model"] == m) & (bat["t_end"] >= t0) & (bat["t_start"] <= t1)]
        axes[1].broken_barh(list(zip(g["t_start"] - t0, g["t_end"] - g["t_start"])), (i - 0.4, 0.8),
                            color=colors[i % len(colors)])
    axes[1].set_yticks(range(len(models)))
    axes[1].set_yticklabels(models)
    axes[1].set_ylabel("batches")
    if len(pw):
        inside = pw[(pw["t"] >= t0) & (pw["t"] <= t1)]
        axes[2].plot(inside["t"] - t0, inside["power_w"], linewidth=0.8)
    axes[2].set_ylabel("power, W")
    axes[2].set_xlabel("seconds since trace start")
    axes[0].set_title("%s: %s, %s, SLO %g ms" % (os.path.basename(os.path.normpath(run_dir)), policy_label(run),
                                                os.path.basename(run["trace"]), run["slo_ms"]))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_md(df, path):
    cols = ["run_id", "label", "trace_kind", "load", "slo_ms", "input", "prep", "status", "latency_ms_p50",
            "latency_ms_p95", "miss_rate", "drop_rate", "goodput_fps", "mean_b", "energy_per_frame_j",
            "dyn_energy_per_frame_j", "gpu_busy_frac", "decide_ms_p95", "arrival_late_ms_p95"]
    cols = [c for c in cols if c in df]
    lines = ["# edge-runtime summary", "", "%d runs. Latency in ms, energy in J per finished frame." % len(df), "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.sort_values(["trace_kind", "slo_ms", "load", "label"]).iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append("%.3g" % v if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def analyze(device_dir, timelines=True):
    runs_dir = os.path.join(device_dir, "runs")
    out_dir = os.path.join(device_dir, "analysis")
    plots = os.path.join(out_dir, "plots")
    os.makedirs(os.path.join(plots, "timelines"), exist_ok=True)
    rows = []
    for run_json in sorted(glob.glob(os.path.join(runs_dir, "*", "run.json"))):
        run_dir = os.path.dirname(run_json)
        try:
            rows.append(run_metrics(run_dir))
        except Exception as e:      # one broken run must not hide the others
            print("skipping %s: %s" % (run_dir, e))
            continue
        if timelines:
            plot_timeline(run_dir, os.path.join(plots, "timelines", os.path.basename(run_dir) + ".png"))
    if not rows:
        print("no complete runs under %s" % runs_dir)
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    write_md(df, os.path.join(out_dir, "summary.md"))
    plot_cdfs(runs_dir, df, plots)
    plot_vs_load(df, plots, "latency_ms_p95", "p95 latency, ms", "p95_vs_load")
    if "energy_per_frame_j" in df:
        plot_vs_load(df, plots, "energy_per_frame_j", "energy per finished frame, J", "energy_vs_load")
    plot_pareto(df, plots)
    plot_batch_hists(df, plots)
    print("%d runs -> %s" % (len(df), out_dir))
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Metrics and plots from edge-runtime run logs.")
    parser.add_argument("device_dir", help="results/<device_tag>")
    parser.add_argument("--timeline", help="only draw this run's timeline")
    parser.add_argument("--span", type=float, help="timeline: first N seconds only")
    parser.add_argument("--no-timelines", action="store_true")
    args = parser.parse_args(argv)
    if args.timeline:
        run_dir = os.path.join(args.device_dir, "runs", args.timeline)
        out = os.path.join(args.device_dir, "analysis", "plots", "timelines", args.timeline + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_timeline(run_dir, out, args.span)
        print(out)
        return
    analyze(args.device_dir, timelines=not args.no_timelines)


if __name__ == "__main__":
    main()

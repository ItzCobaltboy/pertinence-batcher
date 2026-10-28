# Turns the run folders into summary.csv / summary.md, validation.md and the plots. Laptop or board, no torch.
# Run: python3 04_analyze.py   (reads results/<device_tag>/runs, writes results/<device_tag>/analysis)
import sys
sys.dont_write_bytecode = True
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import metrics
from common import profile as profile_lib

PLOTS = os.path.join(config.ANALYSIS_DIR, "plots")


# ---------- loading ----------

def load_all(profile):
    """Every finished run -> (run dict, summary row, checks 1 to 8)."""
    out = []
    for path in sorted(glob.glob(os.path.join(config.RUNS_DIR, "*", "run.json"))):
        run_dir = os.path.dirname(path)
        try:
            run = metrics.load_run(run_dir)
        except (OSError, KeyError) as e:
            print("skipping %s: %s" % (run_dir, e))
            continue
        if run["info"]["status"] != "done":
            print("skipping %s: status %s" % (os.path.basename(run_dir), run["info"]["status"]))
            continue
        row = metrics.summarize(run, profile)
        row["run"] = run["info"]["name"]
        checks = metrics.run_checks(run, profile)
        row["checks_failed"] = sum(1 for c in checks if not c[1])
        out.append((run, row, checks))
    return out


def nearest(values, target):
    return min(values, key=lambda v: abs(v - target))


def md_table(df, cols, digits=1):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(("%.*f" % (digits, v)) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------- summary and validation text ----------

def write_summary(df):
    df.to_csv(os.path.join(config.ANALYSIS_DIR, "summary.csv"), index=False)
    cols = ["mix", "lam", "params", "seed", "n_jobs", "lat_mean_ms", "lat_p99_ms", "energy_j", "energy_per_req_mj",
            "busy_share", "gpu_busy_frac", "span_s", "drain_s", "batch_mean_size", "policy_drops", "goodput_pct"] \
           + ["viol_pct_%d" % s for s in config.SLO_MS_GRID] + ["flags"]
    text = ["# Summary (one row per run)", "",
            "Energy columns use the primary power definition: `%s`. Latency is t_batch_end - t_arrival_target." % config.PRIMARY_POWER, ""]
    for mix in sorted(df["mix"].unique()):
        text += ["## mix %s" % mix, "", md_table(df[df["mix"] == mix].sort_values(["lam", "params", "seed"]), cols, 2), ""]
    text += drain_notes(df)
    with open(os.path.join(config.ANALYSIS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(text) + "\n")


def drain_notes(df):
    """Flag groups where t_last differs between policies by more than a few percent (spec section 13)."""
    notes = []
    for (mix, lam, seed), g in df.groupby(["mix", "lam", "seed"]):
        if len(g) < 2:
            continue
        spread = (g["span_s"].max() - g["span_s"].min()) / g["span_s"].mean()
        if spread > 0.03:
            drain_share = g["drain_s"].max() / g["span_s"].mean()
            notes.append("- %s lam %s seed %s: span differs by %.1f%% between policies (longest drain tail %.1f s, %.1f%% of the span). "
                         "Energy differences here include the drain tail; it is not trimmed."
                         % (mix, lam, seed, spread * 100.0, g["drain_s"].max(), drain_share * 100.0))
    return ["## Span differences between policies", ""] + (notes or ["- none above 3%"]) + [""]


def write_validation(results):
    lines = ["# Validation (claim 3: the harness is trustworthy)", ""]
    per_check = {}
    for run, row, checks in results:
        for name, ok, detail in checks:
            per_check.setdefault(name, []).append(ok)
    lines += ["## Across all %d runs" % len(results), "", "| check | passed | failed |", "|---|---|---|"]
    for name in sorted(per_check):
        lines.append("| %s | %d | %d |" % (name, sum(per_check[name]), len(per_check[name]) - sum(per_check[name])))
    lines += ["", "## Check 9: same job set across policies", ""]
    groups = {}
    for run, row, checks in results:
        groups.setdefault((run["info"]["mix"], run["info"]["lambda_total"], run["info"]["seed"]), []).append(run)
    for key in sorted(groups):
        name, ok, detail = metrics.check_same_jobs(groups[key])
        lines.append("- %s  %s lam %s seed %s: %s" % ("PASS" if ok else "FAIL", key[0], key[1], key[2], detail))
    lines += ["", "## Per run", ""]
    for run, row, checks in results:
        lines.append("### %s" % run["info"]["name"])
        for name, ok, detail in checks:
            lines.append("- %s  %s: %s" % ("PASS" if ok else "FAIL", name, detail))
        lines.append("")
    with open(os.path.join(config.ANALYSIS_DIR, "validation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------- plots ----------

def mean_over_seeds(df):
    return df.groupby(["params", "lam"], as_index=False).mean(numeric_only=True)


def save(fig, name, title=None):
    os.makedirs(PLOTS, exist_ok=True)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name + ".png"), dpi=130)
    plt.close(fig)


def lines_vs_lam(ax, agg, column, ylabel):
    for label, g in agg.groupby("params"):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g[column], marker="o", label=label)
    ax.set_xlabel("offered load, total req/s")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_latency_and_energy(df, mix):
    agg = mean_over_seeds(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lines_vs_lam(axes[0], agg, "lat_mean_ms", "mean latency (ms)")
    lines_vs_lam(axes[1], agg, "lat_p99_ms", "p99 latency (ms)")
    axes[1].legend(fontsize=7)
    axes[0].set_title("mean latency", fontsize=9)
    axes[1].set_title("p99 latency (the slowest 1% of requests)", fontsize=9)
    save(fig, "%s_latency_vs_lam" % mix, "Latency vs offered load, %s mix (lower is better)" % mix)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lines_vs_lam(axes[0], agg, "energy_j", "total energy (J), first arrival to last completion")
    lines_vs_lam(axes[1], agg, "energy_per_req_mj", "energy per request (mJ)")
    axes[1].legend(fontsize=7)
    axes[0].set_title("total energy", fontsize=9)
    axes[1].set_title("energy per request", fontsize=9)
    save(fig, "%s_energy_vs_lam" % mix, "Energy vs offered load, %s mix (lower is better)" % mix)


def plot_busy_idle(df, mix):
    lam = nearest(df["lam"].unique(), config.PLOT_LAM_ENERGY_BAR)
    g = df[df["lam"] == lam].groupby("params", as_index=False).mean(numeric_only=True)
    g = g.sort_values("energy_j").reset_index(drop=True)   # lowest total energy on the left
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(g["params"], g["busy_j"], label="busy (batch on GPU)")
    ax.bar(g["params"], g["idle_j"], bottom=g["busy_j"], label="idle (rest of the span)")
    for i, r in g.iterrows():
        ax.text(i, r["energy_j"] + 8, "%d dropped" % round(r["policy_drops"]), ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_ylim(0, g["energy_j"].max() * 1.25)
    ax.set_ylabel("energy (J)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "%s_busy_idle_energy" % mix,
         "Busy vs idle energy per policy, %s mix, load %.0f req/s\n(sorted by total energy; label = requests dropped in the window)" % (mix, lam))


def pareto_front(points):
    """points: list of (x, y); lower is better on both. Returns the non-dominated ones sorted by x."""
    front = []
    for p in sorted(points):
        if not front or p[1] < front[-1][1]:
            front.append(p)
    return front


def plot_pareto(df, mix):
    lams = sorted(df["lam"].unique())
    fig, axes = plt.subplots(1, len(lams), figsize=(4.5 * len(lams), 4.5), squeeze=False)
    for ax, lam in zip(axes[0], lams):
        g = df[df["lam"] == lam].groupby("params", as_index=False).mean(numeric_only=True)
        ax.scatter(g["lat_mean_ms"], g["energy_j"])
        for _, r in g.iterrows():
            ax.annotate(r["params"], (r["lat_mean_ms"], r["energy_j"]), fontsize=6)
        front = pareto_front(list(zip(g["lat_mean_ms"], g["energy_j"])))
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="k", alpha=0.5)
        ax.set_title("load %.0f req/s" % lam, fontsize=9)
        ax.set_xlabel("mean latency (ms)")
        ax.set_ylabel("total energy (J)")
        ax.grid(alpha=0.3)
    save(fig, "%s_latency_vs_energy_pareto" % mix, "Latency vs total energy per policy, %s mix (bottom-left is better)" % mix)


def plot_queue_p99(df, mix):
    lam = nearest(df["lam"].unique(), config.PLOT_LAM_QUEUE_P99)
    g = df[df["lam"] == lam].groupby("params", as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(config.QUEUES)
    x = np.arange(len(g))
    for i, letter in enumerate(config.QUEUES):
        ax.bar(x + i * width, g["p99_%s_ms" % letter], width, label="queue " + letter)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(g["params"], rotation=60, fontsize=7)
    ax.set_ylabel("p99 latency (ms)")
    ax.legend()
    save(fig, "%s_per_queue_p99" % mix, "p99 latency per queue and policy, %s mix, load %.0f req/s" % (mix, lam))


def plot_batch_hist(results, mix):
    lam = nearest(sorted(set(r[0]["info"]["lambda_total"] for r in results if r[0]["info"]["mix"] == mix)), config.PLOT_LAM_ENERGY_BAR)
    chosen = [r[0] for r in results if r[0]["info"]["mix"] == mix and r[0]["info"]["lambda_total"] == lam and r[0]["info"]["seed"] == min(
        x[0]["info"]["seed"] for x in results)]
    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3 * rows), squeeze=False)
    sizes = np.arange(1, config.B_MAX + 1)
    for ax, run in zip(axes.flat, chosen):
        b = run["batches"]
        width = 0.8 / len(config.QUEUES)
        for q, letter in enumerate(config.QUEUES):
            counts = [int(np.sum((b["queue"] == q) & (b["b"] == s))) for s in sizes]
            ax.bar(sizes + q * width, counts, width, label=letter)
        ax.set_title(run["info"]["policy_label"], fontsize=8)
        ax.set_xlabel("batch size")
    axes.flat[0].legend(fontsize=7)
    save(fig, "%s_batch_size_hist_lam%.0f" % (mix, lam),
         "Batch sizes per policy and queue (how many batches of each size), %s mix, load %.0f req/s" % (mix, lam))


def queue_length_steps(run, q, t_lo, t_hi):
    req = run["requests"]
    sel = (req["queue"] == q) & (req["dropped"] == 0) & ~np.isnan(req["t_batch_start"])
    times = np.concatenate([req["t_enqueued"][sel], req["t_batch_start"][sel]])
    deltas = np.concatenate([np.ones(sel.sum()), -np.ones(sel.sum())])
    order = np.argsort(times, kind="stable")
    return times[order], np.cumsum(deltas[order])


def plot_timeline(results):
    candidates = [r[0] for r in results if r[0]["info"]["policy"] == "lqf"] or [r[0] for r in results]
    lams = sorted(set(r["info"]["lambda_total"] for r in candidates))
    lam = nearest(lams, config.PLOT_LAM_ENERGY_BAR)
    run = [r for r in candidates if r["info"]["lambda_total"] == lam][0]
    info = run["info"]
    t_lo = info["t0"] + info["warmup_s"] + 5
    t_hi = t_lo + config.TIMELINE_S
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for q, letter in enumerate(config.QUEUES):
        t, n = queue_length_steps(run, q, t_lo, t_hi)
        axes[0].step(t - t_lo, n, where="post", label=letter)
    axes[0].set_ylabel("queue length")
    axes[0].legend(ncol=4)
    b = run["batches"]
    for q in range(len(config.QUEUES)):
        sel = (b["queue"] == q) & (b["t_end"] >= t_lo) & (b["t_start"] <= t_hi)
        spans = [(s - t_lo, e - s) for s, e in zip(b["t_start"][sel], b["t_end"][sel])]
        axes[1].broken_barh(spans, (q - 0.4, 0.8))
    axes[1].set_yticks(range(len(config.QUEUES)))
    axes[1].set_yticklabels(config.QUEUES)
    axes[1].set_ylabel("batch on GPU")
    p = run["power"]
    sel = (p["t_monotonic"] >= t_lo) & (p["t_monotonic"] <= t_hi)
    axes[2].plot(p["t_monotonic"][sel] - t_lo, p[config.POWER_COLS[config.PRIMARY_POWER]][sel] / 1000.0)
    axes[2].set_xlim(0, config.TIMELINE_S)
    axes[2].set_ylabel("board power (W)")
    axes[2].set_xlabel("seconds into the window")
    save(fig, "timeline_%s" % info["name"],
         "One run over 10 s: queues, GPU batches, board power (%s, load %.0f req/s)" % (info["policy_label"], info["lambda_total"]))


def plot_slo_violations(df, mix):
    """Share of requests above each deadline, per load. Lines = policies measured against every deadline in the grid;
    EdgeServing runs are tuned to one deadline each, so they appear as points (deadline it was run with, violations at it).
    Runs are put at the nearest load of the current plan (old rho runs at 8.2/16.4/23.3 land on 8/16/24)."""
    df = df.copy()
    loads = sorted(df["lam"].round(0).unique())
    df["panel"] = df["lam"].round(0)
    fig, axes = plt.subplots(1, len(loads), figsize=(4.8 * len(loads), 4.5), squeeze=False)
    for ax, load in zip(axes[0], loads):
        g = df[df["panel"] == load]
        tuned = {}
        for _, r in g.groupby("params", as_index=False).mean(numeric_only=True).iterrows():
            label = r["params"]
            m = re.match(r"^(.*)_slo_ms(\d+)$", label)
            if m:
                slo = int(m.group(2))
                tuned.setdefault(m.group(1), []).append((slo, r["viol_pct_%d" % slo]))
            else:
                ax.plot(config.SLO_MS_GRID, [r["viol_pct_%d" % s] for s in config.SLO_MS_GRID], marker="o", label=label)
        for (family, points), marker in zip(sorted(tuned.items()), ["*", "D", "s"]):
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], marker=marker, markersize=9, color="k",
                    linestyle="--", label="%s (tuned to each deadline)" % family)
        ax.set_xscale("log")
        ax.set_xticks(config.SLO_MS_GRID)
        ax.set_xticklabels([str(s) for s in config.SLO_MS_GRID])
        ax.set_xlabel("deadline (ms)")
        ax.set_ylabel("requests over the deadline (%)")
        ax.set_title("load %d req/s" % load, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=6)
    save(fig, "%s_slo_violations" % mix, "Share of requests over each deadline, %s mix (lower is better)" % mix)


def plot_mix_sweep(df):
    """Goodput and energy per good request against the mix at one fixed load (same req/s for every mix), one panel per deadline,
    one line per policy. Mean over seeds, bars = std over seeds (single seed: no bars)."""
    d = df[df["slo_ms"].notna() & df["mix"].isin(config.MIX_ORDER)].copy()
    if d["mix"].nunique() < 2:
        return
    order = [m for m in config.MIX_ORDER if m in set(d["mix"])]
    d["family"] = d["params"].apply(lambda p: re.sub(r"_slo_ms\d+$", "", p))
    for lam in config.MIX_SWEEP_LAMS:
        keep = d[d["lam"].round(0) == lam]
        if keep["mix"].nunique() < 2:
            continue
        slos = sorted(s for s, g in keep.groupby("slo_ms") if g["mix"].nunique() >= 3)   # skip deadlines only run on one mix
        fig, axes = plt.subplots(2, len(slos), figsize=(4.6 * len(slos), 7), squeeze=False, sharex=True)
        for j, slo in enumerate(slos):
            for i, (col, ylabel) in enumerate([("goodput_pct", "goodput (% of arrivals)"),
                                               ("energy_per_good_req_mj", "energy per good request (mJ)")]):
                ax = axes[i][j]
                for family, g in keep[keep["slo_ms"] == slo].groupby("family"):
                    agg = g.groupby("mix")[col].agg(["mean", "std"]).reindex(order)
                    ax.errorbar(range(len(order)), agg["mean"], yerr=agg["std"].fillna(0), marker="o", capsize=3, label=family)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=45)
                ax.set_ylabel(ylabel)
                ax.set_title("deadline %d ms" % slo, fontsize=9)
                ax.grid(alpha=0.3)
                if col == "energy_per_good_req_mj":
                    ax.set_yscale("log")
        axes[0][-1].legend(fontsize=7)
        save(fig, "mix_sweep_lam%d" % lam, "Policies across the skew ladder at %d req/s "
             "(left = heavy-model dominated, right = cheap-model dominated; mean +/- std over seeds)" % lam)


def make_plots(df, results):
    plot_mix_sweep(df)
    for mix in sorted(df["mix"].unique()):
        d = df[df["mix"] == mix]
        plot_latency_and_energy(d, mix)
        plot_busy_idle(d, mix)
        plot_pareto(d, mix)
        plot_queue_p99(d, mix)
        plot_slo_violations(d, mix)
        plot_batch_hist(results, mix)
    plot_timeline(results)


# ---------- main ----------

def main():
    if not os.path.exists(config.PROFILE_PATH):
        sys.exit("no profile.json at %s" % config.PROFILE_PATH)
    profile = profile_lib.load(config.PROFILE_PATH)
    results = load_all(profile)
    if not results:
        sys.exit("no finished runs in %s" % config.RUNS_DIR)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    df = pd.DataFrame([r[1] for r in results])
    write_summary(df)
    write_validation(results)
    make_plots(df, results)
    failed = int(df["checks_failed"].sum())
    print("%d runs analysed, %d failed checks in total (see validation.md)" % (len(df), failed))
    print("outputs in", config.ANALYSIS_DIR)


main()

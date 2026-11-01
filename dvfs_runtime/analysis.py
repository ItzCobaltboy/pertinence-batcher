"""Plots and tables for the DVFS profiling results (dvfs/results/<device>/).

Run from anywhere:
    python dvfs/analysis.py
        newest models_* run that has a run_summary.csv + every switch_* run that has a switch_raw.csv
    python dvfs/analysis.py --models-dir PATH --switch-dirs PATH PATH ...

Reads run_summary.csv, subruns.csv, meta_*.json (models run) and switch_raw.csv (switch runs).
Writes PNGs, summary.txt and best_clock_table.csv to dvfs/plots/<models run folder name>/.
Needs numpy, pandas and matplotlib only.

dvfs/inspect.py shadows the standard library `inspect`, which matplotlib imports. When this file is run as a script,
Python puts dvfs/ first on sys.path, so that folder is dropped from sys.path below before matplotlib is imported.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import matplotlib
import matplotlib.ticker as mticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS = os.path.join(HERE, "results", "agx_xavier_jp511")
OUT_ROOT = os.path.join(HERE, "plots")

MODELS = ["n", "s", "m", "l"]
MODEL_NAME = {"n": "yolov8n", "s": "yolov8s", "m": "yolov8m", "l": "yolov8l",
              "r18": "resnet18", "r34": "resnet34", "r50": "resnet50", "r101": "resnet101", "r152": "resnet152"}
MODEL_COLOR = {"n": "tab:blue", "s": "tab:green", "m": "tab:orange", "l": "tab:red"}
DPI = 140


# ---------------------------------------------------------------- loading

def newest_models_dir(results_dir):
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "models_*"))):
        if os.path.exists(os.path.join(path, "run_summary.csv")):
            found.append(path)
    if len(found) == 0:
        raise SystemExit("no models_* folder with run_summary.csv in " + results_dir)
    return found[-1]


def switch_dirs_with_raw(results_dir):
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "switch_*"))):
        if os.path.exists(os.path.join(path, "switch_raw.csv")):
            found.append(path)
    return found


def load_models(models_dir):
    summary = pd.read_csv(os.path.join(models_dir, "run_summary.csv"))
    subruns = pd.read_csv(os.path.join(models_dir, "subruns.csv"))
    subruns["freq_mhz"] = subruns["freq_hz"] / 1e6
    meta = {}
    meta_files = sorted(glob.glob(os.path.join(models_dir, "meta_*.json")))
    if len(meta_files) > 0:
        with open(meta_files[0]) as f:
            meta = json.load(f)
    return summary, subruns, meta


def load_switch(dirs):
    frames = []
    for d in dirs:
        raw = pd.read_csv(os.path.join(d, "switch_raw.csv"))
        raw["run"] = os.path.basename(d)
        frames.append(raw)
    allraw = pd.concat(frames, ignore_index=True)
    allraw["from_mhz"] = allraw["from_hz"] / 1e6
    allraw["to_mhz"] = allraw["to_hz"] / 1e6
    allraw["jump_mhz"] = (allraw["to_mhz"] - allraw["from_mhz"]).abs()
    return allraw


def freq_label(f):
    return "%.0f" % f


def freq_index(freqs, f):
    best = 0
    for i in range(len(freqs)):
        if abs(freqs[i] - f) < abs(freqs[best] - f):
            best = i
    return best


# ---------------------------------------------------------------- figures from the models run

def plain_log_axis(ax, which="y"):
    """Log axis with ticks written as plain numbers (100, 200, 500, 1000) instead of 10^2, plus unlabelled minor ticks."""
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: ("%g" % v)))
    axis.set_minor_formatter(mticker.NullFormatter())


def clock_axis(ax, freqs, label=True):
    """x axis showing the real clocks, every panel labelled."""
    ax.set_xticks(freqs)
    ax.set_xticklabels([freq_label(f) for f in freqs])
    ax.tick_params(axis="x", labelbottom=True)
    if label:
        ax.set_xlabel("GPU clock (MHz)")


def order_models(summary):
    """The models present in the run, lightest first (mean batch latency). Unknown names and colours are filled in."""
    present = list(summary["model"].unique())
    present.sort(key=lambda m: summary[summary["model"] == m]["lat_mean_ms"].mean())
    palette = plt.get_cmap("tab10")
    for i, m in enumerate(present):
        MODEL_NAME.setdefault(m, m)
        MODEL_COLOR.setdefault(m, palette(i % 10))
    return present


def grid_axes(n_models, size):
    """2 columns; 2 rows for up to 4 models (as before), more rows beyond that. Always a 2-D array of axes."""
    rows = 2 if n_models <= 4 else (n_models + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(size[0], size[1] * rows / 2.0))
    return fig, np.atleast_2d(axes)


def hide_extra(axes, n_used):
    """Hide unused panels when a run has fewer than 4 models."""
    for ax in axes.flatten()[n_used:]:
        ax.set_visible(False)


def nice_value(v):
    if abs(v) >= 100:
        return "%.0f" % v
    return "%.3g" % v


def plot_scaling(summary, column, ylabel, title, path, log_y=False, show_idle=False, unit=""):
    """One panel per model, x = GPU clock, one line per batch size.
    Numbers are printed on the b=1 line and the biggest-batch line at every clock; red star = the minimum of that panel."""
    freqs = sorted(summary["freq_mhz"].unique())
    idle_by_freq = []
    for f in freqs:
        rows = summary[np.isclose(summary["freq_mhz"], f)]
        idle_by_freq.append(rows["idle_gap_power_w"].mean())
    fig, axes = grid_axes(len(MODELS), (12, 9))
    cmap = plt.get_cmap("viridis")
    handles = []
    hide_extra(axes, len(MODELS))
    for ax, model in zip(axes.flatten(), MODELS):
        pivot = summary[summary["model"] == model].pivot(index="batch", columns="freq_mhz", values=column)
        batches = list(pivot.index)
        handles = []
        for i in range(len(batches)):
            color = cmap(i / max(1, len(batches) - 1))
            line, = ax.plot(pivot.columns, pivot.loc[batches[i]].values, marker="o", ms=4, color=color, label="batch %d" % batches[i])
            handles.append(line)
        box = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75)
        if log_y:
            for i, offset in ((0, (0, -13)), (len(batches) - 1, (0, 8))):
                for f, v in zip(pivot.columns, pivot.loc[batches[i]].values):
                    ax.annotate(nice_value(v), (f, v), xytext=offset, textcoords="offset points", ha="center", fontsize=7,
                                color="black", zorder=6, bbox=box)
        else:
            for col, f in enumerate(pivot.columns):
                row = int(np.argmin(pivot.values[:, col]))
                ax.annotate("b%d: %s" % (batches[row], nice_value(pivot.values[row, col])), (f, pivot.values[row, col]),
                            xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color="black", zorder=6, bbox=box)
        best_value = pivot.values.min()
        place = np.argwhere(pivot.values == best_value)[0]
        best_batch = batches[place[0]]
        best_freq = pivot.columns[place[1]]
        star, = ax.plot([best_freq], [best_value], marker="*", ms=16, color="red", markeredgecolor="black",
                        linestyle="none", zorder=5, label="minimum")
        if show_idle:
            idle, = ax.plot(freqs, idle_by_freq, "k--", label="idle gap power")
            for f, v in zip(freqs, idle_by_freq):
                ax.annotate("%.2f" % v, (f, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5)
        ax.set_title("%s   (min %s %s at batch %d, %s MHz)" % (MODEL_NAME[model], nice_value(best_value), unit, best_batch, freq_label(best_freq)), fontsize=10)
        ax.set_ylabel(ylabel)
        clock_axis(ax, freqs)
        ax.grid(alpha=0.3, which="both")
        if log_y:
            ax.set_yscale("log")
            plain_log_axis(ax, "y")
        ax.margins(y=0.14)
        ax.set_xlim(min(freqs) - 45, max(freqs) + 45)
    legend_note = ("one line per batch size; numbers = value on batch 1 (below) and batch 16 (above)" if log_y
                   else "one line per batch size; numbers = lowest value at each clock (b = its batch size)")
    handles = handles + [star]
    if show_idle:
        handles = handles + [idle]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               title=legend_note, title_fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_energy_u(summary, title, path):
    """The U shape per batch size: energy per frame divided by that line's own minimum over the clocks.
    1.0 = the best clock for that batch size (marked with a dot and its clock); a flat line = clock does not matter."""
    freqs = sorted(summary["freq_mhz"].unique())
    fig, axes = grid_axes(len(MODELS), (12, 9))
    hide_extra(axes, len(MODELS))
    cmap = plt.get_cmap("viridis")
    handles = []
    for ax, model in zip(axes.flatten(), MODELS):
        pivot = summary[summary["model"] == model].pivot(index="batch", columns="freq_mhz", values="energy_per_frame_j")
        batches = list(pivot.index)
        handles = []
        for i in range(len(batches)):
            color = cmap(i / max(1, len(batches) - 1))
            values = pivot.loc[batches[i]].values
            ratio = values / values.min()
            line, = ax.plot(pivot.columns, ratio, marker="o", ms=3.5, color=color, label="batch %d" % batches[i])
            handles.append(line)
            k = int(np.argmin(values))
            ax.plot([pivot.columns[k]], [1.0], marker="o", ms=9, mfc="none", mec=color, mew=2, zorder=5)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_title("%s   (circle = best clock of each batch size)" % MODEL_NAME[model], fontsize=10)
        ax.set_ylabel("energy per frame / that line's minimum (x)")
        clock_axis(ax, freqs)
        ax.grid(alpha=0.3)
        ax.set_xlim(min(freqs) - 45, max(freqs) + 45)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               title="one line per batch size; every line is scaled so its own minimum is 1", title_fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_heatmaps(summary, column, normalised, title, path):
    """One panel per model, rows = batch size, columns = clock. Red box = best cell."""
    fig, axes = grid_axes(len(MODELS), (11, 8))
    hide_extra(axes, len(MODELS))
    for ax, model in zip(axes.flatten(), MODELS):
        pivot = summary[summary["model"] == model].pivot(index="batch", columns="freq_mhz", values=column)
        values = pivot.values.astype(float)
        shown = values
        if normalised:
            shown = values / values.min()
        image = ax.imshow(shown, aspect="auto", cmap="viridis_r")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("energy relative to best (x)" if normalised else "energy per frame (J)", fontsize=8)
        low = shown.min()
        high = shown.max()
        for row in range(shown.shape[0]):
            for col in range(shown.shape[1]):
                fraction = (shown[row][col] - low) / (high - low) if high > low else 0.0
                text_color = "black" if fraction < 0.5 else "white"
                if normalised:
                    label = "%.2fx" % shown[row][col]
                else:
                    label = "%.3f" % shown[row][col]
                ax.text(col, row, label, ha="center", va="center", fontsize=8, color=text_color)
        place = np.argwhere(shown == low)[0]
        ax.add_patch(plt.Rectangle((place[1] - 0.5, place[0] - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([freq_label(f) for f in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(["%d" % b for b in pivot.index])
        ax.set_title("%s   (best %.3g %s at batch %d, %s MHz)" % (
            MODEL_NAME[model], values.min(), "J/frame", pivot.index[place[0]], freq_label(pivot.columns[place[1]])), fontsize=10)
        ax.set_xlabel("GPU clock (MHz)")
        ax.set_ylabel("batch size (frames)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def pareto_front(points):
    """points: (latency, energy, batch, freq). Keep the ones no other point beats on both (lower is better)."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            no_worse = q[0] <= p[0] and q[1] <= p[1]
            strictly_better = q[0] < p[0] or q[1] < p[1]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda r: r[0])
    return front


def plot_tradeoff(summary, title, path, log_x=True):
    """The menu a deadline policy picks from: batch latency (x) against energy per frame (y), one dot per (batch, clock)."""
    freqs = sorted(summary["freq_mhz"].unique())
    cmap = plt.get_cmap("plasma")
    fig, axes = grid_axes(len(MODELS), (11, 8.5))
    hide_extra(axes, len(MODELS))
    for ax, model in zip(axes.flatten(), MODELS):
        rows = summary[summary["model"] == model]
        points = []
        for _, r in rows.iterrows():
            points.append((r["lat_mean_ms"], r["energy_per_frame_j"], int(r["batch"]), r["freq_mhz"]))
        for p in points:
            index = freq_index(freqs, p[3])
            color = cmap(0.9 * index / max(1, len(freqs) - 1))
            ax.scatter(p[0], p[1], color=color, s=15 + 4 * p[2], edgecolor="none", alpha=0.85)
        front = pareto_front(points)
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="black", linewidth=1)
        for p in front:
            ax.annotate("b%d@%s" % (p[2], freq_label(p[3])), (p[0], p[1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        if log_x:
            ax.set_xscale("log")
            plain_log_axis(ax, "x")
        ax.set_title("%s   (%d of %d points on the front)" % (MODEL_NAME[model], len(front), len(points)), fontsize=10)
        ax.set_xlabel("batch latency (ms, log scale)" if log_x else "batch latency (ms)")
        ax.set_ylabel("energy per frame (J)")
        ax.grid(alpha=0.3, which="both")
    handles = []
    for i in range(len(freqs)):
        handles.append(plt.Line2D([], [], marker="o", linestyle="none", color=cmap(0.9 * i / max(1, len(freqs) - 1)),
                                  label="%s MHz" % freq_label(freqs[i])))
    fig.legend(handles=handles, loc="lower center", ncol=len(freqs), fontsize=8,
               title="dot colour = GPU clock, dot size = batch size", title_fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fit_latency(freqs_mhz, times_ms):
    """t = a * f^-b + c. b by a grid search, a and c by a straight-line fit on x = f^-b. Needs a > 0."""
    f = np.array(freqs_mhz) / 1000.0
    t = np.array(times_ms)
    best = None
    for b in np.arange(0.05, 3.0, 0.01):
        x = f ** (-b)
        a, c = np.polyfit(x, t, 1)
        if a <= 0:
            continue
        error = np.sum((a * x + c - t) ** 2)
        if best is None or error < best[0]:
            best = (error, b, a, c)
    if best is None:
        return None
    error, b, a, c = best
    total = np.sum((t - t.mean()) ** 2)
    r2 = 1.0 - error / total if total > 0 else 1.0
    return b, a, c, r2


def latency_fits(summary):
    """Returns {(model, batch): (b, a, c, r2)} and draws nothing."""
    fits = {}
    for model in MODELS:
        pivot = summary[summary["model"] == model].pivot(index="batch", columns="freq_mhz", values="lat_mean_ms")
        for batch in pivot.index:
            fits[(model, int(batch))] = fit_latency(list(pivot.columns), list(pivot.loc[batch].values))
    return fits


def plot_latency_fit(summary, fits, title, path):
    freqs = sorted(summary["freq_mhz"].unique())
    fig, axes = grid_axes(len(MODELS), (11, 8))
    cmap = plt.get_cmap("viridis")
    first_panel = True
    hide_extra(axes, len(MODELS))
    for ax, model in zip(axes.flatten(), MODELS):
        pivot = summary[summary["model"] == model].pivot(index="batch", columns="freq_mhz", values="lat_mean_ms")
        batches = list(pivot.index)
        smooth = np.linspace(min(freqs), max(freqs), 100)
        for i in range(len(batches)):
            color = cmap(i / max(1, len(batches) - 1))
            ax.plot(pivot.columns, pivot.loc[batches[i]].values, marker="o", ms=4, linestyle="none", color=color, label="batch %d" % batches[i])
            fit = fits[(model, int(batches[i]))]
            if fit is not None:
                b, a, c, r2 = fit
                ax.plot(smooth, a * (smooth / 1000.0) ** (-b) + c, "--", color=color, linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_axis(ax, "y")
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks(freqs)
        ax.set_xticklabels([freq_label(f) for f in freqs])
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("GPU clock (MHz, log scale)")
        ax.set_title(MODEL_NAME[model], fontsize=10)
        ax.set_ylabel("batch latency (ms, log scale)")
        ax.grid(alpha=0.3, which="both")
        if first_panel:
            ax.legend(fontsize=7, ncol=2)
            first_panel = False
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_fit_exponent(fits, batches, title, path):
    """Exponent b of the fit per batch size. b = 1 would be latency exactly proportional to 1/clock (plus a constant)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in MODELS:
        xs = []
        ys = []
        for batch in batches:
            fit = fits[(model, int(batch))]
            if fit is not None:
                xs.append(batch)
                ys.append(fit[0])
        ax.plot(xs, ys, marker="o", color=MODEL_COLOR[model], label=MODEL_NAME[model])
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks([int(b) for b in batches])
    ax.set_xlabel("batch size (frames)")
    ax.set_ylabel("fitted exponent b in  t = a * f^-b + c  (1 = latency ~ 1/clock)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def quality_table(subruns):
    """Per (model, batch, clock): how long the power windows were and how much energy per frame varied between the 4 subruns."""
    rows = []
    for key, g in subruns.groupby(["model", "batch", "freq_mhz"]):
        mean_energy = g["energy_per_frame_j"].mean()
        spread = 100.0 * g["energy_per_frame_j"].std(ddof=0) / mean_energy
        rows.append({"model": key[0], "batch": key[1], "freq_mhz": key[2],
                     "window_s": g["duration_s"].mean(),
                     "n_power_samples_min": g["n_power_samples"].min(),
                     "energy_spread_pct": spread})
    return pd.DataFrame(rows)


def plot_quality(quality, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model in MODELS:
        rows = quality[quality["model"] == model]
        axes[0].scatter(rows["window_s"], rows["energy_spread_pct"], s=18, color=MODEL_COLOR[model], alpha=0.8, label=MODEL_NAME[model])
    axes[0].set_xscale("log")
    axes[0].axhline(5.0, color="black", linestyle=":", linewidth=1)
    axes[0].set_xlabel("power window per subrun (s, log)")
    axes[0].set_ylabel("energy-per-frame spread between subruns (%)")
    axes[0].set_title("short windows are noisier?", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].hist(quality["n_power_samples_min"], bins=30, color="tab:gray")
    axes[1].axvline(10, color="black", linestyle=":", linewidth=1)
    axes[1].set_xlabel("fewest tegrastats samples in a subrun window (per point)")
    axes[1].set_ylabel("number of points")
    axes[1].set_title("power samples per window", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figures from the switch runs

def pair_matrix(allraw, column, freqs):
    matrix = np.full((len(freqs), len(freqs)), np.nan)
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            if i == j:
                continue
            rows = allraw[np.isclose(allraw["from_mhz"], freqs[i]) & np.isclose(allraw["to_mhz"], freqs[j])]
            if len(rows) > 0:
                matrix[i][j] = rows[column].mean()
    return matrix


def plot_switch_matrix(allraw, title, path):
    freqs = sorted(allraw["from_mhz"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    columns = [("write_total_ms", "sysfs write (min_freq and max_freq), ms"),
               ("tegra_switch_ms", "until tegrastats GR3D_FREQ shows it, ms")]
    for ax, (column, label) in zip(axes, columns):
        matrix = pair_matrix(allraw, column, freqs)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis_r")
        for i in range(len(freqs)):
            for j in range(len(freqs)):
                if i != j:
                    ax.text(j, i, "%.0f" % matrix[i][j], ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([freq_label(f) for f in freqs])
        ax.set_yticks(range(len(freqs)))
        ax.set_yticklabels([freq_label(f) for f in freqs])
        ax.set_xlabel("to (MHz)")
        ax.set_ylabel("from (MHz)")
        ax.set_title(label, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_switch_detail(allraw, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    direction_color = {"up": "tab:red", "down": "tab:blue"}
    columns = [("write_total_ms", "sysfs write (ms)"), ("tegra_switch_ms", "tegrastats sees it (ms)")]
    for ax, (column, label) in zip(axes[:2], columns):
        for direction in ["up", "down"]:
            rows = allraw[allraw["direction"] == direction]
            ax.scatter(rows["jump_mhz"], rows[column], s=14, alpha=0.6, color=direction_color[direction], label=direction)
        ax.set_xlabel("size of the clock jump (MHz)")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="direction", title_fontsize=7)
        ax.set_title(label + " vs jump size", fontsize=10)
    runs = sorted(allraw["run"].unique())
    width = 0.25
    metrics = [("write_total_ms", "write"), ("cur_switch_ms", "cur_freq"), ("tegra_switch_ms", "tegrastats")]
    for k in range(len(metrics)):
        means = []
        for run in runs:
            means.append(allraw[allraw["run"] == run][metrics[k][0]].mean())
        positions = []
        for i in range(len(runs)):
            positions.append(i + (k - 1) * width)
        axes[2].bar(positions, means, width=width, label=metrics[k][1])
    axes[2].set_xticks(range(len(runs)))
    axes[2].set_xticklabels([r.replace("switch_", "") for r in runs], rotation=15, fontsize=7)
    axes[2].set_xlabel("switch run (date_time)")
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt="%.0f", fontsize=7, padding=1)
    axes[2].set_ylabel("mean switch time (ms)")
    axes[2].set_title("same number in every run?", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_switch_in_batches(summary, switch_ms, title, path):
    """One clock switch divided by the batch latency at each clock: above 1 means switching costs more than a batch."""
    freqs = sorted(summary["freq_mhz"].unique())
    batches = sorted(summary["batch"].unique())
    chosen = [batches[0], batches[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    width = 0.8 / max(len(MODELS), 1)
    for ax, batch in zip(axes, chosen):
        for k in range(len(MODELS)):
            ratios = []
            for f in freqs:
                rows = summary[(summary["model"] == MODELS[k]) & (summary["batch"] == batch) & np.isclose(summary["freq_mhz"], f)]
                ratios.append(switch_ms / rows.iloc[0]["lat_mean_ms"])
            positions = []
            for i in range(len(freqs)):
                positions.append(i + (k - (len(MODELS) - 1) / 2.0) * width)
            bars = ax.bar(positions, ratios, width=width, color=MODEL_COLOR[MODELS[k]], label=MODEL_NAME[MODELS[k]])
            ax.bar_label(bars, fmt="%.2g", fontsize=6, padding=1, rotation=90)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylim(0.015, 12)
        plain_log_axis(ax, "y")
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([freq_label(f) for f in freqs])
        ax.set_xlabel("GPU clock (MHz)")
        ax.set_title("batch size %d" % batch, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("switch time / batch latency (x, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), fontsize=8,
               title="dotted line = 1: above it, one clock switch costs more than a whole batch", title_fontsize=8)
    fig.suptitle(title % switch_ms, fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- tables and text

def best_clock_table(summary):
    """For every (model, batch): the clock with the lowest energy per frame, and what it saves against the top clock.
    Energy per batch (J) = energy per frame x batch size = the total energy of one batch (whole board, active time only)."""
    top = summary["freq_mhz"].max()
    rows = []
    for model in MODELS:
        for batch in sorted(summary["batch"].unique()):
            part = summary[(summary["model"] == model) & (summary["batch"] == batch)]
            best = part.loc[part["energy_per_frame_j"].idxmin()]
            at_top = part[np.isclose(part["freq_mhz"], top)].iloc[0]
            saving = 100.0 * (1.0 - best["energy_per_frame_j"] / at_top["energy_per_frame_j"])
            penalty = 100.0 * (best["lat_mean_ms"] / at_top["lat_mean_ms"] - 1.0)
            rows.append({"model": MODEL_NAME[model], "batch": int(batch), "best_clock_mhz": best["freq_mhz"],
                         "energy_per_frame_j": best["energy_per_frame_j"], "energy_per_frame_at_top_j": at_top["energy_per_frame_j"],
                         "energy_per_batch_j": best["energy_per_frame_j"] * batch,
                         "energy_per_batch_at_top_j": at_top["energy_per_frame_j"] * batch,
                         "energy_saving_pct": saving, "latency_ms": best["lat_mean_ms"],
                         "latency_at_top_ms": at_top["lat_mean_ms"], "latency_penalty_pct": penalty})
    return pd.DataFrame(rows)


def write_energy_tables(summary, out_dir):
    """Raw energy numbers, no plotting: one row per (model, batch), one column per GPU clock.
    energy_per_batch_by_clock.csv (J per batch), energy_per_frame_by_clock.csv (J per frame), energy_raw_long.csv (one row per point)."""
    summary = summary.copy()
    summary["model_name"] = summary["model"].map(MODEL_NAME)
    summary["clock_mhz"] = summary["freq_mhz"].round(2)
    order = {m: i for i, m in enumerate(MODELS)}
    outputs = {}
    for column, name in (("energy_per_batch_j", "energy_per_batch_by_clock.csv"), ("energy_per_frame_j", "energy_per_frame_by_clock.csv")):
        wide = summary.pivot_table(index=["model", "batch"], columns="clock_mhz", values=column)
        wide = wide.reset_index()
        wide["_o"] = wide["model"].map(order)
        wide = wide.sort_values(["_o", "batch"]).drop(columns="_o")
        wide["model"] = wide["model"].map(MODEL_NAME)
        wide.columns = [c if isinstance(c, str) else "%s MHz" % freq_label(c) for c in wide.columns]
        wide.to_csv(os.path.join(out_dir, name), index=False, float_format="%.4f")
        outputs[column] = wide
    keep = ["model_name", "batch", "clock_mhz", "n_subruns", "n_batches", "lat_mean_ms", "avg_power_w", "idle_gap_power_w",
            "energy_per_batch_j", "energy_per_frame_j"]
    keep = [c for c in keep if c in summary.columns]
    long = summary.assign(_o=summary["model"].map(order)).sort_values(["_o", "batch", "clock_mhz"])[keep]
    long.to_csv(os.path.join(out_dir, "energy_raw_long.csv"), index=False, float_format="%.4f")
    return outputs


def write_summary(path, models_dir, switch_dirs, summary, meta, best, fits, quality, allraw):
    lines = []
    lines.append("DVFS analysis")
    lines.append("models run : " + models_dir)
    lines.append("switch runs: " + ", ".join([os.path.basename(d) for d in switch_dirs]))
    lines.append("nvpmodel   : %s (mode id %s), clock cap %s Hz" % (meta.get("nvpmodel"), meta.get("mode_id"), meta.get("cap_hz")))
    lines.append("")
    lines.append("points: %d, clock_held_all==1 on %d, dmesg hits total %d, max temperature %.1f C" % (
        len(summary), int((summary["clock_held_all"] == 1).sum()), int(summary["dmesg_hits"].sum()), summary["t_max_c"].max()))
    lines.append("")
    lines.append("Best clock for energy per frame, against the top clock (%.2f MHz):" % summary["freq_mhz"].max())
    lines.append("%-8s %5s %8s %10s %10s %9s" % ("model", "batch", "best MHz", "saving %", "latency %", "J/frame"))
    for _, r in best.iterrows():
        lines.append("%-8s %5d %8.2f %10.1f %10.1f %9.3f" % (r["model"], r["batch"], r["best_clock_mhz"], r["energy_saving_pct"],
                                                              r["latency_penalty_pct"], r["energy_per_frame_j"]))
    lines.append("")
    lines.append("Median over batch sizes, per model: saving at the best clock and the latency it costs")
    for model in MODELS:
        part = best[best["model"] == MODEL_NAME[model]]
        lines.append("  %s: saving %.1f %%, latency +%.1f %%" % (MODEL_NAME[model], part["energy_saving_pct"].median(), part["latency_penalty_pct"].median()))
    lines.append("")
    lines.append("Latency fit t = a * f^-b + c (f in GHz): exponent b per model, mean over batch sizes; c < 0 means the fit is not physical")
    for model in MODELS:
        exponents = []
        negative_c = 0
        for batch in sorted(summary["batch"].unique()):
            fit = fits[(model, int(batch))]
            if fit is not None:
                exponents.append(fit[0])
                if fit[2] < 0:
                    negative_c += 1
        lines.append("  %s: b = %.2f (range %.2f - %.2f), c < 0 in %d of %d fits" % (
            MODEL_NAME[model], float(np.mean(exponents)), min(exponents), max(exponents), negative_c, len(exponents)))
    lines.append("")
    flagged = quality[(quality["window_s"] < 2.0) | (quality["n_power_samples_min"] < 10) | (quality["energy_spread_pct"] > 5.0)]
    lines.append("Data-quality flags (window < 2 s, or < 10 power samples in some subrun, or energy spread > 5 %%): %d of %d points" % (len(flagged), len(quality)))
    for _, r in flagged.sort_values("energy_spread_pct", ascending=False).head(15).iterrows():
        lines.append("  %s b=%-2d %7.2f MHz  window %.2f s  min samples %3d  spread %.1f %%" % (
            MODEL_NAME[r["model"]], r["batch"], r["freq_mhz"], r["window_s"], r["n_power_samples_min"], r["energy_spread_pct"]))
    lines.append("")
    lines.append("Switching, over %d switches in %d runs (ms):" % (len(allraw), len(switch_dirs)))
    for direction in ["up", "down"]:
        rows = allraw[allraw["direction"] == direction]
        lines.append("  %-4s write %.1f   cur_freq %.1f   tegrastats %.1f" % (
            direction, rows["write_total_ms"].mean(), rows["cur_switch_ms"].mean(), rows["tegra_switch_ms"].mean()))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--switch-dirs", nargs="*", default=None)
    args = parser.parse_args()

    models_dir = args.models_dir
    if models_dir is None:
        models_dir = newest_models_dir(args.results_dir)
    switch_dirs = args.switch_dirs
    if switch_dirs is None:
        switch_dirs = switch_dirs_with_raw(args.results_dir)

    out_dir = os.path.join(OUT_ROOT, os.path.basename(models_dir.rstrip("/\\")))
    os.makedirs(out_dir, exist_ok=True)

    summary, subruns, meta = load_models(models_dir)
    global MODELS
    MODELS = order_models(summary)
    summary["energy_per_batch_j"] = summary["energy_per_frame_j"] * summary["batch"]
    mode = "nvpmodel mode %s" % meta.get("mode_id")
    run_name = os.path.basename(models_dir.rstrip("/\\"))
    tag = "%s, %s, %d points" % (run_name, mode, len(summary))
    batches = sorted(summary["batch"].unique())

    plot_scaling(summary, "lat_mean_ms", "batch latency (ms, log scale)", "Batch latency vs GPU clock  [%s]" % tag,
                 os.path.join(out_dir, "01_latency_vs_clock.png"), log_y=True, unit="ms")
    plot_scaling(summary, "lat_mean_ms", "batch latency (ms)", "Batch latency vs GPU clock, linear axis  [%s]" % tag,
                 os.path.join(out_dir, "01b_latency_vs_clock_linear.png"), log_y=False, unit="ms")
    plot_scaling(summary, "energy_per_frame_j", "energy per frame (J)", "Energy per frame vs GPU clock  [%s]" % tag,
                 os.path.join(out_dir, "02_energy_per_frame_vs_clock.png"), unit="J")
    plot_scaling(summary, "energy_per_batch_j", "energy per batch (J)", "Energy per batch vs GPU clock (fixed work = one batch)  [%s]" % tag,
                 os.path.join(out_dir, "02b_energy_per_batch_vs_clock.png"), log_y=False, unit="J")
    plot_energy_u(summary, "U shape of energy per frame, each batch size scaled to its own minimum  [%s]" % tag,
                  os.path.join(out_dir, "02c_energy_u_shape.png"))
    plot_scaling(summary, "avg_power_w", "average power while running (W)", "Power vs GPU clock  [%s]" % tag,
                 os.path.join(out_dir, "03_power_vs_clock.png"), show_idle=True, unit="W")
    plot_heatmaps(summary, "energy_per_frame_j", False, "Energy per frame (J), red box = best clock+batch per model  [%s]" % tag,
                  os.path.join(out_dir, "04_heatmap_energy.png"))
    plot_heatmaps(summary, "energy_per_frame_j", True, "Energy per frame relative to the model's best  [%s]" % tag,
                  os.path.join(out_dir, "05_heatmap_energy_normalised.png"))
    plot_tradeoff(summary, "Batch latency vs energy per frame, black = Pareto front  [%s]" % tag,
                  os.path.join(out_dir, "06_tradeoff_pareto.png"))
    plot_tradeoff(summary, "Latency vs energy per frame, linear axis, black = Pareto  [%s]" % tag,
                  os.path.join(out_dir, "06b_tradeoff_pareto_linear.png"), log_x=False)
    fits = latency_fits(summary)
    plot_latency_fit(summary, fits, "Batch latency vs clock with t = a * f^-b + c fits (dashed)  [%s]" % tag,
                     os.path.join(out_dir, "07_latency_fit.png"))
    plot_fit_exponent(fits, batches, "Fitted exponent b per batch size (dotted line: b = 1)",
                      os.path.join(out_dir, "08_fit_exponent.png"))
    quality = quality_table(subruns)
    plot_quality(quality, "Measurement quality of the %d points  [%s]" % (len(summary), run_name), os.path.join(out_dir, "09_data_quality.png"))

    allraw = None
    if len(switch_dirs) > 0:
        allraw = load_switch(switch_dirs)
        plot_switch_matrix(allraw, "Clock switch time per (from, to) pair, mean over %d runs" % len(switch_dirs),
                           os.path.join(out_dir, "10_switch_matrix.png"))
        plot_switch_detail(allraw, "Clock switch time: jump size, direction and run-to-run", os.path.join(out_dir, "11_switch_detail.png"))
        plot_switch_in_batches(summary, allraw["cur_switch_ms"].mean(), "One clock switch (%.0f ms) in batches, by model and clock",
                               os.path.join(out_dir, "12_switch_cost_in_batches.png"))

    best = best_clock_table(summary)
    best.to_csv(os.path.join(out_dir, "best_clock_table.csv"), index=False)
    write_energy_tables(summary, out_dir)
    if allraw is not None:
        lines = write_summary(os.path.join(out_dir, "summary.txt"), models_dir, switch_dirs, summary, meta, best, fits, quality, allraw)
        print("\n".join(lines))
    print("\nplots written to " + out_dir)


if __name__ == "__main__":
    main()

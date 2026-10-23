"""
Routing analysis from results/<run>/raw/ alone (numpy, pandas, matplotlib; no
torch), so it runs on a laptop after copying raw/ back.

Plots are for one main operating point (--point, default: the Pareto point
with the highest final_val alpha_sys, ties to the cheaper one). summary.csv
covers every point.

Run: python analyze.py --run results/s01 [--point ID]
Writes results/<run>/analysis/: summary.csv, summary.md, point_<id>/ (plots + CSVs).
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import metrics  # noqa: E402

# n, s, m, l: dataviz categorical slots 1-4 (checked for colour-vision
# deficiency); rows and legends always carry the model name too.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DRIFT_WINDOW_S = 10
RATE_WINDOW_S = 1
SHARE_WINDOW_S = 10
KILL_SHARE = 0.90


# ---------------------------------------------------------------- loading

def load_run(run_dir):
    raw = os.path.join(run_dir, "raw")
    with open(os.path.join(raw, "manifest.json")) as f:
        manifest = json.load(f)
    points = pd.read_csv(os.path.join(raw, "pareto_points.csv"))
    files = sorted(glob.glob(os.path.join(raw, "routing_*.csv.gz")))
    if not files:
        raise SystemExit(f"no routing files in {raw}")
    frames = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frames = frames.sort_values(["scenario", "camera", "frame_idx"]).reset_index(drop=True)
    return manifest, points, frames


def pick_point(points, requested):
    if requested is not None:
        if requested not in set(points["point_id"]):
            raise SystemExit(f"point {requested} not in pareto_points.csv")
        return requested, "chosen with --point"
    if points["final_val_alpha_sys"].notna().any():
        best = points.sort_values(["final_val_alpha_sys", "final_val_cost_gflops"],
                                  ascending=[False, True]).iloc[0]
        return int(best["point_id"]), "highest final_val alpha_sys"
    return int(points["point_id"].iloc[0]), "first point (no final_val numbers found)"


def routing_for_point(frames, point_id, model_names):
    """The identity/time columns plus `model` (0..3) for one point."""
    df = frames[["scenario", "camera", "frame_idx", "fps", "video_time_s", "synced_time_s"]].copy()
    df["model"] = frames[f"model_p{point_id}"].map({n: i for i, n in enumerate(model_names)})
    return df


def cameras_of(df):
    """(scenario, camera, rows in frame order) per camera."""
    return df.groupby(["scenario", "camera"], sort=True)


# ---------------------------------------------------------------- tables

def run_length_table(df, names):
    """Every run, plus mean/median per (scenario, camera, model); camera
    "ALL" pools a scenario's cameras."""
    records = []
    for (scenario, camera), cam in cameras_of(df):
        fps = cam["fps"].iloc[0]
        for model, length in metrics.run_lengths(cam["model"]):
            records.append({"scenario": scenario, "camera": camera, "model": names[model],
                            "frames": length, "seconds": length / fps})
    runs = pd.DataFrame(records)
    stats = (pd.concat([runs, runs.assign(camera="ALL")])
             .groupby(["scenario", "camera", "model"])
             .agg(n_runs=("frames", "size"), mean_frames=("frames", "mean"),
                  median_frames=("frames", "median"), mean_s=("seconds", "mean"),
                  median_s=("seconds", "median"))
             .round(3).reset_index())
    return runs, stats


def scenario_rows(df, names, point_id):
    """summary.csv rows (one per scenario) for one point."""
    k = len(names)
    letters = [n[-1] for n in names]
    rows = []
    for scenario, scen in df.groupby("scenario", sort=True):
        shares = metrics.split(scen["model"], k)
        n_switches, minutes = 0, 0.0
        runs = {m: [] for m in range(k)}
        for _, cam in cameras_of(scen):
            n_switches += metrics.switches(cam["model"])
            minutes += len(cam) / cam["fps"].iloc[0] / 60
            for model, length in metrics.run_lengths(cam["model"]):
                runs[model].append(length)
        merged = metrics.merge_synced(scen)

        row = {"point_id": point_id, "scenario": scenario, "n_cameras": scen["camera"].nunique(),
               "n_frames": len(scen), "camera_minutes": round(minutes, 2)}
        for m, letter in enumerate(letters):
            row[f"share_{letter}"] = round(shares[m], 4)
        for m, letter in enumerate(letters):
            row[f"mean_run_frames_{letter}"] = round(np.mean(runs[m]), 2) if runs[m] else np.nan
        row["switches_per_min"] = round(n_switches / minutes, 3)
        row["max_10s_share"] = round(metrics.max_window_share(
            merged["synced_time_s"], merged["model"], SHARE_WINDOW_S, k), 4)
        row["heaviest_model"] = names[int(np.argmax(shares))]
        row["heaviest_share"] = round(float(shares.max()), 4)
        row["above_90_10"] = bool(shares.max() > KILL_SHARE)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- plots

def draw_strip(ax, models, x0, x1, names):
    """4 rows (one per model), one column per frame, coloured where the frame
    was routed: long runs read as solid bands, switches as breaks."""
    k = len(names)
    models = np.asarray(models)
    image = np.full((k, len(models)), np.nan)
    image[models, np.arange(len(models))] = models
    ax.imshow(image, cmap=ListedColormap(COLORS[:k]), vmin=-0.5, vmax=k - 0.5,
              aspect="auto", interpolation="nearest", extent=(x0, x1, k - 0.5, -0.5))
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_camera_timeline(cam, names, title, path):
    n = len(cam)
    fig, ax = plt.subplots(figsize=(np.clip(n / 150, 8, 40), 2.4))
    frames = cam["frame_idx"].to_numpy()
    draw_strip(ax, cam["model"], frames[0] - 0.5, frames[-1] + 0.5, names)
    ax.set_xlabel("frame")
    ax.set_title(title, loc="left")
    save(fig, path)


def plot_scenario_timeline(scen, names, title, path):
    groups = list(cameras_of(scen))
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 0.9 + 1.0 * len(groups)),
                             sharex=True, squeeze=False)
    for ax, ((_, camera), cam) in zip(axes[:, 0], groups):
        t = cam["synced_time_s"].to_numpy()
        half = 0.5 / cam["fps"].iloc[0]
        draw_strip(ax, cam["model"], t[0] - half, t[-1] + half, [n[-1] for n in names])
        ax.set_ylabel(camera, rotation=0, ha="right", va="center")
    # each imshow resets the shared x limits to its own extent; cover every camera
    axes[0, 0].set_xlim(scen["synced_time_s"].min() - 0.5, scen["synced_time_s"].max() + 0.5)
    axes[-1, 0].set_xlabel("synced time (s)")
    axes[0, 0].set_title(title, loc="left")
    save(fig, path)


def plot_run_lengths(runs, names, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, unit in zip(axes, ["frames", "seconds"]):
        values = runs[unit]
        bins = np.unique(np.geomspace(values.min(), values.max() * 1.01, 30))
        for color, name in zip(COLORS, names):
            subset = runs.loc[runs["model"] == name, unit]
            if len(subset):
                ax.hist(subset, bins=bins, histtype="step", linewidth=2, color=color,
                        label=f"{name} ({len(subset)} runs)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"run length ({unit})")
        ax.set_ylabel("runs")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left")
    save(fig, path)


def plot_split(df, names, title, path):
    k = len(names)
    labels, shares = [], []
    for scenario, scen in df.groupby("scenario", sort=True):
        for (_, camera), cam in cameras_of(scen):
            labels.append(f"{scenario}/{camera}")
            shares.append(metrics.split(cam["model"], k))
        labels.append(f"{scenario} (all)")
        shares.append(metrics.split(scen["model"], k))
    shares = np.array(shares)

    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.35 * len(labels)))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for m, name in enumerate(names):
        ax.barh(y, shares[:, m], left=left, color=COLORS[m], edgecolor="white", linewidth=1.5, label=name)
        for yi, l0, s in zip(y, left, shares[:, m]):
            if s >= 0.08:
                ax.text(l0 + s / 2, yi, f"{100 * s:.0f}%", ha="center", va="center", fontsize=8, color="black")
        left += shares[:, m]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of frames")
    ax.legend(ncol=k, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(title, loc="left", pad=28)
    save(fig, path)


def plot_drift(scen, names, title, path):
    k = len(names)
    groups = list(cameras_of(scen))
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 0.9 + 1.6 * len(groups)), squeeze=False)
    for ax, ((_, camera), cam) in zip(axes[:, 0], groups):
        counts = metrics.window_counts(cam["video_time_s"], cam["model"], DRIFT_WINDOW_S, k, start=0.0)
        shares = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
        # step="post" draws each value up to the next x, so repeat the last
        # window once at its end time or it would not show
        x = np.append(shares.index, shares.index[-1] + DRIFT_WINDOW_S)
        y = np.vstack([shares.to_numpy(), shares.to_numpy()[-1:]]).T
        ax.stackplot(x, y, colors=COLORS[:k], labels=names, step="post")
        ax.set_ylim(0, 1)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel(camera, rotation=0, ha="right", va="center")
    axes[-1, 0].set_xlabel(f"video time (s), {DRIFT_WINDOW_S} s windows")
    axes[0, 0].legend(ncol=k, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[0, 0].set_title(title, loc="left", pad=24)
    save(fig, path)


def plot_transitions(scen, names, title, path):
    k = len(names)
    counts = sum(metrics.transition_counts(cam["model"], k) for _, cam in cameras_of(scen))
    probs = metrics.row_normalize(counts)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(probs, cmap="Blues", vmin=0, vmax=1)
    for a in range(k):
        for b in range(k):
            ax.text(b, a, f"{probs[a, b]:.2f}\n({counts[a, b]})", ha="center", va="center", fontsize=8,
                    color="white" if probs[a, b] > 0.6 else "black")
    ax.set_xticks(range(k))
    ax.set_xticklabels(names)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    ax.set_xlabel("next frame")
    ax.set_ylabel("current frame")
    ax.set_title(title, loc="left")
    save(fig, path)
    return probs


def arrival_rates(scen, names):
    """Merged stream of a scenario's cameras: frames per model queue per
    RATE_WINDOW_S window of synced time (what a shared box's scheduler sees)."""
    merged = metrics.merge_synced(scen)
    counts = metrics.window_counts(merged["synced_time_s"], merged["model"], RATE_WINDOW_S, len(names))
    rates = counts / RATE_WINDOW_S
    rates.columns = names
    rates["total"] = rates.sum(axis=1)
    rates.index.name = "window_start_synced_s"
    return rates


def plot_arrival_rates(rates, names, title, path):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for color, name in zip(COLORS, names):
        ax.plot(rates.index, rates[name], color=color, linewidth=1.5, label=name)
    ax.set_xlabel("synced time (s)")
    ax.set_ylabel(f"frames/s ({RATE_WINDOW_S} s windows)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=len(names), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(title, loc="left", pad=24)
    save(fig, path)


# ---------------------------------------------------------------- main

def fmt_run(value):
    return "-" if pd.isna(value) else f"{value:.1f}"


def write_summary_md(path, run_name, point_id, reason, points, main_rows, df, names):
    point = points.set_index("point_id").loc[point_id]
    letters = [n[-1] for n in names]
    lines = [f"# Routing summary: run `{run_name}`", "",
             f"Main point: **{point_id}** ({reason}): final_val alpha_sys "
             f"{point['final_val_alpha_sys']:.4f}, cost {point['final_val_cost_gflops']:.2f} GFLOPs.",
             "Dispatcher on every frame (D = 1). Routing behaviour only; no correctness on CityFlow.", "",
             "| scenario | cams | frames | " + " | ".join(f"share {x}" for x in letters)
             + " | " + " | ".join(f"mean run {x} (frames)" for x in letters)
             + " | switches/min | max 10 s share |",
             "|" + "---|" * (5 + 2 * len(letters))]
    for r in main_rows:
        lines.append(f"| {r['scenario']} | {r['n_cameras']} | {r['n_frames']} | "
                     + " | ".join(f"{100 * r[f'share_{x}']:.1f}%" for x in letters) + " | "
                     + " | ".join(fmt_run(r[f"mean_run_frames_{x}"]) for x in letters) + " | "
                     + f"{r['switches_per_min']:.2f} | {100 * r['max_10s_share']:.1f}% |")
    lines += ["", "Kill criterion (heaviest-loaded model above 90:10):"]
    for r in main_rows:
        lines.append(f"- {r['scenario']}: heaviest {r['heaviest_model']} at {100 * r['heaviest_share']:.1f}% "
                     f"of frames; above 90:10: {'yes' if r['above_90_10'] else 'no'}")
    overall = metrics.split(df["model"], len(names))
    lines.append(f"- all scenarios: heaviest {names[int(np.argmax(overall))]} at {100 * overall.max():.1f}% "
                 f"of frames; above 90:10: {'yes' if overall.max() > KILL_SHARE else 'no'}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="results/<run_name>")
    parser.add_argument("--point", type=int, default=None)
    args = parser.parse_args(argv)

    manifest, points, frames = load_run(args.run)
    names = manifest["model_names"]
    point_id, reason = pick_point(points, args.point)
    print(f"Main point: {point_id} ({reason})")

    out = os.path.join(args.run, "analysis")
    point_dir = os.path.join(out, f"point_{point_id}")
    os.makedirs(point_dir, exist_ok=True)

    summary = []
    for pid in points["point_id"]:
        summary += scenario_rows(routing_for_point(frames, pid, names), names, int(pid))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    df = routing_for_point(frames, point_id, names)
    tag = f"point {point_id}"
    for (scenario, camera), cam in cameras_of(df):
        plot_camera_timeline(cam, names, f"{scenario}/{camera}: routing per frame ({tag})",
                             os.path.join(point_dir, f"timeline_{scenario}_{camera}.png"))
    runs, run_stats = run_length_table(df, names)
    run_stats.to_csv(os.path.join(point_dir, "run_lengths.csv"), index=False)
    plot_run_lengths(runs, names, f"Run lengths, all cameras ({tag})", os.path.join(point_dir, "run_lengths.png"))
    plot_split(df, names, f"Routing split ({tag})", os.path.join(point_dir, "split.png"))

    for scenario, scen in df.groupby("scenario", sort=True):
        plot_scenario_timeline(scen, names, f"{scenario}: routing per camera on synced time ({tag})",
                               os.path.join(point_dir, f"timeline_{scenario}.png"))
        plot_drift(scen, names, f"{scenario}: split over time per camera ({tag})",
                   os.path.join(point_dir, f"drift_{scenario}.png"))
        probs = plot_transitions(scen, names, f"{scenario}: P(next | current) ({tag})",
                                 os.path.join(point_dir, f"transitions_{scenario}.png"))
        pd.DataFrame(probs, index=names, columns=names).to_csv(
            os.path.join(point_dir, f"transitions_{scenario}.csv"))
        rates = arrival_rates(scen, names)
        rates.to_csv(os.path.join(point_dir, f"arrival_rates_{scenario}.csv"))
        plot_arrival_rates(rates, names, f"{scenario}: arrivals per model queue, cameras merged ({tag})",
                           os.path.join(point_dir, f"arrival_rates_{scenario}.png"))

    main_rows = summary[summary["point_id"] == point_id].to_dict("records")
    write_summary_md(os.path.join(out, "summary.md"), manifest["run_name"], point_id, reason,
                     points, main_rows, df, names)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

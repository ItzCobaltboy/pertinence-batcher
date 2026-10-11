# Cuts the one long power trace into phases and turns it into numbers and plots.
# Only needs numpy and matplotlib (no torch), so it also runs on a laptop.
#
# Inputs (in OUT_DIR):  samples.csv, baseline.json, combos/<model>_bs<b>.json
# Outputs (in OUT_DIR): summary.csv          one row per (model, batch)
#                       phases.csv           one row per run phase
#                       plots/combos/*.png   raw power trace of each combo
#                       plots/models/*.png   power and energy vs batch size, one per model

import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt


def load_samples(path):
    times = []
    watts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            watts.append(float(row["power_w"]))
    return np.array(times), np.array(watts)


def mean_power(times, watts, start, end):
    # average of the samples between start and end (NaN if there are none)
    inside = (times >= start) & (times <= end)
    if inside.sum() == 0:
        return float("nan")
    return float(watts[inside].mean())


def batch_of(row):
    return row["batch"]


def model_and_batch(row):
    return (row["model"], row["batch"])


def get_mark(combo, label):
    for name, t in combo["marks"]:
        if name == label:
            return t
    return None


def summarize_combo(combo, times, watts):
    # one row for the combo, plus one row per run phase
    idle_w = mean_power(times, watts, get_mark(combo, "idle_start"), get_mark(combo, "idle_end"))
    phase_rows = []
    for phase in combo["phases"]:
        power_w = mean_power(times, watts, phase["start"], phase["end"])
        energy_j = power_w * phase["runtime_s"]          # energy = average power x time
        phase_rows.append({
            "model": combo["model"],
            "batch": combo["batch"],
            "phase": phase["phase"],
            "runtime_s": phase["runtime_s"],
            "forwards": phase["forwards"],
            "power_w": power_w,
            "energy_j": energy_j,
            "samples": int(((times >= phase["start"]) & (times <= phase["end"])).sum()),
        })

    # average over the run phases
    power_w = float(np.nanmean([r["power_w"] for r in phase_rows]))
    runtime_s = float(np.mean([r["runtime_s"] for r in phase_rows]))
    forwards = float(np.mean([r["forwards"] for r in phase_rows]))
    energy_j = power_w * runtime_s
    energy_per_batch_j = energy_j / forwards
    row = {
        "model": combo["model"],
        "batch": combo["batch"],
        "load_s": combo["load_s"],
        "idle_power_w": idle_w,
        "run_power_w": power_w,
        "runtime_s": runtime_s,
        "forwards": forwards,
        "ms_per_batch": runtime_s / forwards * 1000.0,
        "energy_j": energy_j,
        "energy_per_batch_j": energy_per_batch_j,
        "energy_per_frame_j": energy_per_batch_j / combo["batch"],
    }
    return row, phase_rows


def plot_combo(combo, row, times, watts, baseline_w, path):
    # the raw power trace of one combo: idle phase, then the run phases
    t0 = get_mark(combo, "idle_start")
    t_end = combo["phases"][-1]["end"]
    inside = (times >= t0 - 2) & (times <= t_end + 2)

    plt.figure(figsize=(10, 4))
    plt.plot(times[inside] - t0, watts[inside], color="tab:blue", linewidth=1)
    plt.axvspan(0, get_mark(combo, "idle_end") - t0, color="lightgrey", alpha=0.5, label="idle (model loaded)")
    for phase in combo["phases"]:
        if phase["phase"] == 0:
            label = "run phases"
        else:
            label = None
        plt.axvspan(phase["start"] - t0, phase["end"] - t0, color="orange", alpha=0.3, label=label)
    plt.axhline(baseline_w, color="black", linestyle="--", linewidth=1, label=f"baseline {baseline_w:.0f} W")
    plt.xlabel("seconds since idle start")
    plt.ylabel("power, W")
    plt.title(f"{combo['model']} batch {combo['batch']}: run {row['run_power_w']:.0f} W, "
              f"idle {row['idle_power_w']:.0f} W, {row['ms_per_batch']:.1f} ms/batch")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_model(model, rows, baseline_w, path):
    # how power and energy scale with batch size, for one model
    rows = sorted(rows, key=batch_of)
    batches = [r["batch"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(batches, [r["run_power_w"] for r in rows], marker="o", label="running")
    axes[0].plot(batches, [r["idle_power_w"] for r in rows], marker="o", label="idle, model loaded")
    axes[0].axhline(baseline_w, color="black", linestyle="--", label="baseline, nothing loaded")
    axes[0].set_ylabel("average power, W")
    axes[0].set_title("Power")
    axes[0].legend(fontsize=8)

    axes[1].plot(batches, [r["energy_per_batch_j"] for r in rows], marker="o")
    axes[1].set_ylabel("energy per batch, J")
    axes[1].set_title("Energy per batch (power x time per batch)")

    axes[2].plot(batches, [r["energy_per_frame_j"] for r in rows], marker="o")
    axes[2].set_ylabel("energy per frame, J")
    axes[2].set_title("Energy per frame (energy per batch / batch size)")

    for ax in axes:
        ax.set_xlabel("batch size")
        ax.grid(alpha=0.3)
    fig.suptitle(model)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(out_dir):
    times, watts = load_samples(os.path.join(out_dir, "samples.csv"))
    with open(os.path.join(out_dir, "baseline.json")) as f:
        baseline = json.load(f)
    baseline_w = mean_power(times, watts, baseline["start"], baseline["end"])
    print(f"baseline (nothing loaded): {baseline_w:.1f} W")

    combo_plot_dir = os.path.join(out_dir, "plots", "combos")
    model_plot_dir = os.path.join(out_dir, "plots", "models")
    os.makedirs(combo_plot_dir, exist_ok=True)
    os.makedirs(model_plot_dir, exist_ok=True)

    summary_rows = []
    all_phase_rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "combos", "*.json"))):
        with open(path) as f:
            combo = json.load(f)
        row, phase_rows = summarize_combo(combo, times, watts)
        row["baseline_power_w"] = baseline_w
        summary_rows.append(row)
        all_phase_rows.extend(phase_rows)
        name = os.path.splitext(os.path.basename(path))[0]
        plot_combo(combo, row, times, watts, baseline_w, os.path.join(combo_plot_dir, name + ".png"))

    summary_rows.sort(key=model_and_batch)
    write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(out_dir, "phases.csv"), all_phase_rows)

    # one plot per model
    models = []
    for row in summary_rows:
        if row["model"] not in models:
            models.append(row["model"])
    for model in models:
        rows = [r for r in summary_rows if r["model"] == model]
        stem = os.path.splitext(model)[0]
        plot_model(model, rows, baseline_w, os.path.join(model_plot_dir, stem + ".png"))

    print(f"{len(summary_rows)} combos -> {out_dir}/summary.csv, phases.csv, plots/")

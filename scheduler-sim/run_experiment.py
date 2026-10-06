"""Runs the load sweeps: for every experiment, policy, load and seed, build a
fresh simulation, run it, save its raw data, and turn that into one row of
metrics. Then writes results/<experiment>/results.csv and 5 plots per
experiment. With SIMULATE = False it skips simulating and recomputes
everything from the saved raw data.

Run from anywhere:  python run_experiment.py
The two knobs are at the top: POLICIES (which schedulers) and each
experiment's workload_class / workload_kwargs (how work arrives).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")  # write PNG files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from sim import Profile, Queue, Simulator, compute_metrics, load_raw_data, save_raw_data
from schedulers import (FCFSBatchScheduler, FCFSNoBatchScheduler, LongestQueueScheduler,
                        TimeoutBatchScheduler)
from workloads import PeriodicRoutedWorkload, StickyWorkload, UniformWorkload, WeightedWorkload

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# True: run every simulation and save its raw data to results/<experiment>/raw/.
# False: skip simulating, load the saved raw data and only recompute metrics and plots
# (use this after adding or changing a metric in compute_metrics).
SIMULATE = True

SEEDS = [1, 2, 3]
HORIZON_MS = 60000.0
SAMPLE_INTERVAL_MS = 100.0

# Knob 2: the schedulers to compare, as (name, class, extra constructor arguments).
POLICIES = [
    ("fcfs_no_batch", FCFSNoBatchScheduler, {}),
    ("fcfs_batch", FCFSBatchScheduler, {}),
    ("longest_queue", LongestQueueScheduler, {}),
    ("timeout_batch", TimeoutBatchScheduler, {"tau_ms": 15.0}),
]

# Share of COCO val2017 images whose cheapest passing model is yolov8 n, s, m, l.
# Derived from yolo-analysis/results/val2017/coco_class_recall_benchmark.csv at
# threshold 0.80 (cheapest model with recall >= 0.80, else the largest model),
# i.e. 2639 / 745 / 417 / 1199 out of 5000 images.
YOLO_WEIGHTS = [0.5278, 0.149, 0.0834, 0.2398]

# Measured YOLOv8 n/s/m/l T_i(b) on the A100, eager PyTorch FP32.
# The TensorRT profiles (yolov8_a100_trt_fp16.csv / trt_fp32) are kept but not used for now:
# the compiled engines' outputs do not match eager (box decoding looks broken in export),
# so their timings are unverified. Switch back once the engines are fixed and checked.
PROFILE_CSV = os.path.join(HERE, "profiles", "yolov8_a100_eager_fp32.csv")

# Loads for the sweeps (jobs per ms).
REAL_LOADS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# Knob 1 lives inside each experiment: workload_class + workload_kwargs.
EXPERIMENTS = [
    # All three use the MEASURED A100 eager FP32 profile (yolo-analysis/batch_sweep).
    # Batch-1 capacity with these routing weights is about 0.17 jobs/ms, max-batch (64)
    # capacity about 0.69 jobs/ms, hence REAL_LOADS.
    {
        "name": "yolo_load_sweep",
        "profile_csv": PROFILE_CSV,
        "loads": REAL_LOADS,
        "num_streams": 8,
        "workload_class": WeightedWorkload,
        "workload_kwargs": {"weights": YOLO_WEIGHTS},
    },
    # Bursty arrivals: a stream keeps hitting the same queue, like real video.
    {
        "name": "yolo_sticky_load_sweep",
        "profile_csv": PROFILE_CSV,
        "loads": REAL_LOADS,
        "num_streams": 8,
        "workload_class": StickyWorkload,
        "workload_kwargs": {"weights": YOLO_WEIGHTS, "stay_probability": 0.9},
    },
    # Camera-like arrivals: each stream sends a frame at a fixed period. All streams
    # start at t=0 with the same period, so frames arrive in simultaneous bursts of 8
    # (a synchronized-cameras worst case until start phases are staggered).
    {
        "name": "yolo_periodic_load_sweep",
        "profile_csv": PROFILE_CSV,
        "loads": REAL_LOADS,
        "num_streams": 8,
        "workload_class": PeriodicRoutedWorkload,
        "workload_kwargs": {"weights": YOLO_WEIGHTS},
    },
]

CSV_COLUMNS = [
    "experiment", "policy", "load_jobs_per_ms", "seed", "num_jobs_completed",
    "num_jobs_dropped", "num_batches", "turnaround_mean_ms", "turnaround_p50_ms",
    "turnaround_p95_ms", "turnaround_p99_ms", "turnaround_max_ms", "wait_mean_ms",
    "busy_time_ms", "compression_ratio", "throughput_jobs_per_ms", "utilization",
    "max_queue_length", "stable",
]


def raw_data_path(experiment, policy_name, load, seed):
    """Where the raw data of one run lives: results/<experiment>/raw/<policy>_load<L>_seed<S>.npz"""
    file_name = policy_name + "_load" + format(load, ".2f") + "_seed" + str(seed) + ".npz"
    return os.path.join(RESULTS_DIR, experiment["name"], "raw", file_name)


def simulate_one(experiment, scheduler_class, scheduler_kwargs, load, seed):
    """Build everything from scratch for one run, run it, and return its raw data."""
    profile = Profile(experiment["profile_csv"])
    queues = []
    for queue_id in range(len(profile.models)):
        queues.append(Queue(queue_id, profile.models[queue_id]))
    rng = np.random.default_rng(seed)
    workload = experiment["workload_class"](queues, rng, experiment["num_streams"], load,
                                            **experiment["workload_kwargs"])
    scheduler = scheduler_class(queues, profile, **scheduler_kwargs)
    simulator = Simulator(queues, workload, scheduler, profile, HORIZON_MS, SAMPLE_INTERVAL_MS)
    simulator.run()
    return simulator.raw_data


def run_one(experiment, policy_name, scheduler_class, scheduler_kwargs, load, seed):
    """Get the raw data of one run (simulate and save it, or load the saved file)
    and turn it into one row of metrics."""
    path = raw_data_path(experiment, policy_name, load, seed)
    if SIMULATE:
        raw = simulate_one(experiment, scheduler_class, scheduler_kwargs, load, seed)
        save_raw_data(raw, path)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(path + " is missing: run once with SIMULATE = True")
        raw = load_raw_data(path)
    metrics = compute_metrics(raw)

    row = {"experiment": experiment["name"], "policy": policy_name,
           "load_jobs_per_ms": load, "seed": seed}
    for column in CSV_COLUMNS[4:]:
        row[column] = metrics[column]
    return row


def run_experiment(experiment):
    os.makedirs(os.path.join(RESULTS_DIR, experiment["name"], "raw"), exist_ok=True)
    rows = []
    for policy_name, scheduler_class, scheduler_kwargs in POLICIES:
        for load in experiment["loads"]:
            for seed in SEEDS:
                row = run_one(experiment, policy_name, scheduler_class, scheduler_kwargs,
                              load, seed)
                rows.append(row)
                print("[" + experiment["name"] + "] " + policy_name.ljust(14)
                      + " load=" + format(load, ".2f") + " seed=" + str(seed)
                      + " turnaround_mean=" + str(row["turnaround_mean_ms"])
                      + " util=" + str(row["utilization"]) + " stable=" + str(row["stable"]))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("wrote " + path)


def values_by_policy_and_load(rows, column):
    """{policy: {load: [value for each seed]}}, skipping missing values."""
    table = {}
    for row in rows:
        if row[column] is None:
            continue
        per_load = table.setdefault(row["policy"], {})
        per_load.setdefault(row["load_jobs_per_ms"], []).append(row[column])
    return table


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("wrote " + path)


def plot_turnaround_vs_busy_time(rows, path):
    """Scatter of every run: x = busy time, y = mean turnaround, one colour per policy."""
    figure, axes = plt.subplots(figsize=(7, 5))
    for policy_name, scheduler_class, scheduler_kwargs in POLICIES:
        busy_times = []
        turnarounds = []
        for row in rows:
            if row["policy"] == policy_name and row["turnaround_mean_ms"] is not None:
                busy_times.append(row["busy_time_ms"])
                turnarounds.append(row["turnaround_mean_ms"])
        axes.scatter(busy_times, turnarounds, label=policy_name, alpha=0.7)
    axes.set_xlabel("accelerator busy time (ms)")
    axes.set_ylabel("mean turnaround (ms)")
    axes.set_title("Mean turnaround vs. busy time, per policy")
    axes.legend()
    save_figure(figure, path)


def plot_metric_vs_load(rows, path, column, label):
    """One line per policy: mean over seeds, with standard-deviation error bars."""
    figure, axes = plt.subplots(figsize=(7, 5))
    table = values_by_policy_and_load(rows, column)
    for policy_name, scheduler_class, scheduler_kwargs in POLICIES:
        if policy_name not in table:
            continue
        loads = sorted(table[policy_name].keys())
        means = []
        stds = []
        for load in loads:
            means.append(np.mean(table[policy_name][load]))
            stds.append(np.std(table[policy_name][load]))
        axes.errorbar(loads, means, yerr=stds, marker="o", capsize=3, label=policy_name)
    axes.set_xlabel("offered load (jobs/ms)")
    axes.set_ylabel(label)
    axes.set_title(label + " vs. load, per policy")
    axes.legend()
    save_figure(figure, path)


def plot_metric_vs_load_log(rows, path, column, label):
    """Same as plot_metric_vs_load but with a log y-axis, so a policy that has
    blown up (thousands of ms) and the ones still coping (a few ms) fit on one chart."""
    figure, axes = plt.subplots(figsize=(7, 5))
    table = values_by_policy_and_load(rows, column)
    for policy_name, scheduler_class, scheduler_kwargs in POLICIES:
        if policy_name not in table:
            continue
        loads = sorted(table[policy_name].keys())
        means = [np.mean(table[policy_name][load]) for load in loads]
        axes.plot(loads, means, marker="o", label=policy_name)
    axes.set_yscale("log")
    axes.set_xlabel("offered load (jobs/ms)")
    axes.set_ylabel(label + ", log scale")
    axes.set_title(label + " vs. load, per policy (log scale)")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    save_figure(figure, path)


def plot_metric_vs_load_stable_only(rows, path, column, label):
    """Linear y-axis, but each policy is drawn only up to the last load where it
    was still stable in at least half the seeds, so one collapsed policy does not
    squash the others. Where each line stops is where that policy breaks down."""
    figure, axes = plt.subplots(figsize=(7, 5))
    table = values_by_policy_and_load(rows, column)
    stable = values_by_policy_and_load(rows, "stable")
    for policy_name, scheduler_class, scheduler_kwargs in POLICIES:
        if policy_name not in table:
            continue
        loads, means = [], []
        for load in sorted(table[policy_name].keys()):
            if np.mean([float(v) for v in stable[policy_name][load]]) < 0.5:
                break
            loads.append(load)
            means.append(np.mean(table[policy_name][load]))
        if loads:
            axes.plot(loads, means, marker="o", label=policy_name + " (stable up to " + format(loads[-1], ".2f") + ")")
    axes.set_xlabel("offered load (jobs/ms)")
    axes.set_ylabel(label)
    axes.set_title(label + " vs. load, stable region only")
    axes.grid(True, alpha=0.3)
    axes.legend()
    save_figure(figure, path)


def rank_policies(mean_per_policy):
    """Policy names ordered from smallest to largest value (ties keep POLICIES order)."""
    pairs = []
    for position in range(len(POLICIES)):
        name = POLICIES[position][0]
        if name in mean_per_policy:
            pairs.append((mean_per_policy[name], position, name))
    pairs.sort()
    ranking = []
    for value, position, name in pairs:
        ranking.append(name)
    return ranking


def plot_ranking_agreement(rows, path):
    """For each load: fraction of policies whose rank by mean busy time equals
    their rank by mean compression ratio (1.0 = both metrics agree fully)."""
    busy = values_by_policy_and_load(rows, "busy_time_ms")
    compression = values_by_policy_and_load(rows, "compression_ratio")
    loads = sorted(set(row["load_jobs_per_ms"] for row in rows))
    agreement = []
    for load in loads:
        busy_means = {}
        compression_means = {}
        for policy_name in busy:
            if load in busy[policy_name]:
                busy_means[policy_name] = np.mean(busy[policy_name][load])
            if policy_name in compression and load in compression[policy_name]:
                compression_means[policy_name] = np.mean(compression[policy_name][load])
        busy_rank = rank_policies(busy_means)
        compression_rank = rank_policies(compression_means)
        # only compare policies that have both numbers (compression ratio is
        # missing when a run completed no jobs)
        compared = 0
        matches = 0
        for name in busy_rank:
            if name in compression_rank:
                compared += 1
                if busy_rank.index(name) == compression_rank.index(name):
                    matches += 1
        if compared == 0:
            agreement.append(np.nan)
        else:
            agreement.append(matches / compared)

    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(loads, agreement, marker="o")
    axes.set_ylim(-0.05, 1.05)
    axes.set_xlabel("offered load (jobs/ms)")
    axes.set_ylabel("fraction of policies with matching rank\n(busy time vs. compression ratio)")
    axes.set_title("Do busy time and compression ratio rank policies the same way?")
    save_figure(figure, path)


def main():
    for experiment in EXPERIMENTS:
        rows = run_experiment(experiment)
        out_dir = os.path.join(RESULTS_DIR, experiment["name"])
        os.makedirs(out_dir, exist_ok=True)
        write_csv(rows, os.path.join(out_dir, "results.csv"))
        plot_turnaround_vs_busy_time(rows, os.path.join(out_dir, "turnaround_vs_busy_time.png"))
        plot_metric_vs_load(rows, os.path.join(out_dir, "turnaround_vs_load.png"),
                            "turnaround_mean_ms", "mean turnaround (ms)")
        plot_metric_vs_load_log(rows, os.path.join(out_dir, "turnaround_vs_load_log.png"),
                                "turnaround_mean_ms", "mean turnaround (ms)")
        plot_metric_vs_load_stable_only(rows, os.path.join(out_dir, "turnaround_vs_load_stable.png"),
                                        "turnaround_mean_ms", "mean turnaround (ms)")
        plot_metric_vs_load_stable_only(rows, os.path.join(out_dir, "turnaround_p95_vs_load_stable.png"),
                                        "turnaround_p95_ms", "p95 turnaround (ms)")
        plot_metric_vs_load(rows, os.path.join(out_dir, "utilization_vs_load.png"),
                            "utilization", "accelerator utilization")
        plot_metric_vs_load(rows, os.path.join(out_dir, "compression_ratio_vs_load.png"),
                            "compression_ratio", "compression ratio (batches / jobs)")
        plot_ranking_agreement(rows, os.path.join(out_dir, "ranking_agreement.png"))


if __name__ == "__main__":
    main()

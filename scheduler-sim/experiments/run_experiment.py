"""Experiment runner: sweeps over load / policy / seed, writes a tidy
results CSV per experiment plus the three plot types the spec asks
for. Per spec, sweep definitions are hardcoded in python (below),
while RuntimeProfile CSVs and everything else stay config-driven.

Run from the scheduler-sim/ directory:

    python experiments/run_experiment.py

Writes to scheduler-sim/results/<experiment>/{results.csv, *.png}.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SCHED_SIM_ROOT = HERE.parent
sys.path.insert(0, str(SCHED_SIM_ROOT))

from scheduler_sim import policies, routers  # noqa: F401  (registers implementations)
from scheduler_sim.engine import SimConfig, Simulator
from scheduler_sim.job import reset_job_id_counter
from scheduler_sim.logging_setup import setup_logging
from scheduler_sim.policy import policy_registry
from scheduler_sim.routers.recall_labeling import (
    DEFAULT_MODEL_ORDER,
    label_counts_to_weights,
    label_images_from_recall_csv,
)
from scheduler_sim.routers.weighted_random import WeightedRandomRouter
from scheduler_sim.runtime_profile import RuntimeProfile
from scheduler_sim.workload import PoissonArrival, Stream, Workload

REPO_ROOT = SCHED_SIM_ROOT.parent
RESULTS_ROOT = SCHED_SIM_ROOT / "results"


# ---------------------------------------------------------------------------
# Hardcoded sweep configs (per spec: sweeps live in python, not YAML).
# ---------------------------------------------------------------------------

POLICY_SPECS: Dict[str, dict] = {
    "fcfs_no_batch": {},
    "fcfs_batch": {},
    "longest_queue": {},
    "timeout_batch": {"tau_ms": 15.0},
}

SEEDS = [1, 2, 3]
HORIZON_MS = 60_000.0
WARMUP_MS = 5_000.0

YOLO_EXPERIMENT = {
    "name": "yolo_synthetic_load_sweep",
    "profile_csv": str(SCHED_SIM_ROOT / "profiles" / "synthetic_4model.csv"),
    "partial_batch_mode": "pad",
    "recall_csv": str(
        REPO_ROOT / "yolo-analysis" / "results" / "val2017" / "coco_class_recall_benchmark.csv"
    ),
    "recall_threshold": 0.80,
    # total offered load in jobs/ms, swept
    "loads_jobs_per_ms": [0.05, 0.10, 0.15, 0.20, 0.25],
    "num_streams": 8,
}

RESNET_EXPERIMENT = {
    "name": "resnet_load_sweep",
    "profile_csv": str(SCHED_SIM_ROOT / "profiles" / "resnet_example.csv"),
    "partial_batch_mode": "pad",
    "weights": [0.5, 0.5],  # 50/50 resnet18/resnet50, no real routing data for this pair
    "loads_jobs_per_ms": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "num_streams": 8,
}


@dataclass
class RunRow:
    experiment: str
    policy: str
    load_jobs_per_ms: float
    seed: int
    num_jobs_completed: int
    num_jobs_dropped: int
    num_batches: int
    turnaround_mean_ms: Optional[float]
    turnaround_p50_ms: Optional[float]
    turnaround_p95_ms: Optional[float]
    turnaround_p99_ms: Optional[float]
    turnaround_max_ms: Optional[float]
    wait_mean_ms: Optional[float]
    busy_time_ms: float
    compression_ratio: Optional[float]
    throughput_jobs_per_ms: float
    utilization: Optional[float]
    max_queue_length: int
    stable: bool


def build_weights_from_recall(recall_csv: str, threshold: float) -> List[float]:
    labels = label_images_from_recall_csv(
        recall_csv, model_order=DEFAULT_MODEL_ORDER, threshold=threshold
    )
    return label_counts_to_weights(labels, model_order=DEFAULT_MODEL_ORDER)


def run_one(
    experiment_name: str,
    profile: RuntimeProfile,
    queue_to_model: List[str],
    weights: List[float],
    policy_name: str,
    policy_kwargs: dict,
    load_jobs_per_ms: float,
    num_streams: int,
    seed: int,
) -> RunRow:
    reset_job_id_counter()
    rng = np.random.default_rng(seed)
    router = WeightedRandomRouter(weights=weights, rng=rng)
    policy = policy_registry.create(policy_name, **policy_kwargs)

    rate_per_stream = load_jobs_per_ms / num_streams
    streams = [
        Stream(stream_id=i, arrival_process=PoissonArrival(rate_per_ms=rate_per_stream))
        for i in range(num_streams)
    ]
    workload = Workload(streams=streams)

    cfg = SimConfig(
        queue_to_model=queue_to_model,
        runtime_profile=profile,
        workload=workload,
        router=router,
        policy=policy,
        seed=seed,
        horizon_ms=HORIZON_MS,
        warmup_ms=WARMUP_MS,
        queue_sample_interval_ms=100.0,
        log_level="ERROR",
    )
    result = Simulator(cfg).run()
    s = result.metrics_summary
    return RunRow(
        experiment=experiment_name,
        policy=policy_name,
        load_jobs_per_ms=load_jobs_per_ms,
        seed=seed,
        num_jobs_completed=s["num_jobs_completed"],
        num_jobs_dropped=s["num_jobs_dropped"],
        num_batches=s["num_batches"],
        turnaround_mean_ms=s["turnaround_mean_ms"],
        turnaround_p50_ms=s["turnaround_p50_ms"],
        turnaround_p95_ms=s["turnaround_p95_ms"],
        turnaround_p99_ms=s["turnaround_p99_ms"],
        turnaround_max_ms=s["turnaround_max_ms"],
        wait_mean_ms=s["wait_mean_ms"],
        busy_time_ms=s["busy_time_ms"],
        compression_ratio=s["compression_ratio"],
        throughput_jobs_per_ms=s["throughput_jobs_per_ms"],
        utilization=s["utilization"],
        max_queue_length=s["max_queue_length"],
        stable=s["stable"],
    )


def run_experiment(spec: dict) -> List[RunRow]:
    name = spec["name"]
    profile = RuntimeProfile.from_csv(spec["profile_csv"], partial_batch_mode=spec["partial_batch_mode"])
    queue_to_model = profile.models

    if "recall_csv" in spec and Path(spec["recall_csv"]).exists():
        weights = build_weights_from_recall(spec["recall_csv"], spec["recall_threshold"])
        print(f"[{name}] routing weights derived from {spec['recall_csv']}: {weights}")
    else:
        weights = spec.get("weights")
        if weights is None:
            weights = [1.0 / len(queue_to_model)] * len(queue_to_model)
        print(f"[{name}] recall CSV not found, falling back to weights: {weights}")

    rows: List[RunRow] = []
    for policy_name, policy_kwargs in POLICY_SPECS.items():
        for load in spec["loads_jobs_per_ms"]:
            for seed in SEEDS:
                row = run_one(
                    experiment_name=name,
                    profile=profile,
                    queue_to_model=queue_to_model,
                    weights=weights,
                    policy_name=policy_name,
                    policy_kwargs=policy_kwargs,
                    load_jobs_per_ms=load,
                    num_streams=spec["num_streams"],
                    seed=seed,
                )
                rows.append(row)
                print(
                    f"[{name}] policy={policy_name:14s} load={load:.3f} seed={seed} "
                    f"turnaround_mean={row.turnaround_mean_ms} util={row.utilization} "
                    f"stable={row.stable}"
                )
    return rows


def write_csv(rows: List[RunRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"wrote {out_path}")


def _aggregate_by_policy_load(rows: List[RunRow], value_fn):
    by_policy: Dict[str, Dict[float, List[float]]] = {}
    for r in rows:
        by_policy.setdefault(r.policy, {}).setdefault(r.load_jobs_per_ms, []).append(value_fn(r))
    return by_policy


def plot_turnaround_vs_busy_time(rows: List[RunRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    by_policy = {}
    for r in rows:
        by_policy.setdefault(r.policy, []).append(r)
    for policy_name, prows in by_policy.items():
        valid = [r for r in prows if r.turnaround_mean_ms is not None]
        busy = [r.busy_time_ms for r in valid]
        turnaround = [r.turnaround_mean_ms for r in valid]
        ax.scatter(busy, turnaround, label=policy_name, alpha=0.7)
    ax.set_xlabel("accelerator busy time (ms)")
    ax.set_ylabel("mean turnaround (ms)")
    ax.set_title("Mean turnaround vs. busy time, per policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_metric_vs_load(rows: List[RunRow], out_path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    by_policy = _aggregate_by_policy_load(rows, lambda r: getattr(r, metric))
    for policy_name, per_load in sorted(by_policy.items()):
        loads = sorted(per_load.keys())
        means = []
        stds = []
        for load in loads:
            vals = [v for v in per_load[load] if v is not None]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        ax.errorbar(loads, means, yerr=stds, marker="o", capsize=3, label=policy_name)
    ax.set_xlabel("offered load (jobs/ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. load, per policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_ranking_agreement(rows: List[RunRow], out_path: Path) -> None:
    """For each load level, rank policies by mean busy_time and by mean
    compression_ratio, and show whether the rankings agree."""
    by_load_policy_busy: Dict[float, Dict[str, List[float]]] = {}
    by_load_policy_comp: Dict[float, Dict[str, List[float]]] = {}
    for r in rows:
        by_load_policy_busy.setdefault(r.load_jobs_per_ms, {}).setdefault(r.policy, []).append(
            r.busy_time_ms
        )
        if r.compression_ratio is not None:
            by_load_policy_comp.setdefault(r.load_jobs_per_ms, {}).setdefault(
                r.policy, []
            ).append(r.compression_ratio)

    loads = sorted(by_load_policy_busy.keys())
    agreement = []
    for load in loads:
        busy_rank = sorted(by_load_policy_busy[load], key=lambda p: np.mean(by_load_policy_busy[load][p]))
        comp_map = by_load_policy_comp.get(load, {})
        if not comp_map:
            agreement.append(np.nan)
            continue
        comp_rank = sorted(comp_map, key=lambda p: np.mean(comp_map[p]))
        # agreement score: fraction of policies whose rank position matches
        common = [p for p in busy_rank if p in comp_rank]
        matches = sum(1 for p in common if busy_rank.index(p) == comp_rank.index(p))
        agreement.append(matches / len(common) if common else np.nan)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loads, agreement, marker="o")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("offered load (jobs/ms)")
    ax.set_ylabel("fraction of policies with matching rank\n(busy time vs. compression ratio)")
    ax.set_title("Do busy time and compression ratio rank policies the same way?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    setup_logging("WARNING")
    for spec in (YOLO_EXPERIMENT, RESNET_EXPERIMENT):
        rows = run_experiment(spec)
        out_dir = RESULTS_ROOT / spec["name"]
        write_csv(rows, out_dir / "results.csv")
        plot_turnaround_vs_busy_time(rows, out_dir / "turnaround_vs_busy_time.png")
        plot_metric_vs_load(
            rows, out_dir / "turnaround_vs_load.png", "turnaround_mean_ms", "mean turnaround (ms)"
        )
        plot_metric_vs_load(
            rows, out_dir / "utilization_vs_load.png", "utilization", "accelerator utilization"
        )
        plot_metric_vs_load(
            rows,
            out_dir / "compression_ratio_vs_load.png",
            "compression_ratio",
            "compression ratio (batches / jobs)",
        )
        plot_ranking_agreement(rows, out_dir / "ranking_agreement.png")


if __name__ == "__main__":
    main()

# Part B step 3. Runs the test scenario (S02) and the stream-only scenario (S06) videos in time order
# at 5 frames per second through the dispatcher and analyses the routing as a stream: distribution,
# stickiness (run lengths, switches/min, transitions) vs the ideal labels, per-model arrival rate and
# burstiness for a scheduler, and the effect of majority-vote smoothing. Traces go to
# results/07_stream/traces/*.csv.gz (these feed the scheduler later).
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from common import detect, extractor, head, paths, report, videos
from common.labels import ideal_letter
from common.boxes import box_recall

STREAM_FPS = 5
SMOKE_FRAMES = 200
SMOOTH_WINDOWS = [3, 5, 10]
LETTERS = config.MODEL_LETTERS


# ---------------------------------------------------------------- tracing

def load_pool(smoke):
    """Fine-tuned n, s, m and stock l detectors."""
    models = {}
    for letter in LETTERS:
        variant = "orig" if letter == "l" else "ft"
        models[letter] = detect.load_model(detect.weights_for(variant, letter, smoke))
    return models


def trace_camera(cam, embedder, saved_head, models, smoke):
    """Per-frame decisions and ideal labels for one camera at STREAM_FPS."""
    fps, _, _, _ = videos.video_info(cam["video_path"])
    stride = max(1, round(fps / STREAM_FPS))
    rows = []
    frames, ids = [], []

    def flush():
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        proba = head.predict_proba(saved_head, extractor.embed_pil(embedder, pil))
        dets = {l: detect.detect(models[l], frames) for l in LETTERS}
        for k, frame_idx in enumerate(ids):
            ref = dets["l"][k]
            recalls = {l: box_recall(ref, dets[l][k], config.MATCH_IOU) for l in LETTERS}
            rows.append({"scenario": cam["scenario"], "camera": cam["camera"], "frame_idx": frame_idx,
                         "time_sec": round((frame_idx - 1) / fps, 3),
                         "pred": LETTERS[int(proba[k].argmax())], "ideal": ideal_letter(recalls),
                         **{f"p_{l}": round(float(proba[k][i]), 4) for i, l in enumerate(LETTERS)},
                         **{f"recall_{l}": round(recalls[l], 4) for l in LETTERS}})
        frames.clear()
        ids.clear()

    max_frames = SMOKE_FRAMES if smoke else None
    for frame_idx, frame in videos.sampled_frames(cam["video_path"], stride=stride, max_frames=max_frames):
        frames.append(frame)
        ids.append(frame_idx)
        if len(frames) == 16:
            flush()
    if frames:
        flush()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analysis helpers

def run_lengths(seq):
    lengths, current = [], 1
    for a, b in zip(seq[:-1], seq[1:]):
        if a == b:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def stickiness_row(sequences, kind, scenario, camera):
    """sequences: list of label lists (one per camera). Run lengths never cross cameras."""
    lengths = [n for seq in sequences for n in run_lengths(seq)]
    total_frames = sum(len(s) for s in sequences)
    switches = len(lengths) - len(sequences)
    minutes = total_frames / STREAM_FPS / 60
    lengths = np.array(lengths)
    return {"scenario": scenario, "camera": camera, "kind": kind, "frames": total_frames,
            "mean_run_frames": round(lengths.mean(), 2), "median_run_frames": float(np.median(lengths)),
            "p90_run_frames": float(np.percentile(lengths, 90)),
            "mean_run_sec": round(lengths.mean() / STREAM_FPS, 2),
            "median_run_sec": round(float(np.median(lengths)) / STREAM_FPS, 2),
            "p90_run_sec": round(float(np.percentile(lengths, 90)) / STREAM_FPS, 2),
            "switches_per_min": round(switches / minutes, 2)}


def transition_rows(sequences, kind, scenario):
    counts = np.zeros((4, 4))
    for seq in sequences:
        idx = [LETTERS.index(x) for x in seq]
        for a, b in zip(idx[:-1], idx[1:]):
            counts[a, b] += 1
    probs = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    return [{"scenario": scenario, "kind": kind, "from": LETTERS[i], "to": LETTERS[j], "prob": round(probs[i, j], 4),
             "count": int(counts[i, j])} for i in range(4) for j in range(4)]


def arrival_rows(traces, scenario, camera):
    """Frames routed to each model per 1 s window: mean rate and index of dispersion (var / mean)."""
    seconds = int(max(t["time_sec"].max() for t in traces)) + 1
    counts = np.zeros((seconds, 4))
    for t in traces:
        for sec, label in zip(t["time_sec"].astype(int), t["pred"]):
            counts[sec, LETTERS.index(label)] += 1
    rows = []
    for i, l in enumerate(LETTERS):
        mean = counts[:, i].mean()
        rows.append({"scenario": scenario, "camera": camera, "model": l, "mean_per_sec": round(mean, 3),
                     "dispersion_index": round(counts[:, i].var() / mean, 3) if mean > 0 else float("nan")})
    return rows


def majority_vote(labels, window):
    """Causal: each decision is the majority of the last `window` raw decisions (tie -> larger model)."""
    idx = np.array([LETTERS.index(x) for x in labels])
    out = []
    for i in range(len(idx)):
        votes = np.bincount(idx[max(0, i - window + 1):i + 1], minlength=4)
        out.append(int(np.flatnonzero(votes == votes.max())[-1]))
    return out


def smoothing_rows(traces, scenario):
    rows = []
    for window in [1] + SMOOTH_WINDOWS:
        sequences, correct = [], []
        for t in traces:
            smoothed = majority_vote(list(t["pred"]), window) if window > 1 else [LETTERS.index(x) for x in t["pred"]]
            sequences.append([LETTERS[i] for i in smoothed])
            recall = np.stack([t[f"recall_{l}"].to_numpy() for l in LETTERS], 1)
            correct.append(recall[np.arange(len(t)), smoothed] >= config.RECALL_THRESHOLD)
        row = stickiness_row(sequences, "pred", scenario, "ALL")
        rows.append({"scenario": scenario, "window": window, "switches_per_min": row["switches_per_min"],
                     "mean_run_sec": row["mean_run_sec"], "alpha_sys": round(float(np.concatenate(correct).mean()), 4)})
    return rows


def plot_rolling_share(traces, scenario, out_dir):
    fig, axes = plt.subplots(len(traces), 1, figsize=(9, 2.2 * len(traces)), sharex=False, squeeze=False)
    for ax, t in zip(axes[:, 0], traces):
        onehot = pd.DataFrame({l: (t["pred"] == l).astype(float) for l in LETTERS})
        rolling = onehot.rolling(60 * STREAM_FPS, min_periods=1).mean()
        ax.stackplot(t["time_sec"], rolling.T.to_numpy(), labels=LETTERS)
        ax.set_ylabel(t["camera"].iloc[0])
        ax.set_ylim(0, 1)
    axes[0, 0].legend(ncol=4, loc="upper right")
    axes[-1, 0].set_xlabel("time (s), rolling 1-minute share of routed model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"rolling_share_{scenario}.png"), dpi=120)
    plt.close()


def plot_run_hist(traces, scenario, out_dir):
    plt.figure(figsize=(6, 4))
    for kind, color in [("pred", "tab:blue"), ("ideal", "tab:orange")]:
        lengths = [n for t in traces for n in run_lengths(list(t[kind]))]
        plt.hist(np.clip(lengths, 0, 60), bins=np.arange(1, 62), alpha=0.5, label=kind, color=color)
    plt.yscale("log")
    plt.xlabel("run length (frames at 5 fps, clipped at 60)")
    plt.ylabel("runs")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"run_hist_{scenario}.png"), dpi=130)
    plt.close()


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke

    saved = torch.load(os.path.join(paths.MODELS, "dispatcher" + ("_smoke" if smoke else ""), "head.pt"), weights_only=False)
    out_dir = paths.result_dir("07_stream", smoke)
    trace_dir = os.path.join(out_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    scenarios = config.SPLITS["test"] + config.STREAM_ONLY_SCENARIOS
    cameras = videos.find_cameras(scenarios)
    if smoke:
        cameras = cameras[:1]
    embedder, models = extractor.load_embedder(), load_pool(smoke)

    traces_by_scenario = {}
    for cam in cameras:
        path = os.path.join(trace_dir, f"{cam['scenario']}_{cam['camera']}.csv.gz")
        if os.path.exists(path) and not args.force:
            trace = pd.read_csv(path)
            print("skip (done):", path)
        else:
            trace = trace_camera(cam, embedder, saved["head"], models, smoke)
            trace.to_csv(path, index=False)
            print(f"{cam['scenario']} {cam['camera']}: {len(trace)} frames", flush=True)
        traces_by_scenario.setdefault(cam["scenario"], []).append(trace)

    dist, stick, trans, arrivals, smooth = [], [], [], [], []
    for scenario, traces in traces_by_scenario.items():
        groups = [(t["camera"].iloc[0], [t]) for t in traces] + [("ALL", traces)]
        for camera, ts in groups:
            allf = pd.concat(ts)
            recall = np.stack([allf[f"recall_{l}"].to_numpy() for l in LETTERS], 1)
            pred_idx = np.array([LETTERS.index(x) for x in allf["pred"]])
            dist.append({"scenario": scenario, "camera": camera, "frames": len(allf),
                         **{f"routed_{l}": round((allf["pred"] == l).mean(), 3) for l in LETTERS},
                         **{f"ideal_{l}": round((allf["ideal"] == l).mean(), 3) for l in LETTERS},
                         "alpha_sys": round(float((recall[np.arange(len(allf)), pred_idx] >= config.RECALL_THRESHOLD).mean()), 4)})
            for kind in ["pred", "ideal"]:
                stick.append(stickiness_row([list(t[kind]) for t in ts], kind, scenario, camera))
            arrivals += arrival_rows(ts, scenario, camera)
        for kind in ["pred", "ideal"]:
            trans += transition_rows([list(t[kind]) for t in traces], kind, scenario)
        smooth += smoothing_rows(traces, scenario)
        plot_rolling_share(traces, scenario, out_dir)
        plot_run_hist(traces, scenario, out_dir)

    tables = {"routing_distribution": pd.DataFrame(dist), "stickiness": pd.DataFrame(stick),
              "transitions": pd.DataFrame(trans), "arrivals": pd.DataFrame(arrivals),
              "smoothing": pd.DataFrame(smooth)}
    for name, table in tables.items():
        table.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    flags = [f"Stream rate is {STREAM_FPS} fps (every 2nd frame of the 10 fps videos), not the stride-5 frames used for training.",
             "Arrival rates are per camera on video-relative time; the scenario ALL row sums cameras on the same "
             "video-relative clock (CityFlow camera start offsets are ignored).",
             "Smoothing is a causal trailing majority vote, ties go to the larger model."]
    all_rows = tables["routing_distribution"].query("camera == 'ALL'")
    stick_all = tables["stickiness"].query("camera == 'ALL'")
    report.write_summary(out_dir, "07 stream analysis", ["models/dispatcher/head.pt", "S02 and S06 videos"],
                         ["traces/*.csv.gz"] + [f"{n}.csv" for n in tables] + ["rolling_share_*.png", "run_hist_*.png"],
                         all_rows, flags,
                         ["Stickiness (ALL cameras):", "", report.table_to_text(stick_all), "",
                          "Smoothing:", "", report.table_to_text(tables["smoothing"])])
    print(all_rows.to_string(index=False))
    print(stick_all.to_string(index=False))
    print(tables["smoothing"].to_string(index=False))
    print("saved", out_dir)


if __name__ == "__main__":
    main()

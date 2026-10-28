"""
Pure routing metrics (numpy/pandas, no torch, no plotting). A routing
sequence is an int array of model indices (0 = yolov8n ... 3 = yolov8l) in
frame order for ONE camera; runs and transitions never cross cameras.
"""

import numpy as np
import pandas as pd


def run_lengths(models):
    """Maximal runs of the same model: list of (model, length) in order.
    [0, 0, 0, 1, 1, 3, 0, 0] -> [(0, 3), (1, 2), (3, 1), (0, 2)]"""
    runs = []
    for m in models:
        if runs and runs[-1][0] == m:
            runs[-1][1] += 1
        else:
            runs.append([m, 1])
    return [(int(m), n) for m, n in runs]


def switches(models):
    """Number of frame-to-frame model changes."""
    models = np.asarray(models)
    return int(np.sum(models[1:] != models[:-1]))


def split(models, n_models):
    """Fraction of frames per model, length n_models."""
    counts = np.bincount(np.asarray(models), minlength=n_models)
    return counts / max(counts.sum(), 1)


def transition_counts(models, n_models):
    """counts[a, b] = how often model a on one frame is followed by b on the next."""
    models = np.asarray(models)
    counts = np.zeros((n_models, n_models), dtype=np.int64)
    np.add.at(counts, (models[:-1], models[1:]), 1)
    return counts


def row_normalize(counts):
    """P(next | current); rows with no frames stay 0."""
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)


def merge_synced(frames):
    """All cameras of one scenario as one stream, ordered by synced time
    (ties broken by camera, then frame). This is the order frames would reach
    a shared box."""
    return frames.sort_values(["synced_time_s", "camera", "frame_idx"], kind="stable").reset_index(drop=True)


def window_counts(times, models, window_s, n_models, start=None):
    """Frames per model in consecutive windows of window_s seconds, starting
    at `start` (default: the first time). Returns a DataFrame indexed by
    window start time, one column per model index, empty windows included."""
    times = np.asarray(times, dtype=float)
    models = np.asarray(models)
    start = times.min() if start is None else start
    window = np.floor((times - start) / window_s).astype(int)
    counts = np.zeros((window.max() + 1, n_models), dtype=np.int64)
    np.add.at(counts, (window, models), 1)
    index = start + window_s * np.arange(len(counts))
    return pd.DataFrame(counts, index=index, columns=range(n_models))


def max_window_share(times, models, window_s, n_models):
    """Largest share any one model takes in any window of window_s seconds.
    Windows with fewer than half the median frame count (the partial edges of
    the recording) are left out so a near-empty window can't read as 100%."""
    counts = window_counts(times, models, window_s, n_models)
    totals = counts.sum(axis=1)
    full = totals >= 0.5 * totals.median()
    shares = counts[full].div(totals[full], axis=0)
    return float(shares.to_numpy().max())

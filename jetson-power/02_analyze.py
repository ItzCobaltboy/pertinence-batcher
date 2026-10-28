# Tables, plots and plain notes from results.csv. Needs only pandas + matplotlib (laptop or board).
# Run: python 02_analyze.py results/<device_tag>   -> summary.md and PNGs in that folder
import sys
sys.dont_write_bytecode = True
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SPREAD_FLAG = 0.10
# (suffix of the energy columns, label); the first is the headline
DEFS = [("", "total (headline)"), ("_excl_sys5v", "total without SYS5V"), ("_gpu", "GPU rail only")]
POWER_COL = {"": "power_mw", "_excl_sys5v": "power_excl_sys5v_mw", "_gpu": "power_GPU_mw"}


def load(folder):
    df = pd.read_csv(os.path.join(folder, "results.csv"))
    df["mode"] = df["mode_id"].astype(str) + " " + df["mode_name"]
    return df


def average_reps(df):
    """One row per (mode, model, batch): mean over repetitions, plus the spread of energy per frame."""
    keys = ["mode_id", "mode", "model", "batch"]
    mean = df.groupby(keys).mean(numeric_only=True).reset_index()
    g = df.groupby(keys)["energy_per_frame_mj"]
    mean["reps"] = g.count().values
    mean["energy_spread"] = ((g.max() - g.min()) / g.mean()).values
    return mean


def plot_group(cells, line_col, group_col, group_value, path, title):
    """2x2 figure: energy per frame, latency per batch, throughput, power vs batch; one line per line_col."""
    sub = cells[cells[group_col] == group_value]
    panels = [("energy_per_frame_mj", "energy per frame (mJ)"), ("latency_ms_mean", "latency per batch (ms)"),
              ("throughput_fps", "throughput (frames/s)"), ("power_mw", "mean power, total (mW)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, label) in zip(axes.flat, panels):
        for name, line in sub.groupby(line_col):
            line = line.sort_values("batch")
            ax.plot(line["batch"], line[col], marker="o", label=str(name))
        ax.set_xlabel("batch size")
        ax.set_ylabel(label)
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def best_batch_table(cells, suffix):
    """Per (mode, model): batch with minimum energy per frame, and the saving at b=8 vs b=1."""
    col = "energy_per_frame%s_mj" % suffix
    rows = []
    for (mode, model), g in cells.groupby(["mode", "model"]):
        best = g.loc[g[col].idxmin()]
        e1 = g[g.batch == 1][col]
        e8 = g[g.batch == 8][col]
        saving = round(100 * (1 - e8.iloc[0] / e1.iloc[0]), 1) if len(e1) and len(e8) else ""
        rows.append({"mode": mode, "model": "yolov8" + model, "best_batch": int(best["batch"]),
                     "min_mj_per_frame": round(best[col], 2), "b1_mj": round(e1.iloc[0], 2) if len(e1) else "",
                     "b8_mj": round(e8.iloc[0], 2) if len(e8) else "", "saving_b8_vs_b1_pct": saving})
    return pd.DataFrame(rows)


def flat_point(g, col):
    """Smallest batch whose energy is within 5% of the minimum for this curve."""
    g = g.sort_values("batch")
    near = g[g[col] <= 1.05 * g[col].min()]
    return int(near["batch"].iloc[0])


def pareto(g):
    """Cells not beaten on both latency per batch and energy per frame."""
    keep = []
    for i, a in g.iterrows():
        beaten = ((g.latency_ms_mean <= a.latency_ms_mean) & (g.energy_per_frame_mj <= a.energy_per_frame_mj)
                  & ((g.latency_ms_mean < a.latency_ms_mean) | (g.energy_per_frame_mj < a.energy_per_frame_mj))).any()
        if not beaten:
            keep.append(i)
    return g.loc[keep].sort_values("latency_ms_mean")


def notes(cells):
    lines = []
    for (mode, model), g in cells.groupby(["mode", "model"]):
        col = "energy_per_frame_mj"
        e1 = g[g.batch == 1][col]
        text = "- mode %s, yolov8%s: energy per frame is lowest at b=%d (%.1f mJ); within 5%% of that from b=%d" % (
            mode, model, g.loc[g[col].idxmin(), "batch"], g[col].min(), flat_point(g, col))
        if len(e1):
            text += "; b=1 costs %.1f mJ" % e1.iloc[0]
        lines.append(text + ".")
    lines.append("")
    for model, g in cells.groupby("model"):
        p = pareto(g)
        count = p.groupby("mode").size().to_dict()
        lines.append("- yolov8%s, latency per batch vs energy per frame, non-dominated cells by mode: %s (of %d cells)."
                     % (model, count, len(p)))
    return lines


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python 02_analyze.py results/<device_tag>")
    folder = sys.argv[1]
    df = load(folder)
    cells = average_reps(df)

    for model in sorted(cells.model.unique()):
        plot_group(cells, "mode", "model", model, os.path.join(folder, "model_yolov8%s.png" % model), "yolov8%s, one line per power mode" % model)
    for mode in sorted(cells["mode"].unique()):
        mode_id = cells[cells["mode"] == mode].mode_id.iloc[0]
        plot_group(cells, "model", "mode", mode, os.path.join(folder, "mode_%d.png" % mode_id), "power mode %s, one line per model" % mode)

    out = ["# Jetson power sweep summary", "", "%d result rows, %d (mode, model, batch) cells." % (len(df), len(cells)), ""]
    for suffix, label in DEFS:
        out += ["## Energy per frame, power definition: %s" % label, "", md_table(best_batch_table(cells, suffix)), ""]
    out += ["## Notes (restating the tables)", ""] + notes(cells) + [""]

    flagged = cells[cells.energy_spread > SPREAD_FLAG]
    out += ["## Repetition spread", "",
            "Cells where the repetitions' energy per frame differ by more than %d%% (max-min over mean): %d of %d."
            % (SPREAD_FLAG * 100, len(flagged), len(cells)), ""]
    if len(flagged):
        t = flagged[["mode", "model", "batch", "reps", "energy_spread"]].copy()
        t["energy_spread"] = (t["energy_spread"] * 100).round(1).astype(str) + "%"
        out += [md_table(t), ""]

    status_path = os.path.join(folder, "status.csv")
    if os.path.exists(status_path):
        st = pd.read_csv(status_path)
        problems = st[~st.status.isin(["done"])]
        out += ["## Cells that were not plain 'done'", "", md_table(problems) if len(problems) else "none", ""]

    with open(os.path.join(folder, "summary.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote summary.md and PNGs in", folder)


main()

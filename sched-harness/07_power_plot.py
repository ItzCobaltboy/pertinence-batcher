#!/usr/bin/env python3
"""Mean board power over the 30 min window against alpha, one line per load, for the three power definitions the harness logs.
Needs analysis/dvfs_summary.csv (05_dvfs_summary.py).   Usage: DEVICE_TAG=agx_xavier_jp511_main python3 07_power_plot.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

df = pd.read_csv(os.path.join(config.ANALYSIS_DIR, "dvfs_summary.csv"))
for key in ("total", "total_excl_sys5v", "gpu"):
    df["p_" + key] = df["energy_j_" + key if key != "total" else "energy_j_total"] / df["span_s"]
PANELS = [("p_total", "A  Total board power (all rails)"),
          ("p_total_excl_sys5v", "B  Excluding the 5 V rail (energy metric)"),
          ("p_gpu", "C  GPU rail only")]
STYLE = {100.0: ("#0072B2", "o"), 200.0: ("#E69F00", "s"), 300.0: ("#D55E00", "^")}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
for ax, (col, title) in zip(axes, PANELS):
    for lam, (color, marker) in STYLE.items():
        g = df[df["lam"] == lam].sort_values("alpha")
        ax.plot(g["alpha"], g[col], color=color, marker=marker, lw=2, ms=7, mec="white", mew=1.2)
        for a, v in zip(g["alpha"], g[col]):
            ax.annotate("%.1f" % v, (a, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
        ax.annotate("%d req/s" % lam, (g["alpha"].iloc[-1], g[col].iloc[-1]), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("alpha  (0 = latency first, 1 = lowest feasible energy)")
    ax.set_ylabel("mean power (W)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim(-0.06, 1.3)
    ax.set_ylim(0, df[col].max() * 1.16)
fig.tight_layout(w_pad=2.5)
out = os.path.join(config.ANALYSIS_DIR, "power_overview.png")
fig.savefig(out, dpi=160, facecolor="white")
print("wrote", out)
print(df.pivot(index="lam", columns="alpha", values="p_total").round(2).to_string())
print(df.pivot(index="lam", columns="alpha", values="p_total_excl_sys5v").round(2).to_string())
print(df.pivot(index="lam", columns="alpha", values="p_gpu").round(2).to_string())
print((df["energy_j_total"] - df["energy_j_total_excl_sys5v"]).div(df["span_s"]).round(2).describe()[["min", "max"]].to_dict(), "= SYS5V rail W")

#!/usr/bin/env python3
"""Energy against latency across alpha, one line per load, with the deadline marked.
Needs analysis/dvfs_summary.csv.   Usage: DEVICE_TAG=agx_xavier_jp511_main python3 08_tradeoff_plot.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

df = pd.read_csv(os.path.join(config.ANALYSIS_DIR, "dvfs_summary.csv"))
SLO = 900
STYLE = {100.0: ("#0072B2", "o"), 200.0: ("#E69F00", "s"), 300.0: ("#D55E00", "^")}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
# label offsets (points) for alpha 0 / 0.5 / 1 so neighbouring labels do not collide
OFFS = {100.0: {0.0: (0, 11), 0.5: (0, 11), 1.0: (0, -17)}, 200.0: {0.0: (0, 11), 0.5: (0, 11), 1.0: (0, -17)},
        300.0: {0.0: (-4, 13), 0.5: (0, -18), 1.0: (0, 12)}}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, col, title in [(axes[0], "lat_mean_ms", "A  Mean latency against energy"), (axes[1], "lat_p99_ms", "B  p99 latency against energy")]:
    for lam, (color, marker) in STYLE.items():
        g = df[df["lam"] == lam].sort_values("alpha")
        ax.plot(g[col], g["energy_per_good_req_mj"], color=color, lw=2, marker=marker, ms=8, mec="white", mew=1.2, zorder=3)
        for _, r in g.iterrows():
            if r["alpha"] in OFFS[lam]:
                ax.annotate("a=%g" % r["alpha"], (r[col], r["energy_per_good_req_mj"]), xytext=OFFS[lam][r["alpha"]],
                            textcoords="offset points", ha="center", fontsize=9, color=MUTED)
        ax.annotate("%d req/s" % lam, (g[col].iloc[0], g["energy_per_good_req_mj"].iloc[0]), xytext=(-12, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=10.5, color=color, fontweight="bold")
    ax.axvline(SLO, color=MUTED, ls="--", lw=1.4)
    ax.text(SLO * 0.97, 0.98, "deadline\n%d ms" % SLO, transform=ax.get_xaxis_transform(), ha="right", va="top", color=MUTED, fontsize=10)
    ax.axvspan(SLO, 1500, color="#f3f3f3", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(12, 1300)
    ax.set_xlabel("%s (ms, log scale)" % ("mean latency" if col == "lat_mean_ms" else "p99 latency"))
    ax.set_ylabel("energy per good request (mJ, lower is better)")
    ax.set_ylim(55, 160)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
fig.text(0.5, 0.005, "each line walks alpha from 0 (left end, latency first) to 1 (lowest feasible energy); labelled points are alpha 0, 0.5, 1",
         ha="center", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=3)
out = os.path.join(config.ANALYSIS_DIR, "tradeoff.png")
fig.savefig(out, dpi=160, facecolor="white")
print("wrote", out)
d = df[["lam", "alpha", "energy_per_good_req_mj", "lat_mean_ms", "lat_p50_ms", "lat_p99_ms"]].round(1)
print(d.to_string(index=False))
for lam, g in df.groupby("lam"):
    a0, a1 = g[g.alpha == 0].iloc[0], g[g.alpha == 1].iloc[0]
    print("load %d: alpha 0 -> 1: energy %.1f -> %.1f mJ (%.0f%%), mean latency %.0f -> %.0f ms (x%.1f), p99 %.0f -> %.0f ms" % (
        lam, a0.energy_per_good_req_mj, a1.energy_per_good_req_mj, (a1.energy_per_good_req_mj / a0.energy_per_good_req_mj - 1) * 100,
        a0.lat_mean_ms, a1.lat_mean_ms, a1.lat_mean_ms / a0.lat_mean_ms, a0.lat_p99_ms, a1.lat_p99_ms))

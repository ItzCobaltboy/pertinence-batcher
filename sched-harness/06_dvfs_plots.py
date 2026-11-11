#!/usr/bin/env python3
"""Overview plots of the DVFS main run from analysis/dvfs_summary.csv and clock_shares.json (run 05_dvfs_summary.py first).
Usage:  DEVICE_TAG=agx_xavier_jp511_main python3 06_dvfs_plots.py     -> analysis/dvfs_overview.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

A = config.ANALYSIS_DIR
df = pd.read_csv(os.path.join(A, "dvfs_summary.csv"))
shares = json.load(open(os.path.join(A, "clock_shares.json")))
LOADS = sorted(df["lam"].unique())
ALPHAS = sorted(df["alpha"].unique())
SLO = 900
# Okabe-Ito (colour-blind safe); markers add a second cue
STYLE = {100.0: ("#0072B2", "o"), 200.0: ("#E69F00", "s"), 300.0: ("#D55E00", "^")}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "font.family": "DejaVu Sans"})


def lines(ax, col, ylabel, title, nudge=None):
    for lam in LOADS:
        g = df[df["lam"] == lam].sort_values("alpha")
        color, marker = STYLE[lam]
        ax.plot(g["alpha"], g[col], color=color, marker=marker, lw=2, ms=7, mfc=color, mec="white", mew=1.2, label="%d req/s" % lam)
        ax.annotate("%d req/s" % lam, (g["alpha"].iloc[-1], g[col].iloc[-1]), xytext=(8, (nudge or {}).get(lam, 0)), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    ax.set_xlabel("alpha   (0 = latency first,  1 = lowest feasible energy)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xticks(ALPHAS)
    ax.set_xlim(-0.05, 1.22)


fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

# A: energy per good request
ax = axes[0, 0]
lines(ax, "energy_per_good_req_mj", "mJ per good request", "A  Energy per good request")
ax.set_ylim(0, None)
ax.annotate("alpha 0 = fixed 1237 MHz,\nalpha 1 = fixed 829 MHz\n(100 and 200 req/s)", xy=(0.03, 0.04), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=9.5, color=MUTED)

# B: p99 latency
ax = axes[0, 1]
lines(ax, "lat_p99_ms", "p99 latency (ms)", "B  Tail latency against the %d ms deadline" % SLO, nudge={300.0: 14})
ax.axhline(SLO, color=MUTED, lw=1.4, ls="--")
ax.text(0.0, SLO + 18, "deadline %d ms" % SLO, color=MUTED, fontsize=10)
ax.set_ylim(0, 1000)

# C: drops at the highest load
ax = axes[1, 0]
hi = df[df["lam"] == max(LOADS)].sort_values("alpha")
bars = ax.bar([str(a) for a in hi["alpha"]], hi["drop_pct"], color=STYLE[max(LOADS)][0], width=0.62)
for rect, v, thr in zip(bars, hi["drop_pct"], hi["throughput_rps"]):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, "%.1f%%" % v, ha="center", color=INK, fontsize=10.5)
    ax.text(rect.get_x() + rect.get_width() / 2, -3.6, "serves\n%.0f req/s" % thr, ha="center", va="top", color=MUTED, fontsize=9)
ax.set_xlabel("alpha", labelpad=34)
ax.set_ylabel("jobs dropped (%)")
ax.set_ylim(0, 30)
ax.set_title("C  Dropped jobs at %d req/s (92%% of capacity)" % max(LOADS), loc="left", fontsize=12, fontweight="bold", color=INK)

# D: where the busy time went
ax = axes[1, 1]
clocks = sorted({int(c) for s in shares.values() for c in s})
ramp = ["#eaf2fb", "#bcd6ee", "#8dbbe0", "#5a9ad0", "#2f6fb5", "#0b2f66"]
assert len(clocks) <= len(ramp), clocks
x, ticks, labels, group_centres = 0.0, [], [], []
bottoms = {}
for lam in LOADS:
    start = x
    for a in ALPHAS:
        s = shares["%s_%s" % (float(lam), float(a))]
        b = 0.0
        for c, col in zip(clocks, ramp):
            v = s.get(str(c), 0.0)
            ax.bar(x, v, bottom=b, color=col, width=0.8, edgecolor="white", linewidth=0.8, label="%d MHz" % c if (lam == LOADS[0] and a == ALPHAS[0]) else None)
            b += v
        ticks.append(x)
        labels.append(("%g" % a))
        x += 1.0
    group_centres.append((start + x - 1) / 2.0)
    x += 0.9
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=9.5)
for gc, lam in zip(group_centres, LOADS):
    ax.text(gc, -0.17, "%d req/s" % lam, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10.5, color=INK, fontweight="bold")
ax.set_ylabel("share of GPU busy time")
ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False)
ax.set_title("D  GPU clock used (bars: alpha 0 to 1 within each load)", loc="left", fontsize=12, fontweight="bold", color=INK)
handles, labs = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labs[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, title="clock")

fig.tight_layout(h_pad=3.2, w_pad=2.5)
out = os.path.join(A, "dvfs_overview.png")
fig.savefig(out, dpi=170, facecolor="white")
print("wrote", out)

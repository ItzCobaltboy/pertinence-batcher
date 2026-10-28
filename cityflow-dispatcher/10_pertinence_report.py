# Reads what 09_pertinence_analysis.py produced for the FINAL split (S02, reported once) and puts the
# Pareto front next to the static policies: always-n/s/m/l, the oracle, and the best that mixing two static
# policies could do (upper hull of the static points). A front point above the hull beats any fixed mix.
# Also reports the routing split of every front point, to check the dispatcher is not collapsed.
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pertinence_config as pc
from common import report

MAX_SHARE, MIN_SHARE = 0.70, 0.05    # "balanced" check, applied after the search (not a search constraint)


def upper_hull(points):
    """Upper concave envelope of (cost, alpha) points, as a list sorted by cost."""
    points = sorted(points)
    hull = []
    for p in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross >= 0:       # hull[-1] lies on or below the line hull[-2] -> p
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def hull_value(hull, cost):
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    return float(np.interp(cost, xs, ys))


def main():
    out_dir = os.path.join(pc.RESULTS_DIR, "report")
    os.makedirs(out_dir, exist_ok=True)
    summary = pd.read_csv(pc.FINAL_VAL_SUMMARY_CSV)
    preds = pd.read_csv(pc.FINAL_VAL_PREDICTIONS_CSV)
    gt = pd.read_csv(pc.FINAL_VAL_GROUND_TRUTH_CSV)
    correct = gt[[f"{n}_correct" for n in pc.MODEL_NAMES]].values.astype(bool)
    label = gt["label"].to_numpy()
    cost = np.array(pc.MODEL_COST)

    static = []
    for k, name in enumerate(pc.MODEL_NAMES):
        static.append({"policy": "always_" + name, "alpha_sys": correct[:, k].mean(), "avg_gflops": cost[k]})
    static.append({"policy": "oracle", "alpha_sys": 1.0, "avg_gflops": cost[label].mean()})
    static = pd.DataFrame(static)
    hull = upper_hull([(r.avg_gflops, r.alpha_sys) for r in static.itertuples() if r.policy.startswith("always_")])

    rows = []
    for r in summary.itertuples():
        pred = preds[f"pred_{r.individual}"].to_numpy()
        share = np.bincount(pred, minlength=pc.NUM_CLASSES) / len(pred)
        balanced = bool(share.max() <= MAX_SHARE and share.min() >= MIN_SHARE)
        rows.append({"individual": r.individual, "alpha_sys": r.alpha_sys, "avg_gflops": r.avg_model_cost,
                     "gain_over_static_hull": r.alpha_sys - hull_value(hull, r.avg_model_cost),
                     "exact_match_vs_ideal": r.accuracy_exact_match_vs_ideal,
                     "share_n": share[0], "share_s": share[1], "share_m": share[2], "share_l": share[3],
                     "balanced": balanced})
    front = pd.DataFrame(rows).sort_values("avg_gflops").round(4)
    static = static.round(4)
    static.to_csv(os.path.join(out_dir, "static_policies.csv"), index=False)
    front.to_csv(os.path.join(out_dir, "front_vs_static.csv"), index=False)

    plt.figure(figsize=(6, 4.5))
    plt.scatter(front["avg_gflops"], front["alpha_sys"], s=18, label="PERTINENCE front (S02)")
    plt.plot([p[0] for p in hull], [p[1] for p in hull], "k--", lw=1, label="best mix of static policies")
    for r in static.itertuples():
        plt.scatter([r.avg_gflops], [r.alpha_sys], marker="x" if r.policy != "oracle" else "*", s=60)
        plt.annotate(r.policy.replace("always_yolov8", ""), (r.avg_gflops, r.alpha_sys), fontsize=8,
                     xytext=(3, 3), textcoords="offset points")
    plt.xlabel("avg GFLOPs per frame (dispatcher includes extractor)")
    plt.ylabel("alpha_sys")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(out_dir, "front_vs_static.png"), dpi=150, bbox_inches="tight")

    best = front.sort_values("gain_over_static_hull", ascending=False).iloc[0]
    balanced_front = front[front["balanced"]]
    extra = [f"Run: {pc.RUN_NAME}{' (SMOKE)' if pc.SMOKE else ''}; final split = S02 (reported once).",
             f"Front points: {len(front)}; above the static hull: {int((front['gain_over_static_hull'] > 0).sum())}; "
             f"balanced (max share <= {MAX_SHARE}, min share >= {MIN_SHARE}): {len(balanced_front)}.",
             f"Best gain over the static hull: {best['gain_over_static_hull']:+.4f} alpha_sys at "
             f"{best['avg_gflops']:.1f} GFLOPs (individual {int(best['individual'])}, shares "
             f"n/s/m/l {best['share_n']:.2f}/{best['share_s']:.2f}/{best['share_m']:.2f}/{best['share_l']:.2f}).",
             "", "Static policies on S02:", "", report.table_to_text(static)]
    shown = front.iloc[np.linspace(0, len(front) - 1, min(12, len(front))).astype(int)]
    report.write_summary(out_dir, "10 PERTINENCE front vs static policies (S02)",
                         [pc.FINAL_VAL_SUMMARY_CSV, pc.FINAL_VAL_PREDICTIONS_CSV, pc.FINAL_VAL_GROUND_TRUTH_CSV],
                         ["static_policies.csv", "front_vs_static.csv", "front_vs_static.png"], shown, [
                             "table shows 12 evenly spaced front points; all of them are in front_vs_static.csv",
                             "dispatcher cost includes extractor + FC; static policies do not (they need no dispatcher)",
                             "balanced rule is checked after the search, it is not a constraint inside NSGA-II"], extra)
    print("\n".join(extra))


if __name__ == "__main__":
    main()

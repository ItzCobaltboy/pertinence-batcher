# Part B step 2. Evaluates the chosen head ONCE on the test split: alpha_sys, GFLOPs and saving vs
# always-l, routing accuracy, under/over-estimate rates, confusion matrix, routing vs ideal split,
# baselines (always n/s/m/l, oracle), per-camera numbers.
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from common import extractor, head, paths, pool, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke
    letters = config.MODEL_LETTERS

    model_dir = os.path.join(paths.MODELS, "dispatcher" + ("_smoke" if smoke else ""))
    saved = torch.load(os.path.join(model_dir, "head.pt"), weights_only=False)
    with open(os.path.join(model_dir, "costs.json")) as f:
        costs = json.load(f)
    gflops = np.array([costs["model_gflops"][l] for l in letters])
    overhead = costs["extractor_gflops"]

    table = pool.pool_table("test", smoke)
    embedder = extractor.load_embedder()
    proba = head.predict_proba(saved["head"], pool.split_embeddings(table, embedder, smoke))
    pred = proba.argmax(1)
    ideal = table["ideal"].map(head.LETTER_INDEX).to_numpy()
    correct = np.stack([table[f"recall_{l}"].to_numpy() >= config.RECALL_THRESHOLD for l in letters], 1)
    rows_idx = np.arange(len(pred))

    def summarize(name, choice, with_overhead):
        alpha = correct[rows_idx, choice].mean()
        cost = gflops[choice].mean() + (overhead if with_overhead else 0.0)
        return {"policy": name, "alpha_sys": round(alpha, 4), "avg_gflops": round(cost, 2),
                "saving_vs_always_l": round(1 - cost / gflops[3], 4)}

    results = [summarize("dispatcher", pred, True)]
    for k, l in enumerate(letters):
        results.append(summarize(f"always_{l}", np.full(len(pred), k), False))
    results.append(summarize("oracle (ideal label)", ideal, False))
    results = pd.DataFrame(results)

    quality = pd.DataFrame([{"routing_accuracy": round((pred == ideal).mean(), 4),
                             "underestimate_rate": round((pred < ideal).mean(), 4),
                             "overestimate_rate": round((pred > ideal).mean(), 4)}])
    confusion = pd.crosstab(pd.Series(ideal, name="ideal"), pd.Series(pred, name="predicted")).reindex(
        index=range(4), columns=range(4), fill_value=0)
    confusion.index = confusion.columns = letters
    split_table = pd.DataFrame({"ideal": np.bincount(ideal, minlength=4) / len(ideal),
                                "routed": np.bincount(pred, minlength=4) / len(pred)}, index=letters).round(4)

    per_camera = []
    for camera, idx in table.groupby("camera").indices.items():
        per_camera.append({"camera": camera, "frames": len(idx),
                           "alpha_sys": round(correct[idx, pred[idx]].mean(), 4),
                           "avg_gflops": round(gflops[pred[idx]].mean() + overhead, 2),
                           "routing_accuracy": round((pred[idx] == ideal[idx]).mean(), 4),
                           **{f"routed_{l}": round((pred[idx] == k).mean(), 3) for k, l in enumerate(letters)},
                           **{f"ideal_{l}": round((ideal[idx] == k).mean(), 3) for k, l in enumerate(letters)}})
    per_camera = pd.DataFrame(per_camera)

    out_dir = paths.result_dir("06_evaluate_dispatcher", smoke)
    results.to_csv(os.path.join(out_dir, "policies.csv"), index=False)
    quality.to_csv(os.path.join(out_dir, "routing_quality.csv"), index=False)
    confusion.to_csv(os.path.join(out_dir, "confusion.csv"))
    split_table.to_csv(os.path.join(out_dir, "routing_vs_ideal_split.csv"))
    per_camera.to_csv(os.path.join(out_dir, "per_camera.csv"), index=False)

    plt.figure(figsize=(4.5, 4))
    plt.imshow(confusion.to_numpy(), cmap="Blues")
    plt.xticks(range(4), letters)
    plt.yticks(range(4), letters)
    plt.xlabel("predicted")
    plt.ylabel("ideal")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, confusion.iloc[i, j], ha="center", va="center")
    plt.savefig(os.path.join(out_dir, "confusion.png"), dpi=150, bbox_inches="tight")

    extra = [f"Head: scheme={saved['scheme']}, penalty={saved['penalty']}", "",
             report.table_to_text(quality), "", "Routing vs ideal split:", "",
             report.table_to_text(split_table.reset_index(names="model")), "",
             "Confusion (rows ideal, cols predicted):", "", report.table_to_text(confusion.reset_index(names="ideal")), "",
             "Dispatcher cost includes the extractor; always-X baselines do not."]
    report.write_summary(out_dir, "06 evaluate dispatcher (test split, evaluated once)",
                         [model_dir, "test split embeddings and pool table"],
                         ["policies.csv", "routing_quality.csv", "confusion.csv", "confusion.png",
                          "routing_vs_ideal_split.csv", "per_camera.csv"], results, None, extra)
    print(results.to_string(index=False))
    print(quality.to_string(index=False))
    print(split_table)
    print("saved", out_dir)


if __name__ == "__main__":
    main()

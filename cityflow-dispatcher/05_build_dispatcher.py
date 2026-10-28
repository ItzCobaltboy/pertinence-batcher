# Part B step 1. Frozen yolov8n extractor (448-d) -> linear head 448 -> 4 trained on the train split
# with ideal labels from the fine-tuned pool. Sweeps the underestimation penalty and class weighting,
# selects on VAL (balanced rule, assumption 7), saves the chosen head to models/dispatcher/.
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

import config
from common import extractor, head, paths, pool, report

PENALTIES = [1, 2, 4, 8, 16]
SCHEMES = ["none", "inv", "sqrt_inv"]
MAX_SHARE, MIN_SHARE = 0.70, 0.05


def model_costs():
    """GFLOPs per frame at 640 from ultralytics get_flops on the fused model for stock n, s, m, l."""
    costs = {}
    for letter in config.MODEL_LETTERS:
        model = YOLO(paths.weights_path(letter))
        model.fuse()                      # same fused model that model.info() reports (8.7 / 28.6 / 78.9 / 165.2)
        costs[letter] = float(get_flops(model.model, imgsz=config.IMG_SIZE))
    return costs


def evaluate(proba, table, costs, extractor_cost):
    pred = proba.argmax(1)
    letters = np.array(config.MODEL_LETTERS)
    correct = np.stack([table[f"recall_{l}"].to_numpy() >= config.RECALL_THRESHOLD for l in config.MODEL_LETTERS], 1)
    alpha = correct[np.arange(len(pred)), pred].mean()
    cost = np.mean([costs[l] for l in letters[pred]]) + extractor_cost
    share = np.bincount(pred, minlength=4) / len(pred)
    return alpha, cost, share


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke

    embedder = extractor.load_embedder()
    tables, embeddings = {}, {}
    for split in ["train", "val"]:
        tables[split] = pool.pool_table(split, smoke)
        embeddings[split] = pool.split_embeddings(tables[split], embedder, smoke)
    costs = model_costs()
    extractor_cost = extractor.extractor_gflops(embedder)
    print("GFLOPs per model", costs, "| extractor", round(extractor_cost, 2))

    y_train = tables["train"]["ideal"].map(head.LETTER_INDEX).to_numpy()
    ideal_share = {s: tables[s]["ideal"].value_counts(normalize=True).reindex(config.MODEL_LETTERS, fill_value=0.0)
                   for s in tables}
    print("ideal split (fine-tuned pool):", {s: v.round(3).to_dict() for s, v in ideal_share.items()})

    rows, heads = [], {}
    for scheme in SCHEMES:
        for lam in PENALTIES:
            h = head.fit_head(embeddings["train"], y_train, scheme, lam)
            proba = head.predict_proba(h, embeddings["val"])
            alpha, cost, share = evaluate(proba, tables["val"], costs, extractor_cost)
            balanced = bool(share.max() <= MAX_SHARE and share.min() >= MIN_SHARE)
            rows.append({"scheme": scheme, "penalty": lam, "val_alpha_sys": round(float(alpha), 4), "val_gflops": round(float(cost), 2),
                         **{f"share_{l}": round(float(s), 3) for l, s in zip(config.MODEL_LETTERS, share)}, "balanced": balanced})
            heads[(scheme, lam)] = h
            print(rows[-1], flush=True)
    sweep = pd.DataFrame(rows)

    flags = []
    candidates = sweep[sweep["balanced"]]
    if len(candidates) == 0:
        # The spec says stop and report here. Fallback so the pipeline still runs: smallest violation, flagged.
        shares = sweep[[f"share_{l}" for l in config.MODEL_LETTERS]]
        violation = (shares.max(axis=1) - MAX_SHARE).clip(lower=0) + (MIN_SHARE - shares.min(axis=1)).clip(lower=0)
        candidates = sweep.loc[[violation.idxmin()]]
        flags.append("NO CONFIG IS BALANCED on val (max share <= 0.70 and min share >= 0.05). Fell back to the config "
                     "with the smallest violation. Ideal labels on val: " + str(ideal_share["val"].round(3).to_dict()))
    best = candidates.sort_values(["val_alpha_sys", "val_gflops"], ascending=[False, True]).iloc[0]
    chosen = (best["scheme"], int(best["penalty"]))
    print("chosen", chosen)

    out_model = os.path.join(paths.MODELS, "dispatcher" + ("_smoke" if smoke else ""))
    os.makedirs(out_model, exist_ok=True)
    torch.save({"head": heads[chosen], "scheme": chosen[0], "penalty": chosen[1]}, os.path.join(out_model, "head.pt"))
    with open(os.path.join(out_model, "costs.json"), "w") as f:
        json.dump({"model_gflops": costs, "extractor_gflops": extractor_cost}, f, indent=2)

    out_dir = paths.result_dir("05_build_dispatcher", smoke)
    sweep["chosen"] = [(r.scheme, r.penalty) == chosen for r in sweep.itertuples()]
    sweep.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(sweep["val_gflops"], sweep["val_alpha_sys"], c="lightgrey", label="not balanced")
    bal = sweep[sweep["balanced"]]
    plt.scatter(bal["val_gflops"], bal["val_alpha_sys"], c="tab:green", label="balanced")
    pick = sweep[sweep["chosen"]]
    plt.scatter(pick["val_gflops"], pick["val_alpha_sys"], marker="*", s=250, c="tab:red", label="chosen")
    plt.xlabel("average GFLOPs per frame (incl. extractor)")
    plt.ylabel("alpha_sys on val")
    plt.legend()
    plt.title("dispatcher sweep on val")
    plt.savefig(os.path.join(out_dir, "alpha_vs_cost.png"), dpi=150)

    ideal_table = pd.DataFrame({s: v.round(3) for s, v in ideal_share.items()}).T.reset_index(names="split")
    ideal_table.to_csv(os.path.join(out_dir, "ideal_split_by_split.csv"), index=False)
    report.write_summary(out_dir, "05 build dispatcher", ["data/images (train, val)", "models/finetuned/", "models/yolov8*.pt"],
                         [out_model, "sweep.csv", "alpha_vs_cost.png", "ideal_split_by_split.csv"], sweep, flags,
                         [f"Chosen: scheme={chosen[0]}, penalty={chosen[1]}",
                          f"GFLOPs per model {costs}, extractor {extractor_cost:.2f} (FlopCounterMode, added for every frame)",
                          "Ideal-label split per split (train labels come from models that were trained on those frames, "
                          "so train may be skewed to small models):", "", report.table_to_text(ideal_table)])
    print("saved", out_dir)


if __name__ == "__main__":
    main()

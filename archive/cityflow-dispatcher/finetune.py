"""
Fine-tunes every COCO Pareto head of yolo-dispatcher on CityFlow labels and
evaluates before/after on held-out CityFlow scenarios.

For each COCO Pareto point:
  - start from its trained head (W, b) instead of a fresh init;
  - keep its chromosome: same penalty matrix and weighting scheme, class
    weights recomputed on the CityFlow train labels;
  - train FT_EPOCHS with the shared penalized loss on TRAIN_SCENARIOS data,
    seeded from the chromosome like every other retraining in this repo.

Data comes from label.py (data/labels, data/embeddings). Train and eval
scenarios never overlap (cf_config asserts it, and this script re-checks).

Outputs mirror yolo-dispatcher's results/ layout, so
`cityflow-routing/route.py --yolo-results ../cityflow-dispatcher/results`
routes video with the fine-tuned heads unchanged:
  results/nsga2/pareto_front.csv             copy of the COCO front (same chromosomes)
  results/eval/model_cache/individual_<id>.npz   fine-tuned W, b, chromosome, scheme
  results/eval/test_summary.csv              fine-tuned, on TRAIN scenarios (in-sample)
  results/eval/final_val_summary.csv         fine-tuned, on EVAL scenarios (held out)
  results/eval/coco_heads_eval_summary.csv   COCO heads unchanged, on EVAL scenarios
  results/eval/baselines_eval.csv            always-n/s/m/l and the oracle, on EVAL
  results/eval/label_distribution.csv, plots/, summary.md, finetune_manifest.json

alpha_sys = fraction of frames where the dispatched model is correct (box
recall vs yolov8l >= RECALL_THRESHOLD). Cost = mean GFLOPs of the dispatched
model + extractor/FC overhead (yolo-dispatcher's measured costs, Eq. 4).

Run: python finetune.py
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

import cf_config as C

LETTERS = [name[-1] for name in C.MODEL_NAMES]


# ---------------------------------------------------------------- data

def load_role(data_dir, role):
    """All labelled frames of one role. Returns (table, embeddings float32
    [n, 448], recalls [n, 4]) in the same row order."""
    tables, embeddings = [], []
    for labels_path in sorted(glob.glob(os.path.join(data_dir, "labels", "*.csv.gz"))):
        table = pd.read_csv(labels_path)
        if table.empty or table["role"].iloc[0] != role:
            continue
        stem = os.path.basename(labels_path)[: -len(".csv.gz")]
        saved = np.load(os.path.join(data_dir, "embeddings", f"{stem}.npz"))
        if not np.array_equal(saved["frame_idx"], table["frame_idx"].values):
            sys.exit(f"{stem}: embeddings and labels are out of sync; delete both and rerun label.py")
        tables.append(table)
        embeddings.append(saved["embeddings"].astype(np.float32))
    if not tables:
        sys.exit(f"No '{role}' data in {data_dir}/labels: run label.py first")
    table = pd.concat(tables, ignore_index=True)
    recalls = table[[f"recall_{l}" for l in LETTERS]].values.astype(np.float64)
    return table, np.concatenate(embeddings), recalls


# ---------------------------------------------------------------- heads

def load_coco_points(coco_results):
    """[(point_id, W, b, chromosome, scheme, weights_path, sha256)] from
    yolo-dispatcher's results/, via cityflow-routing's loader."""
    points, heads = C.cf_heads.load_pareto(coco_results)
    out = []
    for row in points.itertuples():
        saved = np.load(row.weights_path)
        chromosome = saved["chromosome"].astype(np.float64)
        if "scheme" in saved.files:
            scheme = str(saved["scheme"])
        else:
            from weighting_scheme import decode_scheme
            scheme = decode_scheme(chromosome, C.yd)
        W, b = heads[row.point_id]
        out.append((int(row.point_id), W, b, chromosome, scheme, row.weights_path, row.weights_sha256))
    return out


def finetune_head(W, b, embeddings, labels, penalty_matrix, class_weights, seed, device,
                  epochs, learning_rate, batch_size):
    """Continues training a Linear(448, 4) from (W, b) with the shared
    penalized loss. Returns the new (W, b) as numpy."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from loss import penalized_loss   # dispatcher_analysis/loss.py

    torch.manual_seed(seed)
    fc = nn.Linear(W.shape[1], W.shape[0]).to(device)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(np.ascontiguousarray(W)))
        fc.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)))
    loader = DataLoader(TensorDataset(torch.from_numpy(embeddings), torch.from_numpy(labels)),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(fc.parameters(), lr=learning_rate)
    fc.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = penalized_loss(fc(x), y, penalty_matrix, class_weights)
            loss.backward()
            optimizer.step()
    return fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()


# ---------------------------------------------------------------- metrics

def cost_per_model():
    return np.array(C.yd.MODEL_COST, dtype=np.float64), float(C.yd.DISPATCHER_OVERHEAD_COST or 0.0)


def evaluate_choice(chosen, recalls, ideal, with_overhead=True):
    model_cost, overhead = cost_per_model()
    correct = recalls[np.arange(len(chosen)), chosen] >= C.RECALL_THRESHOLD
    row = {
        "alpha_sys": float(correct.mean()),
        "avg_model_cost": float(model_cost[chosen].mean() + (overhead if with_overhead else 0.0)),
        "routing_accuracy": float((chosen == ideal).mean()),
        "underestimate_rate": float((chosen < ideal).mean()),
        "overestimate_rate": float((chosen > ideal).mean()),
    }
    for i, name in enumerate(C.MODEL_NAMES):
        row[f"frac_{name}"] = float((chosen == i).mean())
    return row


def evaluate_head(W, b, embeddings, recalls, ideal):
    chosen = np.argmax(embeddings @ W.T + b, axis=1)
    return evaluate_choice(chosen, recalls, ideal)


def baselines(recalls, ideal):
    """Single-model policies (no dispatcher, so no overhead) and the oracle
    (cheapest correct per frame, with overhead as if a perfect dispatcher ran)."""
    rows = []
    for i, name in enumerate(C.MODEL_NAMES):
        rows.append({"policy": f"always_{name}",
                     **evaluate_choice(np.full(len(ideal), i), recalls, ideal, with_overhead=False)})
    rows.append({"policy": "oracle", **evaluate_choice(ideal, recalls, ideal)})
    return pd.DataFrame(rows)


def nondominated(df):
    """Mask of rows not dominated in (higher alpha_sys, lower avg_model_cost)."""
    a = df["alpha_sys"].values
    c = df["avg_model_cost"].values
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dominated = (a >= a[i]) & (c <= c[i]) & ((a > a[i]) | (c < c[i]))
        keep[i] = not dominated.any()
    return keep


# ---------------------------------------------------------------- outputs

def plot_front(coco_eval, ft_eval, base, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(coco_eval["avg_model_cost"], coco_eval["alpha_sys"], s=18, alpha=0.6,
               label="COCO heads (unchanged)")
    ax.scatter(ft_eval["avg_model_cost"], ft_eval["alpha_sys"], s=18, marker="x",
               label="fine-tuned on CityFlow")
    for _, row in base.iterrows():
        marker = "*" if row["policy"] == "oracle" else "s"
        ax.scatter(row["avg_model_cost"], row["alpha_sys"], marker=marker, s=80, color="black")
        ax.annotate(row["policy"].replace("always_", ""), (row["avg_model_cost"], row["alpha_sys"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("average cost per frame, GFLOPs")
    ax.set_ylabel("alpha_sys (dispatched model correct)")
    ax.set_title(f"Held-out CityFlow {', '.join(C.EVAL_SCENARIOS)}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_labels(dist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bottom = np.zeros(len(dist))
    labels = [f"{s} ({r})" for s, r in zip(dist["scenario"], dist["role"])]
    for name in C.MODEL_NAMES:
        values = dist[name].values
        ax.bar(labels, values, bottom=bottom, label=name)
        bottom += values
    ax.set_ylabel("fraction of frames")
    ax.set_title(f"Ideal label (cheapest model with box recall >= {C.RECALL_THRESHOLD} vs {C.REFERENCE_MODEL})")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path, n_train, n_eval, dist, coco_eval, ft_eval, base, best_id):
    def fmt_split(row):
        return " / ".join(f"{row[f'frac_{m}']:.0%}" for m in C.MODEL_NAMES)

    ft = ft_eval.set_index("individual").loc[best_id]
    coco = coco_eval.set_index("individual").loc[best_id]
    oracle = base.set_index("policy").loc["oracle"]
    lines = [
        "# CityFlow fine-tuning summary",
        "",
        f"- Train scenarios {', '.join(C.TRAIN_SCENARIOS)}: {n_train} frames. "
        f"Eval scenarios {', '.join(C.EVAL_SCENARIOS)}: {n_eval} frames (held out).",
        f"- Correct = box recall vs {C.REFERENCE_MODEL} >= {C.RECALL_THRESHOLD} at IoU {C.MATCH_IOU}, "
        f"classes {C.DETECT_CLASSES}, every {C.FRAME_STRIDE}th frame.",
        f"- Fine-tuning: {C.FT_EPOCHS} epochs, lr {C.FT_LEARNING_RATE}, from each COCO head, same chromosome.",
        "",
        "## Ideal label split (n / s / m / l)",
        "",
    ]
    for _, row in dist.iterrows():
        lines.append(f"- {row['scenario']} ({row['role']}, {int(row['frames'])} frames): "
                     + " / ".join(f"{row[m]:.0%}" for m in C.MODEL_NAMES))
    lines += [
        "",
        f"## Point {best_id} (highest held-out alpha_sys after fine-tuning)",
        "",
        "| heads | alpha_sys | GFLOPs/frame | routed n / s / m / l |",
        "|---|---|---|---|",
        f"| COCO | {coco['alpha_sys']:.3f} | {coco['avg_model_cost']:.1f} | {fmt_split(coco)} |",
        f"| fine-tuned | {ft['alpha_sys']:.3f} | {ft['avg_model_cost']:.1f} | {fmt_split(ft)} |",
        f"| oracle | {oracle['alpha_sys']:.3f} | {oracle['avg_model_cost']:.1f} | {fmt_split(oracle)} |",
        "",
        f"- Fine-tuned points non-dominated on held-out data: {int(ft_eval['nondominated'].sum())} of {len(ft_eval)}.",
        "- The fine-tuned set reuses the COCO chromosomes; it is not a re-searched Pareto front.",
        "",
        "Single-model baselines on held-out data:",
        "",
    ]
    for _, row in base.iterrows():
        if row["policy"] != "oracle":
            lines.append(f"- {row['policy']}: alpha_sys {row['alpha_sys']:.3f}, {row['avg_model_cost']:.1f} GFLOPs")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-results", default=C.COCO_RESULTS_DIR,
                        help="yolo-dispatcher's results/ (nsga2/ and eval/)")
    parser.add_argument("--data-dir", default=C.DATA_DIR, help="label.py output")
    parser.add_argument("--results-dir", default=C.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=C.FT_EPOCHS)
    parser.add_argument("--lr", type=float, default=C.FT_LEARNING_RATE)
    parser.add_argument("--batch", type=int, default=C.FT_BATCH_SIZE)
    args = parser.parse_args(argv)

    import torch
    from penalty_matrix import build_penalty_matrix
    from class_weights import compute_class_weights
    from train_fc import chromosome_seed

    if C.yd.DISPATCHER_OVERHEAD_COST is None:
        sys.exit("yolo-dispatcher/data/model_costs.json is missing: run yolo-dispatcher/measure_costs.py")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_table, train_emb, train_recalls = load_role(args.data_dir, "train")
    eval_table, eval_emb, eval_recalls = load_role(args.data_dir, "eval")
    overlap = set(train_table["scenario"]) & set(eval_table["scenario"])
    if overlap:
        sys.exit(f"scenarios {sorted(overlap)} are in both train and eval data")
    train_labels = train_table["ideal_label"].values.astype(np.int64)
    eval_ideal = eval_table["ideal_label"].values.astype(np.int64)
    print(f"train: {len(train_table)} frames from {sorted(set(train_table['scenario']))}; "
          f"eval: {len(eval_table)} frames from {sorted(set(eval_table['scenario']))}; device {device}")

    nsga2_dir = os.path.join(args.results_dir, "nsga2")
    eval_dir = os.path.join(args.results_dir, "eval")
    cache_dir = os.path.join(eval_dir, "model_cache")
    plots_dir = os.path.join(eval_dir, "plots")
    for d in (nsga2_dir, cache_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    shutil.copyfile(os.path.join(args.coco_results, "nsga2", "pareto_front.csv"),
                    os.path.join(nsga2_dir, "pareto_front.csv"))

    coco_points = load_coco_points(args.coco_results)
    print(f"fine-tuning {len(coco_points)} COCO Pareto heads, {args.epochs} epochs, lr {args.lr}")
    coco_rows, ft_train_rows, ft_eval_rows = [], [], []
    for k, (point_id, W, b, chromosome, scheme, weights_path, sha) in enumerate(coco_points):
        penalty = build_penalty_matrix(chromosome, C.yd)
        class_weights = compute_class_weights(train_labels, scheme, C.yd)
        W_ft, b_ft = finetune_head(W, b, train_emb, train_labels, penalty, class_weights,
                                   chromosome_seed(chromosome), device, args.epochs, args.lr, args.batch)
        np.savez(os.path.join(cache_dir, f"individual_{point_id}.npz"), W=W_ft, b=b_ft,
                 chromosome=chromosome, scheme=scheme, init_weights=weights_path, init_sha256=sha)
        coco_rows.append({"individual": point_id, **evaluate_head(W, b, eval_emb, eval_recalls, eval_ideal)})
        ft_train_rows.append({"individual": point_id,
                              **evaluate_head(W_ft, b_ft, train_emb, train_recalls, train_labels)})
        ft_eval_rows.append({"individual": point_id, **evaluate_head(W_ft, b_ft, eval_emb, eval_recalls, eval_ideal)})
        if (k + 1) % 10 == 0 or k == len(coco_points) - 1:
            print(f"  {k + 1}/{len(coco_points)}")

    coco_eval = pd.DataFrame(coco_rows)
    ft_train = pd.DataFrame(ft_train_rows)
    ft_eval = pd.DataFrame(ft_eval_rows)
    ft_eval["nondominated"] = nondominated(ft_eval)
    base = baselines(eval_recalls, eval_ideal)

    coco_eval.to_csv(os.path.join(eval_dir, "coco_heads_eval_summary.csv"), index=False)
    ft_train.to_csv(os.path.join(eval_dir, "test_summary.csv"), index=False)      # in-sample (train scenarios)
    ft_eval.to_csv(os.path.join(eval_dir, "final_val_summary.csv"), index=False)  # held out (eval scenarios)
    base.to_csv(os.path.join(eval_dir, "baselines_eval.csv"), index=False)

    both = pd.concat([train_table, eval_table], ignore_index=True)
    dist = (both.groupby(["scenario", "role"])["ideal_model"].value_counts(normalize=True)
            .unstack(fill_value=0).reindex(columns=C.MODEL_NAMES, fill_value=0).reset_index())
    sizes = both.groupby(["scenario", "role"]).size().rename("frames").reset_index()
    dist = dist.merge(sizes, on=["scenario", "role"])
    dist.to_csv(os.path.join(eval_dir, "label_distribution.csv"), index=False)

    plot_front(coco_eval, ft_eval, base, os.path.join(plots_dir, "front_eval.png"))
    plot_labels(dist, os.path.join(plots_dir, "label_split.png"))
    best_id = int(ft_eval.sort_values(["alpha_sys", "avg_model_cost"], ascending=[False, True])["individual"].iloc[0])
    write_summary(os.path.join(eval_dir, "summary.md"), len(train_table), len(eval_table),
                  dist, coco_eval, ft_eval, base, best_id)

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "coco_results": os.path.abspath(args.coco_results),
        "coco_pareto_front_sha256": file_sha256(os.path.join(args.coco_results, "nsga2", "pareto_front.csv")),
        "data_dir": os.path.abspath(args.data_dir),
        "train_scenarios": sorted(set(train_table["scenario"])), "eval_scenarios": sorted(set(eval_table["scenario"])),
        "train_frames": len(train_table), "eval_frames": len(eval_table),
        "epochs": args.epochs, "learning_rate": args.lr, "batch_size": args.batch,
        "reference_model": C.REFERENCE_MODEL, "recall_threshold": C.RECALL_THRESHOLD, "match_iou": C.MATCH_IOU,
        "torch": torch.__version__, "device": str(device),
    }
    with open(os.path.join(eval_dir, "finetune_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. See {os.path.join(eval_dir, 'summary.md')}")


if __name__ == "__main__":
    main()

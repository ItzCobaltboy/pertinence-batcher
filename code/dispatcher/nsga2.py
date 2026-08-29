"""
NSGA-II search — Step 2 of the dispatcher pipeline.

Evolves the penalty matrix P and class-weighting exponent α jointly to trace
the Pareto front of (routing accuracy, avg FLOPs per image).

Chromosome: 13 floats
  [p01, p02, p03,       ← true=0 overestimation penalties
   p10, p12, p13,       ← true=1 (under p10, over p12/p13)
   p20, p21, p23,       ← true=2
   p30, p31, p32,       ← true=3 underestimation penalties
   α]                   ← weighting exponent: weight_i ∝ 1/count_i^α
                           α=0 → uniform, α=0.5 ≈ ISNS, α=1.0 = INS

Objectives (both minimised):
  obj1 = underestimation_rate  (predicted cheaper model than needed → accuracy risk)
  obj2 = avg_flops_G           (mean GFLOPs of routed model per image)

Key optimisation: backbone is frozen → precompute all embeddings once,
then each individual only trains/evals a Linear(512→4) on in-memory tensors.
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

CLEANED_CSV    = "./results/cleaned-data/dispatcher_labels_clean.csv"
OUTPUT_DIR     = "./results/nsga2/"
LOG_DIR        = "./results/nsga2logs/"

POP_SIZE       = 50
GENERATIONS    = 50
FC_EPOCHS      = 30       # inner FC training epochs per individual
BATCH_SIZE     = 128
LR             = 1e-3

# SBX crossover
SBX_ETA        = 20
SBX_PROB       = 0.9

# Polynomial mutation
PM_ETA         = 25

# Chromosome bounds: 12 penalty values [0, 5] + 1 alpha [0, 1]
N_GENES        = 13
LOWER          = np.array([0.0] * 12 + [0.0])
UPPER          = np.array([5.0] * 12 + [1.0])

MODELS         = ["resnet18", "resnet34", "resnet50", "resnet152"]
FLOPS_G        = np.array([1.824, 3.679, 4.134, 11.604])
NUM_CLASSES    = 4

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"nsga2_{run_id}.log")
    logger   = logging.getLogger("nsga2")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh  = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.info(f"Log → {log_file}")
    return logger


# ── Chromosome helpers ────────────────────────────────────────────────────────

def decode_chromosome(chrom: np.ndarray) -> tuple[torch.Tensor, float]:
    """
    Returns (penalty 4×4 tensor, alpha float).
    Off-diagonal layout:
      chrom[0..2]  → row 0 cols 1,2,3
      chrom[3..5]  → row 1 cols 0,2,3
      chrom[6..8]  → row 2 cols 0,1,3
      chrom[9..11] → row 3 cols 0,1,2
      chrom[12]    → alpha
    """
    P = torch.zeros(4, 4, dtype=torch.float32)
    idx = 0
    for row in range(4):
        for col in range(4):
            if row != col:
                P[row, col] = float(chrom[idx])
                idx += 1
    alpha = float(chrom[12])
    return P, alpha


def compute_sample_weights(labels: np.ndarray, alpha: float) -> np.ndarray:
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts  = np.clip(counts, 1, None)
    class_w = 1.0 / (counts ** alpha) if alpha > 0 else np.ones(NUM_CLASSES)
    class_w = class_w / class_w.sum() * NUM_CLASSES
    return class_w[labels]


# ── Backbone + embedding precomputation ──────────────────────────────────────

class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.transform(img), int(row["label"])


def precompute_embeddings(
    cleaned_csv: str,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run all images through frozen ResNet18 backbone once → (N,512) embeddings."""
    df  = pd.read_csv(cleaned_csv)
    ds  = _ImageDataset(df, TRANSFORM)
    ldr = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    feat     = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    for p in feat.parameters():
        p.requires_grad = False

    embs, lbls = [], []
    logger.info(f"Precomputing embeddings for {len(df)} images...")
    with torch.no_grad():
        for imgs, labels in ldr:
            out = feat(imgs.to(device)).squeeze(-1).squeeze(-1)  # (B,512)
            embs.append(out.cpu())
            lbls.append(labels)

    embs = torch.cat(embs, dim=0)   # (N, 512)
    lbls = torch.cat(lbls, dim=0)   # (N,)
    logger.info(f"Embeddings shape: {embs.shape}")

    # Explicitly free backbone and loader workers before the GA loop starts
    del feat, ldr
    torch.cuda.empty_cache()

    return embs, lbls


# ── Fitness evaluation ────────────────────────────────────────────────────────

class PenalizedCE(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        preds    = logits.argmax(dim=1)
        ce_loss  = self.ce(logits, targets)
        penalty  = self.P[targets, preds]
        is_wrong = (preds != targets).float()
        return (ce_loss * penalty * is_wrong).mean()


def evaluate_individual(
    chrom: np.ndarray,
    embs: torch.Tensor,
    lbls: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train Linear(512→4) on precomputed embeddings, then evaluate.
    Returns (obj1=1-acc, obj2=avg_flops_G).
    """
    P, alpha = decode_chromosome(chrom)
    labels_np = lbls.numpy()

    sample_w = compute_sample_weights(labels_np, alpha)
    sampler  = WeightedRandomSampler(
        torch.from_numpy(sample_w).float(),
        num_samples=len(lbls),
        replacement=True,
    )

    ds_train = TensorDataset(embs, lbls)
    ldr_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler)

    fc        = nn.Linear(512, NUM_CLASSES).to(device)
    criterion = PenalizedCE(P).to(device)
    optimizer = optim.Adam(fc.parameters(), lr=LR)

    fc.train()
    for _ in range(FC_EPOCHS):
        for e, t in ldr_train:
            e, t = e.to(device), t.to(device)
            optimizer.zero_grad()
            criterion(fc(e), t).backward()
            optimizer.step()

    # Evaluate on full dataset (no sampler)
    fc.eval()
    ldr_eval = DataLoader(TensorDataset(embs, lbls), batch_size=256, shuffle=False)
    preds_all = []
    with torch.no_grad():
        for e, _ in ldr_eval:
            preds_all.append(fc(e.to(device)).argmax(1).cpu())
    preds_all = torch.cat(preds_all).numpy()

    avg_flops = FLOPS_G[preds_all].mean()

    # obj1 = underestimation rate: predicted cheaper model than needed → accuracy risk
    under_rate = (preds_all < labels_np).mean()

    # Explicitly free all CUDA tensors — critical over 2500 iterations
    del fc, criterion, optimizer, ldr_train, ldr_eval
    torch.cuda.empty_cache()

    return float(under_rate), float(avg_flops)


# ── NSGA-II operators ─────────────────────────────────────────────────────────

def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_set    = [[] for _ in range(n)]
    fronts           = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            o_i, o_j = objectives[i], objectives[j]
            i_dom_j  = np.all(o_i <= o_j) and np.any(o_i < o_j)
            j_dom_i  = np.all(o_j <= o_i) and np.any(o_j < o_i)
            if i_dom_j:
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_set[j].append(i)
                domination_count[i] += 1

    fronts[0] = [i for i in range(n) if domination_count[i] == 0]
    current   = fronts[0]
    while current:
        next_front = []
        for i in current:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current = next_front

    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for individuals in a front."""
    n    = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist

    for m in range(objectives.shape[1]):
        vals  = objectives[front, m]
        order = np.argsort(vals)
        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        rng = vals[order[-1]] - vals[order[0]]
        if rng == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng

    return dist


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    eta: float = SBX_ETA,
    prob: float = SBX_PROB,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover."""
    if np.random.rand() > prob:
        return p1.copy(), p2.copy()

    c1, c2 = p1.copy(), p2.copy()
    for i in range(N_GENES):
        if np.random.rand() < 0.5 or abs(p1[i] - p2[i]) < 1e-10:
            continue
        lo, hi = LOWER[i], UPPER[i]
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = np.random.rand()

        beta_q = lambda u, y, bound, sign: (
            (2 * u + (1 - 2 * u) * (1 - sign * (y - bound) / (y2 - y1)) ** (eta + 1))
            ** (1 / (eta + 1))
        )
        b1 = (2 * np.random.rand()) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        b2 = (2 * np.random.rand()) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))

        c1[i] = np.clip(0.5 * ((p1[i] + p2[i]) - b1 * abs(p2[i] - p1[i])), lo, hi)
        c2[i] = np.clip(0.5 * ((p1[i] + p2[i]) + b2 * abs(p2[i] - p1[i])), lo, hi)

    return c1, c2


def polynomial_mutation(
    ind: np.ndarray,
    eta: float = PM_ETA,
    prob: float = None,
) -> np.ndarray:
    """Polynomial mutation."""
    if prob is None:
        prob = 1.0 / N_GENES
    out = ind.copy()
    for i in range(N_GENES):
        if np.random.rand() > prob:
            continue
        lo, hi = LOWER[i], UPPER[i]
        delta  = hi - lo
        u      = np.random.rand()
        if u < 0.5:
            delta_q = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        out[i] = np.clip(ind[i] + delta_q * delta, lo, hi)
    return out


def tournament_select(
    objectives: np.ndarray,
    ranks: np.ndarray,
    distances: np.ndarray,
) -> int:
    """Binary tournament selection."""
    a, b = np.random.choice(len(objectives), 2, replace=False)
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    return a if distances[a] >= distances[b] else b


# ── Main NSGA-II loop ─────────────────────────────────────────────────────────

def run_nsga2(
    cleaned_csv: str  = CLEANED_CSV,
    output_dir: str   = OUTPUT_DIR,
    pop_size: int     = POP_SIZE,
    generations: int  = GENERATIONS,
    fc_epochs: int    = FC_EPOCHS,
):
    log    = setup_logging()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  Pop: {pop_size}  |  Gen: {generations}  |  FC epochs: {fc_epochs}")

    os.makedirs(output_dir, exist_ok=True)

    # Precompute embeddings once
    embs, lbls = precompute_embeddings(cleaned_csv, device, log)

    # Initialise population randomly
    population  = np.random.uniform(LOWER, UPPER, size=(pop_size, N_GENES))
    objectives  = np.zeros((pop_size, 2))

    log.info(f"\nEvaluating initial population ({pop_size} individuals)...")
    t0 = time.time()
    for i, chrom in enumerate(population):
        objectives[i] = evaluate_individual(chrom, embs, lbls, device)
        log.info(f"  [{i+1:>3}/{pop_size}]  under={objectives[i,0]:.4f}  flops={objectives[i,1]:.3f}G")
    log.info(f"Initial population done in {time.time()-t0:.0f}s")

    for gen in range(1, generations + 1):
        t_gen = time.time()
        log.info(f"\n── Generation {gen}/{generations} ──")

        # Compute ranks and crowding distances
        fronts  = non_dominated_sort(objectives)
        ranks   = np.zeros(pop_size, dtype=int)
        dists   = np.zeros(pop_size)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            cd = crowding_distance(objectives, front)
            for k, idx in enumerate(front):
                dists[idx] = cd[k]

        # Generate offspring
        offspring_pop = []
        offspring_obj = []
        while len(offspring_pop) < pop_size:
            p1_idx = tournament_select(objectives, ranks, dists)
            p2_idx = tournament_select(objectives, ranks, dists)
            c1, c2 = sbx_crossover(population[p1_idx], population[p2_idx])
            c1     = polynomial_mutation(c1)
            c2     = polynomial_mutation(c2)
            for child in [c1, c2]:
                if len(offspring_pop) < pop_size:
                    obj = evaluate_individual(child, embs, lbls, device)
                    offspring_pop.append(child)
                    offspring_obj.append(obj)
                    log.info(f"  offspring [{len(offspring_pop):>3}/{pop_size}]  "
                             f"under={obj[0]:.4f}  flops={obj[1]:.3f}G")

        # Combine parent + offspring, select next generation
        combined_pop = np.vstack([population, offspring_pop])
        combined_obj = np.vstack([objectives, offspring_obj])

        all_fronts    = non_dominated_sort(combined_obj)
        new_pop_idx   = []
        new_ranks     = np.zeros(len(combined_pop), dtype=int)
        new_dists     = np.zeros(len(combined_pop))
        for rank, front in enumerate(all_fronts):
            for idx in front:
                new_ranks[idx] = rank
            cd = crowding_distance(combined_obj, front)
            for k, idx in enumerate(front):
                new_dists[idx] = cd[k]

        for front in all_fronts:
            if len(new_pop_idx) + len(front) <= pop_size:
                new_pop_idx.extend(front)
            else:
                remaining = pop_size - len(new_pop_idx)
                sorted_front = sorted(front, key=lambda i: -new_dists[i])
                new_pop_idx.extend(sorted_front[:remaining])
                break

        population = combined_pop[new_pop_idx]
        objectives = combined_obj[new_pop_idx]

        # Log Pareto front stats this generation
        fronts_new     = non_dominated_sort(objectives)
        pareto_indices = fronts_new[0]
        pareto_obj     = objectives[pareto_indices]
        best_under     = pareto_obj[:, 0].min()
        best_flops     = pareto_obj[:, 1].min()
        log.info(f"Gen {gen} done in {time.time()-t_gen:.0f}s  |  "
                 f"Pareto size={len(pareto_indices)}  "
                 f"best_under={best_under:.3f}  best_low_flops={best_flops:.3f}G")

        # Save checkpoint every 5 generations
        if gen % 5 == 0 or gen == generations:
            _save_checkpoint(population, objectives, gen, output_dir, log)

    # Final Pareto front
    final_fronts = non_dominated_sort(objectives)
    pareto_idx   = final_fronts[0]
    log.info(f"\nFinal Pareto front: {len(pareto_idx)} individuals")
    _save_pareto(population[pareto_idx], objectives[pareto_idx], output_dir, log)

    return population[pareto_idx], objectives[pareto_idx]


def _save_checkpoint(population, objectives, gen, output_dir, log):
    path = os.path.join(output_dir, f"checkpoint_gen{gen:03d}.npz")
    np.savez(path, population=population, objectives=objectives, gen=gen)
    log.info(f"Checkpoint → {path}")


def _save_pareto(pareto_pop, pareto_obj, output_dir, log):
    rows = []
    for i, (chrom, obj) in enumerate(zip(pareto_pop, pareto_obj)):
        P, alpha = decode_chromosome(chrom)
        rows.append({
            "individual": i,
            "underestimation_rate": round(obj[0], 4),
            "avg_flops_G":         round(obj[1], 4),
            "alpha":            round(alpha, 4),
            **{f"P_{r}{c}": round(float(P[r, c]), 4)
               for r in range(4) for c in range(4) if r != c},
        })
    df   = pd.DataFrame(rows).sort_values("underestimation_rate", ascending=True)
    path = os.path.join(output_dir, "pareto_front.csv")
    df.to_csv(path, index=False)
    log.info(f"Pareto front CSV → {path}")
    log.info("\nPareto front (accuracy vs FLOPs):")
    for _, row in df.iterrows():
        log.info(f"  under={row['underestimation_rate']:.3f}  flops={row['avg_flops_G']:.3f}G  alpha={row['alpha']:.3f}")


if __name__ == "__main__":
    run_nsga2()

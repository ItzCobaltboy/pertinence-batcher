# dispatcher_analysis

Step 3: evaluation only of whatever Pareto front `../dispatcher/` last produced.

**Does**: for every Pareto individual, retrains its FC head directly from its
chromosome (never from a copied weight cache — see below), predicts on train and a
held-out val split, and computes `alpha_sys`, `avg_flops_G`, and per-class
recall/precision on both splits. Produces a Pareto scatter and confusion matrices.
**Does not**: run any search — it never touches `pymoo` or modifies the penalty-matrix
chromosomes. It only reads `../dispatcher/`'s output files (copied into `data/`, not
read live) and shares no code with it.

## How to run

```
cd code/dispatcher_analysis
python main.py
```

Four phases, run in order by `main.py`:
1. `src/build_models.py` — retrains every individual's FC head from
   `data/pareto_front.csv`. Always rebuilt from the chromosome, never from a cached
   `.npz` copy — a prior stale-checkpoint bug means copied weight files are no longer
   trusted. Cheap (~3-5s/individual, ~3 min for 50).
2. `src/cache_predictions.py` — predicts on train + val embeddings, caches to
   `predictions/{train,val}_predictions.csv`.
3. `src/summarize.py` — computes `alpha_sys`, `avg_flops_G`, per-class recall/precision
   per individual per split → `results/{train,val}_summary.csv`.
4. `src/plots.py` — Pareto scatter + confusion matrices for 3 representative
   individuals (cheapest / highest-alpha_sys / middle) → `results/plots/`.

## Inputs

- `data/pareto_front.csv` — copied from `../dispatcher/results/nsga2/pareto_front.csv`
  (the front to evaluate).
- `data/train_ground_truth.csv`, `data/val_ground_truth.csv` — same underlying data as
  `../dispatcher/`, val split added here since dispatcher never sees held-out data.

## Outputs

- `model_cache/individual_*.npz` — retrained FC weights (rebuilt every run).
- `predictions/{train,val}_predictions.csv` — cached raw predictions.
- `results/{train,val}_summary.csv` — alpha_sys, avg_flops_G, per-class metrics.
- `results/plots/pareto_scatter.png`, `results/plots/confusion_matrices.png`.

## Why

The `alpha_sys` metric definition and why copied weight files aren't trusted are
narrated in `../../../Journel/Week1.md` (original ImageNette pipeline). This CIFAR-10
reproduction's own run and why alpha_sys sits in a narrow near-ceiling band here are in
`../../../Journel/Week2.md`.

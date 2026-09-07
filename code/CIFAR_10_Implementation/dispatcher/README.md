# dispatcher

Step 1-2: NSGA-II search for the dispatcher's penalty matrix / class-weighting scheme.

**Does**: trains a `Linear(64→4)` FC head on frozen resnet20 embeddings under an
NSGA-II search over 12-gene penalty-matrix chromosomes, minimizing
`(alpha_sys_loss, avg_flops_G)`. Saves the resulting Pareto front and one set of FC
weights per Pareto individual.
**Does not**: evaluate the front (train/val alpha_sys, per-class metrics, plots) — that
is `../dispatcher_analysis/`'s job only. This folder has no held-out val data and no
eval code by design; that split was violated once and had to be undone — keep it split.

## How to run

```
cd code/dispatcher
python main.py
```

Single phase: `src/nsga2_search.py` wires embeddings, penalty matrix, loss, and FC
training into `pymoo`'s `NSGA2`. Backbone embeddings are computed/cached once
(`embeddings_cache/train_embeddings.npz`, ~26s); each fitness evaluation only
trains/evals the FC head on cached tensors (~3-5s).

Expect ~2-3 hours for a full run (pop/gen/epochs are set in `src/constants.py`).

## Inputs

- `data/train_ground_truth.csv` — per-image labels + per-model correctness columns,
  used both for the dispatcher's ideal-label target and for looking up `alpha_sys`.

## Outputs

- `results/nsga2/pareto_front.csv` — one row per Pareto individual
  (`individual, alpha_sys, avg_flops_G, P_*` penalty-matrix genes).
- `results/nsga2/models/individual_*.npz` — trained FC weights per individual.
- `results/nsga2/checkpoint_gen*.npz` — per-generation checkpoints (for resuming/audit).
- `results/logs/` — live run logs.

## Why

Dispatcher methodology, the `alpha_sys` objective, and class-imbalance (INS) fix are
narrated in `../../../Journel/Week1.md` (original ImageNette pipeline). This CIFAR-10
reproduction's own run is in `../../../Journel/Week2.md`.

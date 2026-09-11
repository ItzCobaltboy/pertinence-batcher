# dispatcher

Step 1-2: NSGA-II search for the dispatcher's penalty matrix. **Shared, generic code
only** — no dataset-specific constants, paths, or logic live here.

**Does**: trains a `Linear(EMBEDDING_DIM→NUM_CLASSES)` FC head on frozen TRAIN embeddings
under an NSGA-II search over 12-gene penalty-matrix chromosomes, then evaluates each
individual's fitness `(alpha_sys_loss, avg_model_cost)` by predicting on the held-out VAL
set — matching the paper's own methodology ("we perform the fitness evaluation for each
individual on the test set"), see `fitness.py`'s module docstring. Saves the resulting
Pareto front and one set of FC weights per Pareto individual (that final save/retrain
step is train-only — only per-generation fitness touches val).
**Does not**: run the FINAL evaluation of the front (per-class metrics, confusion
matrices, plots, `accuracy_exact_match_vs_ideal`) — that is `../dispatcher_analysis/`'s
job only, and it shares no code with this folder by design. This folder reads val data
only through `config.VAL_GROUND_TRUTH_CSV`/`config.VAL_EMBEDDINGS_NPZ` — it never copies
val data into this folder itself. Also does not know anything about ImageNette or
CIFAR-10 specifically — see "Config injection" below.

## Config injection

Every function here that needs a path, a hyperparameter, or the model pool takes an
explicit `config` argument — a plain Python module (`imagenette/config.py` or
`cifar-10/config.py`), passed in by that track's entry-point script
(`imagenette/run_dispatcher.py` / `cifar-10/run_dispatcher.py`). Nothing in this folder
does `import constants` or reads a per-dataset value implicitly.

`config` must define: `MODEL_NAMES`, `MODEL_COST` + `MODEL_COST_UNIT`, `NUM_CLASSES`,
`EMBEDDING_DIM`, `build_feature_extractor(device)` (returns the frozen backbone with its
classifier head removed — the *mechanism* for loading this differs by dataset, e.g.
torchvision's pretrained ResNet18 vs. a local CIFAR-10 checkpoint loader, so it's a
function the config provides, not a constant), `IMAGE_TRANSFORM`, `DATASET_DIR`,
`TRAIN_GROUND_TRUTH_CSV`, `TRAIN_EMBEDDINGS_NPZ`, `VAL_GROUND_TRUTH_CSV`,
`VAL_EMBEDDINGS_NPZ`, `LOGS_DIR`, `NSGA2_DIR`, `PARETO_FRONT_CSV`, plus the GA/FC
hyperparameters (`N_GENES`, `PENALTY_LOWER_BOUND`, `PENALTY_UPPER_BOUND`, `FC_EPOCHS`,
`BATCH_SIZE`, `LEARNING_RATE`, `POPULATION_SIZE`, `GENERATIONS`, `SBX_ETA`,
`SBX_CROSSOVER_PROBABILITY`, `MUTATION_ETA`, `CHECKPOINT_EVERY_N_GENERATIONS`,
`EMBEDDING_NUM_WORKERS`).

## How to run

```
python imagenette/run_dispatcher.py
```
or
```
python cifar-10/run_dispatcher.py
```

Single phase: `nsga2_search.py` wires embeddings, penalty matrix, loss, and FC training
into `pymoo`'s `NSGA2`. Backbone embeddings are computed/cached once per track
(`<track>/embeddings_cache/{train,val}_embeddings.npz`); each fitness evaluation only
trains the FC head on cached train tensors then evals it on cached val tensors.

Expect on the order of hours for a full run (pop/gen/epochs are set in each track's
`config.py`).

## Inputs

- `<track>/data/train_ground_truth.csv` — per-image labels + per-model correctness
  columns; used for the dispatcher's ideal-label target and to train each individual's
  FC head.
- `<track>/data/val_ground_truth.csv` — per-model correctness columns for the held-out
  set; used to evaluate each individual's fitness (`alpha_sys`, `avg_model_cost`) once
  its FC head is trained.

## Outputs

- `<track>/results/nsga2/pareto_front.csv` — one row per Pareto individual
  (`individual, alpha_sys, avg_model_cost, P_*` penalty-matrix genes). `alpha_sys` here
  is the VAL-set fitness value from the search, not a train-set number.
- `<track>/results/nsga2/models/individual_*.npz` — trained FC weights per individual.
- `<track>/results/nsga2/checkpoint_gen*.npz` — per-generation checkpoints (for
  resuming/audit).
- `<track>/results/logs/` — live run logs.

## Why

Dispatcher methodology, the `alpha_sys` objective, and the completed NSGA-II runs are
narrated in `../Journel/`. This is shared, generic code across both tracks — only the
model pool, cost units, embedding dim/backbone, and image transform differ between
ImageNette and CIFAR-10, and those live in each track's own `config.py`.

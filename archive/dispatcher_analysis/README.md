# dispatcher_analysis

Step 3: evaluation only of whatever Pareto front `../dispatcher/` last produced.
**Shared, generic code only** — no dataset-specific constants, paths, or logic live
here (same config-injection scheme as `../dispatcher/`, see below).

**Does**: for every Pareto individual, retrains its FC head directly from its
chromosome (never from a copied weight cache — see below), predicts on train and a
held-out val split, and computes `alpha_sys`, `accuracy_exact_match_vs_ideal`,
`avg_model_cost`, and per-class recall/precision on both splits. Produces a Pareto
scatter and confusion matrices.
**Does not**: run any search — it never touches `pymoo` or modifies the penalty-matrix
chromosomes. It reads `../dispatcher/`'s output (`<track>/results/nsga2/pareto_front.csv`,
read directly, not copied) and shares no code with it.

## Three metrics, spelled out

- **`alpha_sys`** (Eq. 3 of the PERTINENCE paper): fraction of images where the model
  the dispatcher *actually picked* classifies correctly. Overestimating (routing to a
  bigger, still-correct model) costs nothing here, only `avg_model_cost`.
- **`accuracy_exact_match_vs_ideal`**: fraction of images where the dispatcher's routing
  decision was an *exact match* to `ideal_label` (the argmin-cheapest-correct
  reference). Strictly stricter than `alpha_sys` — punishes overestimation exactly as
  hard as an actual misclassification.
- **`recall_<model>` / `precision_<model>`**: per-class versions of the same
  exact-match-against-`ideal_label` question, broken down per class.

See `summarize.py`'s module docstring for the full explanation — it's intentionally
repeated there since that's where the column names actually get written.

## Config injection

Same scheme as `../dispatcher/`: every function takes an explicit `config` argument
(that track's `config.py`, passed in by `imagenette/run_dispatcher_analysis.py` /
`cifar-10/run_dispatcher_analysis.py`). In addition to everything `../dispatcher/`
needs, this folder's `config` must also define `VAL_GROUND_TRUTH_CSV`,
`VAL_EMBEDDINGS_NPZ`, `MODEL_CACHE_DIR`, `TRAIN_PREDICTIONS_CSV`,
`VAL_PREDICTIONS_CSV`, `TRAIN_SUMMARY_CSV`, `VAL_SUMMARY_CSV`, `PLOTS_DIR`.

## How to run

```
python imagenette/run_dispatcher_analysis.py
```
or
```
python cifar-10/run_dispatcher_analysis.py
```

Four phases, run in order by the entry-point script:
1. `build_models.py` — retrains every individual's FC head from
   `<track>/results/nsga2/pareto_front.csv`. Always rebuilt from the chromosome, never
   from a cached `.npz` copy. Cheap (~3-5s/individual).
2. `cache_predictions.py` — predicts on train + val embeddings, caches to
   `<track>/results/eval/predictions/{train,val}_predictions.csv`.
3. `summarize.py` — computes `alpha_sys`, `accuracy_exact_match_vs_ideal`,
   `avg_model_cost`, per-class recall/precision per individual per split →
   `<track>/results/eval/{train,val}_summary.csv`.
4. `plots.py` — Pareto scatter + confusion matrices for 3 representative individuals
   (cheapest / highest-alpha_sys / middle) → `<track>/results/eval/plots/`.

## Inputs

- `<track>/results/nsga2/pareto_front.csv` — the front to evaluate (written by
  `../dispatcher/`, read directly — no copy step).
- `<track>/data/train_ground_truth.csv`, `<track>/data/val_ground_truth.csv` — the same
  files `../dispatcher/` reads.

## Outputs

- `<track>/results/eval/model_cache/individual_*.npz` — retrained FC weights (rebuilt
  every run).
- `<track>/results/eval/predictions/{train,val}_predictions.csv` — cached raw
  predictions.
- `<track>/results/eval/{train,val}_summary.csv` — alpha_sys,
  accuracy_exact_match_vs_ideal, avg_model_cost, per-class metrics.
- `<track>/results/eval/plots/pareto_scatter.png`, `confusion_matrices.png`.

## Why

The `alpha_sys` metric definition and the majority-class-bias finding are narrated in
`../Journel/`. This is shared, generic code across both tracks, same as `../dispatcher/`.

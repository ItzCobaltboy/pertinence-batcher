# eda

Shared, generic ground-truth EDA — runs directly on the labeled ground-truth CSVs
(`<model>_correct` columns + `label` column) produced for either track, before the
dispatcher search or evaluation runs. Same config-injection scheme as `../dispatcher/`
and `../dispatcher_analysis/`: `ground_truth_eda.py`'s `run_eda(config)` takes that
track's `config.py`.

**Not** the same thing as `../archive/model_analysis/eda/`, which compares
model_analysis's own quantization-precision benchmark CSVs (accuracy/latency/size per
precision) and has nothing to do with dataset ground truth.

**Does**, per split (train and val), for whichever dataset's `config` it's given:
1. Per-model standalone accuracy (mean of each `<model>_correct` column, over every
   image including "impossible" ones).
2. Theoretical maximum ("oracle") accuracy — the fraction of images where at least one
   pool model is correct — computed directly from the `<model>_correct` columns (not
   from `label`, which can't distinguish an impossible image from a genuine
   highest-cost-model route — see `label_data.py`'s docstring in either track).
3. Class balance: distribution of the routing target (`label`) across the pool. Note
   impossible images are folded into the highest-cost model's count here; use the
   oracle rate (point 2) to see that split out explicitly.
4. Pairwise model error overlap (Jaccard index over each pair's wrong-image sets) — how
   independent the models' mistakes are.

## How to run

```
python imagenette/run_eda.py
```
or
```
python cifar-10/run_eda.py
```

## Inputs

- `<track>/data/train_ground_truth.csv`, `<track>/data/val_ground_truth.csv` — no other
  files needed; both label_data.py scripts write every raw image into these (including
  "impossible" ones, routed to the highest-cost model), so the oracle rate and every
  other stat here comes straight from the CSV.

## Outputs

Currently prints a report per split to stdout; nothing is written to disk yet (add an
output path here if/when this needs to feed a presentation deck directly rather than
being read off the terminal).

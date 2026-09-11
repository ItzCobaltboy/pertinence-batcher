# imagenette

ImageNet-pretrained ResNet18/34/50/152 evaluated on ImageNette. This folder holds only
what's specific to this track: its config, entry-point scripts, raw dataset, and
results. The actual dispatcher/eval/EDA logic is shared code in `../dispatcher/`,
`../dispatcher_analysis/`, `../eda/` — see those folders' READMEs.

**Model pool**: FP32 ResNet18 / ResNet34 / ResNet50 / ResNet152 (see
`../archive/model_analysis/` for the precision-benchmark work that locked this pool).

## How to run (in order)

```
python imagenette/label_data.py                 # optional -- see "Labeling script" below
python imagenette/run_eda.py                     # ground-truth EDA (per-model/oracle accuracy, class balance)
python imagenette/run_dispatcher.py              # NSGA-II search (multi-hour on a full run)
python imagenette/run_dispatcher_analysis.py     # evaluate the resulting front
```

`data/train_ground_truth.csv` and `data/val_ground_truth.csv` already exist and are
valid — `label_data.py` does not need to be run before the other three steps.

## Labeling script

`label_data.py` walks `dataset/imagenette2-320/{train,val}/`, runs the 4 pool models
(full 1000-class ImageNet head, not the classifier-stripped embedding extractor
`config.build_feature_extractor()` returns) over every image, remaps ImageNette's
synset-ID folder names to real ImageNet class indices via `config.IMAGENETTE_LABEL_MAP`,
and writes `data/{train,val}_ground_truth.csv` — same schema and methodology as
`cifar-10/label_data.py`. Every raw image ends up in the CSV, including ones no pool
model classifies correctly ("impossible" images) — those get routed to the highest-cost
model (resnet152) rather than dropped, so nothing needs filtering downstream.

Running `label_data.py` overwrites the existing `data/{train,val}_ground_truth.csv`
with freshly computed values (same schema, same argmin-cheapest-correct labeling rule
as `cifar-10/label_data.py`; train count is 9469 images). Any existing
`results/nsga2/pareto_front.csv` was trained against whatever ground-truth CSVs were
present at the time, so re-running `label_data.py` and getting different numbers means
the front needs a re-run too.

ImageNette's `train`/`val` split is the dataset's own official division, not a re-mixed
one.

## Requirements

Root `requirements.txt` — no extras.

## Why

Model pool selection and the Torch-TensorRT precision benchmark are narrated in
`../Journel/Week0.md` and `../Journel/Week1.md`; the dispatcher methodology and the
three completed NSGA-II runs in `../Journel/Week1.md`. See `../CLAUDE.md` for current
status.

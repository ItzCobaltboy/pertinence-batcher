# cifar-10

Runs the PERTINENCE methodology on CIFAR-10, using CIFAR-10-pretrained models, to
validate the pipeline against a second dataset beyond ImageNet/ImageNette. This folder
holds only what's specific to the CIFAR-10 track: its config, entry-point scripts,
labeling script, model checkpoints/loader, raw dataset, and results. The actual
dispatcher/eval/EDA logic is shared code in `../dispatcher/`, `../dispatcher_analysis/`,
`../eda/` — see those folders' READMEs.

**Model pool** (chenyaofo/pytorch-cifar-models, ordered by MAdds): `shufflenetv2_x0_5`
(10.90M, 90.13% top-1) · `resnet20` (40.81M, 92.60%) · `resnet32` (69.12M, 93.53%) ·
`vgg11_bn` (153.29M, 92.79%). This is not the exact pool the PERTINENCE paper uses on
CIFAR-10 (`resnet8, resnet14, shufflenetv2_x0_5, vgg16_bn`) — no pretrained CIFAR-10
checkpoints exist publicly for `resnet8`/`resnet14`, so this track uses this pretrained
substitute pool instead.

## How to run (in order)

```
python cifar-10/label_data.py          # downloads CIFAR-10, builds ground-truth CSVs
python cifar-10/run_eda.py             # ground-truth EDA (per-model/oracle accuracy, class balance)
python cifar-10/run_dispatcher.py      # NSGA-II search (multi-hour on a full run)
python cifar-10/run_dispatcher_analysis.py   # evaluate the resulting front
```

Each step reads the previous step's output directly from `cifar-10/data/` and
`cifar-10/results/` — no copy step between them.

## Split methodology

Uses CIFAR-10's own official train (50,000) / test (10,000) division as-is, with no
re-mixing. Official train is the only pool labeling and dispatcher FC training ever see;
official test is the held-out `val` split, used by NSGA-II's per-individual fitness
evaluation and by `run_dispatcher_analysis.py`'s final eval. See `label_data.py`'s module
docstring for the full split rationale.

## Requirements

Root `requirements.txt` — nothing extra. `pytorch-cifar-models` is not pip-installable;
`models/model_loader.py` pulls it via
`torch.hub.load("chenyaofo/pytorch-cifar-models", ...)`, which needs network access on
first run (cached after).

## Key differences from the ImageNette track

- **Embeddings**: frozen `shufflenetv2_x0_5` (this pool's cheapest model) instead of
  ResNet18 — 1024-dim, not 512. The backbone-loading mechanism itself differs (a local
  checkpoint via `models/model_loader.py`, not a torchvision pretrained download) — see
  `config.py`'s `build_feature_extractor()`.
- **Preprocessing**: CIFAR mean/std, native 32×32, no resize/crop, no ImageNet label
  remapping.
- **Cost unit**: MAdds (millions), not GFLOPs — `config.MODEL_COST_UNIT` makes this
  explicit everywhere cost is printed or plotted.
- Penalty-matrix chromosome, INS class weighting, the `alpha_sys` objective, and
  NSGA-II hyperparameters are the same as the ImageNette track — this track validates
  the methodology, it doesn't redesign it.

## Why

Full run narrative is in `../Journel/`. See `../CLAUDE.md`'s open threads for current
status.

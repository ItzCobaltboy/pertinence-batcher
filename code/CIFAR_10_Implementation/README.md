# CIFAR_10_Implementation

Reproduces the PERTINENCE paper's exact experimental setup — the paper's own
CIFAR-10-pretrained model pool on CIFAR-10 — to demonstrate the methodology is
implemented faithfully, not just adapted for the ImageNet/ImageNette pool used
elsewhere in this project (`../dispatcher/`, `../dispatcher_analysis/`). Fully
self-contained: separate model pool, dataset, dispatcher, and analysis; those two
folders are untouched by this work.

**Model pool** (chenyaofo/pytorch-cifar-models, ordered by MAdds): `resnet20` (40.81M,
92.60% top-1) · `resnet32` (69.12M, 93.53%) · `shufflenetv2_x2_0` (187.81M, 93.81%) ·
`vgg16_bn` (313.73M, 94.16%).

## How to run (in order)

```
cd code/CIFAR_10_Implementation
python label_data.py              # downloads CIFAR-10, builds ground-truth CSVs
cd dispatcher && python main.py   # NSGA-II search (~7-8h on a full run)
# copy dispatcher/results/nsga2/pareto_front.csv into dispatcher_analysis/data/
cd ../dispatcher_analysis && python main.py   # evaluate the front
```

`dispatcher/` and `dispatcher_analysis/` each have their own README with per-phase
detail — same structure/responsibilities as the root pipeline's, see there for specifics.

## Requirements

Root `requirements.txt` plus `scikit-learn` (stratified split) — both already pinned
there. `pytorch-cifar-models` is not pip-installable; the model loader pulls it via
`torch.hub.load("chenyaofo/pytorch-cifar-models", ...)`, which needs network access on
first run (cached after).

## Key differences from the ImageNette pipeline

- **Split**: CIFAR's official train+test (60k) combined into one pool, then a fresh
  class-stratified 70:30 split — not CIFAR's own division (the official test set was
  also used to report the checkpoints' accuracy, so reusing it as held-out would
  double-count it).
- **Embeddings**: frozen `resnet20` (this pool's cheapest model) instead of ResNet18 —
  64-dim, not 512.
- **Preprocessing**: CIFAR mean/std, native 32×32, no resize/crop, no ImageNet label
  remapping.
- Penalty-matrix chromosome, INS class weighting, `alpha_sys` objective, and NSGA-II
  hyperparameters are unchanged from the root pipeline — this track validates the
  methodology, it doesn't redesign it.

## Outputs

- `dispatcher/results/nsga2/pareto_front.csv` — 30-individual front, alpha_sys
  99.09–99.42%, avg MAdds 42.3–233.0M.
- `dispatcher_analysis/results/{train,val}_summary.csv` + `results/plots/` — eval
  confirming the front (val alpha_sys 99.01–99.33%).

## Why

Full run narrative (split rationale, the Windows/Linux CSV path bug and fix, and why
alpha_sys sits in a narrow near-ceiling band here) is in `../../Journel/Week2.md`. See
`../CLAUDE.md`'s open threads for current status.

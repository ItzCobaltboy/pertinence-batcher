# cifar-100

Reproduces two specific results from the PERTINENCE paper's CIFAR-100 exploration (Fig. 9,
page 8), rather than inventing a new experiment the paper never runs. The paper never trains
a single dispatcher across its whole six-model CIFAR-100 pool at once — every reported front
comes from a separate MOEA search over a hand-picked 2- or 3-model subset. This track
originally tried the full-six-model approach anyway; that attempt was dropped once it became
clear it wasn't actually reproducing anything the paper reports, and in favor of directly
reproducing Fig. 9(c) and Fig. 9(d) — see `Journel/Week3.md` for the pivot.

## Layout

```
cifar-100/
  dataset/            shared raw CIFAR-100 images (both official train and test splits),
                       downloaded once, used by every variant below
  models/             shared checkpoints + loader for all 7 CIFAR-100 models this track
                       has ever needed across both variants
  label_data.py        shared labeling logic, takes a `config` argument (see below)
  analyze_model_overlap.py   shared overlap-analysis script, auto-detects a variant's
                       model columns from whatever CSV it's pointed at
  fig9c/              variant reproducing Fig. 9(c): shufflenetv2_x0_5, mobilenetv2_x0_75,
                       repvgg_a2
  fig9d/              variant reproducing Fig. 9(d): shufflenetv2_x0_5, mobilenetv2_x1_4,
                       repvgg_a2
```

`fig9c/` and `fig9d/` are each a fully self-contained thin track in the same shape as
`imagenette/`/`cifar-10/` — own `config.py`, own `data/`, own `embeddings_cache/`, own
`results/`. They share the raw dataset images and the model checkpoints (nothing about a raw
CIFAR-100 image or a downloaded `.pt` file depends on which 3-model subset a variant
dispatches to), but **not** ground-truth CSVs, embeddings caches, or results — a variant's
routing labels depend entirely on its own model subset, so sharing an embeddings cache
between variants would silently serve one variant's labels to the other. Embeddings
themselves are cheap enough to just recompute per variant (~1 minute each) rather than build
a cache-sharing scheme with a real correctness trap in it.

**Model pool**: 7 CIFAR-100 checkpoints total across both variants (chenyaofo/pytorch-cifar-models) —
`shufflenetv2_x0_5`, `mobilenetv2_x0_75`, `mobilenetv2_x1_4`, `repvgg_a2` are actually used
(2 models shared between the variants, `mobilenetv2_x0_75` unique to fig9c, `mobilenetv2_x1_4`
unique to fig9d); the other 3 (`mobilenetv2_x0_5`, `shufflenetv2_x1_0`, `repvgg_a1`) were
downloaded for the abandoned full-pool attempt and are kept in `models/` since they're
harmless to have around, just unused by either variant. `mobilenetv2_x0_75` is a real,
separately published chenyaofo checkpoint, not a typo for `mobilenetv2_x0_5` — confirmed
against the chenyaofo repo's full CIFAR-100 model list, and it's not one of the six models
plotted on the paper's Fig. 4b (that plot doesn't show every model the paper's Fig. 9
experiments actually use).

## How to run (per variant, in order)

```
python cifar-100/fig9c/run_label_data.py         # downloads CIFAR-100 (shared), builds this variant's 3 ground-truth CSVs
python cifar-100/fig9c/run_eda.py                # ground-truth EDA on all 3 splits
python cifar-100/fig9c/run_dispatcher.py         # NSGA-II search (7-gene chromosome, multi-hour on a full run)
python cifar-100/fig9c/run_dispatcher_analysis.py   # evaluate the resulting front, 2 passes -- writes
                                                      # accuracy_vs_mflops_{test,final_val}.png in the
                                                      # paper's own Fig. 6-10 plotting style, among others
```

Same four commands under `fig9d/` for the other variant. The dataset download only actually
happens once — `label_data.py`'s image dump is idempotent and both variants point at the same
`cifar-100/dataset/`.

## Split methodology — three-way, not the other tracks' two-way train/val

The paper's own methodology (Section II-C, page 7) evaluates each NSGA-II individual's
fitness on what it calls the **test set** during the search, then separately reports final
numbers using a **validation set** "that was not used during model training or the MOEA
process." That's genuinely three disjoint sets, and the paper's naming is backwards from
what you'd guess — its "test set" is touched constantly (every fitness eval, every
generation) and its "validation set" is the one actually held out until the end.

CIFAR-100 only ships an official train (50,000) / test (10,000) division, no third split to
lean on. An early version of this track carved the validation set out of official train — a
class-stratified 10% slice — and the EDA on it came back looking wrong: 100% oracle accuracy,
dead routing classes. Root cause: the pool checkpoints are pretrained on the *entire*
official 50k train set, so anything carved out of train is already memorized, no matter how
it's stratified. Fixed by carving both the test-set role and the validation-set role out of
the official **test** split instead — the only 10,000 images the checkpoints never saw during
their own training. Official test has exactly 100 images/class, split class-stratified 70/30
with a fixed seed: 7,000 for NSGA-II fitness, 3,000 held out as the real final validation.
See `label_data.py`'s module docstring for the full rationale, and `Journel/Week3.md` for how
this was actually discovered (by running the EDA and noticing the numbers looked wrong, not
by reasoning about it in advance).

## Matches the paper that the other two tracks don't

- **Weighting-scheme gene, searched for real.** The paper's chromosome searches penalties AND a
  choice of class-weighting scheme (INS/ISNS/ENS) together. `fig9c`/`fig9d` reverse the earlier
  "INS only" decision that ImageNette/CIFAR-10 still use — `N_GENES=7` (6 penalty genes + 1
  scheme selector), decoded by `dispatcher/weighting_scheme.py`. Whether a track searches the
  scheme at all is inferred from `N_GENES` itself, not a flag, so ImageNette/CIFAR-10 are
  completely unaffected.
- **Cost metric is locally-measured FLOPs, not chenyaofo's table.** `MODEL_COST`/
  `DISPATCHER_OVERHEAD_COST` come from this repo's own `thop` measurement (MACs doubled to
  FLOPs — the paper states it uses THOP "to obtain MACs... and the resulting FLOPS"), not
  chenyaofo's published MAdds table. The switch happened after chenyaofo's table numbers
  produced dispatched-cost values that didn't line up with where the paper's own Fig. 9 plots
  actually put PERTINENCE's points; the local thop-doubled measurement, by contrast, matches
  the paper's own published Table 5 dispatcher-overhead figure (24.17 MFLOPS for CIFAR-100/
  ShuffleNetV2) to within 1%. `MODEL_COST_UNIT = "MFLOPs-M"` now, not `"MAdds-M"`.
- **`avg_model_cost` matches the paper's Eq. 4.** Includes `config.DISPATCHER_OVERHEAD_COST`
  (feature extractor + FC head cost) on top of the dispatched model's own cost — shared
  `dispatcher/fitness.py`/`dispatcher_analysis/summarize.py` code, applies to every track. Fixed
  offset per variant, doesn't change a front's shape or ranking, only its absolute cost axis.

## Still approximated

- **Continuous penalty search, not discretized.** The paper specifies penalty values in
  `[0, 100]` with a step size in `[0.5, 1]`. `config.py` uses the paper's `[0, 100]` range,
  but pymoo's SBX/polynomial-mutation operators search it as a continuous interval — there's
  no native step-size knob, so this is an approximation of the paper's discretized space. The
  weighting-scheme gene has the same category of approximation: the paper doesn't say how it
  encodes a discrete 3-way choice as a continuous MOEA gene, so `[0, 3)` split into 3 equal
  bins is this repo's own reasonable choice to fill that gap.

## Requirements

Root `requirements.txt` — nothing extra. `pytorch-cifar-models` is not pip-installable;
`models/model_loader.py` pulls it via `torch.hub.load("chenyaofo/pytorch-cifar-models", ...)`,
which needs network access on first run (cached after). The 7 checkpoint `.pt` files are
downloaded once into `models/` directly from the repo's GitHub releases.

## Why

Full run narrative, including the abandoned full-pool attempt and the pivot to fig9c/fig9d,
is in `../Journel/Week3.md`. See `../CLAUDE.md`'s open threads for current status, and
`PIPELINE_REPORT.md` for a complete lifecycle walkthrough.

# CIFAR-100 track — pipeline lifecycle report

This walks through the full pipeline end to end, from raw CIFAR-100 through smoke-tested
NSGA-II searches to final evaluation, for both variants this track now builds. Neither
variant's real, full-hyperparameter search has been run — that's a multi-hour job for the
A100, and it's mine to trigger, not something to run unattended. Everything below has been
smoke-tested (tiny population, 1-2 generations, 2 FC epochs) to confirm the whole chain
actually wires together and produces sane numbers, not just that it doesn't crash.

## 0. Why two variants instead of one six-model dispatcher

The first version of this track built a single dispatcher across all six models on the
paper's Fig. 4b CIFAR-100 Pareto front. That was a mistake, caught before the real search
ran: the paper's actual CIFAR-100 experiments (Fig. 9, page 8) never dispatch across the
whole pool at once. Every reported front comes from a separate MOEA search over a hand-picked
2- or 3-model subset — "we report results on cases where the input can be dispatched to two
CNN models (Fig. 9(a)-(b)), or three CNN models (Fig. 9(c)-(d))." The paper gives an explicit
reason for this restriction (stated for CIFAR-10, same logic applies): a cheap model becomes
pointless to include once a strictly-better, similarly-cheap alternative is already in the
subset — "extracting features with resnet14 makes resnet8 useless since resnet14 is more
accurate and is supposed to be able to handle all inputs resnet8 can handle."

A single six-way dispatcher isn't wrong on its own terms, but it isn't reproducing anything
the paper reports, and it's a much harder search problem on the same compute budget: the
paper's subset runs search 2-6 continuous penalty genes with population 50 / 50 generations;
a six-model dispatcher searches 30 genes with the identical budget — a much sparser
exploration of a much bigger space. Once this was pointed out, the decision was to drop the
six-model attempt and reproduce Fig. 9(c) and Fig. 9(d) specifically instead — two separate,
independent three-model dispatchers, each searching 6 genes, which is a far closer match to
what the paper's own subset runs actually search.

Reading Fig. 9's captions closely surfaced one more thing worth flagging: Fig. 9(a) and 9(c)
both use `mobilenetv2_x0_75`, a model that is NOT one of the six plotted on Fig. 4b and was
not part of the original six-checkpoint download. Confirmed it's a real, separately published
chenyaofo checkpoint (not a typo for `mobilenetv2_x0_5`) before downloading it — this is
exactly the kind of gap that hit the CIFAR-10 track with `resnet8`/`resnet14`, so it got the
same scrutiny rather than being assumed away.

- **fig9c/** reproduces Fig. 9(c): "Inputs dispatched to either shufflenetv2_x0_5,
  mobilenetv2_x0_75, or repvgg_a2."
- **fig9d/** reproduces Fig. 9(d): "Inputs dispatched to either shufflenetv2_x0_5,
  mobilenetv2_x1_4, or repvgg_a2."

Both share the same feature extractor (`shufflenetv2_x0_5`), the same three-way split logic,
and the same NSGA-II hyperparameters — they differ only in which second and third model round
out the three-model pool.

## 1. Data: what's downloaded, how it's split, what each split is for

`label_data.py` (shared, takes a `config` argument so both variants call the same code with
their own model list) pulls CIFAR-100 through `torchvision.datasets.CIFAR100`, which gives
the dataset's own official division: 50,000 train images, 10,000 test images, 100 classes,
500/100 images per class in train/test respectively. Every image gets dumped to a PNG under
`cifar-100/dataset/images/` — shared between both variants, since the raw images and their
filenames don't depend on which 3-model subset a variant dispatches to. Everything downstream
reads PNGs off disk through `image_path` columns in each variant's own ground-truth CSVs.

The PERTINENCE paper's methodology (Section II-C, page 7) doesn't use a two-way train/val
split, it uses three disjoint sets, and its naming is backwards from what you'd guess:

- paper's **"training set"** trains the FC head for each individual.
- paper's **"test set"** is what NSGA-II's own fitness function evaluates against, every
  single generation, during the search itself.
- paper's **"validation set"** is touched exactly once, after the search is completely
  finished, to report final numbers — "a validation set that was not used during model
  training or the MOEA process."

CIFAR-100 doesn't ship a third official split, so I had to manufacture one, and my first
attempt at this got it wrong in an instructive way. The obvious move is to carve the
validation set out of official train, the way you'd usually split off a dev set. I did that
first — a stratified 10% slice, 50 images/class — and the ground-truth EDA on it came back
looking almost fraudulent: 100% oracle accuracy, most routing classes with zero or
near-zero images. The reason is that every pool checkpoint (`chenyaofo/pytorch-cifar-models`)
was trained on the *entire* official 50k train set. Any slice you cut out of train, no matter
how carefully stratified, is data the pool has already memorized. A "held-out" validation
pass against it would report near-ceiling numbers that say nothing real about generalization.

The fix was to carve both the test-set role and the validation-set role out of the official
**test** split instead, since that's the only 10,000 images none of the checkpoints saw
during their own training. The official test set has exactly 100 images per class, so a
stratified 70/30 split lands exactly on 70/30 per class:

- **7,000 images** (70/class) → `data/test_ground_truth.csv` → paper's "test set" → what
  `dispatcher/fitness.py` evaluates every individual against during the search.
- **3,000 images** (30/class) → `data/final_val_ground_truth.csv` → paper's "validation set"
  → touched only by `run_dispatcher_analysis.py`'s second pass, after everything else is
  done.
- **The full, untouched official train set** (50,000 images) → `data/train_ground_truth.csv`
  → paper's "training set" → what FC-head training and NSGA-II's search actually see for
  gradient signal.

Neither the 70/30 split point nor the original 10% figure are values out of the paper — it
doesn't give split sizes for any of its three datasets, on any of the tracks. These are
project-level calls, and I ran the EDA after making each one specifically to catch a bad call
before it propagated into a multi-hour search. After the fix, test and final_val track each
other closely on both variants — fig9c: 86.01% vs. 86.53% oracle accuracy; fig9d: 86.61% vs.
86.60% — which is what two independent stratified draws off the same 10k pool should look
like, and is the actual evidence the fix worked, not just a theoretical argument.

The split point (`FINAL_VAL_FRACTION_OF_TEST`, `FINAL_VAL_SEED`) is identical between both
variants' `config.py` files, so both variants carve the exact same 7,000/3,000 image split of
the official test set — only the correctness columns computed against that split differ,
since each variant's own three-model pool decides which images each model gets right.

## 2. Labeling: argmin-cheapest-correct, and what happens to unsolvable images

`label_data.py` runs every model in `config.MODEL_NAMES` (three per variant) over every image
in each split and records, per image, whether each model's own prediction matches the
CIFAR-100 ground-truth class. The routing target is then

    label(x) = argmin_j { MAdds_j | model_j(x) is correct }

— whichever correct model costs the least, computed independently per split and per variant.

Images where *no* pool model in a variant's subset gets the right answer aren't dropped.
They're routed to the highest-MAdds model in that variant's pool (`repvgg_a2` for both
variants, since it's the priciest model in each) instead, on the reasoning that if nothing in
the pool is right, the biggest/most capable model available is the most defensible fallback.
With only 3 models instead of 6, this happens noticeably more often than the abandoned
six-model attempt saw: fig9c's test-fitness split has 13.99% impossible images, fig9d's has
13.39% — versus ~9.6% when all six models were available to catch a hard image. That's a
real, structural consequence of a smaller pool, not a labeling bug: fewer models means fewer
chances that at least one of them gets a given image right. Both variants' test and
final_val splits track each other closely on this figure too (fig9c: 13.99%/13.47%; fig9d:
13.39%/13.40%), which is the same split-quality check as the oracle-accuracy numbers above.

On the official train split, both variants land at essentially zero impossible images (5/50000
for fig9c, 3/50000 for fig9d) — expected, since that's the data the pool checkpoints were
trained on.

## 3. Feature extractor: what it is, why this one, how it's stripped

The embedding extractor is `shufflenetv2_x0_5` with its final classifier layer (`.fc`)
removed — identical for both variants, since Fig. 9's subsets change which models get
dispatched to, not which model extracts features. This matches the paper's explicit
statement for CIFAR-100 (page 8): "we used shufflenetv2_x0_5 as feature extractor."

Stripping the classifier head isn't just `list(model.children())[:-1]`. ShuffleNetV2's
architecture does its global average pooling inline inside `forward()`
(`x = x.mean([2, 3])`), not as a stored submodule the way ResNet's `.avgpool` is — so
truncating at `children()[:-1]` alone leaves the feature map spatial, and flattening it
directly would multiply the true channel count by the spatial extent instead of giving you
one vector per image. `models/model_loader.py`'s `strip_classifier_head` inserts an explicit
`AdaptiveAvgPool2d(1)` before the flatten to correct for this — the same fix the CIFAR-10
track already needed for the same architecture.

Output dimension is measured with a real forward pass over a dummy 32x32x3 tensor rather than
assumed, and each variant's `config.py` asserts the measured value matches the hardcoded
`EMBEDDING_DIM = 1024` at import time. 1024 is the same dimension as the CIFAR-10 track's
extractor, which makes sense — it's architecturally the exact same ShuffleNetV2 backbone.

## 4. Embeddings: what gets cached, and why they aren't shared between variants

`dispatcher/embeddings.py` (shared code, unmodified) runs the frozen extractor over every
image in a ground-truth CSV through a plain `torch.utils.data.DataLoader`, batch size 64, and
caches the result — embeddings AND labels together — to a `.npz` keyed by split. This
happens three times per variant (train/test-fitness/final-val), on the first call to that
variant's `run_dispatcher.py` or `run_dispatcher_analysis.py`; every call after that loads
the cache straight off disk.

This is the one place worth being explicit about a real trap I almost built into this track:
the embeddings themselves (the 1024-dim feature vectors) genuinely don't depend on which
model subset a variant dispatches to — they're purely a function of the frozen extractor and
the raw image. It would have been tempting to share one embeddings cache between fig9c and
fig9d to save the ~1 minute of recomputation. The problem is that `compute_embeddings` caches
the routing *labels* in the exact same `.npz` file as the embeddings, and those labels are
computed from each variant's own ground-truth CSV — fig9c's 3-way argmin-cheapest label for a
given image is not necessarily the same as fig9d's, since the two variants have different
third models in their pool. A shared cache would silently serve one variant's labels
alongside the other's embeddings, and nothing about that mismatch would throw an error — the
shapes all line up fine, the numbers would just be quietly wrong. Each variant gets its own
`embeddings_cache/` directory specifically to make this impossible, at the cost of a
negligible amount of duplicate compute.

The data loader itself does no augmentation of any kind — no random crop, no flip, nothing
stochastic, since nothing in this pipeline ever trains a pool model; they're pretrained and
frozen. The only preprocessing is `ToTensor()` followed by channel-wise normalization using
`mean=[0.5070, 0.4865, 0.4409]`, `std=[0.2673, 0.2564, 0.2761]` — the exact values from
`chenyaofo/image-classification-codebase`'s own `conf/cifar100.conf`, the actual training
config used to produce the checkpoints this track loads, not a generic CIFAR-100 statistic
pulled from memory. Images are used at their native 32x32 resolution — no resize, no crop.

## 5. FC head training: what gets learned, and how class imbalance shows up with 3 models

The only trainable component in either variant is a single `nn.Linear(1024, 3)` — 1024-dim
embedding in, 3 logits out, one per pool model. `dispatcher/dispatcher_model.py` (shared,
unmodified) trains it with Adam, learning rate `1e-3`, batch size 128, for `FC_EPOCHS=20`
epochs, matching the paper's stated value exactly.

The loss (`dispatcher/loss.py`) is zero whenever the FC head's argmax prediction matches the
true routing label, and otherwise `cross_entropy(logits, true) * penalty[true, pred] *
class_weight[true]`. The penalty term comes from the chromosome (next section). The
class-weight term is INS — inverse number of samples, computed once from each variant's own
train-split label distribution.

Both variants show the same structural class-imbalance pattern the paper itself names (page
6, Table 3): `shufflenetv2_x0_5` (cheapest, correct 96.8% of the time on train, its own
training data) absorbs 96.8% of routing labels in both variants by construction — the
`argmin`-cheapest rule always collapses to the cheapest correct model, and on data the pool
has memorized, that model is almost always correct. With only 3 models, this leaves very
little train signal for the two pricier models: fig9c's `repvgg_a2` gets 27 train examples
(0.054%); fig9d's `repvgg_a2` gets 9 (0.018%). Neither is a hard zero the way the abandoned
six-model attempt's `repvgg_a1` was (0 examples, structurally unlearnable), but both are thin
enough that INS weighting is doing real, necessary work to keep the FC head from ignoring
that class entirely — worth watching in the real run's per-class recall numbers.

## 6. Chromosome → penalty matrix + weighting scheme

Three pool models means a 3x3 penalty matrix, and since the diagonal (correct predictions) is
always zero, that only needs 6 off-diagonal genes. On top of that, the chromosome carries one
more gene selecting the class-weighting scheme — `N_GENES = 3² - 3 + 1 = 7`, identical for both
variants — matching the paper's own chromosome description exactly: "we need (num. of DNNs)²
chromosomes to encode the P matrix and an extra one for the weighting scheme." This is a
reversal of an earlier decision: the first version of this track (and ImageNette/CIFAR-10,
unchanged) hardcoded INS and searched only the penalty genes. `dispatcher/penalty_matrix.py`
(shared, unmodified) fills the first 6 genes in row-major order, true class first, predicted
class second, and simply ignores whatever comes after — it was already written generically
enough that adding a trailing gene needed no change there.

Penalty range is `[0, 100]`, taken directly from the paper's Section II-C. The paper also
specifies a discretized step size in `[0.5, 1]` for penalty values; pymoo's SBX crossover and
polynomial mutation (what `dispatcher/nsga2_search.py` wires in, shared and unmodified)
operate over continuous real-valued genes with no native step-size or discretization control,
so this searches the same `[0, 100]` interval as a continuous space rather than a discretized
one — an approximation of what the paper describes, not an exact match.

The 7th gene picks the weighting scheme: `dispatcher/class_weights.py` implements all three the
paper names — INS (`1/count`), ISNS (`1/sqrt(count)`), and ENS (Cui et al. 2019,
`(1-beta)/(1-beta^count)`, `beta=0.999` — the paper doesn't state its own beta, this is a
project choice, not a paper-matched value). `dispatcher/weighting_scheme.py` decodes the gene
(range `[0, 3)`, chosen for this track — the paper doesn't say how it encodes a discrete 3-way
choice as a continuous MOEA gene) into one of the three by floor-and-clamp into equal bins.
Whether a track searches this gene at all is inferred structurally from `config.N_GENES`
(penalty-only count vs. that +1), not a separate flag, so ImageNette/CIFAR-10 stay on
unconditional INS without any change to their own configs.

This ripples further than just adding a gene: class weights can no longer be computed once,
upfront, for the whole search, since each individual's own chromosome now picks its own scheme.
`fitness.py`, `save_models.py`, and `dispatcher_analysis/build_models.py` all moved their class-
weight computation inside the per-individual loop. `save_results.py` records both the decoded
scheme name and the raw gene value per Pareto individual, so a front's exact chromosome is fully
recoverable from `pareto_front.csv` alone, not just its penalty values.

With `N_GENES=7` instead of the abandoned full-pool attempt's 30, `POPULATION_SIZE=50` /
`GENERATIONS=50` (the paper's exact values) searches a space close to the same order of
magnitude the paper's own subset runs search (a 3-model subset in the paper's own counting
formula needs 3²+1=10 chromosomes; a 2-model one needs 2²+1=5) — a real improvement in how
faithfully the search budget matches what's being reproduced, not just a smaller number for
its own sake.

## 7. NSGA-II: how the search is wired to the fitness function

`dispatcher/nsga2_search.py` (shared, unmodified) is the actual driver, run once independently
per variant: loads that variant's train + test-fitness embeddings (cached after the first
run), reads the test-fitness split's `<model>_correct` columns into a boolean correctness
matrix, then hands everything to a `DispatcherProblem` (`dispatcher/dispatcher_problem.py`)
wrapping `dispatcher/fitness.py`'s `evaluate_individual` as a pymoo `Problem`. Gene bounds are
now per-gene, not a single scalar pair — the 6 penalty genes get `[0, 100]`, the 7th gets
`[0, 3)` — `dispatcher_problem.py` builds a vector `xl`/`xu` for tracks that search the scheme
and falls back to the old scalar pair for tracks that don't.

For every individual pymoo asks about, `evaluate_individual` builds that individual's 3x3
penalty matrix from its first 6 genes, decodes its own weighting scheme from the 7th, computes
that scheme's class weights fresh from train labels, trains a fresh FC head from scratch on
train embeddings under the penalized+weighted loss for 20 epochs, then predicts on the
**test-fitness** embeddings (not train) and computes two objectives:

- `alpha_sys_loss = 1 - alpha_sys`, where `alpha_sys` is the fraction of test-fitness images
  where the model the dispatcher actually routed to is itself correct.
- `avg_model_cost`, the mean MFLOPs of whichever model each image got routed to, using that
  variant's own `MODEL_COST` list, **plus `config.DISPATCHER_OVERHEAD_COST`** (23.91 MFLOPs-M
  for both variants — the extractor + FC head cost, measured via `thop`, MACs doubled to
  FLOPs). This matches the paper's Eq. 4, which defines the cost objective as inclusive
  of "all operations performed by both the dispatcher and the selected model," and page 7,
  which confirms this inclusive figure is what decides Pareto dominance during the search
  itself, not just what gets reported afterward. Since this overhead is identical for every
  chromosome (same extractor, same FC head shape), it's a fixed constant added uniformly
  across the whole population — shifts a front's cost axis by one offset, doesn't change
  which individuals are non-dominated or the front's shape.

  `MODEL_COST` and `DISPATCHER_OVERHEAD_COST` are measured locally via `thop` (MACs doubled to
  FLOPs — see `models/model_loader.py`'s module docstring), not taken from chenyaofo's own
  published MAdds table the way an earlier version of this track did. The switch happened
  after chenyaofo's table-derived costs produced dispatched-cost values that didn't line up
  with where the paper's own Fig. 9 plots actually put PERTINENCE's points for the same
  subsets; the local, doubled thop measurement instead reproduces the paper's own published
  Table 5 dispatcher-overhead figure (24.17 MFLOPS for CIFAR-100/ShuffleNetV2) to within 1%,
  which is strong evidence the methodology (thop, doubled, 32x32 input) matches what the paper
  itself used. `MODEL_COST_UNIT` is `"MFLOPs-M"` now, not `"MAdds-M"`.

Both objectives are minimized. pymoo's `NSGA2` (population 50, 50 generations, SBX
eta=20/probability=0.9, polynomial mutation eta=25 — all four from the paper's Section II-C)
drives selection, crossover, and mutation. `progress_logger.py` checkpoints the population
every 5 generations. At the end, `save_results.py` writes the Pareto front's chromosomes +
objectives (plus scheme columns) to `pareto_front.csv`, and `save_models.py` retrains and
saves each Pareto individual's actual FC weights.

The smoke test (population 4, 2 generations, 2 FC epochs) confirmed this entire chain end to
end for **both** variants independently, on the current 7-gene/MFLOPs setup:

- **fig9c**: embeddings loaded (train 50000×1024, test-fitness 7000×1024, correctness matrix
  shape (7000, 3)), search log reported "chromosome-selected INS/ISNS/ENS class-weighted
  LOSS", front of 4 individuals saved with all three schemes actually represented (INS, ISNS,
  ENS each picked at least once — confirms the gene is live, not just present and ignored),
  cost ranged roughly 48-2845 MFLOPs-M across evaluations.
- **fig9d**: same shapes and scheme-detection log line, front of 4 individuals saved, cost
  ranged roughly 48-3081 MFLOPs-M.

Neither result is meaningful as a trade-off (4 individuals, 2 generations is far too small a
budget to have converged to anything), but both confirm the full mechanical chain — embedding
shapes, correctness matrix shapes, 7-gene chromosome decode, per-individual scheme selection,
FLOPs-based cost accounting — is wired correctly for a 3-model pool.

## 8. Final evaluation: two passes per variant, because there are three splits

`dispatcher_analysis/` (shared, unmodified) only has one held-out "val" slot in its config
contract. Each variant's own `run_dispatcher_analysis.py` runs the shared evaluation logic
**twice**, once per held-out set, instead of touching the shared code to add a second slot.

**Pass 1** points the shared code's `VAL_*` config attributes at the test-fitness split (7,000
images) — the same data NSGA-II's own fitness function used during the search. Useful as a
sanity check that retraining from the saved chromosome reproduces something close to what the
search's own fitness function reported, but not a genuinely held-out number.

**Pass 2** builds a lightweight view of the same config object
(`types.SimpleNamespace(**vars(config))`, four attributes overridden to point at
`FINAL_VAL_*`) and reruns the exact same shared functions against it — evaluating the *same*
retrained FC weights Pass 1 did, just against a different, genuinely-untouched 3,000-image
set. This trick lives entirely in each variant's own `run_dispatcher_analysis.py`; nothing in
`dispatcher_analysis/` itself knows a second pass is happening.

The smoke test ran both passes on both variants' 4-individual fronts:

- **fig9c**: Pass 1 alpha_sys ranged 70.2-74.5% across the 4 individuals; Pass 2 ranged
  71.7-75.4%. Close between passes, as expected from two independent stratified draws off the
  same 10k pool.
- **fig9d**: Pass 1 ranged 70.7-76.4%; Pass 2 ranged 72.4-76.3%. Same consistency.

Pass 2's `final_val_summary.csv` is what should get reported as each variant's actual result —
not Pass 1's `test_summary.csv`, which exists for continuity with the search's own fitness
metric, not as a held-out claim.

Each pass also renders `accuracy_vs_mflops_{test,final_val}.png` — the paper's own Fig. 6-10
plotting convention directly, not the loss-framed `pareto_scatter_*.png` this repo already had.
Accuracy (%) on the y-axis, cost on the x-axis, both "higher/lower is better," with three series:
PERTINENCE's own dispatcher configs (black), each individual pool model's own standalone
accuracy/cost — the paper's "SOTA CNNs" — (green, labeled by name), and the Pareto front
computed jointly across both series (red dashed line + markers). A SOTA point uses only that
model's own `MODEL_COST`, no `DISPATCHER_OVERHEAD_COST` — a standalone deployment never runs the
feature extractor or FC head, so it doesn't pay for either. `dispatcher_analysis/plots.py`
gained this as `plot_accuracy_vs_cost`, generic across every track (not cifar-100-specific),
though only fig9c/fig9d's `run_dispatcher_analysis.py` call it so far.

Building this surfaced a real, pre-existing bug in the shared `dispatcher_analysis/embeddings.py`:
its `ImageDataset` class was defined *inside* `_compute_embeddings()`, which Windows' spawn-based
multiprocessing can't pickle once `EMBEDDING_NUM_WORKERS>0` and no cache exists yet to skip the
DataLoader entirely. Fixed by moving it to module level and having it take `dataset_dir`/
`image_transform` directly rather than a whole `config` object — the latter mattered because
Pass 2's `types.SimpleNamespace` config view copies real imported modules (torchvision's
`transforms`) as plain attribute values, which don't pickle the same safe way a real module
does. Both issues are Windows-spawn-only (the A100's fork-based Linux multiprocessing never
needs to pickle anything here) but were fixed as general robustness, not platform appeasement.

## What matches the paper exactly, what's approximated, what doesn't match

**Matches exactly:**
- Model subsets: fig9c and fig9d use the exact three models named in Fig. 9(c)/9(d)'s
  captions, same checkpoint source the paper itself cites.
- Feature extractor: `shufflenetv2_x0_5`, exactly as the paper states for CIFAR-100, shared
  between both variants.
- Labeling rule: `argmin`-cheapest-correct, identical logic to every other track/variant.
- Loss structure: zero-if-correct, else `cross_entropy * penalty * class_weight` — same
  asymmetric penalty-matrix design as the paper's Eq. 6.
- Chromosome structure: penalty matrix genes PLUS a weighting-scheme selector gene
  (INS/ISNS/ENS), matching the paper's own chromosome description exactly — unlike ImageNette/
  CIFAR-10, which stay INS-only.
- NSGA-II hyperparameters where the shared code exposes a knob: population 50, generations
  50, SBX eta=20/probability=0.9, polynomial mutation eta=25, FC epochs 20, penalty range
  [0, 100] — all taken directly from Section II-C.
- Three-way train/test/validation split, matching the paper's own methodology text.
- Reproducing per-subset MOEA runs rather than one full-pool dispatcher — matching how the
  paper actually structures its CIFAR-100 experiments (Fig. 9), not just its final numbers.
- `avg_model_cost` — Eq. 4 defines the paper's cost objective as inclusive of the dispatcher's
  own overhead, and `dispatcher/fitness.py`/`dispatcher_analysis/summarize.py` (shared code,
  applies to all tracks) now add `config.DISPATCHER_OVERHEAD_COST` to match.
- Cost unit itself: measured locally via `thop`, MACs doubled to FLOPs, matching how the paper
  states it derives its own MFLOPS figures — and cross-validated against the paper's own
  published Table 5 dispatcher-overhead number (24.17 MFLOPS) to within 1%, not just assumed
  to match.
- Plot style: `accuracy_vs_mflops_*.png` reproduces the paper's own Fig. 6-10 layout —
  accuracy(%) vs. cost, PERTINENCE points, standalone "SOTA CNN" reference points, joint
  Pareto front.

**Close, deliberately approximated:**
- Penalty search space is continuous `[0, 100]`, not discretized to the paper's `[0.5, 1]`
  step size — no native support for that in pymoo's SBX/polynomial-mutation operators. The
  weighting-scheme gene has the same category of gap: the paper doesn't say how it encodes a
  discrete 3-way choice as a continuous MOEA gene, so `[0, 3)` split into 3 equal bins is this
  repo's own choice to fill that in.
- ENS's beta hyperparameter (0.999) is a project choice — the paper doesn't state which beta
  it used.
- Test/validation split point (70/30 of official test) and the earlier, now-abandoned
  10%-of-train figure are both project choices; the paper gives no split sizes for any
  dataset.

**Doesn't match anything left:**
- Nothing currently in this "doesn't match" category for fig9c/fig9d specifically — the
  weighting-scheme gap that used to be listed here is now implemented (see above). ImageNette
  and CIFAR-10 still hardcode INS only, unrevisited, which is a real gap for those two tracks,
  not for this one.

## What's still open

Neither variant's real search (population 50, 50 generations, 20 FC epochs — the paper's
actual numbers) has been run. That's the next step, on the A100, and it's a multi-hour job
per variant (two runs total) I need to trigger myself. Once each is done,
`run_dispatcher_analysis.py` needs to run for real against the resulting front — the
two-pass structure described above doesn't change, only the numbers do.

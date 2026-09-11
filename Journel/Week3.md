# Week 3

## Meeting notes & tasks

**Tasks**:
1. Redo the presentation of everything done so far, this time as an actual structured
   walkthrough (graphs, visualizations, tabulated data) instead of a codebase screen-share.
2. Nail down terminology upfront: accuracy vs alpha_sys vs "absolute accuracy" need to be
   clearly distinguished so the same confusion doesn't happen again.
3. Fix the CIFAR-10 reproduction's train/test mixing bug — the 70:30 split currently pools
   the official train+test sets before re-splitting, which leaks test images the pool
   checkpoints were trained on into the val split.
4. Add proper eval to the redo: per-model accuracy plus other relevant metrics, with
   conclusions drawn from them, not just alpha_sys/flops numbers presented in isolation.
5. Side-quest (open-ended, not a hard requirement): explore reducing the NSGA-II search's
   bias toward overestimation as a safety net.

---

## [MEETING] Redo requested — presentation, not results, was the problem

Meeting was over video call with Dr. Traiola. Presented by screen-sharing the codebase and
walking through it live, which didn't land, the flow was too hard to follow without a
structured walkthrough. Got a week to redo and re-present properly.

Two specific things caused confusion in the meeting:
- Terminology: accuracy vs alpha_sys vs "absolute accuracy" weren't clearly distinguished,
  so we ended up talking past each other on what each number actually meant.
- The CIFAR-10 alpha_sys ceiling (documented in `Journel/Week2.md`'s dispatcher_analysis
  eval entry — 99.0–99.4% across the whole Pareto front) read as suspicious in the meeting,
  even though it's already understood and explained (resnet20 alone gets ~99% of CIFAR-10
  val images right, so the pool's accuracy spread is just narrow). Same root cause as above,
  it wasn't presented with the supporting EDA, so it looked like an unexplained anomaly
  instead of an already-diagnosed dataset/pool property.

Neither issue was about the underlying work being wrong, both were about how it was
communicated. Redo requirements: short presentation, chronological (what was set up, what
ran, what came out), graphs and tables instead of code on screen, and a clear definitions
slide up front so terminology doesn't derail things again.

Also got an open-ended side-quest, not a hard requirement: the NSGA-II search currently
maximizes alpha_sys against avg_flops, which structurally biases the dispatcher toward
overestimation, since routing to a pricier-but-safe model is never penalized the way an
actual misclassification is. Worth exploring whether/how to reduce that bias.

## [DECISION] Fixing the CIFAR-10 split leak before the redo

`code/CIFAR_10_Implementation/label_data.py` pools CIFAR-10's official train (50k) + test
(10k) into one 60k set and draws a fresh stratified 70:30 split over the combined pool. The
`chenyaofo/pytorch-cifar-models` checkpoints were trained on the official 50k train set, so
any official-test images landing in the val split leak train-time-seen data into what's
supposed to be held-out evaluation, likely part of why the alpha_sys ceiling reads as
suspicious.

Fix: stop re-mixing. Official train (50k) is the only pool visible to labeling, NSGA-II
search, and dispatcher FC training. Official test (10k) stays untouched until a single
final held-out eval pass. No pre-filtering of the raw 60k either, everything runs through
as-is on both official splits. Labeling, NSGA-II, and dispatcher_analysis eval all need a
re-run on the corrected split before the redo presentation's numbers are trustworthy.

## [DECISION] Idea for the overestimation side-quest

The NSGA-II penalty matrix's initial population is currently random, which under-penalizes
overestimation early in the search purely by chance. Idea: seed the initial population
deliberately instead, a mix of individuals with absolute-strict penalty matrices (heavily
punish overestimation) and absolute-lax ones (barely punish it), so the search actually
covers that spectrum from generation zero rather than hoping crossover/mutation stumbles
onto it. Worth pursuing alongside, not instead of, a possible third objective or asymmetry
term that penalizes overestimation directly in the fitness function itself, biased init
helps exploration but doesn't necessarily change what the search converges to if the
objective itself still rewards overestimation.

## [SETUP] Repo restructure kicked off for the redo

Handed a refactor prompt to Claude Code covering: moving `quantization_experiments/` into
an `archive/` folder, fixing the CIFAR-10 split leak, adding a ground-truth EDA step
(per-model accuracy + theoretical max/oracle accuracy per split), adding an explicit
`accuracy` column to `dispatcher_analysis`'s eval output (exact-match against the ideal
label, distinct from alpha_sys), and collapsing the duplicated ImageNette/CIFAR-10
dispatcher and dispatcher_analysis code into one shared, generic codebase at the repo root
with thin per-dataset folders (`imagenette/`, `cifar-10/`) for config and entry points.
Sequenced so the pure file-move work (restructure + archive) happens first while the code
is still functionally unchanged, then the split fix, then EDA and the accuracy column last
on top of the corrected data and new layout. Re-running labeling/NSGA-II/eval on the fixed
split is still mine to trigger separately, not something to run unattended.

## [PIVOT] CIFAR-10 model pool swapped, then found the actual paper pool

Noticed the corrected-split ground-truth EDA showed the cheapest pool model
(`resnet20`) soaking up ~92-99% of routing labels on train/val — `argmin-cheapest-correct`
labeling always ties to the cheapest model, so whichever model is cheapest dominates
almost by construction. First swapped the pool to `shufflenetv2_x0_5/resnet20/resnet32/vgg11_bn`
(all pretrained, no training needed) to widen the spread — this just moved the dominance
onto the new cheapest model (`shufflenetv2_x0_5`, 90-99% of routes) rather than actually
fixing anything, since the root cause is structural to the labeling rule, not the specific
pool.

Then went and actually checked the PERTINENCE paper (arXiv 2507.01695) for its real CIFAR-10
pool: `resnet8, resnet14, shufflenetv2_x0_5, vgg16_bn` — not the `chenyaofo/pytorch-cifar-models`
pool this repo has been using (`resnet20/32/44/56`, no `resnet8`/`resnet14`). No public
pretrained checkpoints exist anywhere for CIFAR-10 resnet8/resnet14 (neither chenyaofo nor
akamaster's repo publish weights below `resnet20`) — matching the paper's exact pool means
defining the CIFAR-ResNet architecture at n=1/n=2 (`6n+2` depth family, so layers=[1,1,1]/[2,2,2])
and training both from scratch, not just swapping a config list. Not done yet — deferred,
this is a real chunk of new work (architecture + training script + actual GPU training time),
not a quick fix.

## [RESULT] Paper's fitness evaluation is on held-out data, not train — fixed to match

Checked the paper's actual methodology text for whether each NSGA-II individual's fitness
is computed on train or held-out data during the search (this repo's `dispatcher/fitness.py`
was evaluating on train). Paper states directly: "we perform the fitness evaluation for each
individual on the test set." Also confirms 20 FC-training epochs per individual (this repo
currently uses 30 — not changed, flagged only). No sign of a three-way train/val/test split
anywhere in the paper — it evaluates fitness on the very same set it reports final numbers
on, which is itself a mild leak in the paper's own methodology; this repo doesn't replicate
that specific leak but does now match the intent (fitness measured on unseen data at
FC-training time, not the same data the head was just trained on).

Fixed `dispatcher/fitness.py`, `dispatcher_problem.py`, `nsga2_search.py`, `embeddings.py`
(shared code, so this applies to both tracks): each individual's FC head still trains on
TRAIN embeddings, but `alpha_sys`/`avg_model_cost` for the NSGA-II objectives are now
computed by predicting on VAL embeddings/correctness instead. `dispatcher/` now reads
`config.VAL_GROUND_TRUTH_CSV`/`VAL_EMBEDDINGS_NPZ` for this — doesn't reopen the earlier
folder-boundary violation, since that was about physically copying val data into
`dispatcher/`, not about reading it via the same config-injection path every other value
already goes through. `save_models.py`'s final per-individual weight save is unaffected
(still trains on train only, no fitness eval involved there). All three completed NSGA-II
runs documented in CLAUDE.md predate this fix and need a re-run before being trusted as
the paper-matching numbers.

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

## [RESULT] First CIFAR-10 run on the corrected pool/split/fitness — real tradeoff, one dead class

Reran labeling, NSGA-II search, and dispatcher_analysis eval on the official split, the
substitute pool (`shufflenetv2_x0_5/resnet20/resnet32/vgg11_bn`), and the val-based
fitness fix. Result: a genuinely wide front this time — val alpha_sys 90.7-92.7% against
avg_model_cost 11.1-39.7 MAdds (~3.6x spread), instead of the old pool's 99.0-99.4%
near-ceiling band. The two ends of the front trade off very differently: the cheap end
(cost 12.1, alpha_sys 90.87%) has 88.5% exact-match-vs-ideal accuracy (mostly routes
correctly); the expensive end (cost 39.7, alpha_sys 92.70%) has only 25.8% exact-match
accuracy — it buys its extra alpha_sys almost entirely through overestimation, not better
routing decisions.

**Finding**: `recall_vgg11_bn` is exactly 0.0 for every one of the 50 Pareto individuals,
on both train and val. Root cause: `vgg11_bn`'s ideal-label count on TRAIN is exactly 0 —
`shufflenetv2_x0_5`/`resnet20`/`resnet32` alone already reach 100% oracle accuracy on
train (expected, since the chenyaofo checkpoints were trained on this exact 50k set), so
nothing ever needs the pool's most expensive model there. With zero training examples for
that class, the FC head has no gradient signal for it regardless of chromosome or INS
weighting — structurally unlearnable from this train set, not a bug to tune away. Shows
up on val as 256 guaranteed-miss images (196 genuinely impossible + 60 others) that no
dispatcher config on this front can route to correctly.

**Decision**: reporting both of these as-is rather than working around them — the
substitute pool (no public `resnet8`/`resnet14` CIFAR-10 checkpoints exist anywhere) and
the `vgg11_bn` dead-routing (an expected consequence of the data, not the dispatcher) are
both explainable limitations for the writeup, not things to fix before presenting.

## [SETUP] Built the CIFAR-100 track

Added a third track, `cifar-100/`, reproducing the paper's own CIFAR-100 experiment
(Figure 4b, page 5) as closely as the shared `dispatcher`/`dispatcher_analysis` code
allows, using `cifar-10/` as the structural template. Pool is the paper's actual
CIFAR-100 pool this time — `shufflenetv2_x0_5`, `mobilenetv2_x0_5`, `shufflenetv2_x1_0`,
`mobilenetv2_x1_4`, `repvgg_a1`, `repvgg_a2` — and unlike CIFAR-10, every one of these six
has a public pretrained checkpoint on `chenyaofo/pytorch-cifar-models`, confirmed against
the actual GitHub release assets before writing any code. No substitute-pool problem this
time. Embedding extractor is `shufflenetv2_x0_5` (cheapest pool model), matching the
paper's explicit statement for CIFAR-100 — measured 1024-dim, same as CIFAR-10's
extractor, since it's architecturally the same ShuffleNetV2 backbone.

Read the paper's actual hyperparameters for this track rather than reusing the other
tracks' `config.py` defaults: NSGA-II population 50 / generations 50 / SBX eta=20,
probability=0.9 / polynomial mutation eta=25 / FC epochs 20 / penalty range [0, 100] — all
Section II-C values, several of which genuinely differ from what ImageNette/CIFAR-10 use
(FC epochs 30 there, penalty range [0, 5]). Six-model pool means a 30-gene chromosome
(`6²-6` off-diagonal penalty entries) instead of 12 — `penalty_matrix.py` already
generalized cleanly since it loops over `NUM_CLASSES`, no shared-code change needed there.

## [DECISION] Three-way split, and getting it wrong once before getting it right

The paper's methodology (Section II-C, page 7) uses three disjoint sets, not the two-way
train/val the other tracks use — its own terminology is backwards from what you'd expect:
its "test set" is what fitness evaluates against every generation during the search, and
its "validation set" is the one genuinely held out until the very end ("a validation set
that was not used during model training or the MOEA process"). CIFAR-100 only ships an
official train(50k)/test(10k) division, so the paper's extra split had to be manufactured.

First attempt: carve the validation set as a stratified 10% slice (50/class) out of
official train, the obvious move for a dev split. The ground-truth EDA on that slice came
back looking wrong — 100% oracle accuracy, three of six routing classes with zero images.
Root cause: the six pool checkpoints are pretrained on the *entire* official 50k train
set, so anything carved out of train is already memorized by every pool model, no matter
how it's stratified. A "held-out" pass against it would report near-ceiling numbers that
say nothing about generalization — structurally the same category of problem as the
train/test mixing bug from earlier this week, just introduced fresh instead of inherited.

Fix: carve both the test-set role and the validation-set role out of the official **test**
split instead, since that's the only 10k images none of the six checkpoints saw during
their own training. Official test has exactly 100 images/class, split stratified 70/30
with a fixed seed: 7,000 for NSGA-II fitness, 3,000 held out as the real final validation.
Official train (50,000, untouched) is purely the FC-training pool. Re-ran the EDA after
the fix — test and final_val now track each other closely (67-77% per-model accuracy,
~90% oracle on both), which is the actual evidence the fix worked, not just a theoretical
argument. Neither the original 10%-of-train figure nor the 70/30-of-test split point are
paper values — it doesn't give split sizes for any of its three datasets, on any track.

Also surfaced, while looking at the corrected train-split class balance: `repvgg_a1` has
**zero** training examples (label never gets assigned to it on the 50k train set), and
`mobilenetv2_x1_4`/`repvgg_a2` have exactly one each. INS class weighting can't manufacture
gradient signal that isn't there — with zero examples, `repvgg_a1`'s weight is forced to
0.0 (division by zero clamped in `class_weights.py`), so no chromosome on this pool can
teach the dispatcher to route there. Same category of finding as this week's `vgg11_bn`
dead-routing on CIFAR-10, but caused structurally rather than incidentally, and worse (a
true zero, not just a thin minority class) — the paper's own class-imbalance discussion
(Table 3, page 6) never reports hitting an exact zero, so this reads as this pool's wider
accuracy/cost spread pushing a problem the paper already names further than it goes.

## [DECISION] `avg_model_cost` needed the dispatcher's own overhead added in

While writing up the CIFAR-100 pipeline report, went back to confirm exactly what the
paper's cost objective is measuring. It's not just how the paper *reports* MFLOPS
afterward — Eq. 4 defines the objective itself as "the average number of operations per
input sample, considering all operations performed by both the dispatcher and the selected
model," and page 7's description of the actual fitness step ties this inclusive figure
directly to Pareto-front updates during the search: "we measure ... MFLOPS ... considering
all operations, i.e., the input dispatcher (feature extraction and fully connected layer)
and the subsequent opportunistic DNN execution. Hence, we update the current Pareto front."
`dispatcher/fitness.py` and `dispatcher_analysis/summarize.py` were only counting the
dispatched model's own cost, undercounting relative to what the paper actually optimizes.

Fixed both (shared code, applies to all three tracks): `avg_model_cost` now adds a new
`config.DISPATCHER_OVERHEAD_COST` — feature extractor + FC head forward-pass cost, measured
per track via `thop`, the same tool `archive/model_analysis` already uses for its own FLOPs
numbers. Since this overhead is architecturally identical for every chromosome (same
extractor, same FC head shape regardless of penalty matrix), it's a fixed constant added
uniformly across the whole population — shifts a front's cost axis by one offset, doesn't
change which individuals are non-dominated or the front's shape. Practical consequence: old
fronts are still structurally valid, but no prior run's `avg_model_cost` (ImageNette Runs
#1-3, any CIFAR-10 run) is numerically comparable to a fresh run's absolute cost axis
anymore — on top of already predating the earlier val-based-fitness fix. Measured values:
1.824 GFLOPs (ImageNette, near-identical to ResNet18's own MODEL_COST entry since the
extractor is that same backbone), 11.96 MAdds-M (both CIFAR tracks — same ShuffleNetV2
extractor in both). Noted a local-`thop`-vs-chenyaofo's-own-published-table discrepancy
while measuring this (my local full-model measurement: 11.94M vs. their table's 10.90M) —
a known category of difference between FLOPs-counting tools/conventions, not a bug in how
the extractor gets stripped.

## [RESULT] CIFAR-100 track smoke-tested end to end

Full chain confirmed working on tiny hyperparameters (population 4, 2 generations, 2 FC
epochs) — not the real search, which is still to come on the university A100. Labeling ran
for real (not smoke-scale) since it's cheap relative to the search: 60k images x 6 models,
a few minutes on GPU. NSGA-II search completed, Pareto front + FC weights saved.
`run_dispatcher_analysis.py`'s two-pass design (test-fitness split, then the genuinely
held-out final-validation split, via a `types.SimpleNamespace` view of `config` rather than
touching shared code) ran cleanly both passes — alpha_sys landed in the high-60s/low-70s%
range on both passes, consistent with two independent stratified draws off the same 10k
pool, which is the actual evidence the split fix holds up end to end, not just in the EDA.

Full pipeline lifecycle writeup, including exactly what matches the paper, what's
approximated, and what's a settled divergence: `cifar-100/PIPELINE_REPORT.md`.

## [PIVOT] Dropped the six-model CIFAR-100 dispatcher, reproducing Fig. 9(c)/9(d) instead

Before committing a 12-hour A100 run to the smoke-tested six-model dispatcher above, went
back through the paper's actual CIFAR-100 results section (Fig. 9, page 8) to re-verify
every hyperparameter one more time. Found something that should have been caught earlier:
the paper never runs a dispatcher across its whole six-model pool. Every reported CIFAR-100
front comes from a separate MOEA search over a hand-picked 2- or 3-model subset — "we report
results on cases where the input can be dispatched to two CNN models (Fig. 9(a)-(b)), or
three CNN models (Fig. 9(c)-(d))." The paper's own stated reason (given for the CIFAR-10
case, same logic applies here): a cheap model becomes pointless once a strictly-better,
similarly-cheap alternative is already in the subset.

A six-way dispatcher isn't wrong in isolation, but it isn't reproducing anything the paper
actually reports, and it's a much harder search problem for the same budget — the paper's
subset runs search 2-6 continuous penalty genes at population 50/generation 50; the
six-model attempt searches 30 genes at the identical budget, a much sparser exploration of a
much bigger space. Decision: drop the six-model dispatcher and reproduce Fig. 9(c) and
Fig. 9(d) specifically instead, since those are real, named experiments in the paper rather
than a broader question the paper never asks.

Reading Fig. 9's captions closely (not just the surrounding text) surfaced a second finding:
Fig. 9(a) and 9(c) both use `mobilenetv2_x0_75`, a model that isn't one of the six plotted on
Fig. 4b and wasn't part of the original six-checkpoint download. Confirmed it's a real,
separately published chenyaofo checkpoint (not a typo for `mobilenetv2_x0_5`) against the
full chenyaofo CIFAR-100 model list before downloading it, then verified the checkpoint
loads cleanly (strict state_dict match, 100-class output) before trusting it. Same category
of gap the CIFAR-10 track hit with `resnet8`/`resnet14`, caught the same way — by actually
checking rather than assuming Fig. 4b's plotted set was exhaustive.

Rebuilt `cifar-100/` around this: shared infrastructure (raw dataset, checkpoints/loader,
config-driven `label_data.py`) at the root, two fully self-contained thin sub-tracks
(`fig9c/`, `fig9d/`) underneath, each with its own config/data/embeddings/results — not
shared, since a variant's routing labels depend on its own model subset and a shared
embeddings cache would silently serve one variant's labels alongside the other's embeddings
(same shape, wrong data, no error thrown — caught this before building it, not after).
Six-gene chromosome per variant (`3²-3`) instead of the abandoned attempt's 30.

Both sub-tracks labeled for real and smoke-tested end to end (NSGA-II + the two-pass
dispatcher_analysis). Both show the same split-quality signature the six-model version did —
test-fitness and final-validation oracle accuracy track within half a point of each other on
both variants (fig9c: 86.01% vs 86.53%; fig9d: 86.61% vs 86.60%) — confirming the
test/validation carve fix generalizes correctly to a smaller pool, not just the specific
six-model case it was originally diagnosed on.

## [DECISION] Cost metric switched from chenyaofo's published MAdds table to locally-measured thop FLOPs

Sanity-checked one of fig9d's own generation-12 log lines against the paper's actual Fig. 9(a)
plot before letting the real 12-hour search run any further, and the numbers didn't line up —
PERTINENCE's own dispatched-cost cluster in the paper's plot sits around 2000-3000 MFLOPs for a
subset whose individual models (per chenyaofo's published MAdds table, doubled to FLOPs) should
cap out under 400. Traced it to a genuine screenshot/caption misread at first (the panel shown
was actually Fig. 9(a), not 9(b) — captions sit below their own plot, not above, so the crop
made it look like the wrong panel), which resolved the apparent contradiction for that specific
case. But decided not to leave it there — measured every pool model directly with `thop` instead
of trusting chenyaofo's table, and cross-checked against a number the paper itself publishes
outright: Table 5's CIFAR-100/ShuffleNetV2 dispatcher overhead, 24.17 MFLOPS. My own thop
measurement of the extractor+FC overhead, doubled (MACs -> FLOPs, since the paper states it uses
THOP "to obtain MACs... and the resulting FLOPS"), comes out to 23.91 -- within 1%. That match is
strong enough evidence the measurement methodology (thop, doubled, 32x32 input) is right, and
worth switching the whole cost metric over to it rather than chenyaofo's table, which doesn't
carry the same direct paper cross-check.

Switched `MODEL_COST`/`DISPATCHER_OVERHEAD_COST` in both `fig9c/config.py` and `fig9d/config.py`
to this locally-measured, doubled-FLOPs convention (`MODEL_COST_UNIT = "MFLOPs-M"`, replacing
`"MAdds-M"`). `cifar-100/models/model_loader.py`'s `MADDS_M_BY_NAME` dict became
`MFLOPS_M_BY_NAME`. This is scoped to `fig9c`/`fig9d` only — ImageNette and CIFAR-10 keep their
existing MAdds/GFLOPs conventions, not revisited here, no shared-code change needed for the unit
switch itself (`MODEL_COST_UNIT` is just a display label, the shared `dispatcher/fitness.py`
doesn't care what unit the numbers are in).

## [DECISION] Implemented the weighting-scheme gene for real, in fig9c/fig9d

Reversed the earlier "INS only, no weighting-scheme gene" decision — for these two sub-tracks
specifically, not retroactively for ImageNette/CIFAR-10. Added ISNS and ENS (Cui et al. 2019,
beta=0.999 — the paper doesn't state its own beta, this is a project choice) to
`dispatcher/class_weights.py` alongside the existing INS, and a new
`dispatcher/weighting_scheme.py` that decodes a chromosome's trailing gene into one of the three.

Made this opt-in per track rather than a hard requirement everywhere, inferred structurally from
`config.N_GENES` (penalty-only count vs. that +1) instead of a separate boolean flag threaded
through every function — keeps ImageNette/CIFAR-10 completely unaffected (verified: both still
report `searches_weighting_scheme=False` and get unconditional INS) while `fig9c`/`fig9d` opt in
by setting `N_GENES=7` and defining `SCHEME_GENE_LOWER_BOUND`/`SCHEME_GENE_UPPER_BOUND=[0,3)`.

This touched more of the shared `dispatcher`/`dispatcher_analysis` code than expected once class
weights stopped being a fixed, precomputed-once array: `fitness.py`, `dispatcher_problem.py`
(per-gene xl/xu bounds now, not a single scalar pair), `nsga2_search.py`, `save_models.py`, and
`dispatcher_analysis/build_models.py` all needed their per-individual class-weight computation
moved inside the per-chromosome loop instead of hoisted above it, since each chromosome can now
pick its own scheme. `save_results.py` gained `scheme`/`scheme_gene` columns on
`pareto_front.csv` so a front's chromosome is fully recoverable from the CSV alone.

Re-ran both sub-tracks' smoke tests after this — pareto_front.csv for fig9c's tiny 4-individual
front shows all three schemes actually got selected (INS, ISNS, ENS each appearing at least
once), confirming the gene is live, not just present and ignored.

## [DEAD-END, fixed] dispatcher_analysis/embeddings.py's Dataset broke on Windows with no cache

Hit this rerunning fig9d's real `run_dispatcher_analysis.py` end to end (not just the smoke-test
harness): `_compute_embeddings()` defined its `ImageDataset` class *inside* the function, and
Windows' spawn-based multiprocessing can't pickle a class scoped like that when
`EMBEDDING_NUM_WORKERS>0` and no embeddings cache exists yet to skip the DataLoader path
entirely. Pre-existing bug in shared code (used by every track), just never triggered before
now — fig9c's earlier smoke test happened to have its embeddings already cached from a prior
run, so it never touched the code path where the bug lives.

Fixed by moving the class to module level (matching how `dispatcher/embeddings.py`, the
NSGA-II-side copy, already did it correctly) and having it take `dataset_dir`/`image_transform`
directly instead of a whole `config` object — the second half of the fix mattered because
`run_dispatcher_analysis.py`'s final-validation pass builds a `types.SimpleNamespace(**vars(config))`
view of config for its second pass, and that view copies real imported modules (torchvision's
`transforms`) as plain attribute values, which pickle by value and fail, unlike a real config
module (which pickles fine by reference, workers just re-import it by name). Both bugs are
Windows-spawn-only artifacts that wouldn't occur on the A100's fork-based Linux multiprocessing,
but fixed them anyway since they're general robustness issues, not platform appeasement.

## [RESULT] fig9c/fig9d re-smoke-tested end to end on the new chromosome/cost/plots

Confirmed clean on both variants after all three changes above: NSGA-II runs with the 7-gene
chromosome and reports "chromosome-selected INS/ISNS/ENS class-weighted LOSS" instead of the old
fixed-INS log line; costs now print in the MFLOPs-M range consistent with the doubled-FLOPs
convention (tens to low thousands, matching the pool's actual MFLOPs spread); and the real
`run_dispatcher_analysis.py` entry points (not just the scratch smoke-test harness) run both
passes cleanly end to end, producing the new `accuracy_vs_mflops_{test,final_val}.png` plots —
visually an exact match to the paper's own Fig. 6-10 style (PERTINENCE points, green "SOTA CNNs"
reference points per pool model, joint Pareto front). Neither variant's real 50-population/
50-generation/20-epoch search has been run yet — that's still the next step, on the A100.

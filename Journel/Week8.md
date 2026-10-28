# Week 8

## Meeting notes & tasks

No meeting this week: I was away from college (meet 8 skipped). Tasks carry over from Week 7:
1. Power profiling on a Jetson (no access yet).
2. Fix the CityFlow routing: run `cityflow-dispatcher/` on the A100 (no SSH from home).
3. Reimplement a couple of existing scheduling approaches and run them on real hardware.

## Log

## [DECISION] No simulator: evaluate on real hardware only

The discrete event simulator (`scheduler-sim/`) is no longer the evaluation path. Policies run on
real models on real hardware, so the costs a simulator never models show up:
- CPU preprocessing (decode, letterbox) and Python / launch overhead per batch;
- jitter in batch start and finish times, and in when arrivals actually land;
- thermal throttling and clock changes over long runs;
- CPU/GPU contention on the Jetson's shared memory;
- the cost of switching between models.

Development happens on the laptop RTX 5070 Ti (numbers there are not data); real measurements on
the Jetson. Real-time replay makes wall-clock time the budget: policies x loads x SLOs x run
length has to fit a Jetson visit.

## [DECISION] Promotion is not PERTINENCE-specific

Promotion (moving queued jobs into a bigger model's batch) only needs a pool of variants of the
same task, where a bigger variant is not worse on average. That covers model-variant serving,
cascades, early-exit networks and any accuracy-SLO setup, not just PERTINENCE routing. Per input a
bigger model can occasionally be worse, but on COCO the dominant disagreement pattern is "nano
alone fails", so it is a footnote. It fits the generic scheduler as a third action next to serve
(queue, b) and WAIT: promote k jobs from queue i into a batch of j > i, at the visible extra cost
T_j(b) instead of T_i(b). Not possible across unrelated tasks.

## [SETUP] Spec for `edge-runtime/`, a real-hardware scheduling harness

Wrote `research/spec_edge_runtime.md`, to hand to Claude Code. Main points:
- **Runtime**: an arrival thread replays traces in real time into per-model FIFO queues; one GPU
  thread runs one batch at a time. A policy returns Serve(queue, b), Wait(until), Promote or Drop;
  the executor validates every decision.
- **Power**: nvidia-smi backend (laptop, A100) and tegrastats backend (Jetson, every rail parsed,
  board power = VDD_IN if present). The Jetson power profile is a port of
  `yolo-analysis-a100/power_analysis/`, plus the nvpmodel mode recorded (changed only if sudo).
- **Calibration**: T_i(b) measured on the device (p50/p95/p99), plus model switch cost.
- **Workload**: periodic multi-camera traces (random phases, fixed or sticky split), Poisson, and
  CityFlow replay exported from the routing CSVs. Input as GPU tensors or real JPEG frames.
- **Policies, first batch**: FCFS no-batch, FCFS batch, LQF, Triton dynamic batcher (per-queue
  timeout grid), EdgeServing, Clockwork (INFER only), Symphony; my cost v1 as a stretch. Each
  checked against its paper before implementing. Needs per-job deadlines (SLOs) back.
- **Hidden-cost logging**: arrival lateness, decide() time, preprocessing time, model switches,
  clock and temperature drift.
- **Jetson visit**: `preflight.py` (board, JetPack, rails, sudo, models load) in under a minute,
  then `plan.py visit --budget-min 90` fitting calibration, power and policy runs into the visit,
  ending with one tarball.
- **Code style rules**: plain readable Python (explicit loops, plain classes, descriptive names,
  no comprehension chains, decorators, kwargs pass-through, inheritance or clever tricks), but no
  excess verbosity either.
- **Tests on real hardware only**: laptop GPU with real weights and real nvidia-smi; no CPU-only
  or fake-model testing. Policy decision asserts and tegrastats line fixtures are the exceptions,
  since they need no hardware.

`edge-runtime` covers the runtime only. The CityFlow routing fix is `cityflow-dispatcher/` (built,
needs an A100 run); its rerouted S02 traces then feed the CityFlow replay.

## [SETUP] Week 6-7 work pushed

Committed and pushed `cityflow-dispatcher/`, the A100 power results, the yolo-dispatcher and
CityFlow routing results (routing CSVs and analysis; per-frame embeddings git-ignored), the
yolo-analysis recall CSVs, the spec, and the Week 6-7 journal.

## [SETUP] Jetson access: AGX Xavier

Got a Jetson AGX Xavier (Volta GPU, 512 CUDA cores). Consequences for `edge-runtime`:
- JetPack 5.1 is the last release for Xavier, so CUDA 11.4 and Python 3.8 at best; the code must
  avoid 3.9+ syntax. If the board is on JetPack 4 (Python 3.6), ultralytics will not run and the
  board needs reflashing to JetPack 5.
- No `VDD_IN` rail: tegrastats reports GPU / CPU / SOC / CV / DDR / SYS5V rails, so board power is
  a sum of rails (whether SYS5V overlaps the others still to check).
- Power modes (nvpmodel) exist, which is the knob the A100 could not show.

Spec updated for this board (platform, Python 3.8 rules, rail handling, fixtures). Next: run the
device check (model, L4T/JetPack, Python, torch + CUDA, sudo, nvpmodel, tegrastats sample).

[SETUP] Repo restructure: all code folders and pptx/ moved into archive/ (history kept in git), venv, requirements.txt and __pycache__/.pytest_cache moved to _to_delete/ (git-ignored, deleted by hand), .gitignore paths prefixed with archive/. Reason: the repo had accumulated parts of old directions and I could no longer tell what each folder was for. The new codebases are written task-specific from scratch. The A100 was unreachable at the time, so nothing there was touched. Model caches (model_cache, embeddings_cache) were kept, since they may hold the only copy of the trained heads.

[DECISION] Start afresh. I could no longer tell what each folder in the repo was for, and part of it belonged to directions that are over (CIFAR/imagenette tracks, the simulator, the old batching pitch). Instead of cleaning in place, everything code-related is archived and each remaining task gets its own small codebase that I have the full context of: (1) Jetson profiling (latency and power vs batch size), (2) the CityFlow stream with the PERTINENCE dispatcher, (3) reimplemented policies on real hardware. Archived code stays in git history and can be pulled back when needed. Goal for this week: define the problem, and write the preface so that every claim is backed by a citation or a result I measured myself. Context: this is my first R&D project, so part of the work is figuring out what the problem actually is, and the repo had stopped helping with that.

[SETUP] edge-runtime/ (built by Claude Code from the spec, PR #8) archived too, to archive/edge-runtime/. Starting the Jetson codebase afresh so that I know every line of it.

## [RESULT] CityFlow routing: PERTINENCE NSGA-II dispatcher does not beat static policies

Built `cityflow-dispatcher/` out as a faithful port of the original PERTINENCE search (13-gene chromosome, pop 50,
50 generations, penalty-matrix loss, INS/ISNS/ENS class weighting, fitness on a held-out set, final set reported
once). An earlier "Part B" stand-in (linear-head grid, no NSGA-II) was my mistake and its numbers are not
PERTINENCE results; they are not used.

Labels: a frame is correct for a model if box recall vs yolov8l is >= 0.80 (IoU 0.5, road classes, conf 0.25);
ideal label = cheapest correct model; l is always correct by definition. CityFlow GT has no classes and only covers
vehicles seen by 2 or more cameras, so yolov8l is the reference.

Protocol (the clean one): head trained on S05 camera-fold A, NSGA-II fitness on S05 fold B, final report once on S02.
v1 fine-tuned pool (n, s, m; l stock). 2,500 evaluations, front over all evaluations. Run on CPU (the FC trainer
is a numpy port of the torch one, same maths); the laptop GPU was not reachable.

Static policies on S02 (alpha_sys, GFLOPs): n 0.518 @ 8.7, s 0.780 @ 28.6, m 0.867 @ 78.9, l 1.0 @ 165.1;
oracle 1.0 @ 32.1.

| protocol | front pts | above static hull | best gain over hull |
|---|---|---|---|
| s05 (clean) | 109 | 8 | +0.013 alpha_sys at 146 GFLOPs (84% routed to l) |
| train split (old COCO-style protocol) | 119 | 28 | +0.038 alpha_sys at 21.8 GFLOPs (50% n, 50% s) |

No front point is balanced (max share <= 0.7 and min share >= 0.05) in either protocol. The train-split variant is
not trustworthy: train labels are 88% n because the models were fine-tuned on those frames (val is 42% n).
Dispatcher cost includes the 3.156 GFLOP extractor, which static policies do not pay.

Reading: headroom is real (oracle is about 80% cheaper than always-l at full correctness), but a frame-embedding
dispatcher does not capture it. Not shown yet: whether labels, the weak pool or single-frame features are the cause.
Write-up: `cityflow-dispatcher/results/REPORT_PERTINENCE_v1.md`.

## [RESULT] Fine-tuning flaw found (v1)

The v1 fine-tune used AdamW lr0 1e-4 but the ultralytics default `warmup_bias_lr=0.1`, so the bias group ran at lr
0.067 then 0.033 in epochs 1 to 2. Saved checkpoints were epoch 10 (n), 3 (s), 1 (m), chosen by mAP vs pseudo-labels
on S05. m is weak on val. Fixed in `03_finetune.py` (`--warmup-bias-lr`, `--val-every`, `CF_TAG=_v2`); the v2 rerun
and the dispatcher rerun on v2 labels still need the laptop GPU and have not run.

## [DECISION] CityFlow thread parked, not closed (pending my call)

Open causes if I come back to it: label criterion (candidates: per-frame F1 vs confident l boxes, count tolerance,
temporal window, a reference beyond l), the v1 pool, single-frame features. A label shoot-out on the cached data
needs no GPU. For now the week goes to the Jetson work: the generic scheduler needs hardware numbers, not routing.

## [RESULT] Jetson power sweep, first numbers (partial)

MAXN, yolov8n, p50 batch latency: b=1 12.5 ms, b=2 15.5, b=4 28.2, b=8 53.7. Energy per frame 292 mJ at b=1, 235 at
b=2, 224 at b=4, 222 at b=8. So batch time is NOT size-independent on this board: roughly a launch floor plus a
linear term per frame, and the energy saving from batching is gone by b=4 for n. The size-independent-cost
assumption of Xia et al. and Chen and Wang does not hold here, which changes which prior art applies (SMDP-style
size-dependent batch service). The sweep did NOT finish: results stop at yolov8n b=8 in MAXN, and the board crashed (a MAXN run took it down before too). nvpmodel mode persists across reboots, so the board probably came back in MAXN and then crashed again on the harness preflight (SESSION_MODE None + locked clocks). Next: check `nvpmodel -q`, force mode 7, set SESSION_MODE explicitly, and keep clocks unlocked for the first run.

## [SETUP] Spec for `sched-harness/` (the proof harness for the meeting)

Wrote `research/spec_harness.md` for Claude Code: a smaller, plainer replacement for the archived edge-runtime.
One queue per model (n, s, m, l), seeded Poisson arrivals per queue, eager FP16, all models resident, ready input pool
(no camera or decode step), one batch on the GPU at a time. Metrics: mean and p99 latency, total energy, energy per request.
Policies first: fcfs_nobatch, fcfs_batch, LQF, Triton-style timer (delay grid), fixed_batch_b; index_T slot reserved.
Three claims to prove: batching matters on this board, standard policies leave something on the table, and the
harness is trustworthy (validation.md: job accounting, no overlapping batches, service time vs calibration,
Little's law, arrival rates, lateness, power coverage). Calibration fits T_i(b) = a + c*b per model from
measurements. Tier 1 is 18 runs (about 1 hour).

[DECISION] Harness energy metric: total energy = board energy from the first arrival of the measurement window to the completion of the last job of that window (idle included), identical arrival list across policies so only the end time and energy differ. Energy per request = total / jobs. Per-job lifecycle energy deferred (attribution rule needed first). Added to `research/spec_harness.md` with busy/idle split and two more validation checks.

## [SETUP] `sched-harness/` built and run on the AGX Xavier

Built from `research/spec_harness.md`: `01_calibrate.py` (T_i(b) per model, fit a + c*b, idle power, model-switch cost), `02_run.py` (one run), `03_plan.py` (Tier 1, 18 runs, resumable), `04_analyze.py` (summary, validation, plots), `common/` (policies, runtime, traffic, metrics, inputs, tegrastats sampler). Python 3.8.10, torch 2.0.0+nv23.05, ultralytics 8.3.186, 15W mode (`MODE_15W_DESKTOP`, mode 7) with `jetson_clocks` locked. Policies implemented: fcfs_nobatch, fcfs_batch, lqf, triton (per-queue timer), fixed_batch_b; symphony, clockwork, edgeserving and index_T are stubs. Unit tests (11) cover the policies.

Board-specific problems met on the way: (1) MAXN (~33 W) crashed the board on preflight, so the session mode is forced to 7 and clocks locked; (2) 0.5 s campus power dropouts also crash it, so everything runs in `tmux` and `go.sh` is rerunnable; (3) the PyPI torchvision 0.15.1 wheel does not match NVIDIA's torch build ("Couldn't load custom C++ ops"), so NMS falls back to a plain-torch implementation (`nms_plain_torch`, pairwise IoU in torch plus a greedy pass), recorded as `nms_backend` in every `run.json`.

## [DEAD-END] Threaded harness: arrival thread starved by the executor

First smoke runs failed check 3 (in-run service time 13-14% over calibration) and check 6 (arrival lateness p99 139-152 ms against a 5 ms limit). The arrival thread and the GPU executor shared one Python process, so the arrival thread waited for the GIL during each batch; jobs were enqueued late and in bursts. `sys.setswitchinterval(0.001)` did not help. Fix: the job maker is now its own process (`ArrivalProcess`, no torch, forked before CUDA init, pinned to its own core, sends tuples through a pipe), the scheduler and runtime stay in one single-threaded process on the other cores. Arrival lateness p99 is now about 0.7 ms. Two things changed at once (separate GIL and separate core), so I cannot say how much each contributed; unpinned-process test not run.

## [RESULT] Tier 1 sweep: 18 runs, skewed mix, one seed (`results/agx_xavier_jp511/`)

Skewed mix weights n/s/m/l = 0.528/0.149/0.083/0.240, Lambda_cap 27.4 req/s (balanced mix: 22.1). Loads were set as nominal rho 0.3/0.6/0.85 (= 8.2/16.4/23.3 req/s); 15 s warmup + 120 s measured per run.

- rho 0.3: all six policies are the same (mean 83-104 ms, about 670 mJ/request, mean batch size 1.06); nothing to batch. Energy is mostly idle.
- rho 0.6: fcfs_nobatch mean 273 ms, p99 840 ms, 522 mJ/req; batching policies 189-212 ms mean, 499-506 mJ/req. lqf p99 808 ms vs fcfs_batch 631 ms.
- rho 0.85: fcfs_nobatch collapses (mean 17.9 s, p99 30 s, 30 s drain tail, 512 mJ/req); batching policies 612-675 ms mean, p99 2.4-2.8 s, 427-432 mJ/req. lqf starves the m queue (per-queue p99 about 5 s vs 1.7 s for n).
- Batches stay small (mean 1.06 / 1.5 / 4.1 at the three loads; at rho 0.6 almost all batches are size 1-3), so nominal rho, which assumes full B_max = 16 batches, is optimistic.
- The batching policies are within a few percent of each other with one seed; a 20 ms Triton delay costs latency for about 1.5% energy. Claim 1 (batching matters) is supported from rho 0.6 up. Claim 2 (standard policies leave something on the table) is NOT yet shown by this data.

## [RESULT] Validation of the harness (checks 1-9 over the 18 runs)

Passing everywhere: counts, no overlap, arrival rates, arrival lateness (p99 about 0.7 ms), energy books (0.00% apart). Findings:
- Check 3 fails on 10 runs and it is real: calibration measured b in {1,2,3,4,6,8,12,16} and the check interpolates between them, but odd batch sizes in the runs are slower than the interpolation, and slower than the next even size (n forward: b=7 115 ms vs b=8 104 ms; b=15 271 ms vs b=16 198 ms). Suspected cause (not tested): eager FP16 kernel selection per batch size. A pad-to-a-good-size policy is a possible lever. Next: calibrate every b in 1..16.
- Check 7 fails on 14 runs: tegrastats delivers about 8.2 Hz when 10 Hz is asked (regular 121 ms gaps, so the tool itself runs at interval + about 21 ms; cause not confirmed), longest gap about 315 ms vs a 300 ms limit. Energy integrates on the real timestamps, so totals are unaffected; fix is to ask for about 80 ms or loosen the check.
- Check 4 fails only on the overloaded fcfs_nobatch run at rho 0.85 (no steady state): report as overloaded.
- Check 9 failed on a rounding bug in my checker (compared times at 1e-6 s); fixed to millisecond rounding, now passes for all three loads.
- Energy uses the five rails excluding SYS5V (`total_excl_sys5v`); whether SYS5V overlaps with the other rails is still unchecked on the board.

## [DECISION] Offered load is a hardcoded Lambda in req/s, not rho

rho was confusing, so `02_run.py` takes `--lam` (total req/s) and Lambda_cap is only printed for reference. Tier 1 skewed: 8, 16, 24 req/s (29%, 58% and 88% of capacity). Tier 2 values per mix (skewed 6, 11, 16, 22, 25; balanced 4, 9, 13, 18, 20) are proposed, not reviewed. Run folders are named `<mix>_lam<L>_<policy>_s<seed>`. The 18 finished runs keep their old `rho` folder names and were not rerun; the analysis groups by `lambda_total`, so they show as 8.2/16.4/23.3. All plots now have titles; the busy/idle legend sits outside the bars.

## [DECISION] EdgeServing baseline: plan (not built)

Read the paper (arXiv 2605.05527). One-step greedy, run whenever the GPU frees: one candidate per non-empty queue; batch B = min(|Q|, B_max = 10) fixed; exit = deepest one with w_max + L(m, e, B) <= tau; the candidate's predicted service time L* is added to every remaining task's wait in all queues; urgency f(w) = min(exp(w/tau - 1), C) summed over all tasks gives the score S; the candidate with the lowest S is dispatched. A single global tau (50 ms on RTX 3080 / GTX 1650, 100 ms on the Jetson Orin Nano), no drops (late tasks stay queued, capped at C so they cannot dominate), arrivals during service ignored, clocks locked at max. C is not stated (about 10 is implied by the "3.3 tau" cap). Unspecified in the paper: tie-break, what to do when no exit is feasible, FIFO order inside a queue, and whether the profile stores the average or P95 (the text says both).

Adaptation, since we have no exit dimension: the decision is (queue, batch size). For each non-empty queue pick the largest b <= min(|Q|, B_max) whose predicted completion keeps the oldest job within the SLO (the analogue of the deepest feasible exit), then score candidates with the same urgency sum. The SLO goes in `config.py` and is swept from tight to lenient. Fallback when no b is feasible is still open (my default would be b = 1). Everything else in the harness stays unchanged.

## [SETUP] EdgeServing policy implemented (not yet run on the board)

`EdgeServing(slo_ms, clip=10)` in `common/policies.py`, reduced to (queue, batch size). Per non-empty queue: the largest b <= min(length, B_max = 16) with oldest wait + T(b) <= SLO (T = p50 from the profile; the paper is ambiguous between average and P95); if none fits, b = 1 (my choice, the paper is silent). Score = sum over every job that stays queued, in all queues, of min(exp((wait + T(b)) / SLO - 1), 10); lowest score wins, ties go to the older head. It never waits and never drops. Decision cost is about 1.2 ms with 500 queued jobs (check `decide_ms` in `batches.csv`). 5 new unit tests, 16 pass.

SLO deadlines live in `config.SLO_MS_GRID = [150, 300, 600, 1200, 2400]` ms. Every run now also reports the share of requests above each deadline (`viol_pct_<slo>` in summary, plus a new `slo_violations` plot), so all policies are measured against the same deadlines; on the 18 existing runs, lqf is lowest at 150-600 ms and fcfs_nobatch is above everything at 16 and 24 req/s. EdgeServing runs are tuned to one deadline each: Tier 3 = 15 runs (skewed, Lambda 8/16/24 req/s, five SLOs, one seed), about 55 minutes; `ACTIVE_TIERS = [3]`. Caveat: the old Tier 1 runs sit at 8.2/16.4/23.3 req/s, so their arrival lists differ from Tier 3's. The clean comparison needs Tier 1 rerun at 8/16/24 (about 1 hour).

## [SETUP] Clockwork INFER policy implemented (not yet run on the board)

Read the scheduling part of Clockwork (OSDI 2020, arXiv 2006.02464; the authors' repo was not reachable, so this is from the paper text). Kept: per-model queues, batch sizes {1, 2, 4, 8, 16}, deadline = arrival + SLO, strategy (model, latest start = oldest deadline - predicted time), smallest latest start first, speculative increase to the largest batch that fits, and cancelling requests that cannot meet the deadline even at batch 1. Dropped (single GPU, all models resident): LOAD/UNLOAD and placement, scheduling ahead with a 5 ms lead time, the earliest/latest window. Execution times are the static calibrated p99 (the paper uses a rolling p99 over the last 10 actions). All five batch sizes are calibrated points, so nothing is interpolated and the odd-size slowness is avoided.

[DECISION] Greedy reading of an ambiguity: the batch is always the oldest b requests, so a bigger batch is never filled by skipping older ones; the batch size is capped by what the oldest live request tolerates. Ties: lower queue, then larger b. [DECISION] Cancelled requests are counted, not hidden: they are logged as dropped = 2, count as a miss at every deadline in `viol_pct_<slo>`, and new columns `policy_drops`, `goodput_pct` (served within the policy's own SLO over all window arrivals) and `energy_per_good_req_mj` exist for the SLO policies. Energy per request and latency stay over served requests only, so a policy that drops a lot looks cheaper on them; goodput is the figure to read next to them. Check 9 now compares arrivals, not served jobs. Little's law (check 4) is unchanged because it already covers served jobs only.

Plumbing: `service_ms(q, b, stat=None)` takes an optional statistic ('p99'); the runtime applies a policy's `drops` before serving; a stub-GPU run at 160 req/s cancelled 632 of 1322 jobs with logged == planned and no hang. Tier 4 = Clockwork at the five SLOs, 8/16/24 req/s, 15 runs; `ACTIVE_TIERS = [3, 4]` (30 runs, about 1 h 50). 21 unit tests pass.

Observation (stub only, not measured): my 2-D EdgeServing shrinks the batch whenever the oldest job is close to its deadline. Under backlog that lowers throughput, which makes jobs later still (in the stub at 160 req/s it ran 1307 batches for 1322 jobs). The paper's exit choice shortens service time, mine does the opposite. Worth watching in the Tier 3 results.

## [SETUP] Symphony policy implemented (not yet run on the board)

Read the paper (arXiv 2308.07470, Algorithm 1 and the appendix pseudocode). Symphony is multi-model and multi-GPU, not single-queue: each model keeps its own queue and a candidate batch, and a freed GPU takes the ready candidate with the smallest latest-start time. [DECISION] Run it multi-queue on the existing skewed mix, with that matchmaking rule as the paper states it; no new single-queue mix.

Per queue: drop jobs that cannot finish even alone; b = largest calibrated size that fits the live jobs and lets the oldest meet its deadline if started now; latest = d - T(b); frontrun = d - T(next size). A candidate is ready only after its frontrun, so the GPU idles on purpose until then (not work-conserving); the ready candidate with the smallest latest wins, otherwise `Wait` until the first frontrun. Dropped: staggering across GPUs, autoscaling and consolidation, network-delay terms, the scalability threads. Choices of mine where the paper is silent: execution time = calibrated p99 (it says only "profiled latency"); batch sizes limited to the calibrated ones {1,2,3,4,6,8,12,16} (the paper assumes every size is profiled; interpolated odd sizes run slower than predicted); drops counted like Clockwork (`dropped` = 2). Same SLO grid, Tier 5 = 15 runs at 8/16/24 req/s, `ACTIVE_TIERS = [3, 4, 5]` (45 runs, about 2 h 40). 27 unit tests pass.

Observation (stub GPU only, not measured): with four queues deferring to their last moment on ONE GPU, windows collide: a queue's batch is released at its frontrun, finds the GPU busy with another queue's batch, and by the time the GPU frees its deadline has passed. In the stub, SLO 150 ms at 20 req/s dropped 20 of 171 requests and SLO 600 ms dropped 6. The paper's staggering across many GPUs is what absorbs this, so on a single GPU Symphony may drop more than the others at tight SLOs. To confirm on the board.

## [RESULT] EdgeServing, Clockwork and Symphony on the board: skewed mix, seed 0, 45 runs

All three ran at SLO 150/300/600/1200/2400 ms and 8/16/24 req/s. Goodput = served within the SLO over all window arrivals (drops count as misses). At 16 req/s: SLO 150 ms Clockwork 83%, Symphony 70%, EdgeServing 60%; at 600 ms 99/88/98%; at 2400 ms 100/97/100%. At 24 req/s EdgeServing collapses at every SLO (goodput 1-15%, mean latency about 10 s, mean batch size 1.0): once the oldest job has outlived the SLO no b fits, the rule falls back to b = 1, service falls below the arrival rate, the queue never drains and the policy never sheds. This is my 2-D adaptation without the paper's exit/depth knobs, not EdgeServing itself. Clockwork gives the best goodput at tight SLOs (24 req/s, 150 ms: 75% vs Symphony 62% vs EdgeServing 8%); it cancels 25% of arrivals there but served requests finish at about the SLO (p99 close to the SLO). Symphony is non-work-conserving: its mean latency sits near the SLO (1.4 s at 2400 ms) and it batches deep (mean batch 7.6 at 16 req/s, 2400 ms), giving the lowest energy per good request at loose SLOs (445 vs 512 mJ, -13%, at 16 req/s and 2400 ms) but dropping a lot at tight ones (30% of arrivals at 150 ms, 16 req/s; 17% at 8 req/s). Per-queue drops for Symphony at 16 req/s: 600 ms n 11%, s 21%, m 24%, l 4%; 2400 ms n 1%, s 7%, m 12%, l 0%; Clockwork at 600 ms about 1% everywhere. My explanation (window collisions of the sparse, slower queues behind long batches on one GPU) is a hypothesis, not checked on `batches.csv`.

Validation over the 45 runs: checks 1, 2, 5, 6, 8 and 9 pass everywhere; check 3 fails on 6 (five are EdgeServing choosing odd sizes 5 and 7, 34-38% above the calibration), check 4 on 5 (all overloaded EdgeServing at 24 req/s), check 7 on 21 (8.25 Hz). The `drops` column counts every run's drops, `policy_drops` only those in the window. Busy/idle split of the energy (16 req/s): busy intervals draw about 8.6 W and the gaps between batches about 7.4 W against a calibrated idle floor of 2.9 W (total_excl_sys5v), so "idle" is not idle (CPU work between batches, sensor averaging at 8 Hz); use it as a guide, not as a measured share of waste. The busy/idle plot (`skewed_busy_idle_energy`) is now sorted by total energy with the dropped count on each bar.

## [DECISION] Seeds, skew ladder and fixed loads; fcfs/fixed/no-batch out of the comparison

Plan after the first results: compare only the three paper policies (EdgeServing, Clockwork, Symphony) at SLO 150/600/2400 ms, because fcfs_nobatch collapses and skews every plot and the work-conserving baselines tie. Three seeds everywhere (seed 0 existed for skewed). A skew ladder of eight mixes, one cheap-model-dominated direction (n40, n55, n70, n85 = share of the n queue, rest equal), balanced, the current skewed, and a heavy-model-dominated direction (l55, l85). Lambda_cap in req/s: l85 13.3, l55 16.7, balanced 22.4, n40 25.8, skewed 27.5, n55 30.3, n70 36.8, n85 46.9. My first plan had loads at about 60/85/100% of each mix's capacity; I changed it because I need every mix at the SAME req/s: 8, 16, 24 and 36 for all mixes (so l-dominant mixes are overloaded from 16 req/s up and n85 is at most 77% loaded). Run order (resumable): tier 6 (skewed seeds 1 and 2, 54 runs), then tier 7 seed by seed. Total 864 runs, 837 to go, about 50 h at 3.5 min per run; `ACTIVE_TIERS = [6, 7]`, `LADDER_LAMS = [8, 16, 24, 36]`, `MIX_ORDER` for the plots. New plots `mix_sweep_lam16/24/36` (goodput and energy per good request against the mix, one line per policy). The board ran it unattended; 636 runs were done on Tuesday morning and 682 run folders pulled (seeds 0 and 1 complete for every mix and load, seed 2 up to balanced and part of n40), the rest (about 10 h) still running when I analysed. The full `04_analyze.py` does not finish within a few minutes on 680 runs; I summarised in chunks and wrote `analysis/summary_partial_682runs.csv`. Rerun `04_analyze.py` on the laptop once the board is done.

## [RESULT] Policies across the skew ladder (682 runs, 2 to 3 seeds per point)

Seed noise on the skewed mix (3 seeds): goodput spread at most 1-2 points, energy per request about 2-3% (EdgeServing in its collapse is the exception), so differences above about 5% are real. The other mixes still have 2 seeds at some points.

- Clockwork has the best goodput at SLO 150 ms in 30 of 32 mix/load points (ties in the other 2), beating Symphony by 8-15 points at moderate load (balanced, 16 req/s: 76 vs 64%).
- Symphony's energy saving at loose SLO holds in every mix: SLO 2400 ms, 16 req/s, energy per good request Clockwork vs Symphony: balanced 575/531 (-8%), n40 530/472 (-11%), skewed 506/437 (-14%), n55 468/416 (-11%), n70 411/371 (-10%), n85 352/325 (-8%); at 8 req/s -4% to -18%, largest on the heavy-model mixes (l85 1051/867, l55 915/778). It costs 1-6 goodput points at light load. At 600 ms the energy is about equal and Symphony loses 8-10 goodput points, so deferral pays only with enough slack.
- Under overload (36 req/s) with loose SLOs Symphony matches or beats Clockwork on goodput and energy (balanced, 2400 ms: 65 vs 62%, 441 vs 463 mJ). Its single-GPU window-collision problem is a mid-load, tight-deadline problem.
- EdgeServing-2D spirals once load passes about 65-85% of a mix's capacity (balanced at 16 req/s, 600 ms: 60% goodput; skewed at 24 req/s: collapse at every SLO; fine on n70 and n85 below about 78% of capacity). Below that its energy ties with Clockwork.
- Energy per request falls with load (n85: 526 mJ at 8 req/s, 246 at 36), because the idle floor is spread over more requests; compare policies only at the same load.

The gap: no policy dominates. Clockwork wins goodput at tight and mid deadlines, Symphony wins energy by 8-18% at loose ones, and they converge in overload. A policy that defers like Symphony only when slack is large and the GPU is not needed by other queues, and sheds like Clockwork otherwise, is open space; that is a hypothesis for `index_T`, not a result.

Validation over the 682 runs: check 1 fails on 34 runs, all EdgeServing in overload (queue cap 512 overflows, drain timeout cuts the run, unfinished requests count as misses); check 4 on 107 (overload, no steady state); check 7 on 293 (8.25 Hz); check 3 on 47 (odd sizes). Still open: calibrate every b in 1..16, check 7 limit or interval, SYS5V overlap, the per-queue fairness of Clockwork's drops and the collision hypothesis for Symphony, Tier 1 baselines rerun at the new loads (no longer needed for the comparison).


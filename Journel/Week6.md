# Week 6

## Meeting notes & tasks

The problem itself is now understood (problem definition, measured T_i(b), simulator, baselines,
policy v1). This week is about the **why**: the motivation and background of the paper.

**Tasks**:
1. Find the mainstream inference-serving work, starting from Clockwork (OSDI 2020) and the systems
   derived from it, with a focus on ones built for edge devices.
2. Understand how each of them batches and schedules, and find the gap myself: "mainstream work
   does X and Y, but not Z, and that is what this work addresses".
3. Collect the material for the Background and Motivation sections. Research only this week, no
   writing yet.

Next meeting (meet 7): about a week away.

## Log

## [SETUP] Clockwork lineage: literature list

Targeted search for papers that build on, compare against, or cite Clockwork (Gujarati et al.,
OSDI 2020, https://www.usenix.org/conference/osdi20/presentation/gujarati), each checked against
its full text for the actual Clockwork mention. Citation-graph APIs were rate-limited, so this is a
targeted list, not every citing paper. Finding the gap from it is my job.

**Edge papers, closest to Clockwork's design first**:

| Paper | Link to Clockwork | Hardware | Mechanism | Objective |
|---|---|---|---|---|
| Jellyfish (RTSS 2022; Real-Time Systems 2024), https://doi.org/10.1007/s11241-024-09418-4 | Reuses its ideas (lazy dropping, queueing capped at execution time) and a Clockwork-inspired EDF baseline | edge server, 2x RTX 2080 Ti | Profiled latency; DP picks batch size per model under the SLO; picks model variants; drops late requests | accuracy under end-to-end latency SLOs over dynamic networks |
| EdgeServing (arXiv 2605.05527, ICCCN 2026), https://arxiv.org/abs/2605.05527 | Cites it as cloud-oriented; "orthogonal", not a baseline (uses Symphony) | RTX 3080, GTX 1650, Jetson Orin Nano | Time-division; stability score; batch = min(\|Q\|, 10); early exit; never waits; no drops | SLO violations, P95, accuracy |
| DREAM (ASPLOS 2023), https://doi.org/10.1145/3623278.3624753 | Related work: FCFS plus drop "not suitable" for dynamic multi-model workloads | simulated AR/VR multi-accelerator | Score-based scheduling, smart frame drop, no batching | deadline violations + energy (UXCost) |
| Adaptive Scheduling for Edge-Assisted DNN Serving (arXiv 2304.09961), https://arxiv.org/abs/2304.09961 | Contrast: no layer-wise batching in Clockwork | P100 edge server + Jetson Nano clients | DP over layer-wise batching segments; FIFO | completion time, on-time ratio |
| BCEdge (IEEE TNSM 2024), https://doi.org/10.1109/TNSM.2024.3409701 | Related work only | Jetson Xavier NX / Nano / TX2 / Orin | RL picks batch size + number of concurrent instances; interference predictor | throughput/latency utility under SLOs |
| Ibis (ACM TAAS 2023), https://par.nsf.gov/servlets/purl/10409959 | Related work only | Jetson Nano, GTX 1080, Edge TPU cluster | Queueing models for app placement across accelerators, not per-request batching | capacity under latency SLOs |
| Edge-Inference Governors Need Memory-Clock State (arXiv 2606.16106, 2026), https://arxiv.org/abs/2606.16106 | Cites it as motivation (predictability) | Jetson Orin Nano Super / Orin NX | Deadline-aware DVFS governor incl. memory clock; no batching | deadline-miss rate |

**Non-edge lineage**:
- Orloj (arXiv 2209.00159): built on Clockwork's code; distribution-aware batching for DNNs with
  unpredictable latency. https://arxiv.org/abs/2209.00159
- Shepherd (NSDI 2023): Clockwork as baseline; largest feasible batch across queues, preempts when a
  much bigger batch is ready; up to 4.6x Clockwork's goodput.
  https://www.usenix.org/conference/nsdi23/presentation/zhang-hong
- Symphony (arXiv 2308.07470): Clockwork as baseline; deliberately defers batches inside a window to
  grow them; drops requests that would miss the SLO. https://arxiv.org/abs/2308.07470
- AlpaServe (OSDI 2023): baseline "Clockwork++". https://www.usenix.org/conference/osdi23/presentation/li-zhouhan
- Paella (SOSP 2023): Clockwork as an overhead baseline only. https://vincen.tl/files/ng23paella.pdf
- REEF (OSDI 2022): baseline "SEQ" is Clockwork-style sequential execution.
  https://www.usenix.org/conference/osdi22/presentation/han
- DARIS (arXiv 2504.08795): related work only. https://arxiv.org/abs/2504.08795

**Not fully verified**: Octopus (ICSOC 2023; cites Clockwork, context paywalled); Interference-Aware
DNN Serving on Heterogeneous Processors in Edge Systems (Jetson AGX Xavier; cites Clockwork for the
goodput metric; venue probably ICCD 2024, unconfirmed); Sponge (EuroMLSys 2024); DEFT (preemptive
edge GPU sharing); SLO-aware scheduler for heterogeneous edge processors (TACO 2021); Abacus (SC
2021). Clockwork link not confirmed for these.

**Checked, do not cite Clockwork**: DeepRT, Miriam, Pantheon, OCTOPINF, Rim, Ekya, SLICE, WISP,
JigsawServe.

Next: read these myself and find the gap (what decides batch size, whether it ever waits on
purpose, what it optimises).

## [SETUP] Background and Motivation: first draft and positioning

Wrote a first draft of the Background and Motivation section, saved as-is in
`research/background_motivation_draft1.md` (working draft, not paper text; `[TODO]` = needs data,
`[VERIFY]` = taken from an abstract or secondhand, check against the full paper).

- **Structure**: 2.1 multi-model inference at the edge (PERTINENCE turns one stream into several
  per-model workloads on one accelerator); 2.2 batching and energy on edge accelerators; 2.3 the
  batch fragmentation problem (routing across N models splits arrivals over N queues, so batches
  shrink; waiting on one queue delays all others); 2.4 limitations of existing approaches; 2.5
  opportunity (reimplement Clockwork, Symphony, EdgeServing in one Jetson runtime, same traces).
- **Positioning (Table 1)**: compared on multi-model, edge hardware, energy objective, chooses
  batch size, waits deliberately. Clockwork (datacenter, deadline-greedy largest batch, no
  waiting), Symphony (deferral per queue, deadline-driven), EdgeServing (edge, always max batch, no
  waiting, clocks locked at max), E4 / PolyThrottle (energy on edge, single network, no batching).
  Ours: all five, with waiting across coupled queues and energy in the objective.
- 10 references. Pending: verify Table 1 cells against full papers, Jetson figures (Fig. 1 to 3),
  ResNet-primary / YOLO-secondary split to confirm with the profs.

## [DECISION] Plan to measure the gap: Jetson harness, baselines, energy floor

Todo list from here on. Each step lists what it produces. Steps 1 to 6 are also the Evaluation
section's infrastructure; our scheduler later becomes policy #5 in step 4.

0. **While waiting for the Jetson**
   - [ ] Read Symphony properly (can't reimplement a policy only seen secondhand in EdgeServing).
   - [ ] Get the PERTINENCE routing split for the ResNet pool: fraction of images the dispatcher
     sends to 18 / 34 / 50 / 152.
1. **Profile the Jetson**
   - [ ] T_i(b) per model, b = 1 to ~16, clocks locked, power mode written down. ResNets first,
     YOLO after.
   - [ ] p95 vs median per (model, batch): predictability check.
   - [ ] Power idle and while running each (model, batch), via `tegrastats`.
   - Produces: profile table, Fig. 1, energy cost per batch.
2. **Make the workload**
   - [ ] Total frame rate x routing split -> arrival rate per queue.
   - [ ] Arrival traces at a few loads (~20%, 50%, 80% of what the Jetson sustains), saved to files
     so every policy sees the exact same arrivals.
3. **Build the harness**
   - [ ] One executor, one batch at a time.
   - [ ] Policy interface: `decide(queues, now, profile) -> (queue, batch_size)` or `WAIT`.
   - [ ] Log every request (arrival, start, finish) and every batch (model, size), `tegrastats`
     logging alongside.
4. **Baselines, in this order**
   - [ ] Plain longest-queue-first (sanity check; EdgeServing uses it as a baseline too).
   - [ ] EdgeServing: stability score + `B = min(|Q|, B_max)`, no early exits.
   - [ ] Clockwork's INFER policy.
   - [ ] Symphony.
5. **Fidelity check**
   - [ ] EdgeServing shows its score beating longest-queue-first as load rises; the
     reimplementation should show the same ordering.
6. **Sweep**
   - [ ] Every policy x every load x a few SLO values. Measure p95 latency, SLO violation ratio,
     energy per frame, batch-size histogram, GPU busy fraction.
7. **Measure the gap**
   - [ ] Energy floor per model = energy per frame at max batch size (from the profile, no oracle).
     Compare each policy's actual energy per frame against it, per load.
8. **Decision point (bring to the profs)**
   - Baselines well above the floor at realistic loads while meeting latency -> real gap, build
     our scheduler in the same harness.
   - Baselines near the floor everywhere -> idea is weak, found out cheaply.
   - Gap only at some loads -> that is exactly where our policy needs to win.

## [RESULT] Issues with the current premise (external review of draft 1, Oct 8)

Two external reviews of `research/background_motivation_draft1.md` (the second read every paper in
full and supersedes the first). Issues found with the premise as written:

1. **Novelty is only a combination.** Every column of draft 1's Table 1 is covered by some
   published work. What survives is the conjunction: online serve-(queue, b)-or-WAIT across N
   coupled per-model queues, energy measured on device, on Jetson, with content-routed arrivals.
   A reviewer can call that incremental.
2. **Missed prior work.** Draft 1 cites none of: Xu, Zhou, Niu SMDP (arXiv 2301.12865,
   2501.02181), EdgeBatch (likely RTSS 2019), Fulcrum (2509.20205), DeepRT (SEC 2021),
   CascadeServe (2406.14424), Inoue (Perf. Eval., 1912.06322), Triton dynamic batching, and
   MOSAIC (2305.03222) / OCTOPINF (2502.01277) for the multi-camera setting. EdgeBatch already does
   energy + delay with a learned batch size and hold time on a Jetson TX2, but for a single model.
3. **Single-queue serve-or-wait is prior art.** Xu et al. solve optimal serve-b-or-WAIT under
   latency + power for one queue and call multi-queue an open problem. Only the multi-queue
   generalization can be claimed, and it must beat per-queue Xu threshold + queue selection.
4. **Single stream has nothing to batch.** At 15 FPS the next frame is 66.67 ms away, frames are
   processed one at a time and the chosen model is held for D frames. "One stream becomes several
   workloads" (2.1) is wrong. The primary setting has to be a multi-camera box; drones only for
   latency motivation.
5. **Energy is not busy time.** On Jetson, power rises with batch size and static power is a large
   share, so busy time overstates batching gains (~1.8x in Fulcrum's data). Needs measured board
   power: dynamic energy (P_i(b) - P_idle) x T_i(b) and total board energy, same power mode for
   every policy. WAIT costs ~0 dynamic energy on an always-on device; the accounting has to be
   stated either way.
6. **Wait cost is EdgeServing's.** The uncapped A·exp(w/τ) with A = e^-1 is EdgeServing's
   f(w) = min(exp(w/τ - 1), C) without the cap. Cite EdgeServing for the form and justify no cap,
   or switch to Xu's integrated holding cost.
7. **"Just tune the baseline."** Triton with grid-searched per-queue timeouts must be a baseline,
   and the work has to show those timeouts fail when the routing split shifts. EdgeBatch-style hold
   and per-queue Xu are also needed.
8. **Routing may not fragment much.** The split may be skewed and sticky, so queues barely thin
   out. Not measured yet; PERTINENCE's 1146/1800 TinyYOLOv3 figure is an oracle count, not a
   routing measurement.
9. **Draft 1 sentences to fix.**
   - Clockwork picks the most urgent valid (model, size) then grows it, not the largest feasible
     batch.
   - Symphony's windows are per model but dispatch is arbitrated centrally; cite Symphony itself.
   - EdgeServing also picks the exit point and has an accuracy objective.
   - PolyThrottle does tune batch size (offline).
   - The knobs of E4 / PolyThrottle / Pagoda are coupled with batching, not orthogonal:
     "complementary but coupled".
   - Rey et al. say cloud latency "limits" real-time use, not "prohibitive"; Rey has no
     multi-camera setup.
   - Boroujerdian is true but weak for energy: the effect is through latency and mass, compute is
     3 to 24% of drone power. Move energy motivation to fixed / solar camera boxes.
   - EcoLens tunes camera-side knobs; inference runs off-device.
   - EdgeServing and PolyThrottle have no verified venue: cite as arXiv.
   - Table 1 is replaced by the corrected comparison from the second review.

**Kill criteria**: if the routing split is above ~90:10 and stays fixed for seconds, or batching
energy gains on the Orin flatten by b = 2 to 4, the problem is too small; fallback is joint
power-mode + batching control (same harness).

**Possible uniquifier: promotion.** In PERTINENCE, overestimation (sending a frame to a bigger
model than it needs) is accuracy-safe; only underestimation costs accuracy. So a queued frame can
be promoted into a bigger model's batch to fill it without hurting accuracy: the scheduler gains a
lever (where a frame runs, not just when). Not yet checked against prior work: EdgeServing already
picks model and exit point per batch, and INFaaS-style variant selection is adjacent. Cost: the promoted frame runs on a more expensive model, and the
decision gets more complex, since each frame can now go to its own queue or any larger one, so the
option set grows well beyond M x k + 1.

## [SETUP] yolo-analysis renamed to yolo-analysis-a100; A100 power sweep built

Renamed `yolo-analysis/` to `yolo-analysis-a100/`: everything in it (recall benchmark, timing
sweeps) ran on the A100, and Jetson numbers will need their own folder. References are updated in
code, READMEs, `.gitignore` and CLAUDE.md; journal weeks 0-5 keep the old path. The `.gitignore`
rules now point at the folders that exist (`batch_sweep_DENSE/`, `batch_sweep_OLD/`: raw/,
engines/, logs/) instead of the old `results/batch_sweep/`.

Built `sweep.py power` in `yolo-analysis-a100/batch_sweep/` as the A100 stand-in for E_i(b)
until there is a Jetson. Measured:
- eager_fp32 only (the TRT outputs are wrong);
- n/s/m/l x the dense grid (1, 2..32 step 2, 40, 48, 56, 64) x 3 runs.

Method:
- Power from `nvidia-smi -lms 50`, no pynvml. `power.draw.instant` when the driver gives it,
  else `power.draw`, which is a ~1 s average on Ampere. The field used is recorded everywhere,
  and each window records how often the reading really changes (the sensor may refresh slower
  than 50 ms).
- Two idle levels: no CUDA context (60 s), and per model with context + model loaded (30 s).
  The second is what WAIT costs and is the baseline for dynamic energy.
- Model load and unload, 3 runs per model, one process each, every step timed with its power
  and energy:
  - load: process start, imports, CUDA context, weights from disk, copy to GPU, first forward;
  - then the 30 s idle above;
  - unload: free the model, context teardown at exit, 5 s tail back to idle.

  This is what swapping a model in and out would cost the scheduler.
- Load runs: 5 s+ warmup, then 20 s of back-to-back forwards with one sync at the end.
- Outputs: J/frame (total and dynamic), mean/p95 power, throttle flags (`sw_power_cap` expected
  at large b), and per-step load/unload time and energy.

Tested on CPU against a fake nvidia-smi whose power is a known sine of wall-clock time: window
energy matches the closed form to <1%. A worker killed mid-window (timeout or abort) leaves no
sampler behind. Ran once on the A100 (copied over by scp); the results looked right.

## [DECISION] Power sweep rebuilt from scratch as a small separate tool

The first power sweep worked, but the code was far heavier than the problem: about 2,800 lines
plus 650 lines of tests. It had four layers of sampler cleanup, two clocks per sample, Windows
support, an 11-step lifecycle, nine plot types and many diagnostics. For a measurement I need to
read and trust myself, that is the wrong trade.

Changes:
- Rolled it back completely. The timing code is back to its original state, apart from the path
  rename.
- Renamed `batch_sweep/` to `batch_timing_analysis/`. The result folders keep their
  `batch_sweep_*` names.
- Rebuilt the power measurement as `yolo-analysis-a100/power_analysis/`: 4 files, ~520 lines,
  plain functions, one entry point (`python power_sweep.py`).
- Relaxed one requirement: model load/unload cost is no longer measured. The model is loaded
  before the idle phase starts.

Design:
- One `nvidia-smi -lms 50 -f` sampler for the whole sweep, so there is one continuous power
  trace.
- 20 s baseline with nothing loaded.
- Then every (model, batch) once, in random order, each in its own worker process:
  - load the model and a real COCO batch, one warmup forward;
  - 20 s idle with the model loaded;
  - 5 run phases (200 forwards of the batch, however long that takes, then sync), with 3 s
    buffers. On the A100 that is ~1 s for yolov8n batch 1 up to ~45 s for yolov8l batch 64.
- The worker only records `time.time()` marks. Afterwards the trace is cut with them:
  - phase power = mean of the samples inside the phase;
  - energy = power x runtime;
  - energy per batch = energy / forwards;
  - energy per frame = energy per batch / batch size.
- Outputs: one raw trace plot per combo, and per model, power / energy per batch / energy per
  frame vs batch size.

Smoke-tested on the laptop's RTX 5070 Ti (real nvidia-smi, `power.draw.instant`, untrained
yolov8n/l, batch 1 and 16, 3 phases of 100 forwards). The power steps in the raw traces line up
with the idle and run phases, and every run phase had at least 13 samples.

The test also showed that nvidia-smi writes its `-f` file in ~4 KB chunks (~6 s of samples), so
the trace stopped ~6 s before the end. The sweep now samples 10 s past the last combo.

The laptop's numbers are not data: it was power-limited, at ~20x slower than the A100. About
2 h 10 min on the A100; not run there yet.

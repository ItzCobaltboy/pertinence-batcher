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

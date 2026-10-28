# Week 9

## Meeting notes & tasks

[MEETING] Latest meeting: the room left in the current batching + deadline model is small, so the supervisors pointed me at DVFS on the Jetson. Goal order: deadline first (primary), energy saving second.

**Tasks, in order**:
1. Part 1: profile YOLO latency and energy across locked GPU frequencies (an offline table that acts as the knob), plus the cost of switching frequency. Gaps between runs so a hot board does not bias power.
2. Part 2: a first-iteration policy on top of that table.
3. Check whether rival papers already cover this.

**Schedule**: Part 1 profiling Oct 29 - Nov 2, switching cost Nov 3, literature check Nov 4, Part 2 policy Nov 5 - 11.

**Formal problem** (from the follow-up message after the meeting): given K DNN model variants with per-model queues, jobs routed by input complexity and arriving in bursts, and a single non-preemptive edge GPU with discrete core and memory frequencies, choose at each dispatch point (queue, batch size, GPU frequency, and optionally a complexity-aware downgrade) to minimise energy per job, subject to every admitted job meeting its deadline and an accuracy floor. Reading list from the same message: BEAM (MLSys 2026, https://openreview.net/pdf?id=BfNBXM8CCT, slides https://mlsys.org/media/mlsys-2026/Slides/3849_mCVfQGX.pdf; built for vLLM, but a fine-grained controller co-optimising resources and power under per-request SLOs) and EdgeInfer.

## Log

## [DECISION] The problem as formalised, and what it changes for Part 1

Why combining existing pieces naively fails, in three points:
1. Slack is shared across queues, so frequency is a global decision. A batch at lower frequency runs longer on a non-preemptive GPU and delays the head of every other queue.
2. Batching and frequency compete for the same slack. Batching creates slack (batch 1 to 10 costs only 2-3x the latency per EdgeInfer), slack can become lower frequency, but waiting to form a batch also spends slack. The best (batch size, frequency) depends on every queue's deadlines.
3. The cost of downgrading depends on the input: simple inputs lose almost nothing, complex ones lose a lot. Under overload, raising frequency (energy) versus downgrading (accuracy) can be priced per job.

What it changes for me:
- Memory frequency (EMC) is now part of the problem. Part 1 as planned pins only the GPU clock. Plan: EMC at max for the first sweep, then a small second sweep (about 3 EMC levels x a few GPU frequencies). A full cross-product would take days of board time.
- "Every admitted job" makes rejection part of the problem, so the metric rule below matters even more.
- The novelty claim has to survive PolyThrottle (frequency + batch size on Jetson, single model, offline). The gap rests on multi-queue, per-job deadline guarantees and input-aware downgrade, not on frequency + batching. Checked properly on Nov 4.
- The downgrade needs a per-input accuracy signal and the CityFlow dispatcher is parked, so it may stay an extension.

## [DECISION] Energy metric: total board energy over the fixed window, dropped requests shown separately

Energy per good request (energy / goodput) was wrong: a policy that drops more serves fewer requests, so the figure gets inflated or flattered depending on the drops. Now: total board energy over the fixed 120 s window [t0+15, t0+135]; goodput = % of in-window arrivals served within their deadline (drops count as misses); % dropped plotted next to the energy, never energy alone. The earlier reading "Clockwork dominates Symphony on energy" came from that metric and is retracted. The energy-per-good-request numbers in Week 8's `[RESULT]` entries are superseded by this one.

## [RESULT] Policies on the board, re-read with the corrected metric (skew ladder, 2 to 3 seeds)

- Symphony uses the least energy: 11-19% less than Clockwork at SLO 2400 ms at low load, 12-16% less on three mixes at medium load; the gap shrinks at high load. It drops more.
- Clockwork buys goodput: +11-14 points at SLO 150 ms for 3-7% more energy; at 2400 ms, 10-19% more energy buys only 2-4 points.
- Greedy-2D collapses at or near capacity (3-17% goodput at 24 req/s) while still spending about 1.1-1.2 kJ.
- Goodput caps under overload: 62-64% (l55 at 24 req/s), 84% (balanced at 24 req/s).
- Seed 2 was still running when I wrote this, so the l55 and n70 points have 2 seeds. Differences under about 5% are not trusted.

## [DECISION] Naming: Greedy-2D, and Clockwork is "Clockwork-style"

"Greedy-2D" replaces "EdgeServing (2-D)": it is my 2-D adaptation without the paper's exit/depth knobs, not EdgeServing itself. My Clockwork INFER implementation is Clockwork-style: the oldest live job caps the batch size, where the paper's larger-size queue excludes old requests. A synthetic replay against a paper-style version (queue per batch size) shows 0-0.7% of decisions differing at light and medium load or loose deadlines, but 12% (skewed 24 req/s at 600 ms), 26% (balanced 24 at 600 ms), 71% (balanced 24 at 2400 ms) and 20-27% (l55 16 at 600 and 2400 ms) under pressure; when they differ the paper-style batch is about 2-3x larger. Faithful per-size-queue rerun (about 36 runs, about 2 h of board time) not done. Symphony and Greedy-2D are not audited against their papers either.

## [RESULT] Energy per batch is an estimate, not a measurement

Calibration measures time only. Energy per batch was estimated by regression over 682 runs: E_run = P_gap*(T - busy) + sum_q(kappa*a_q*N_q + B_q*F_q), with P_gap about 3.95 W, kappa about 6.8 W and B (J per frame) of n 0.155, s 0.272, m 0.643, l 1.05. Per-frame energy from b = 1 to 16 falls 27 / 24 / 17 / 16% for n / s / m / l. Labelled as an estimate on the deck. Part 1 replaces it with direct measurement.

## [SETUP] Week 8 supervisor deck

`research/week8_meeting_deck.pptx`, 21 slides: overview, how the harness works, arrival model, calibration, energy vs batch size (estimated), policies, goodput and total energy at low / medium / high load with % dropped on the bars, goodput vs energy scatters, CityFlow method / result / why, limits, next steps. Rules I set for it: nothing from `results_old/` (stale), no repo or code details, "Greedy-2D" as the name. Needs a refresh once seed 2 finishes.

## [SETUP] DVFS literature scan and Xavier GPU frequency facts

Scanned, with what each does not cover:
- PolyThrottle (arXiv 2310.19991): Jetson TX2 / Orin, minimum energy per query under a latency SLO, CPU / GPU / memory frequency + batch size, constrained Bayesian optimisation, offline, single model, about 17 ms per config write (up to about 150 ms for all three), no thermal control.
- arXiv 2502.06295: Xavier NX / Orin Nano, latency model t = a*f^-b + c, E ~ kappa*f^3*t, no batching.
- E4 (AAAI 2025): CPU/GPU DVFS + early exit on Jetsons, switch overhead not quantified.
- BatchDVFS (Nabavinejad, TPDS 2022): Tesla P40, about 100 ms per DVFS change, no deadlines.
- LATEST (arXiv 2502.20075): GPU frequency switching takes 5-480 ms on A100 / GH200 / Quadro, no Jetson.

Not read yet: "Leveraging DVFS for Energy-Efficient and QoS-aware Edge Video Analytics" (ACM fetch failed), SEEB-GPU, SparseDVFS, DEFT, plus BEAM and EdgeInfer from the supervisor.

Provisional gap: online per-decision DVFS inside a multi-queue, deadline-driven scheduler on a Jetson, with measured switch cost and directly measured energy.

Xavier GPU facts: devfreq dir `/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/` (`available_frequencies`, `cur_freq`, `min_freq`, `max_freq`); pin by writing the same value to `min_freq` and `max_freq` (raise: max first; lower: min first). Range about 114.75 MHz to 1377 MHz in MAXN. Mode 7's own cap is not in the table I found; read it from `/etc/nvpmodel.conf` and `max_freq`. `cur_freq` can mislead, so verify with tegrastats `GR3D_FREQ x%@MHz`. `jetson_clocks` pins the GPU at max, so the GPU pin is applied after it.

## [SETUP] Part 1 profiling plan

Pin GPU frequency (min = max), grid = frequencies x yolov8 n/s/m/l x batch {1, 2, 4, 8, 16}; back-to-back batches for at least 8 s and 50 batches per point; energy per batch by integrating tegrastats power over the timed window; idle power per frequency; randomised point order, fixed fan; thermal gate (wait until near the idle baseline + 2 C and flat, at least 20 s gap), a reference-point drift check every about 20 points, flag points that heat by more than 5 C, abort above 80 C or on a dmesg over-current / throttle message. Resumable CSV. Switching cost per (from, to) pair, up vs down, 20 repeats: write time, settle time (within 5% of steady for 3 batches), stall. Sanity check against `profile.json` within about 5%. Stay in nvpmodel mode 7. Full grid is about 280 points (6-8 h of board time); start with every other frequency. Code goes under `sched-harness/dvfs/`, written by Claude Code from a spec; smoke test (one frequency, n and l, b = 1 and 16) before the full run. Part 2 sketch: Clockwork-style queue and batch choice for the deadline, then the lowest feasible frequency from the p99 table plus switch cost with hysteresis, compared against the same policy at fixed max frequency.

## [DEAD-END] MAXN crashed the board; cause not established

The board crashed in MAXN. Candidates, in the order I would check: weak power supply or brownout, over-current protection, thermal, driver. I have not confirmed any of them (symptom, fan setting and adapter not yet checked). Part 1 stays in mode 7 and skips the 1377 MHz point. MAXN only gets retried after a PSU and thermal check, as a short run with tegrastats open.

## [DECISION] Jetson run results stay out of git

Run outputs are local only. `.gitignore` gained `sched-harness/results/`, `results_old/`, `sched-harness/**/__pycache__/` and `weights/`. Only small summaries get committed, if ever.

## Open

- Board status: is the seed-2 run done? Frequency list and mode 7 cap still to be read from the board (`nvpmodel -q`, `available_frequencies`, the mode 7 block of `/etc/nvpmodel.conf`).
- Part 1 prompt for Claude Code, then smoke test, then the full run.
- Cost-aware shedding idea (shed expensive requests first under overload) parked, no prototype.
- CityFlow dispatcher fix needs the A100; Background and Motivation draft 2 waits for Jetson numbers; TensorRT YOLOv8 export still broken (eager only).

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

## [SETUP] Part 1 profiling code built and run: `dvfs/` (as built vs as planned)

Built by Claude Code (PR #9, `dvfs-profiling`) as a top-level `dvfs/` folder, not under `sched-harness/`: `inspect`, `profile_switch`, `profile_models`, `gpu.py` (pin and restore clocks), `tegrastats.py` (sampler), `board_info.py`, `run_all.sh` (inspect, switch, models in one go, logging to `runlog.txt`). It ran from a renamed copy on the board (`~/cs24bt033/dvfs_runtime`, python from the `jetson-power` venv).

What differs from the plan above:
- Power mode **4 (MODE_30W_6CORE)**, GPU cap 905.25 MHz, not mode 7. The harness results are in mode 7, so the two are not yet on the same footing (see Open).
- 6 clocks: 216.75, 318.75, 420.75, 522.75, 675.75, 905.25 MHz (left out 114.75, 624.75 and 828.75: too slow, or too close to a neighbour). Batch sizes {1, 2, 3, 4, 6, 8, 12, 16} (not {1, 2, 4, 8, 16}), 4 models, so 192 points.
- Each point is 4 subruns of 20 back-to-back batches, with fixed gaps (10 s between subruns, 20 s after a point). No thermal gate and no abort limit: temperature, `cur_freq` and dmesg are recorded next to every subrun, never acted on.
- EMC, CPU clocks, fan and `jetson_clocks` are not touched, so the EMC question from the problem statement is still open.
- Switch cost: 30 ordered pairs x 5 repeats = 150 switches.
- CRLF line endings made `run_all.sh` fail on the board; converted to LF.

## [DEAD-END] Two days lost to a hang in the sampler (Oct 28 - 30)

`tegrastats.Sampler` runs a background thread that polled dmesg every second with `subprocess.run(["dmesg"])`, while the main process was running torch/CUDA. Forking from a thread of a CUDA process can deadlock (the fork copies a lock held by another thread). The model profiling hung on its own; 11 `models_*` result folders between Oct 29 and Oct 30 morning hold only meta and raw logs, no `run_summary.csv`. Fix (`tegrastats.py`): read new kernel log lines straight from `/dev/kmsg` with non-blocking reads, no subprocess at all. The first run after it finished. I read the cause off the fix and its comment; I did not reproduce the hang. Lesson: no `subprocess` from a sampler thread in a process that holds CUDA; and look at `runlog.txt` timestamps every hour, so a hang costs one hour.

## [RESULT] Part 1: latency and energy across locked GPU clocks (mode 4, 192 points)

Run Oct 30 13:36:56 - 18:36:32 (5.0 h, as estimated), `dvfs/results/agx_xavier_jp511/models_20261030_133704/` (`run_summary.csv` 192 rows, `subruns.csv`, `batches.csv`). Clocks restored afterwards (min/max/governor equal to before). Checks: `clock_held_all` = 1 on all 192 points, 0 dmesg throttle or over-current hits, max temperature 42.8 C (range 37.5 - 42.8). First look only, not yet compared with `profile.json` from the harness.

| model | lowest J/frame | highest J/frame (216.75 MHz) |
|---|---|---|
| yolov8n | 0.129 at 905.25 MHz, b=2 | 0.216 (b=2) |
| yolov8s | 0.252 at 905.25 MHz, b=1 | 0.426 (b=1) |
| yolov8m | 0.584 at 675.75 MHz, b=16 | 0.961 (b=1) |
| yolov8l | 0.883 at 675.75 MHz, b=16 | 1.491 (b=1) |

The lowest clock is the worst energy per frame for every model, and the best is not always the top clock (m and l bottom out at 675.75 MHz). yolov8n at b=1: 0.213 / 0.178 / 0.146 / 0.144 / 0.152 / 0.165 J/frame from 216.75 up to 905.25 MHz, so there is a flat minimum around 420 - 520 MHz. Latency at b=1 drops 58.2 ms to 29.2 ms over the same range. Caveats: only 4 subruns x 20 batches per point, so the fastest points have very short power windows (check `n_power_samples` before trusting them), and this is mode 4 with EMC and CPU clocks free.

## [RESULT] Part 1: cost of switching the GPU clock

Three complete runs (`switch_20261028_193920`, `switch_20261029_154127`, `switch_20261029_191106`, 150 switches each; `switch_20261029_190404` has no summary). The sysfs write takes about 88 ms (87 - 90 ms, std about 1 ms) and `cur_freq` shows the new clock about 1 ms later; tegrastats `GR3D_FREQ` shows it only after about 125 - 157 ms (sampling lag, noisier). The three runs agree within 1 - 2 ms per pair. So one frequency change costs about the time of 1.5 - 3 yolov8n batches at b=1 (29 - 58 ms): Part 2 cannot change frequency per batch for free; it needs hysteresis or a per-window frequency plan.

## [SETUP] `dvfs/analysis.py`: plots and tables for Part 1 (Oct 30)

`python dvfs/analysis.py` (from anywhere). It drops `dvfs/` from `sys.path` before importing matplotlib, because `dvfs/inspect.py` shadows the stdlib `inspect`. It reads the newest `models_*` folder that has a `run_summary.csv` plus every `switch_*` folder with a `switch_raw.csv`, and writes 16 files to `dvfs/plots/<models run>/`: latency, energy per frame and power against clock with one line per batch size (01 log axis, 01b linear, 02, 03), heatmaps batch x clock (04, 05 normalised to the best), the latency-energy menu with its Pareto front (06 log, 06b linear), the t = a*f^-b + c fits (07, 08), data quality (09), switch time per pair, detail and cost in batches (10 - 12), `summary.txt` and `best_clock_table.csv`. numpy, pandas and matplotlib only. Whether the PNGs go into git is not decided.

## [RESULT] Part 1 first read of the plots (mode 4, 192 points, 4 switch runs)

- Whole-board energy per frame (`total_excl_sys5v`) is lowest at 675.75 or 905.25 MHz for every model; 216.75 MHz is the worst for every model.
- Saving of the best clock against the top clock, median over batch sizes: whole board n 0.3%, s 0.7%, m 1.8%, l 3.1%, at +5 to +27% latency. On the GPU rail alone (`power_gpu_w`): n 11.8%, s 12.0%, m 13.4%, l 14.0%. The GPU rail is only 26% (216.75 MHz) to 44% (905.25 MHz) of the measured power (GPU 1.9 to 6.6 W of 7.3 to 14.8 W), so most of the measured energy is CPU, SOC and DRAM that keep burning while a slow batch runs. "DVFS barely saves energy" holds for whole-board energy only.
- Latency fit t = a*f^-b + c: b is 1.04 - 1.14 per model. Idle-gap power rises from about 2.19 W (216.75 MHz) to 3.11 W (905.25 MHz).
- yolov8n b=1: forward time stops falling from 675.75 to 905.25 MHz (21.1 to 22.1 ms). NMS is about 74 ms at b=16 at every clock (CPU side). Small models and small batches therefore do not follow the GPU clock; tentatively CPU or launch bound, not confirmed.
- 53 of 192 points are noisy (window under 2 s, under 10 power samples, or energy spread over 5%); the worst spread is 22.5% (yolov8s b=1 at 905.25 MHz, 0.63 s window). The 12.8% saving at yolov8n b=1 is one of them.
- Temperature 37.0 - 42.8 C with no drift against run order (correlation -0.1; mean start temperature 38.0 - 38.5 C at every clock). The run order was seeded random.
- Switching, 482 switches in 4 runs: sysfs write 88 ms, `cur_freq` 89 ms, `tegrastats` 143 - 145 ms. The `tegrastats` figures cluster near 180 ms, which looks like sampling quantisation, so quote the write time. One switch = 1.5 - 3 yolov8n b=1 batches, but only 0.02 - 0.4 of a batch at b=16.

## [RESULT] Part 1 methodology: second opinion (Claude, Oct 30)

Sound: clocks pinned by min = max and checked with `cur_freq` and `GR3D_FREQ` (`clock_held` 192 of 192), seeded random order, clocks restored afterwards, direct power measurement (not a regression), idle-gap power measured per clock, subrun spread reported, temperature checked.

Weak points, by impact:
1. Metric. Whole-board energy and GPU-rail energy tell different stories (3% against 13% at the best clock). Report both, and decide which one the paper minimises. Check whether the CPU spins in the CUDA sync during long batches, since that would inflate the non-GPU share at low clocks.
2. Saturated back-to-back batches measure active energy only. The scheduler's quantity is energy over a window with idle gaps: E = E_active + P_idle * (T - busy). The fair baseline is "run at one clock, drop to the lowest clock when idle", at one 88 ms switch per idle period. Part 2 should be judged in window energy, using the idle power per clock already measured.
3. Eager PyTorch with CPU-side NMS (TensorRT export is still broken). The result for small models at small batches mostly measures the CPU. CPU clocks (mode 4 governor) and EMC are free and not recorded; record both per subrun at least. State "eager PyTorch" as a limit of the table.
4. Fixed 20 batches per subrun gives the fastest points 0.55 s windows with 8 - 15 power samples. Use a minimum time per subrun (about 5 s), or rerun only the 53 flagged points.
5. The switch stall is not measured. 88 ms is how long the sysfs write takes, not how much a switch delays the next or the in-flight batch. The plan had stall and settle time; measure the latency of a batch issued right after a switch.
6. One pass, 4 subruns per point, no repeat on another day, no confidence intervals. Treat differences under about 3% as ties. A latency fit on 6 points with 3 parameters is a summary, not a law.
7. Mode 4 against mode 7 is still unreconciled (see Open).

## [SETUP] Paper to cite: Dutt et al., energy impact of device parameters on edge DNN inference (Oct 30)

A. Dutt, S. P. Rachuri, A. Lobo, N. Shaik, A. Gandhi, Z. Liu (Stony Brook), "Evaluating the energy impact of device parameters for DNN inference on edge", https://www3.cs.stonybrook.edu/~anshul/igsc23-dnnedge.pdf (file name `igsc23-dnnedge.pdf`; the venue, probably IGSC 2023, is not confirmed: check before citing). To cite in Background and Motivation.

What it measures (from the paper's abstract and body, as far as I read them): Jetson Nano and Xavier NX, CPU and GPU clocks swept, batch size fixed (16, or 8 for MobileNet-v2), six workloads, energy for a fixed 3,200 inferences; memory (EMC) frequency is not studied at all. Default settings are not energy-optimal: the energy minimum over CPU + GPU clocks is 9.9 - 19.3% below the stock DVFS governor on the Nano and 2.5 - 15.1% on the Xavier NX, at 28 - 35% higher latency on the Nano. They find CPU frequency has a smaller impact than GPU frequency (AlexNet on the Nano: power span 4.7 W across GPU clocks against 1 W across CPU clocks) and energy changes little with CPU frequency at a given GPU clock. The energy-optimal GPU clock is about 614 MHz on the Nano and about 803 MHz on the Xavier NX; the optimal CPU clock varies by workload.

The figure the paper shows (energy against GPU and CPU frequency, flat "DVFS" plane = stock governor), minimum against stock DVFS, read off the plot: ResNet-18 208.3 J vs 213.4 J (2.4% lower), YOLOv4-Tiny 614.7 J vs 705.6 J (12.9%), BERT-Tiny 73.9 J vs 87.1 J (15.2%). All three surfaces are bowls in GPU frequency (minimum around 0.8 - 0.9 GHz). The figure (Figure 3 of the paper) is the Xavier NX, batch size 16, a fixed 3,200 inferences. The paper says the CPU axis matters less than the GPU axis, so decision (Oct 30): do not sweep CPU or EMC clocks yet, but log both per subrun so a confound would show. Memory frequency has no evidence either way in this paper: deferred, not ruled out.

How it relates to Part 1: same U shape in energy against GPU clock, and the saving against the stock governor can be small (ResNet-18 here) or large depending on the model. The baseline to beat is the stock governor, not a pinned top clock. It does not do batching, deadlines or several queues.

## [SETUP] ResNet run support in `dvfs/` (Oct 30, same stack, not run yet)

Decision: run the Part 1 grid on ResNets as well (lighter models, no NMS, a same-task variant ladder), with the old stack and no CPU or EMC scaling (CPU frequency matters less than GPU frequency in Dutt et al.; EMC is untested there, deferred not ruled out). Done by editing `dvfs/` directly, default behaviour unchanged (YOLO dry run still 192 runs, about 5.0 h):
- `settings.py`: `FAMILY` from the environment variable `DVFS_FAMILY` ("yolo" default, or "resnet"). ResNet: queues r18, r50, r152 (r34 and r101 are in the name table, add them to `QUEUES`), torchvision with random weights in FP16, 224x224 input, no post-processing.
- `models.py`: `load_resnet`, NMS setup skipped, `run_batch` returns nms_ms = 0 for a classifier.
- `profile_models.py`: new `MIN_SUBRUN_S` (a subrun runs at least `SUBRUN_BATCHES` batches and at least that many seconds: 5.0 for ResNet, 0.0 for YOLO, so the YOLO run is reproduced exactly). Without it 20 ResNet batches would be a window with almost no power samples and the run would abort. Folder name gets a `_resnet` suffix. Dry run: 144 runs, about 2.9 h (more at the low clocks for resnet152).
- `analysis.py`: models are discovered from the run (ordered lightest first), names and colours are filled in, layouts handle any number of models.
- Inputs: with no files in `dvfs/data/inputs/` the ResNet pool is `RANDOM_IMAGES = 10` seeded random-noise images (224x224), cycled; a classifier's compute does not depend on content. Real jpg/png files placed in that folder override it. Caveat: noise may draw slightly more dynamic power than natural images. The YOLO run still uses the two Ultralytics sample images. `profile_models.py` prints the pool source as `inputs:`.
Not tested on the board (no torch here): the dry runs, a compile of every file and the plots on fake ResNet data were. Run: `cd ~/cs24bt033/dvfs_runtime && sudo env DVFS_FAMILY=resnet STEPS=models bash run_all.sh` in tmux, after copying `dvfs/` over.

## [RESULT] ResNet Part 1: same hockey stick, a real U only at batch 1-2 (Oct 31)

Run `models_20261030_205537_resnet`: r18, r50, r152 x 6 clocks x batch {1..16} = 144 points, 5 s windows, 0 of 144 flagged, no dmesg hits, max 39.8 C. Whole-board energy per frame, best clock vs the top clock (905 MHz).
- Batch >= 3: the curve is flat from 676 to 905 MHz (energy within 0-2.6%), the best clock is 676 (or 905). The latency cost of 676 is +28% for r18/r50. The roughly 20% gap read off the paper's ResNet-18 surface is not reproduced here. r18 b=16 gives 1.7%, the paper's ResNet-18 b=16 gave 2.4% (208.3 vs 213.4 J, but against stock DVFS), so they agree.
- Batch 1 (and mildly 2): a real interior optimum. r152 b=1 best at 421 MHz, 14.8% below the top clock for +7.6% latency (0.332 vs 0.390 J/frame); r50 b=1 9.2% at 421; r18 b=1 7.8% at 523. r152 b=2 6.8% at 676. Median over batch sizes: r18 1.8%, r50 0.8%, r152 0.8%.
- Mechanism: at b=1 latency stops falling above about 420-676 MHz (r152 b=1: 58.4 ms at 421, 53.5 at 676, 54.3 at 905) while power keeps rising (5.7 to 7.2 W). Eager PyTorch launch overhead is the floor, so extra GPU clock only burns power. A graph-compiled engine would probably flatten this (TensorRT export is still broken, so it is untested).
- r152 b=16: latency 254.8 to 197.4 ms (-22.5%) while power goes 10.3 to 13.4 W (+30%): a wash (0.165 vs 0.166 J/frame), the same trade as yolov8l b=16.
- Caveats: single pass, no confidence intervals; subrun power scatter about 2-3%, so 421 vs 523 at b=1 is within noise (the r152 b=1 latency at 523 is above that at 421); whole-board metric only, GPU rail not re-examined.
- Policy reading: slack exists at low load, and low load means small batches, which is exactly where the U is. At high load batches are large, the curve is flat, and running fast costs at most about 2%.

## [RESULT] ResNet in MAXN up to 1377 MHz: the right arm of the curve exists (Oct 31)

Run `models_20261031_100223_resnet` (nvpmodel MAXN, mode 0, 8 cores, GPU cap 1377 MHz): r18, r50, r152 x 11 clocks (420.75 to 1377) x batch {1..16} = 264 points, 264 of 264 done, 0 flagged, no dmesg hits, max 47.0 C. The mode was switched to MAXN after the mode-4 ResNet run (no reboot). Results for the earlier mode-4 runs live in `dvfs_runtime/results_part run/`.
- Energy per frame has a flat bottom at 676-905 MHz (within about 3%) and a steep right arm above about 1000 MHz. Against the top clock the best clock saves 22-27% at every batch size (median r18 24.2%, r50 23.1%, r152 22.6%); best clock is 829 or 905 MHz for batch >= 2, 676-829 for batch 1.
- r18 b=16 energy per frame relative to 905 MHz: 1033 +6%, 1198 +15%, 1377 +30%. Latency 31.3 to 22.7 ms (-27%) while power 15.3 to 27.4 W (+79%). r152 b=16 is the same (+28% energy for -27% latency).
- Batch 1 is launch-bound: r152 b=1 latency 31.7 ms at 905 vs 30.6 at 1377 (-3.5%), energy +31%. Nothing is gained above about 830 MHz there.
- So the earlier "flat from 676 to 905" (mode 4) and this result agree: the U is a flat bottom plus an expensive top end that mode 4 could not reach.
- Caveat, mode confound: at the same clock MAXN costs 13-18% more energy per frame than mode 4 (8 cores at 2.26 GHz) and is 1-2% faster at b >= 3, 11-16% faster at b <= 2 (faster CPU shortens the launch-bound batches). MAXN best point (r18 b=16, 29.8 mJ at 829) is about 12% above mode 4's best (26.6 mJ at 676). Absolute numbers are not comparable across modes. Switching cost was measured in mode 4 only.
- Policy reading: 905 to 1377 buys 27% latency for about 30% energy, so the clock above about 905 is a deadline-only resource. Default 829-905, go up only when slack requires.
- YOLO in MAXN (`models_20261031_152516`) did not run: the board went down about 10 s after start, during model loading (tegrastats and dmesg logs truncated, GPU idle at 114 MHz, all 8 CPUs at 100% at 2259 MHz, board draw about 13 W, well below the ResNet run's peak). Cause not established; a high GPU clock is not the trigger. Candidates: CPU-load transient or the supply.

## [DECISION] Policy v1 written up; documents now go out as artifacts (Nov 2)

Plan for the week: (1) build the theoretical policy, (2) the literature check; the policy is run on the board next week. Documents are artifacts from now on (nothing new goes into the project's `claude/` folder except the weekly slides).
- Policy spec v1 (artifact, "Scheduling policy v1: deadline first, energy second"): https://claude.ai/artifact/PHjhkeprpfqE1vaLLaCe4U
- Rule in one line: EDF for the queue, clock = max(f*, f_feas) with f* the fastest clock within 2% of the energy minimum, an admission test that prices the round-trip switch, hysteresis, max clock under overload, no lookahead.
- Claim to test offline: v1 lands near fixed-f* on energy and near always-max on deadline misses (miss rate never above always-max).
- Board inputs still missing: MAXN switch cost, stall during the write, switch energy, YOLO tables in MAXN, a repeat pass for the noise level.
- Decisions for Gayathri: energy metric (board vs GPU rail), power mode for the evaluation, idle clock.

## [SETUP] Week 9 supervisor deck: DVFS only (Nov 3)

`pertinence_week9_dvfs.pptx`, 12 slides with speaker notes, native charts from the run data: setup and runs, mode-4 YOLO (flat bottom), latency vs clock (batch 1 floor), MAXN energy curve, why the top is expensive, savings vs the top clock (22-24%), the 88 ms switch, policy v1 in words (two slides), offline test plan, caveats and next steps. Built outside the repo; a copy goes to the project's `claude/` folder only after Affan has looked at it.

## Open

- **Power mode:** the Part 1 table is mode 4 (30W, cap 905.25 MHz), the harness results are mode 7 (15W). Decide which mode Part 2 runs in; if mode 7, rerun a reduced Part 1 there. Until then the table cannot be combined with the Week 8 policy results.
- Check the Part 1 table against `profile.json` (sanity) and `n_power_samples` on the fast points; EMC question (EMC at max first, then a small sweep).
- From the methodology review: choose the energy metric (whole board vs GPU rail, report both); record CPU and EMC clocks per subrun; rerun the 53 noisy points with longer windows; measure the post-switch stall; judge Part 2 in window energy with idle at the lowest clock.
- Commit `dvfs/` as it ran: it has uncommitted line-ending changes, the `tegrastats.py` fix, `board_info.py` and the results folder; decide what of `results/` goes into git (the Week 8 rule kept board results out). Add `.gitattributes` (`*.sh text eol=lf`).
- Literature check (Nov 4): BEAM, EdgeInfer, PolyThrottle, the DVFS papers. Part 2 policy (Nov 5 - 11).
- Seed 2 of the policy runs is on disk (882 runs); rerun `04_analyze.py` and refresh the deck.
- Cost-aware shedding idea (shed expensive requests first under overload) parked, no prototype.
- CityFlow dispatcher fix needs the A100; Background and Motivation draft 2 waits for Jetson numbers; TensorRT YOLOv8 export still broken (eager only).

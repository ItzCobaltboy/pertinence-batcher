# batch_sweep: measured T_i(b) for YOLOv8 n/s/m/l

Times batched inference of YOLOv8n/s/m/l at batch sizes 1, 2, 4, 8, 12, 16, 32, 48 in three
variants, and turns the numbers into scheduler-sim profile CSVs (the T_i(b) table in
`scheduler-sim/sim.py`'s `Profile`). It replaces `scheduler-sim/profiles/synthetic_4model.csv`
with real curves once it has run on the A100.

| variant | what runs |
|---|---|
| `eager_fp32` | the plain PyTorch model, torch defaults (same setup as `yolo-analysis/inference.py`) |
| `trt_fp32` | `torch_tensorrt.compile(..., ir="dynamo")`, FP32 only, no quantization |
| `trt_fp16` | the same compile with FP16 enabled |

Self-contained in this folder. It imports only `../config.py` (dataset folders, model pool,
image size, confidence threshold), so it reads the exact images yolo-analysis already downloaded.
It never downloads data and never installs anything.

## Files

    run_sweep.sh  launcher: preflight -> measure -> aggregate, logs appended with &>>
    sweep.py      SINGLE ENTRY POINT: preflight | measure | aggregate | profile
    settings.py   every default (batch sizes, warmup, repeats, runs, pool size, paths, flag thresholds)
    worker.py     one compile or one timed run, always in its own subprocess
    models.py     model loading, the ForwardOnly wrapper (what "forward" means), TRT compile/save/load, NMS
    images.py     seeded image pool, letterbox preprocessing, pre-built device batches
    gpu_log.py    nvidia-smi snapshots (clocks, temperature, power, memory, throttle, other processes)
    store.py      raw file naming, atomic writes, resume index, session log
    aggregate.py  raw -> summary CSVs, plots, sanity flags, profile CSV (no torch, runs on a laptop)

## What is timed

**Forward** = one call of the fused ultralytics `DetectionModel` on a `(B, 3, 640, 640)` float32
batch that is already on the GPU: backbone + neck + the full Detect head, including DFL and box
decoding, returning the decoded `(B, 84, 8400)` tensor NMS consumes. NMS is not included. All
three variants go through the same `ForwardOnly` wrapper (`models.py`), so the definition is the
same for all of them.

- The timed loop runs `torch.cuda.synchronize()` → `forward` → `synchronize()` for every single
  batch, with `time.perf_counter()` around it. That's the `src/benchmark.py` convention applied per
  batch, and it matches the simulator: one batch at a time, alone on the GPU.
- Extra columns, which never replace forward-only:
  - `forward_nms_ms`: forward plus ultralytics NMS (conf 0.25, iou 0.7, max_det 300).
  - `loop_mean_ms`: the old benchmark.py style, one sync around 100 back-to-back forwards.
- Outside the timed region: image loading, JPEG decode, letterbox, host-to-device copy, model
  load, TRT compile and warmup.
- **TRT fallback**: each compile records how many TensorRT subgraphs and PyTorch fallback
  subgraphs the compiled module has, and which ops fell back. Any fallback raises a sanity flag,
  so "TRT" never silently means "partly PyTorch".
- **Compile check**: each compile also compares its output against eager on one real batch
  (max abs diff and NMS detection counts).
- **TF32**: on the A100, torch's default "fp32" convs run in TF32 (cuDNN default) and so do
  TensorRT FP32 builds. I kept the defaults; each run records the flags. `--no-tf32` turns it off.
- **torch 2.3 export fix**: its `export` (the first step of the dynamo path) rejects YOLOv8
  because `DetectionModel.stride` and `Detect.stride` are the same tensor object. `load_eager`
  gives the model-level copy its own clone. That copy is metadata, and forward never reads it.

## How a sweep runs

1. **Image pool**: 128 seeded images from `val2017` and 128 from the `train2017` subset, saved to
   `raw/<session>/image_pool.json`. Each run picks its own images from the pool (different per
   run, the same on a rerun), letterboxes them like `predict()` (640x640 square, pad 114, BGR to
   RGB, /255) and builds up to 8 distinct batches on the GPU before timing starts.
2. **Phase 1, compile**: every (TRT variant, model, batch size) is compiled once with a fixed
   input shape and cached to `engines/` (64 compiles). Compile time is logged. A config that
   fails to compile is recorded and its runs are skipped.
3. **Phase 2, timed runs**: `--runs` rounds (default 5). Each round runs every
   (variant, model, batch size) config once, in a seeded shuffled order, so slow drift on the
   shared GPU is spread over all configs. Every run is a fresh process, which means a fresh CUDA
   context, a fresh model or engine load, and its own warmup (default 20), then 100 timed batches.
4. **Aggregate**: runs automatically at the end, and can be rerun any time.

**Robustness**:
- Every job is its own subprocess, so a Python error, OOM, C++ `terminate()`, segfault or
  timeout kills only that job. The parent writes a failed record with the error or log tail and
  continues.
- Records are written atomically.
- **Resume**: rerunning `measure` skips every job that already has an ok record and retries
  failed ones until `--max-attempts` (2) failures.
- A lock file stops two `measure` processes from running on one session.

## Outputs

    yolo-analysis/results/batch_sweep/
      raw/<session>/image_pool.json
      raw/<session>/compile/<variant>__<model>__bs<b>__<timestamp>__<ok|failed|skipped>.json
      raw/<session>/runs/<variant>/<model>/bs<b>__run<k>__<timestamp>__<ok|failed>.json
      gpu_logs/<session>/sweep_{start,end}_<timestamp>.txt   full nvidia-smi at sweep start/end
      logs/<session>/sweep.log, logs/<session>/jobs/*.log   everything a job printed
      summary/<session>/run_summary.csv                     one row per run: mean/median/std/p95/min/max, GPU state
      summary/<session>/aggregated.csv                      per config: mean of run means, std across runs, n_runs
      summary/<session>/failures.csv, sanity_flags.csv/.txt
      summary/<session>/plots/T_of_b_<variant>.png, variant_comparison.png
      engines/<variant>/<model>_bs<b>_img640.ep (+ .json)   cached TRT modules (git-ignored, ~4 GB)
    scheduler-sim/profiles/yolov8_a100_<variant>.csv        only when you run `profile --variant`

**Each run file holds**:
- The 100 per-batch forward times, plus the forward+NMS times and the loop mean.
- Summary stats for the run.
- Model, batch size, variant, run number and timestamps.
- GPU name, compute capability and driver.
- Versions: torch, torch_tensorrt, tensorrt, ultralytics, numpy, cuDNN.
- Precision, image size, TF32 flags, and the weights path plus sha256.
- The exact images used, peak memory and the git commit.
- Engine compile/load time and the fallback report.
- nvidia-smi snapshots at run start and end: SM/mem clocks and their max, temperature, power,
  memory used, throttle reasons, P-state and other compute processes.

**Sanity flags**:
- T(b) not increasing.
- T(b)/b not decreasing.
- Std across runs above 5% of the mean.
- Failed, skipped or partial configs.
- TRT PyTorch fallback.
- Possible interference: other GPU processes, SM clock below 90% of max at run end, active
  throttle reasons, or temperature at 80 C or above.

**Profile CSV**:
- Format: the `Profile` format, meaning `#` comment header, `model,1,2,4,8,12,16,32,48`, one row
  per model in n/s/m/l order, ms per batch.
- Cell value: the mean across runs of each run's mean forward time. A failed config is an empty
  cell, which the sim treats as "no engine" and pads over.
- `--variant` has no default. The command refuses to overwrite an existing file without `--force`.

## Run it on the server (inside tmux)

Pre-flight:

    cd <repo>/yolo-analysis/batch_sweep
    df -h <repo>                          # need ~5 GB free for engines + outputs (engines dominate)
    nvidia-smi                            # GPU idle? no other users' processes on it?
    python sweep.py preflight             # versions, CUDA, torch_tensorrt, data folders, weights, disk

Run (one script does preflight -> measure -> aggregate, each step appending with `&>>` to its
own log under `results/batch_sweep/logs/<session>/`: `launcher.log`, `environment.log`
(df + nvidia-smi at start and end), `preflight_stdout.log`, `measure.log`, `aggregate.log`,
plus the per-job logs in `jobs/`). It refuses to start the sweep if preflight finds a problem.

    tmux new -s batchsweep
    bash <repo>/yolo-analysis/batch_sweep/run_sweep.sh
    # pick a GPU:   CUDA_VISIBLE_DEVICES=N bash run_sweep.sh
    # env knobs:    SESSION (default main), PYTHON (default python), DATASET_ROOT
    # extra args go to `sweep.py measure`, e.g.  bash run_sweep.sh --runs 3
    # detach: Ctrl-b d   reattach: tmux attach -t batchsweep
    # watch:  tail -f <repo>/yolo-analysis/results/batch_sweep/logs/main/measure.log
    # killed or disconnected? run the same command again, it resumes

Afterwards:

    cat <repo>/yolo-analysis/results/batch_sweep/summary/main/sanity_flags.txt

**Expected cost**:

| item | estimate |
|---|---|
| TRT compiles (64) | 1 to 4 min each, about 2 to 4 h total. Larger models and batch sizes are slower |
| timed runs (480) | about 15 to 30 s each, mostly process start, imports and engine load. About 2 to 4 h total |
| whole sweep | about 4 to 8 h |
| `engines/` disk | about 4 GB (deletable after the sweep, not copied back) |
| `raw/` + `summary/` + `logs/` disk | under 100 MB |

**Copy back** (everything except `engines/`):

    yolo-analysis/results/batch_sweep/raw/
    yolo-analysis/results/batch_sweep/summary/
    yolo-analysis/results/batch_sweep/gpu_logs/
    yolo-analysis/results/batch_sweep/logs/

The profile CSVs can be built on the laptop from `raw/` (`python sweep.py profile --variant
trt_fp16`). If you build them on the server instead, also copy
`scheduler-sim/profiles/yolov8_a100_*.csv`.

## Useful flags

`--models`, `--batch-sizes`, `--variants`, `--runs`, `--warmup`, `--repeats` (smaller sweeps),
`--session` (a separate campaign with its own resume state), `--dataset-root` (images somewhere
else), `--engine-dir` (put the ~4 GB cache on another disk), `--no-nms`, `--no-tf32`,
`--max-attempts`, `--compile-timeout`, `--run-timeout`.

## CPU test (what was checked before the first server run)

On a CPU box, with `BATCH_SWEEP_OUT_DIR` pointed at a scratch folder, `yolov8{n,s}.yaml`
(untrained, no download), fake JPEGs under `--dataset-root`, `--imgsz 64` and tiny counts:

- torch 2.3.1 `export` of the full wrapped model (n and l), Detect head included.
- Raw files, a hard `abort()`, an injected exception, and give-up after `--max-attempts`.
- Kill mid-run, then resume (no half files, no duplicate runs), and the session lock.
- Aggregation, plots and sanity flags.
- A profile with a failed cell loaded by scheduler-sim's real `Profile` class.
- All 16 scheduler-sim tests still pass.

`BATCH_SWEEP_TEST_FAIL` / `BATCH_SWEEP_TEST_CRASH` (`variant:model:batch`) are test-only hooks.
TRT itself cannot be tested without a GPU; its first real run is the server sweep.

# power_analysis: power and energy of YOLOv8 n/s/m/l vs batch size

Measures board power with `nvidia-smi` while each (model, batch size) runs, and turns it into
average power, runtime and energy per batch and per frame. eager FP32 (plain PyTorch), the same
model setup as the timing sweep in `../batch_timing_analysis/`.

## Run it

    cd yolo-analysis-a100/power_analysis
    python power_sweep.py              # measure everything, then aggregate (about 2 h 10 min on the A100)
    python power_sweep.py --aggregate  # only redo the tables and plots from saved data (works on a laptop)

Edit `settings.py` for the models, batch sizes, phase lengths, the GPU index and the power field.
If the GPU is not 0, set both `CUDA_VISIBLE_DEVICES` and `GPU_INDEX` to it.

## How it works

Files: `power_sweep.py` (the entry point), `worker.py` (one combo), `aggregate.py` (numbers and
plots), `settings.py` (all knobs).

1. `power_sweep.py` starts `nvidia-smi -lms 50 -f samples_raw.csv` in the background. It writes
   one power sample every 50 ms into one file for the whole sweep: one long power trace.
2. Baseline: it sleeps `BASELINE_S` (20 s) with nothing loaded on the GPU.
3. It runs every (model, batch) combo once, in a random order, each in a new process
   (`worker.py`). The worker only writes down `time.time()` at the start and end of each phase:

   | phase | what happens |
   |---|---|
   | load (not measured) | load the model and 1 batch of real COCO images (resized to 640), one warmup forward |
   | idle | model loaded, GPU doing nothing, for `IDLE_S` (same length as the baseline) |
   | run 0..4 | forward the batch `FORWARDS_PER_PHASE` (200) times, however long that takes, then sync |
   | buffer | sleep `BUFFER_S` (3 s) between run phases |

4. It stops `nvidia-smi` and converts its timestamps (local time on the server) into the same
   seconds that `time.time()` gives, in `samples.csv`.
5. `aggregate.py` cuts the trace with the worker's start/end times:
   - power of a phase = the mean of the samples inside it;
   - energy of a phase = power x runtime;
   - per combo: average over the run phases, energy per batch = energy / forwards, energy per
     frame = energy per batch / batch size.

The phase edges and the 50 ms samples are not perfectly aligned. That is accepted. With 200
forwards a run phase lasts from ~1 s (yolov8n, batch 1, ~20 samples) to ~45 s (yolov8l, batch 64)
on the A100, going by the measured eager FP32 times.

**End tail**: nvidia-smi writes its `-f` file in ~4 KB chunks (~6 s of samples), and the last
chunk can be lost when it is stopped. So the sweep keeps sampling `END_TAIL_S` (10 s) after the
last combo, and only that tail is at risk.

**Power field**: `power.draw` works on every driver, but on the A100 it is a ~1 s moving average.
If `nvidia-smi --query-gpu=power.draw.instant --format=csv` prints a number on your server, set
`POWER_FIELD = "power.draw.instant"` in `settings.py`.

## Outputs (`yolo-analysis-a100/results/power_analysis/`)

    samples_raw.csv          nvidia-smi output as written (git-ignored)
    samples.csv              time (s), power_w: the whole power trace
    baseline.json            start/end of the baseline
    combos/<model>_bs<b>.json  the worker's marks, run phases (start, end, runtime, forwards), load time
    combos/<model>_bs<b>.log   everything the worker printed
    summary.csv              one row per combo: idle_power_w, run_power_w, runtime_s, forwards,
                             ms_per_batch, energy_j, energy_per_batch_j, energy_per_frame_j, baseline_power_w
    phases.csv               one row per run phase (to see how power drifts from phase to phase)
    plots/combos/<model>_bs<b>.png   the raw power trace of each combo, idle and run phases shaded
    plots/models/<model>.png         power, energy per batch and energy per frame vs batch size

## What is not here (on purpose)

- No resume: a rerun starts over and overwrites the output folder.
- No lock, no retries: a failed or stuck worker (killed after `WORKER_TIMEOUT_S`) is printed and
  skipped, and its combo is missing from `summary.csv`.
- Model load and unload cost are not measured. The model is loaded before the idle phase starts.

## Checked before the first server run

On the laptop's RTX 5070 Ti (real `nvidia-smi`, `power.draw.instant`, CUDA), with untrained
`yolov8n.yaml` / `yolov8l.yaml`, batch 1 and 16, 3 run phases of 100 forwards and short idle
phases:
- the whole flow ran;
- the power steps in each raw trace line up with the shaded idle and run phases;
- every run phase had at least 13 samples;
- the trace covers the last phase.

That run is also how the ~4 KB chunking was found: without the 10 s tail, the file stopped about
6 s early. The laptop GPU's absolute numbers mean nothing (it is power-limited), only the pipeline
was being checked.

# Spec: `jetson-power/` (written on my laptop, run on the Jetson AGX Xavier)

For Claude Code running on my laptop. You write the code and I copy the folder to the Jetson with `scp` and
run it there. You cannot run it yourself, so everything must be checked by `00_preflight.py` and `--smoke` on
the board, and the code has to be right by reading, not by trial. Read this whole file first.

## 0. Goal

For each YOLOv8 model (n, s, m, l) and for each Jetson power mode, measure how **runtime, power and energy**
change with **batch size**. This answers whether batching saves energy on a real edge board (the A100 could
not show this: it saturates at its power cap and a datacenter part sits at the top of its curve).

Output per (power mode, model, batch size): latency per batch, throughput, power (total and per rail), energy
per batch, energy per frame, total and dynamic (idle subtracted).

## 1. Rules

- **Hard stop (my standing rule, laptop side):** never touch C drive system areas, Program Files or the
  Windows folder. Work only in the project folder.
- On the Jetson the code may use `sudo` (I will start the sweep with it). It may call `nvpmodel` and
  `jetson_clocks`, and it **must restore the original power mode and clocks at the end, also on error or
  Ctrl+C** (`try/finally`).
- Do not edit `Journel/` or `CLAUDE.md`.
- `archive/` is old code. Do not import from it. Ideas worth reading are in section 9.
- **No fake testing.** No mocked tegrastats in the shipped code paths. The only fixtures allowed are text
  lines of real tegrastats output for testing the parser (section 10). Real checks happen on the board.
- **Flag, do not guess.** Unknown fact -> write it in the summary under Flags, and make the code stop with a
  clear message instead of continuing on a guess.

### 1b. Code style

Plain readable Python that I can follow. One idea per function, short functions, descriptive names. Prefer a
simple loop to a clever one-liner. No decorators, inheritance, `**kwargs` pass-through or framework-style
layers. Comments explain why. Each file starts with 2 to 4 comment lines on what it does. Scripts print
progress and say where outputs landed. Not verbose either.

**Python 3.8 only** (the board has 3.8.10, and the stack stops there): no `list[int]` or `dict[str, int]`
annotations, no `X | Y` types, no `match`, no `str.removeprefix`, no `dict | dict`, no parenthesised
multi-line `with`. Use `typing.List` etc. or leave annotations off.

## 2. Target board (checked on the board)

- Jetson AGX Xavier, 32 GB RAM, L4T R35.3.1 = JetPack 5.1.1, Ubuntu 20.04, Python 3.8.10, aarch64.
- torch is **not installed**. `nvpmodel -q` currently reports mode 7 `MODE_15W_DESKTOP`; the board boots
  there. Do not assume a mode id means the same thing on another board: parse `/etc/nvpmodel.conf`.
- `tegrastats` shows power rails **only when run as root**. A sample as root looks like this (one line):

```
10-23-2026 18:33:47 RAM 1766/31003MB (lfb 7001x4MB) SWAP 0/15502MB (cached 0MB) CPU [1%@1180,0%@1180,1%@1178,0%@1177,off,off,off,off] EMC_FREQ 0%@800 GR3D_FREQ 0%@318 VIC_FREQ 1036 APE 150 AUX@26.5C CPU@27.5C thermal@27.3C Tboard@29C AO@27.5C GPU@27.5C Tdiode@31.75C PMIC@50C GPU 0mW/0mW CPU 466mW/466mW SOC 933mW/933mW CV 0mW/0mW VDDRQ 310mW/310mW SYS5V 2363mW/2363mW
```

  Rails: `GPU`, `CPU`, `SOC`, `CV`, `VDDRQ`, `SYS5V`, written `NAME current_mW/average_mW`. There is no total
  rail. Use the **current** value (first number); the second is tegrastats' own running average.
  Without root the same line has no rail fields at all.
- The board has internet. sudo works with a password (so `sudo -n` fails; I start the run myself).

## 3. Layout

```
jetson-power/
  README.md                 steps: scp to board, setup, preflight, smoke, full run, scp results back
  setup_env.sh              venv + packages (section 4), safe to rerun
  requirements-jetson.txt   pins you verified (section 4)
  config.py                 models, batch grid, durations, thresholds, paths
  common/
    tegrastats.py           start/stop sampler, parse lines, rails -> columns
    powermode.py            list modes, read, set, save/restore original mode and clocks
    bench.py                load model, make input batch, timed forward loop
  00_preflight.py
  01_power_sweep.py         the measurement (run with sudo)
  02_analyze.py             tables and plots from the CSVs; runs on any machine (laptop too)
  results/<device_tag>/     outputs
```

`device_tag` example: `agx_xavier_jp511`. Modes are a column, not part of the tag.

## 4. setup_env.sh

Safe to rerun, prints each command before running it. Creates `.venv` next to the scripts.

1. If `python3 -m venv` is missing: `sudo apt-get install -y python3-venv python3-pip` (nothing else via apt,
   plus `libopenblas-dev` only if the torch wheel needs it; say so when you do).
2. **torch / torchvision:** never `pip install torch` from PyPI (that gives a CPU-only aarch64 build) and
   never upgrade them once installed. Use NVIDIA's JetPack 5 wheel. Look at
   `https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/` (and NVIDIA's "PyTorch for Jetson"
   page) to find the wheel for `cp38` aarch64 and pin the exact file name you found in
   `requirements-jetson.txt`. I could not verify the exact version, so **you check it** on the laptop with
   internet before writing the file, and write down where you read it.
3. torchvision: NVIDIA ships no JetPack 5 wheel. We only time the network forward pass (no NMS), but
   ultralytics imports torchvision. Try, in this order: (a) the torchvision release that matches the torch
   version, installed with `--no-deps`, check `import torchvision` works; (b) building from source, which is
   slow on a Xavier (tens of minutes) and which you describe in the README with the exact commands. Choose
   (a) first, and make preflight check the import. Do not let pip pull its own torch.
4. numpy must stay below 1.25 on Python 3.8. Install ultralytics with `--no-deps`, then install only what it
   needs (opencv-python-headless, pyyaml, scipy, psutil, py-cpuinfo, tqdm, pillow, requests, pandas,
   matplotlib), each compatible with 3.8. Choose the **newest ultralytics release whose Requires-Python
   allows 3.8** (check with `pip index versions ultralytics` and the release metadata) and pin it. If none
   works with the installed torch, stop and tell me instead of upgrading torch.
5. End of script: run `00_preflight.py`.

## 5. 00_preflight.py

Prints and saves `results/<device_tag>/preflight.json`:
- `/proc/device-tree/model`, `/etc/nv_tegra_release`, Python, torch version and `torch.cuda.is_available()`
  (must be True, otherwise stop), torchvision import, ultralytics version;
- am I root (`os.geteuid()`), tegrastats present, a 3 second tegrastats sample parsed: which rails were found,
  the total power, cadence in samples per second. No rails found -> say "run with sudo" and stop;
- `nvpmodel -q` and the list of all modes parsed from `/etc/nvpmodel.conf` (id, name);
- `jetson_clocks --show` output saved raw;
- free disk and RAM, swap, thermal now;
- the four models (yolov8n/s/m/l, stock COCO weights from ultralytics, downloaded into `models/`) load and
  do one FP16 forward at b=1 and b=8 on the GPU.

## 6. 01_power_sweep.py

Run as `sudo .venv/bin/python 01_power_sweep.py [flags]`. At the end it `chown`s everything it created back
to the invoking user (`SUDO_UID`, `SUDO_GID`) so I can scp it.

Flags: `--modes 0,7` (default all modes from nvpmodel.conf), `--models n,s,m,l`, `--batches 1,2,4,8,16`,
`--reps 3`, `--min-seconds 10`, `--lock-clocks / --no-lock-clocks` (default lock), `--budget-min N` (picks a
smaller grid that fits and prints it first), `--smoke` (one mode, yolov8n, batches 1 and 4, short; for me to
check on the board), `--resume` (default on: skip finished cells).

**What is measured.** The network forward pass only: FP16, input tensor already on the GPU, no image
decoding, no NMS, no post-processing, `torch.no_grad()`, `model.half().eval()`. Input is real image content
(ultralytics ships `assets/bus.jpg` and `zidane.jpg`), letterboxed to 640x640 and repeated to fill the batch,
prepared once before timing. State this in the README, since T_i(b) for scheduling is the network time.

**Batch grid** default 1, 2, 3, 4, 6, 8, 12, 16, 24, 32. Stop a model's grid at the first CUDA out-of-memory,
record "OOM at b" and continue with the next model. `torch.cuda.empty_cache()` between cells.

**Per power mode** (one continuous tegrastats trace per mode, written raw to
`results/<device_tag>/raw/<mode>_tegrastats.log.gz`):
1. Save the original mode id and the current clocks. Set the mode with `nvpmodel -m ID`, wait 10 s to settle.
   If it fails, record the error, skip the mode, go on.
2. If `--lock-clocks`: `jetson_clocks`; save `jetson_clocks --show` for the mode. (Locked clocks make
   runs reproducible and match what EdgeServing does. The DVFS variant is the same sweep with the flag off.)
3. Idle baseline: 20 s with CUDA initialised and no model: mean rail powers.
4. For each model: load it, then idle-with-model-loaded for 20 s.
5. For each batch size: 10 warm-up forwards, then `reps` repetitions. A repetition = run forwards back to back
   for at least `min-seconds` and at least 30 iterations, with `torch.cuda.synchronize()` before and after,
   then a 3 s gap. Per iteration latency by CUDA events. Record the wall-clock start and end of each
   repetition from the same clock used to stamp tegrastats lines.
6. Thermal guard: if `thermal` exceeds 80 C, pause until below 65 C, log that it happened and which cell it
   delayed. Record temperature at the start and end of every repetition.
7. At the end (and in `finally`): stop tegrastats, `jetson_clocks --restore` to the saved state, restore the
   original nvpmodel mode.

**Sampler.** Run `tegrastats --interval 100` (flag) via `stdbuf -oL` or a pty so lines arrive line by line;
a plain pipe buffers about 4 KB and stalls. Stamp each line with `time.monotonic()` when read (do not parse
tegrastats' own timestamp). Parse the six rails generically (`NAME cur/avg mW`), plus `GR3D_FREQ`, `EMC_FREQ`,
the CPU load list, `thermal`, `GPU@` temperature. Unknown tokens kept raw, never a crash. Keep sampling 3 s
after the last mark. Compute and record the effective sampling cadence per mode.

**Power definition.** Per sample: `total_mw` = sum of all six rails. Also store `total_excl_sys5v_mw` and the
GPU rail alone. Reason: SYS5V (about 2.3 W idle) is board overhead and may overlap other rails; I have not
verified this. Report all three in every table so I can decide later; mark `total_mw` as the headline.

**Energy.** For each repetition, integrate power over [start, end] by the trapezoid rule on the stamped
samples, per rail and total. Energy per batch = energy / iterations. Energy per frame = per batch / b.
Dynamic energy per frame = (mean power in repetition minus idle-with-model-loaded power) x time per batch / b,
for the headline total. Never integrate samples outside the repetition window.

**Outputs** (`results/<device_tag>/`):
- `results.csv`: one row per (mode, model, batch, rep): mode_id, mode_name, clocks_locked, model, batch,
  iterations, wall_s, latency_ms_mean/p50/p95, throughput_fps, power_mw (total, excl_sys5v, per rail),
  energy_per_batch_mj, energy_per_frame_mj, dynamic_energy_per_frame_mj, gpu_freq, temp_start/end, cadence.
- `idle.csv`: idle baselines per mode and model.
- `status.csv`: per cell done / OOM / skipped / error text.
- `raw/` traces, `meta.json` (preflight, versions, command line, start/end time).

## 7. 02_analyze.py (runs on laptop or board)

Reads `results.csv`, writes `summary.md` and PNGs (matplotlib, Agg backend):
- per model: energy per frame vs batch, one line per power mode;
- per model: latency per batch vs batch, throughput vs batch, mean power vs batch;
- per power mode: the same with models as lines;
- a table: per (mode, model) the batch with minimum energy per frame, and the energy saving at b=8 vs b=1;
- plain-language notes that only restate what the numbers show (where the curve flattens, which modes
  are Pareto-better in latency vs energy). No claims beyond the data.
Report the spread across repetitions; mark cells where repetitions disagree by more than 10 percent.

## 8. Time estimate
Print before the full run: cells x (reps x (min-seconds + 3 s gap) + warm-up) + idle phases + mode settle
times. Roughly 3 to 4 hours for all modes with 3 reps and 10 batch sizes. `--budget-min` exists so I can
shrink it; resuming must work if I stop with Ctrl+C.

## 9. Reference only
- `archive/yolo-analysis-a100/power_analysis/` : the A100 version of this sweep (nvidia-smi). Read it for
  the structure of idle phases, marks and trace cutting, and for the known gotcha that raw tool output is
  buffered so sampling must continue past the last mark. Do not copy.

## 10. Checks you can do on the laptop
- Parser test with **real tegrastats lines** (the sample in section 2 plus a no-root line without rails):
  rails found, values correct, unknown tokens kept.
- Energy integration test on a small hand-made power series with a known integral.
- Everything else is verified on the board by me running `00_preflight.py` and `--smoke`; make their output
  clear enough that I can paste it back to you.

## 11. After the run
`README.md` has the exact commands:
```
scp -r jetson-power agx2@<ip>:~/            (exclude .venv and results)
ssh agx2@<ip>
cd jetson-power && bash setup_env.sh
sudo .venv/bin/python 00_preflight.py
sudo .venv/bin/python 01_power_sweep.py --smoke
sudo .venv/bin/python 01_power_sweep.py
scp -r agx2@<ip>:~/jetson-power/results/<device_tag> .
python 02_analyze.py results/<device_tag>
```

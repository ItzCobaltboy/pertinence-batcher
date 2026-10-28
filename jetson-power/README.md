# jetson-power

Measures how runtime, power and energy change with batch size for YOLOv8 n/s/m/l, in every Jetson power
mode, on a Jetson AGX Xavier (L4T R35.3.1 = JetPack 5.1.1, Python 3.8.10). Written on the laptop, run on the board.

**What is timed:** the network forward pass only. FP16 (`model.half().eval()`, BN fused into conv as ultralytics
does for inference), `torch.no_grad()`, input tensor already on the GPU, no image decoding, no NMS, no
post-processing. The input is real image content (`bus.jpg`, `zidane.jpg` shipped with ultralytics) letterboxed
to 640x640 and repeated to fill the batch, prepared once before timing. So T_i(b) here is the network time.

**Power:** `tegrastats --interval 100` as root, one continuous trace per mode, each line stamped with
`time.monotonic()` when read. Per sample `total_mw` = sum of the six rails (GPU, CPU, SOC, CV, VDDRQ, SYS5V),
the headline. `total_excl_sys5v_mw` and the GPU rail alone are reported next to it, because SYS5V (about 2.3 W
idle) is board overhead and may overlap other rails (not verified). Energy of a repetition = trapezoid
integral of the samples inside its start/end marks (edges interpolated). Dynamic energy subtracts the idle power
measured with that model loaded.

## Steps

```
scp -r jetson-power agx2@<ip>:~/            # do not copy .venv or results
ssh agx2@<ip>
cd jetson-power && bash setup_env.sh        # venv, packages, then runs preflight
sudo .venv/bin/python 00_preflight.py
sudo .venv/bin/python 01_power_sweep.py        # smoke run first (config.SMOKE = True)
nano config.py                                 # set SMOKE = False
sudo .venv/bin/python 01_power_sweep.py        # full sweep
scp -r agx2@<ip>:~/jetson-power/results/<device_tag> .
python 02_analyze.py results/<device_tag>
```

Paste back the output of `00_preflight.py` and the smoke run. The smoke run writes to `results/<device_tag>_smoke/`
so it never mixes with the real run.

## Sweep settings

No command line flags: edit the "sweep settings" block in `config.py` (`SMOKE`, `SWEEP_MODES`, `SWEEP_MODELS`,
`LOCK_CLOCKS` (False = DVFS variant), `BUDGET_MIN`) and `BATCHES`, `REPS`, `MIN_SECONDS` above it.
`SMOKE = True` runs the current mode only, yolov8n, b=1 and 4, short. It prints a time estimate first (3 to 4
hours for all modes, 3 reps, 10 batch sizes). Ctrl+C is safe: the original mode and clocks are restored, and
rerunning skips finished cells (a cell is the three repetitions of one mode/model/batch; an interrupted cell is
redone). To start over, delete the results folder.

## Outputs (`results/<device_tag>/`)

`results.csv` (one row per mode, model, batch, rep), `idle.csv`, `status.csv` (done / OOM / error / thermal
pauses), `meta.json`, `preflight.json`, `raw/` (gzipped tegrastats traces with our timestamps, `jetson_clocks
--show` per mode), and after `02_analyze.py`: `summary.md`, `model_yolov8*.png`, `mode_*.png`.

## If torchvision does not import (setup option b)

`setup_env.sh` installs PyPI's `torchvision==0.15.1` aarch64 wheel with `--no-deps` (a CPU build; we never call
its ops). If `import torchvision` still fails, build it from source (tens of minutes on a Xavier):

```
sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision-src
cd torchvision-src
BUILD_VERSION=0.15.1 ../.venv/bin/python setup.py install
cd .. && rm -rf torchvision-src
```

## Laptop tests

`python -m pytest tests` (parser on real tegrastats lines, energy integration on a known series).

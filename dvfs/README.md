# dvfs

Standalone GPU-frequency profiling for the Jetson AGX Xavier (JetPack 5.1.1, nvpmodel mode 7, mode never changed).
Python 3.8, run as root from the repo root, in `tmux`, with the same venv python as sched-harness
(`sudo .venv/bin/python -m dvfs.<script>`; never run from inside `dvfs/`, `inspect.py` would shadow the stdlib).
`tegrastats.py`, `models.py`, `inputs.py` are copies from `sched-harness/common/` (only the config import changed).
Weights: copy `sched-harness/models/*.pt` to `dvfs/models/` to skip the download. All constants are in `settings.py`.

| step | command | time |
|---|---|---|
| look at the board (read only) | `sudo python3 -m dvfs.inspect` | ~1 min |
| plan and estimate, touches nothing | `python3 -m dvfs.profile_switch --dry-run`, `python3 -m dvfs.profile_models --dry-run` | instant |
| clock switch cost | `sudo python3 -m dvfs.profile_switch` | ~10 min |
| latency + power per freq/model/batch | `sudo python3 -m dvfs.profile_models [--resume DIR]` | ~3.3 h (20 batches per subrun) |
| recompute the averages | `python3 -m dvfs.profile_models --summary-only DIR` (same for `profile_switch`) | seconds |
| manual clock recovery | `sudo python3 -m dvfs.gpu --restore` | instant |

Outputs go to `dvfs/results/<DEVICE_TAG>/<switch_|models_><timestamp>/`:
`switch_raw.csv`, `switch_tegra_trace.csv`, `switch_summary.csv`; `subruns.csv` (one row per subrun), `batches.csv`
(every batch), `run_summary.csv` (averages per freq/model/batch), `tegrastats_raw.log.gz` (whole session, monotonic stamps),
`clock_dmesg_raw.log.gz`, `meta_*.json`.

Runtime vs measurement: the profilers just run; `tegrastats.Sampler` measures in parallel and only records. Besides the power
rails and temperatures it reads sysfs `cur_freq` every `CUR_FREQ_INTERVAL_S` (the clock actually held) and polls dmesg every second
for new lines (those matching throttle / thermal / over-current are flagged). Everything goes raw to `tegrastats_raw.log.gz`
(power lines) and `clock_dmesg_raw.log.gz` (`cur_freq`, `dmesg`, `dmesg_hit` lines). Each subrun row carries `n_cur_freq_reads`,
`cur_freq_min_hz`, `cur_freq_max_hz`, `clock_held` (all reads equal the pinned clock), `gr3d_mhz_median` / `gr3d_ok` (tegrastats agrees)
and `dmesg_hits`; the run prints a WARNING line when any of them is off. Nothing aborts a run on its own.
Session safety (not measurement): the original min/max/governor/3D-scaling go to `gpu_state_backup.json` before the first write and are
restored on every exit path (normal, error, Ctrl-C, SIGTERM, ssh hangup). It refuses to start as non-root, outside mode 7, next to
another python/tegrastats process, or with a frequency not in `available_frequencies` / above the mode's `max_freq`.
Never touched: `nvpmodel -m`, `jetson_clocks`, fan, governor, 3D scaling, CPU and EMC clocks (the raw tegrastats log records them).

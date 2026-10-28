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
`meta_*.json`, `abort.txt` (only after a safety abort).

Safety: the original min/max/governor/3D-scaling go to `gpu_state_backup.json` before the first write and are restored on
every exit path (normal, error, Ctrl-C, SIGTERM, ssh hangup, abort). Refuses to start as non-root, outside mode 7, next to another
python/tegrastats process, or with a frequency not in `available_frequencies` / above the mode's `max_freq`.
Two run-time checks, both abort and restore: `cur_freq` must stay on the pinned clock (read before every batch; tegrastats `GR3D_FREQ`
is also compared in each warm-up and subrun), and dmesg must show no new over-current / throttle / thermal line (every 5 s and at
the end of each subrun). Every subrun row records `n_cur_freq_reads`, `cur_freq_min_hz`, `cur_freq_max_hz` and `n_dmesg_checks`,
so a clean run shows the checks ran. Temperatures are recorded, never acted on.
Never touched: `nvpmodel -m`, `jetson_clocks`, fan, governor, 3D scaling, CPU and EMC clocks (the raw tegrastats log records them).

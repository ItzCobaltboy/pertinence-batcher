# clock_settle: what does a GPU clock change really cost?

Question: the policy charges `DVFS_SWITCH_MS = 88` ms for every clock change. Week 9 measured 88 ms as the time of the sysfs write with
an idle GPU; in the runs the same write returns in about 1.7 ms. Neither number says when the GPU actually runs at the new clock.

## Stage A (done, no board): `stageA_post_switch.py`
Reads `batches.csv` of existing runs and compares the first batches after a clock change with steady state, and the write time with
the idle time before the change. Run: `python3 stageA_post_switch.py` (about 4 s). Results are in `Journel/Week11.md`.

## Stage B (needs the board): `clock_probe.cu`
A one-thread CUDA kernel publishes `clock64()` (the SM cycle counter, so its rate is the GPU clock) into pinned mapped memory (volatile
on both sides). A reader thread pinned to one core stamps each new value with `CLOCK_MONOTONIC`. The main thread, pinned to another
core, does the clock change with the same writes and order as `common/dvfs.py` `Clock.set` and stamps start, middle and end. The dump
gives the GPU clock against time with about 20 us between samples.

- `--mode busy`: the kernel runs when the clock changes (what the policy does between batches).
- `--mode idle --idle-ms N`: no kernel for N ms before the change; the kernel starts when the write returns (tests the idle penalty).

Files: `clock_probe.cu`, `run_probe.sh` (builds, runs 5 clock pairs busy and idle plus a 3 s idle case, analyses), `analyze_probe.py`,
`test_analyze_probe.py` (checks the analysis on synthetic dumps with known behaviour; run it anywhere with numpy and pandas).

### Run (on the Jetson, nothing else on the GPU)
    nvcc -O2 -arch=sm_72 -o clock_probe clock_probe.cu -lpthread     # as your normal user: root usually has no nvcc on its PATH
    sudo ./run_probe.sh                 # about 15 min;  REPEATS=10 for a quick test first
then copy `probe_results/<stamp>/` to the PC and, if it was not analysed on the board, `python3 analyze_probe.py <prefix>`.

### Reading the result
| Outcome | Meaning for the policy |
|---|---|
| busy: `settle_ms` about equal to `write_ms` (about 2 ms), `after_write_ms` near 0 | the clock is at the new value when the write returns; cost is the write; set the switch cost to about 2 ms (plus the p95) |
| busy: `after_write_ms` of several ms | the clock ramps after the write; the cost is the settle time, and the batch right after runs partly at the old clock |
| `min_ratio` far below 1, or `max_gap_ms` of ms | the GPU pauses during the change (a stall); the cost includes it |
| idle200 / idle3000: large `write_ms` or low `first_ratio` | the idle penalty is real; it matters only for changes after long idle (start-up, a quiet period) |

## Not tested here
`clock_probe.cu` was only syntax-checked on the host side against a stub header (no nvcc and no GPU in the workspace); the kernel and
the CUDA calls have never been compiled. Expect to fix a compile error or two on the board. Assumptions to confirm on the board:
`nvcc` exists (JetPack: `/usr/local/cuda/bin`), the GPU is `sm_72`, the devfreq path is the one in `common/dvfs.py`, and a long-running
kernel is not killed by a watchdog (headless board).

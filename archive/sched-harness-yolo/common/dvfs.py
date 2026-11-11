# DVFS policy (v2: prune, then rank) and what it needs: the latency/energy table and the GPU clock.
# Pure python except Clock (sysfs). No torch. Python 3.8.
import bisect
import csv
import json
import os
import time

from common.policies import Serve


# ---------- table: L(q, b, f) p95 ms and E(q, b, f) J per batch, from a dvfs_runtime run_summary.csv ----------

class Table:
    """rows[(model, b, f_mhz)] = (lat_ms, energy_j). Batch sizes between measured ones are interpolated linearly."""

    def __init__(self, rows):
        self.rows = rows
        self.models = sorted(set(k[0] for k in rows))
        self.freqs = sorted(set(k[2] for k in rows))
        self.sizes = {m: sorted(set(k[1] for k in rows if k[0] == m)) for m in self.models}
        self.cache = {}

    def get(self, model, b, f):
        key = (model, b, f)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if key in self.rows:
            value = self.rows[key]
        else:
            sizes = self.sizes[model]
            i = bisect.bisect_left(sizes, b)
            if i == 0 or i >= len(sizes):
                raise KeyError("batch %d outside the measured sizes %s for %s" % (b, sizes, model))
            lo, hi = sizes[i - 1], sizes[i]
            w = (b - lo) / float(hi - lo)
            a, c = self.rows[(model, lo, f)], self.rows[(model, hi, f)]
            value = (a[0] + w * (c[0] - a[0]), a[1] + w * (c[1] - a[1]))
        self.cache[key] = value
        return value

    def lat(self, model, b, f):
        return self.get(model, b, f)[0]

    def energy(self, model, b, f):
        return self.get(model, b, f)[1]


def load_table(path, lat_col="lat_p95_ms", energy_col="energy_per_batch_j"):
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[(r["model"], int(r["batch"]), int(round(float(r["freq_mhz"]))))] = (float(r[lat_col]), float(r[energy_col]))
    if not rows:
        raise ValueError("empty table %s" % path)
    return Table(rows)


# ---------- the policy ----------

class DvfsAlpha:
    """Stage 1 prunes (queue, batch, clock) actions that would make a deadline miss, stage 2 ranks what is left:
    least-slack queue, largest feasible batch, clock with the smallest J = alpha*E_hat + (1-alpha)*L_hat.
    alpha = 1: lowest energy that is feasible. alpha = 0: fastest feasible clock (the latency-first ranking)."""
    name = "dvfs_alpha"

    def __init__(self, alpha, slo_ms, table, queue_models, switch_ms=88.0, keep_within=0.02):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.slo_s = slo_ms / 1000.0
        self.table = table
        self.models = queue_models              # queue index -> table model label
        self.switch_s = switch_ms / 1000.0
        self.keep_within = keep_within
        self.f_max = max(table.freqs)
        self.n_clean = 0                        # dispatches with both checks passing
        self.n_risky = 0                        # check 1 held, check 2 did not
        self.n_miss = 0                         # nothing feasible: least-slack queue at the top clock

    # --- helpers
    def lat_s(self, q, b, f):
        return self.table.lat(self.models[q], b, f) / 1000.0

    def switch(self, f_from, f_to):
        return 0.0 if f_from == f_to else self.switch_s

    def others_ok(self, state, q, tau, f):
        """Check 2: every other non-empty queue, in deadline order, at the top clock, one batch each (arrivals ignored)."""
        t, cur = tau, f
        order = sorted((r for r in state.nonempty() if r != q), key=lambda r: (state.head(r) + self.slo_s, r))
        for r in order:
            t += self.switch(cur, self.f_max) + self.lat_s(r, min(state.length(r), state.b_max), self.f_max)
            cur = self.f_max
            if t > state.head(r) + self.slo_s:
                return False
        return True

    def feasible(self, state, q, b, f, f0, with_check2):
        tau = state.now + self.switch(f0, f) + self.lat_s(q, b, f)
        if tau > state.head(q) + self.slo_s:
            return False
        return (not with_check2) or self.others_ok(state, q, tau, f)

    def score(self, q, b, f, f_best_energy, e_min):
        e_hat = self.table.energy(self.models[q], b, f) / e_min
        l_hat = self.table.lat(self.models[q], b, f) / self.table.lat(self.models[q], b, self.f_max)
        return self.alpha * e_hat + (1.0 - self.alpha) * l_hat

    def options(self, state, f0, with_check2):
        """{queue: (b, [feasible clocks])} with the largest b that has any feasible clock."""
        out = {}
        for q in state.nonempty():
            for b in range(min(state.length(q), state.b_max), 0, -1):
                fs = [f for f in self.table.freqs if self.feasible(state, q, b, f, f0, with_check2)]
                if fs:
                    out[q] = (b, fs)
                    break
        return out

    def slack(self, state, q):
        n = min(state.length(q), state.b_max)
        return state.head(q) + self.slo_s - state.now - self.lat_s(q, n, self.f_max)

    def pick_clock(self, q, b, fs, f0):
        e_min = min(self.table.energy(self.models[q], b, f) for f in self.table.freqs)
        scores = {f: self.score(q, b, f, None, e_min) for f in fs}
        best = min(fs, key=lambda f: (scores[f], -f))
        if f0 in scores and scores[f0] <= scores[best] * (1.0 + self.keep_within):
            return f0
        return best

    def decide(self, state):
        f0 = state.freq_now if state.freq_now is not None else self.f_max
        opts = self.options(state, f0, True)
        if opts:
            self.n_clean += 1
        else:
            opts = self.options(state, f0, False)
            if opts:
                self.n_risky += 1
        if opts:
            q = min(opts, key=lambda r: (self.slack(state, r), r))
            b, fs = opts[q]
            return Serve(q, b, freq=self.pick_clock(q, b, fs, f0))
        self.n_miss += 1
        q = min(state.nonempty(), key=lambda r: (self.slack(state, r), r))
        return Serve(q, min(state.length(q), state.b_max), freq=self.f_max)

    def counters(self):
        return {"clean": self.n_clean, "risky": self.n_risky, "miss": self.n_miss}


class DvfsFixed:
    """Baseline: same queue rule (least slack at the chosen clock), batch = whole queue up to B, one fixed clock."""
    name = "dvfs_fixed"

    def __init__(self, freq_mhz, slo_ms, table, queue_models):
        self.f = int(freq_mhz)
        self.slo_s = slo_ms / 1000.0
        self.table = table
        self.models = queue_models

    def decide(self, state):
        def slack(q):
            n = min(state.length(q), state.b_max)
            return state.head(q) + self.slo_s - state.now - self.table.lat(self.models[q], n, self.f) / 1000.0
        q = min(state.nonempty(), key=lambda r: (slack(r), r))
        return Serve(q, min(state.length(q), state.b_max), freq=self.f)

    def counters(self):
        return {}


# ---------- GPU clock (Jetson AGX Xavier devfreq). Needs root. ----------

DEVFREQ = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b"


def _read(name):
    with open(os.path.join(DEVFREQ, name)) as fh:
        return fh.read().strip()


def _write(name, value):
    t0 = time.monotonic()
    with open(os.path.join(DEVFREQ, name), "w") as fh:
        fh.write("%s\n" % value)
    return time.monotonic() - t0


class Clock:
    """Pins the GPU to one clock at a time (min_freq = max_freq). The original min/max are saved to backup_path first
    and written back at exit. An existing backup means an earlier session died pinned: it holds the real original."""

    def __init__(self, backup_path):
        self.backup_path = backup_path
        self.available_hz = sorted(int(x) for x in _read("available_frequencies").split())
        self.available_mhz = [int(round(h / 1e6)) for h in self.available_hz]
        self.original = None
        if os.path.exists(backup_path):
            try:
                with open(backup_path) as fh:
                    saved = json.load(fh)
                if "min_freq" in saved and "max_freq" in saved:
                    self.original = saved
                    print("NOTICE: %s exists; an earlier session did not restore. Using it as the original state." % backup_path)
            except ValueError:
                pass
            if self.original is None:
                print("NOTICE: %s is empty or unreadable (the board probably died while it was written); ignoring it and "
                      "taking the current clock state as the original." % backup_path)
        if self.original is None:
            self.original = {"min_freq": _read("min_freq"), "max_freq": _read("max_freq")}
            tmp = backup_path + ".tmp"
            with open(tmp, "w") as fh:                  # write then rename, so a crash never leaves a half-written backup
                json.dump(self.original, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, backup_path)
        self.cap_hz = int(self.original["max_freq"])
        self.cap_mhz = int(round(self.cap_hz / 1e6))
        self.current_mhz = None
        self.switches = 0
        self.switch_ms = []
        self.restored = False

    def to_hz(self, f_mhz):
        """The table says 829, sysfs has 828750000: take the nearest available clock, refuse if it is more than 1 MHz away."""
        hz = min(self.available_hz, key=lambda h: abs(h - f_mhz * 1e6))
        if abs(hz - f_mhz * 1e6) > 1e6 or hz > self.cap_hz:
            raise ValueError("clock %s MHz not usable: available %s MHz, mode cap %d MHz" % (f_mhz, self.available_mhz, self.cap_mhz))
        return hz

    def check(self, freqs_mhz):
        for f in freqs_mhz:
            self.to_hz(f)

    def set(self, f_mhz):
        """Returns the time the change took in ms (0 when already there). Raising writes max first, lowering min first."""
        if f_mhz == self.current_mhz:
            return 0.0
        hz = self.to_hz(f_mhz)
        cur_max = int(_read("max_freq"))
        order = ["max_freq", "min_freq"] if hz >= cur_max else ["min_freq", "max_freq"]
        spent = sum(_write(name, hz) for name in order)
        self.current_mhz = f_mhz
        self.switches += 1
        self.switch_ms.append(spent * 1000.0)
        return spent * 1000.0

    def restore(self):
        if self.restored:
            return True
        self.restored = True
        ok = True
        for name in ("max_freq", "min_freq"):
            try:
                _write(name, self.original[name])
            except OSError as e:
                ok = False
                print("COULD NOT WRITE %s: %s ; fix by hand: echo %s > %s/%s" % (name, e, self.original[name], DEVFREQ, name))
        if ok and os.path.exists(self.backup_path):
            os.replace(self.backup_path, "%s.restored_%s" % (self.backup_path, time.strftime("%Y%m%d_%H%M%S")))
        return ok

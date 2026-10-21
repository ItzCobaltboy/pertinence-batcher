# Scheduling policies. A policy is a small class with
#
#   decide(state) -> Serve | Wait | Promote | Drop
#   on_batch_done(model, b, ms)        optional: measured time of a batch it ran
#
# and is only asked when the GPU is idle and at least one queue is non-empty.
# `state` (built fresh by the executor for every call) gives read-only access to:
#   state.now              run clock, seconds
#   state.queues[m]        tuple of Job, oldest first (t_arrival, t_enqueued, deadline, slo_ms, job_id)
#   state.models           pool, cost-ascending
#   state.t(m, b, stat)    profiled batch time in SECONDS (linear interpolation between sizes)
#   state.b_max            largest batch allowed
#   state.batch_sizes      profiled batch sizes
#   state.sizes(n)         allowed batch sizes with n jobs waiting: profiled sizes <= min(n, b_max)
#                          plus min(n, b_max) itself
# Times: arrival = when the camera produced the frame (trace time), deadline =
# arrival + SLO. Each reimplemented policy says what it changed from its paper; the
# README has the same as a table.

import math
from collections import deque, namedtuple

Serve = namedtuple("Serve", ["queue", "b", "job_ids"], defaults=[None])   # job_ids: explicit pick
Wait = namedtuple("Wait", ["until"])
Promote = namedtuple("Promote", ["src", "dst", "k"])
Drop = namedtuple("Drop", ["queue", "job_ids"])


class State:
    def __init__(self, now, queues, models, profile, b_max, stat=None):
        self.now = now
        self.queues = queues
        self.models = models
        self.profile = profile
        self.b_max = b_max
        self.batch_sizes = profile.batch_sizes
        self.stat = stat

    def t(self, model, b, stat=None):
        return self.profile.t(model, b, stat or self.stat) / 1000.0

    def sizes(self, n):
        top = min(n, self.b_max)
        return sorted(set([s for s in self.batch_sizes if s <= top] + [top]))

    def nonempty(self):
        return [m for m in self.models if self.queues[m]]


def oldest_head(state, models):
    """Model whose head job arrived first (ties: pool order)."""
    return min(models, key=lambda m: (state.queues[m][0].t_arrival, state.models.index(m)))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
class FcfsNoBatch:
    """Serve the queue holding the oldest job, one job at a time."""

    def decide(self, state):
        return Serve(oldest_head(state, state.nonempty()), 1)


class FcfsBatch:
    """Serve the queue holding the oldest job, as large a batch as allowed."""

    def decide(self, state):
        m = oldest_head(state, state.nonempty())
        return Serve(m, min(len(state.queues[m]), state.b_max))


class Lqf:
    """Longest queue first (ties: oldest head), as large a batch as allowed."""

    def decide(self, state):
        models = state.nonempty()
        longest = max(len(state.queues[m]) for m in models)
        m = oldest_head(state, [m for m in models if len(state.queues[m]) == longest])
        return Serve(m, min(len(state.queues[m]), state.b_max))


# ---------------------------------------------------------------------------
# Triton dynamic batcher (NVIDIA docs, user_guide/batcher.html)
# ---------------------------------------------------------------------------
class Triton:
    """One dynamic batcher per model, no coordination between models.
    A queue is ready when it can form a max_batch batch, a preferred-size batch,
    or its oldest request has waited max_queue_delay since it entered the
    scheduler (t_enqueued). Batch formed: max_batch if possible, else the largest
    formable preferred size, else everything up to max_batch. Real Triton would run
    ready models concurrently; here the one GPU serves the ready queue with the
    oldest head. Nothing ready: wait for the earliest delay to expire (or an arrival)."""

    def __init__(self, max_queue_delay_ms=0.0, max_batch=None, preferred_batch_sizes=None, delays_ms=None):
        self.delay_s = float(max_queue_delay_ms) / 1000.0
        self.max_batch = max_batch
        self.preferred = sorted(preferred_batch_sizes or [])
        self.delays_s = {m: float(v) / 1000.0 for m, v in (delays_ms or {}).items()}

    def _max_b(self, state):
        return min(self.max_batch or state.b_max, state.b_max)

    def batch_size(self, n, max_b):
        if n >= max_b:
            return max_b
        prefs = [p for p in self.preferred if p <= n]
        return prefs[-1] if prefs else n

    def decide(self, state):
        max_b = self._max_b(state)
        ready = []
        next_timeout = float("inf")
        for m in state.nonempty():
            q = state.queues[m]
            delay = self.delays_s.get(m, self.delay_s)
            timeout_at = q[0].t_enqueued + delay
            preferred_ok = any(p <= len(q) for p in self.preferred)
            if len(q) >= max_b or preferred_ok or state.now >= timeout_at:
                ready.append(m)
            else:
                next_timeout = min(next_timeout, timeout_at)
        if ready:
            m = oldest_head(state, ready)
            return Serve(m, self.batch_size(len(state.queues[m]), max_b))
        return Wait(next_timeout)


# ---------------------------------------------------------------------------
# EdgeServing (Cao et al., arXiv 2605.05527, Sec. V)
# ---------------------------------------------------------------------------
class EdgeServing:
    """Work-conserving one-step greedy. For each non-empty queue m: B = min(|Q_m|,
    B_max) (Eq. 5); predict every queued job's waiting time after the batch
    finishes, w + L(m, B), with the B served jobs removed (future arrivals ignored);
    stability score S_m = sum of f(w) = min(exp(w / tau - 1), C) (Eqs. 3-4); serve
    argmin S_m (Eq. 7). tau = the job's SLO. C = 10: the paper clips tasks past
    tau (1 + ln 10). L uses the P95 profile, as the paper's scheduler does (Sec. VI-A).
    Exit-point selection (Eq. 6) is dropped: these models have no early exits.
    Ties (unspecified in the paper): oldest head, then pool order."""

    def __init__(self, b_max=None, C=10.0, stat="p95"):
        self.b_max = b_max
        self.C = float(C)
        self.stat = stat

    def urgency(self, w, tau):
        x = w / tau - 1.0
        return self.C if x >= math.log(self.C) else min(math.exp(x), self.C)

    def score(self, state, m, B):
        done_at = state.now + state.t(m, B, self.stat)
        s = 0.0
        for m2 in state.models:
            jobs = state.queues[m2][B:] if m2 == m else state.queues[m2]
            for job in jobs:
                s += self.urgency(done_at - job.t_arrival, job.slo_ms / 1000.0)
        return s

    def decide(self, state):
        b_max = min(self.b_max or state.b_max, state.b_max)
        best = None
        for m in state.nonempty():
            B = min(len(state.queues[m]), b_max)
            key = (self.score(state, m, B), state.queues[m][0].t_arrival, state.models.index(m))
            if best is None or key < best[0]:
                best = (key, m, B)
        return Serve(best[1], best[2])


# ---------------------------------------------------------------------------
# Clockwork (Gujarati et al., OSDI 2020, Sec. 5.3 "Scheduling INFER", and the
# strategy-queue scheduler in its source, controller/concurrent_infer_and_load_scheduler.cpp)
# ---------------------------------------------------------------------------
class Clockwork:
    """INFER scheduling only, one executor, all models resident (no LOAD).
    Every queued job j makes one strategy per supported batch size s with
    latest = d_j - T(s), d_j = deadline - buffer (Clockwork aims 5 ms early).
    Strategies are tried most-urgent first (smallest latest). A strategy (j, s) is
    valid if it finishes in time (now + T(s) <= d_j) and at least s jobs are queued
    from j onward. It then grows to the largest supported size those jobs allow; if
    that larger batch would miss d_j the strategy is skipped (the source returns no
    action rather than running the smaller batch). Older jobs that cannot make the
    batch's completion time are dropped ("skipped"); the batch is the s oldest
    remaining jobs. Jobs that cannot finish even alone are dropped (cancelled).
    T(s) is the p99 of the last `window` measured runs of (model, s), seeded with the
    profile, like Clockwork's action profiles. Supported sizes = profiled sizes
    <= B_max (Clockwork's default kernels are 1, 2, 4, 8, 16).
    commit_ahead_ms pads the start time (Clockwork schedules ~5-10 ms ahead to hide
    RPC latency; with a local executor it is 0 by default).
    no-drop: late jobs are not dropped; they are skipped by strategies and served
    one at a time (oldest first) only when no strategy is valid."""

    def __init__(self, buffer_ms=5.0, window=10, percentile=0.99, commit_ahead_ms=0.0, drop=True,
                 online=True, stat=None):
        self.buffer_s = float(buffer_ms) / 1000.0
        self.window = int(window)
        self.percentile = float(percentile)
        self.commit_ahead_s = float(commit_ahead_ms) / 1000.0
        self.drop = drop
        self.online = online
        self.stat = stat
        self.samples = {}

    def on_batch_done(self, model, b, ms):
        if self.online:
            self.samples.setdefault((model, b), deque(maxlen=self.window)).append(ms / 1000.0)

    def est(self, state, m, s):
        window = self.samples.get((m, s))
        if not window:
            return state.t(m, s, self.stat)
        values = sorted(window)
        pos = self.percentile * (len(values) - 1)      # SlidingWindow::get_percentile
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    def decide(self, state):
        free_at = state.now + self.commit_ahead_s
        sizes = [s for s in state.batch_sizes if s <= state.b_max]

        def d(job):
            return job.deadline - self.buffer_s

        feasible = {}
        late = {}
        for m in state.nonempty():
            t1 = self.est(state, m, sizes[0])
            feasible[m] = [j for j in state.queues[m] if d(j) >= free_at + t1]
            late[m] = [j for j in state.queues[m] if d(j) < free_at + t1]
            if self.drop and late[m]:
                return Drop(m, [j.job_id for j in late[m]])

        strategies = []
        for m, jobs in feasible.items():
            for i, j in enumerate(jobs):
                for s in sizes:
                    strategies.append((d(j) - self.est(state, m, s), state.models.index(m), i, s, m))
        strategies.sort()

        for _, _, i, s, m in strategies:
            jobs = feasible[m]
            j = jobs[i]
            if d(j) < free_at + self.est(state, m, s):
                continue
            available = len(jobs) - i
            if available < s:
                continue
            candidate = max(x for x in sizes if x <= available)
            if candidate > s:
                if free_at + self.est(state, m, candidate) > d(j):
                    continue
                s = candidate
            completion = free_at + self.est(state, m, s)
            skipped = [k for k in jobs[:i] if d(k) < completion]
            if skipped and self.drop:
                return Drop(m, [k.job_id for k in skipped])
            batch = [k for k in jobs if k not in skipped][:s]
            return Serve(m, s, [k.job_id for k in batch])

        # nothing valid (only reachable with drop=False): serve the oldest late job alone
        m = min(state.nonempty(), key=lambda q: (state.queues[q][0].t_arrival, state.models.index(q)))
        return Serve(m, 1, [state.queues[m][0].job_id])


# ---------------------------------------------------------------------------
# Symphony (Chen et al., arXiv 2308.07470, Sec. 3.1-3.3, Algorithm 1)
# ---------------------------------------------------------------------------
class Symphony:
    """Deferred batch scheduling. Per model a candidate batch B = GetBatch(Q): from
    the head, keep adding the next job while the batch still finishes by its
    earliest deadline d if started now (capped at B_max). The candidate may launch
    from exec = max(now, d - l(|B| + 1)) (the frontrun point: waiting for one more
    job would make the head miss d) until latest = d - l(|B|). At |B| = B_max it
    cannot grow, so exec = now. When the GPU is free, serve the due candidate
    (exec <= now <= latest) with the smallest latest (most urgent); if none is due,
    wait until the earliest exec (or an arrival, which updates the candidates).
    Jobs that cannot finish even alone are dropped. l(b) is the measured profile
    (the paper fits an affine l(b) = alpha b + beta per model).
    no-drop: late jobs are kept and served first, oldest queue first, up to B_max."""

    def __init__(self, drop=True, stat=None):
        self.drop = drop
        self.stat = stat

    def candidate(self, state, m, jobs):
        batch = []
        d = float("inf")
        for job in jobs:
            if len(batch) >= state.b_max:
                break
            d_new = min(d, job.deadline)
            if state.now + state.t(m, len(batch) + 1, self.stat) > d_new:
                break
            batch.append(job)
            d = d_new
        if not batch:
            return None
        if len(batch) >= state.b_max:
            exec_t = state.now
        else:
            exec_t = max(state.now, d - state.t(m, len(batch) + 1, self.stat))
        latest = d - state.t(m, len(batch), self.stat)
        return batch, exec_t, latest

    def decide(self, state):
        cands = {}
        late = {}
        for m in state.nonempty():
            t1 = state.t(m, 1, self.stat)
            late[m] = [j for j in state.queues[m] if state.now + t1 > j.deadline]
            if late[m] and self.drop:
                return Drop(m, [j.job_id for j in late[m]])
            c = self.candidate(state, m, [j for j in state.queues[m] if state.now + t1 <= j.deadline])
            if c is not None:
                cands[m] = c
        with_late = [m for m in late if late[m]]
        if with_late:
            m = min(with_late, key=lambda q: (late[q][0].t_arrival, state.models.index(q)))
            ids = [j.job_id for j in late[m][:state.b_max]]
            return Serve(m, len(ids), ids)
        due = [m for m, (_, exec_t, latest) in cands.items() if exec_t <= state.now <= latest]
        if due:
            m = min(due, key=lambda q: (cands[q][2], state.models.index(q)))
            batch = cands[m][0]
            return Serve(m, len(batch), [j.job_id for j in batch])
        return Wait(min(exec_t for _, exec_t, _ in cands.values()))


# ---------------------------------------------------------------------------
# cost_v1 (CLAUDE.md, "Policy v1 defined")
# ---------------------------------------------------------------------------
class CostV1:
    """argmin over (queue i, batch size b) of
    beta * sum over jobs left waiting of u(age at completion) + (1 - beta) * T_i(b),
    u(J) = (exp(lam J) - 1) / lam (lam = 0: plain sum). Ages and T in ms, lam in 1/ms.
    No Wait option, no padding (b <= |Q_i|)."""

    def __init__(self, beta=0.5, lam=0.0):
        self.beta = float(beta)
        self.lam = float(lam)

    def u(self, age_ms):
        if self.lam == 0.0:
            return age_ms
        return math.expm1(self.lam * age_ms) / self.lam

    def decide(self, state):
        best = None
        for m in state.nonempty():
            q = state.queues[m]
            for b in state.sizes(len(q)):
                T = state.t(m, b)
                done = state.now + T
                left = 0.0
                for m2 in state.models:
                    for job in (state.queues[m2][b:] if m2 == m else state.queues[m2]):
                        left += self.u((done - job.t_arrival) * 1000.0)
                key = (self.beta * left + (1 - self.beta) * T * 1000.0, state.models.index(m), -b)
                if best is None or key < best[0]:
                    best = (key, m, b)
        return Serve(best[1], best[2])


POLICIES = {
    "fcfs_nobatch": FcfsNoBatch,
    "fcfs_batch": FcfsBatch,
    "lqf": Lqf,
    "triton": Triton,
    "edgeserving": EdgeServing,
    "clockwork": Clockwork,
    "symphony": Symphony,
    "cost_v1": CostV1,
}


def make_policy(name, params=None):
    if name not in POLICIES:
        raise ValueError("unknown policy %r (have: %s)" % (name, ", ".join(POLICIES)))
    return POLICIES[name](**(params or {}))

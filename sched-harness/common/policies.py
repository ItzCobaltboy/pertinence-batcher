# Scheduling policies. Each is a small class with decide(state) -> Serve(queue, b) or Wait(until).
# The executor calls decide only when at least one queue is non-empty. No torch here.
# Nothing inherits from anything; shared helpers are plain functions.
import math


class Serve:
    def __init__(self, queue, b, drops=None):
        self.queue = queue
        self.b = b
        self.drops = drops      # optional {queue: n}: cancel the n oldest jobs of that queue first (logged as dropped = 2)


class Wait:
    def __init__(self, until, drops=None):
        self.until = until      # monotonic time; the executor also wakes on the next arrival
        self.drops = drops


class State:
    """Read-only snapshot taken by the executor under the lock."""

    def __init__(self, now, arrivals, service_ms, b_max, draining=False):
        self.now = now
        self.arrivals = arrivals        # per queue: enqueue times of its jobs, oldest first
        self.service_ms = service_ms    # function (queue, b, stat=None) -> milliseconds; stat 'p50' (default) or 'p99'
        self.b_max = b_max
        self.draining = draining        # True once arrivals have stopped

    def length(self, q):
        return len(self.arrivals[q])

    def head(self, q):
        return self.arrivals[q][0]

    def nonempty(self):
        return [q for q in range(len(self.arrivals)) if self.arrivals[q]]


def oldest_queue(state, candidates):
    """Among the candidate queues, the one whose oldest job arrived first (tie: lower index)."""
    return min(candidates, key=lambda q: (state.head(q), q))


def full_batch(state, q):
    return min(state.length(q), state.b_max)


class FcfsNoBatch:
    name = "fcfs_nobatch"

    def decide(self, state):
        return Serve(oldest_queue(state, state.nonempty()), 1)


class FcfsBatch:
    name = "fcfs_batch"

    def decide(self, state):
        q = oldest_queue(state, state.nonempty())
        return Serve(q, full_batch(state, q))


class Lqf:
    # longest queue first (Xia et al. 2002); tie: the queue with the older head
    name = "lqf"

    def decide(self, state):
        q = min(state.nonempty(), key=lambda i: (-state.length(i), state.head(i), i))
        return Serve(q, full_batch(state, q))


class Triton:
    # Triton-style dynamic batching: each queue has its own timer (max_queue_delay)
    name = "triton"

    def __init__(self, delay_ms):
        self.delay_s = delay_ms / 1000.0

    def decide(self, state):
        ready = []
        for q in state.nonempty():
            if state.length(q) >= state.b_max or state.now - state.head(q) >= self.delay_s:
                ready.append(q)
        if ready:
            q = oldest_queue(state, ready)
            return Serve(q, full_batch(state, q))
        return Wait(min(state.head(q) + self.delay_s for q in state.nonempty()))


class FixedBatchB:
    # sanity baseline: only ever serve exactly b jobs (what does a fixed target size cost)
    name = "fixed_batch_b"

    def __init__(self, b):
        self.b = b

    def decide(self, state):
        enough = [q for q in state.nonempty() if state.length(q) >= self.b]
        if enough:
            return Serve(oldest_queue(state, enough), self.b)
        if state.draining:
            # no more arrivals will come, so waiting would hang the run: flush what is left
            q = oldest_queue(state, state.nonempty())
            return Serve(q, state.length(q))
        return Wait(state.now + 3600.0)     # effectively "until the next arrival"


class EdgeServing:
    # EdgeServing (arXiv 2605.05527) reduced from its (model, exit, batch) decision to (queue, batch size).
    # Per non-empty queue: the largest batch b <= min(length, b_max) whose service still lets the oldest job meet the SLO
    # (the analogue of "deepest exit that fits"); if no b fits, b = 1 (the paper does not say what to do). Then score
    # the candidate: add its service time to the wait of every job that stays queued, in all queues, and sum the urgency
    # f(w) = min(exp(w / slo - 1), clip). The lowest total wins (tie: older head, lower index). Work-conserving: never waits.
    # Like the paper, arrivals during the service are ignored and late jobs are never dropped.
    name = "edgeserving"

    def __init__(self, slo_ms, clip=10.0):
        self.slo_s = slo_ms / 1000.0
        self.clip = clip

    def urgency(self, wait_s):
        return math.exp(min(wait_s / self.slo_s - 1.0, math.log(self.clip)))      # same as min(exp(.), clip), no overflow

    def pick_b(self, state, q):
        w_max = state.now - state.head(q)
        for b in range(min(state.length(q), state.b_max), 0, -1):
            if w_max + state.service_ms(q, b) / 1000.0 <= self.slo_s:
                return b
        return 1

    def decide(self, state):
        waits = [[state.now - t for t in queue] for queue in state.arrivals]
        best = None
        for q in state.nonempty():
            b = self.pick_b(state, q)
            service_s = state.service_ms(q, b) / 1000.0
            score = 0.0
            for other, ws in enumerate(waits):
                for w in (ws[b:] if other == q else ws):      # the served jobs leave
                    score += self.urgency(w + service_s)
            key = (score, state.head(q), q)
            if best is None or key < best[0]:
                best = (key, q, b)
        return Serve(best[1], best[2])


class Clockwork:
    # Clockwork's INFER scheduler (Gujarati et al., OSDI 2020), reduced to one GPU with every model resident: no LOAD/UNLOAD,
    # no scheduling ahead (lead time) and no earliest/latest window, the decision is taken when the GPU is free.
    # Deadline of a job = enqueue time + SLO. Execution times are the calibrated p99 (the paper predicts with a rolling p99).
    # Per queue: jobs that cannot meet the deadline even alone (now + T99(1) > deadline) are cancelled. A strategy (queue, b)
    # exists for b in {1, 2, 4, 8, 16} when the queue holds at least b live jobs and the OLDEST live job still meets its
    # deadline with a batch of b; its latest start is deadline(oldest) - T99(b). The strategy with the smallest latest wins
    # (tie: lower queue, larger b), then the batch is raised to the largest such b of that queue (speculative increase).
    # The batch is always the b oldest jobs, so a bigger batch is never filled by skipping older jobs (greedy).
    name = "clockwork"
    SIZES = (1, 2, 4, 8, 16)

    def __init__(self, slo_ms):
        self.slo_s = slo_ms / 1000.0

    def decide(self, state):
        drops = {}
        valid = {}                  # queue -> list of (b, latest)
        for q in state.nonempty():
            times = state.arrivals[q]
            t1 = state.service_ms(q, 1, "p99") / 1000.0
            dead = 0
            while dead < len(times) and state.now + t1 > times[dead] + self.slo_s:
                dead += 1
            if dead:
                drops[q] = dead
            live = len(times) - dead
            if live == 0:
                continue
            deadline = times[dead] + self.slo_s
            for b in self.SIZES:
                if b > live or b > state.b_max:
                    continue
                t = state.service_ms(q, b, "p99") / 1000.0
                if state.now + t <= deadline:
                    valid.setdefault(q, []).append((b, deadline - t))
        if not valid:
            return Wait(state.now + 3600.0, drops or None)
        best = min(((latest, q, -b) for q, options in valid.items() for b, latest in options))
        q = best[1]
        return Serve(q, max(b for b, _ in valid[q]), drops or None)


class Symphony:
    # Symphony deferred batch scheduling (Algorithm 1 of the paper), one GPU, all models resident: no staggering across GPUs,
    # no autoscaling, no network-delay terms. Deadline of a job = enqueue time + SLO; times are the calibrated p99.
    # Per queue (the "candidate batch"): jobs that cannot finish even alone are dropped; b = the largest calibrated size that
    # fits the live jobs and still lets the oldest one meet its deadline if started now; latest = d - T(b) (last start that
    # meets the deadline); frontrun = d - T(next size above b) (after it, one more job could not be added). The candidate is
    # ready once frontrun has passed, never earlier: the GPU idles on purpose until then (not work-conserving). Among ready
    # candidates the one with the smallest latest wins (the paper's GPU-timer rule); with none ready, Wait until the first
    # frontrun (the runtime also wakes on an arrival, which re-forms the batch). Sizes are the calibrated ones because the
    # paper assumes every batch size is profiled and interpolated odd sizes run slower than predicted.
    name = "symphony"
    SIZES = (1, 2, 3, 4, 6, 8, 12, 16)

    def __init__(self, slo_ms):
        self.slo_s = slo_ms / 1000.0

    def decide(self, state):
        drops = {}
        ready = []              # (latest, queue, b)
        soonest = None          # earliest frontrun among candidates that are not ready yet
        sizes = [s for s in self.SIZES if s <= state.b_max]
        for q in state.nonempty():
            times = state.arrivals[q]
            ell = lambda s: state.service_ms(q, s, "p99") / 1000.0
            dead = 0
            while dead < len(times) and state.now + ell(1) > times[dead] + self.slo_s:
                dead += 1
            if dead:
                drops[q] = dead
            live = len(times) - dead
            if live == 0:
                continue
            d = times[dead] + self.slo_s
            b = max(s for s in sizes if s == 1 or (s <= live and state.now + ell(s) <= d))
            bigger = [s for s in sizes if s > b]
            frontrun = d - ell(bigger[0]) if bigger else float("-inf")
            if frontrun <= state.now:
                ready.append((d - ell(b), q, b))
            elif soonest is None or frontrun < soonest:
                soonest = frontrun
        if ready:
            _, q, b = min(ready)
            return Serve(q, b, drops or None)
        return Wait(soonest if soonest is not None else state.now + 3600.0, drops or None)


class NotBuilt:
    # second policy set and my own policy: the interface exists, the logic does not (yet)
    def __init__(self, name):
        self.name = name

    def decide(self, state):
        raise NotImplementedError("policy %s is not implemented yet (see spec section 6)" % self.name)


STUBS = ["index_T"]


def make_policy(name, params):
    if name == "fcfs_nobatch":
        return FcfsNoBatch()
    if name == "fcfs_batch":
        return FcfsBatch()
    if name == "lqf":
        return Lqf()
    if name == "triton":
        return Triton(float(params["delay_ms"]))
    if name == "fixed_batch_b":
        return FixedBatchB(int(params["b"]))
    if name == "edgeserving":
        return EdgeServing(float(params["slo_ms"]))
    if name == "clockwork":
        return Clockwork(float(params["slo_ms"]))
    if name == "symphony":
        return Symphony(float(params["slo_ms"]))
    if name in STUBS:
        return NotBuilt(name)
    raise ValueError("unknown policy %r" % name)


def label(name, params):
    """e.g. triton_delay_ms5, fixed_batch_b_b4, lqf."""
    text = name
    for key in sorted(params):
        value = params[key]
        if float(value) == int(float(value)):
            value = int(float(value))
        text += "_%s%s" % (key, value)
    return text


def run_name(mix, lam, name, params, seed):
    return "%s_lam%g_%s_s%d" % (mix, lam, label(name, params), seed)

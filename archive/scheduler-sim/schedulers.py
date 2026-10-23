"""Concrete schedulers: knob 2 of the simulator (which queue and batch size to
run next). Each class only implements decide(now). Shared helpers
(nonempty_queue_ids, oldest_queue_id, full_batch_size) live in the Scheduler
base class in sim.py.
"""
from sim import Scheduler


class FCFSNoBatchScheduler(Scheduler):
    """Run the single oldest job overall, batch size 1. The no-batching
    reference baseline: lowest latency when load is light, but wastes the
    accelerator's ability to batch, so it overloads first."""

    def decide(self, now):
        candidates = self.nonempty_queue_ids()
        if len(candidates) == 0:
            return None
        return ("run", self.oldest_queue_id(candidates), 1)


class FCFSBatchScheduler(Scheduler):
    """Pick the queue holding the oldest job overall, then run as many of its
    jobs as fit in one batch. Fair in arrival order and batches for free
    whatever has piled up, without ever waiting on purpose."""

    def decide(self, now):
        candidates = self.nonempty_queue_ids()
        if len(candidates) == 0:
            return None
        queue_id = self.oldest_queue_id(candidates)
        return ("run", queue_id, self.full_batch_size(queue_id))


class LongestQueueScheduler(Scheduler):
    """Serve the longest queue first with the biggest batch it can fill.
    Maximises jobs per batch, but a short queue can starve under load."""

    def decide(self, now):
        best_id = None
        for queue_id in self.nonempty_queue_ids():
            # strictly longer only, so on a tie the lower queue_id wins
            if best_id is None or self.queues[queue_id].length() > self.queues[best_id].length():
                best_id = queue_id
        if best_id is None:
            return None
        return ("run", best_id, self.full_batch_size(best_id))


class TimeoutBatchScheduler(Scheduler):
    """Triton-style dynamic batching. A queue is ready when it can fill its
    max batch size, or its oldest job has waited tau_ms. Among ready queues
    run the one with the oldest job; if none are ready, wait for the earliest
    timeout. Trades a bounded extra delay for bigger batches."""

    def __init__(self, queues, profile, tau_ms):
        Scheduler.__init__(self, queues, profile)
        if tau_ms < 0:
            raise ValueError("tau_ms must be >= 0")
        self.tau_ms = tau_ms

    def decide(self, now):
        candidates = self.nonempty_queue_ids()
        if len(candidates) == 0:
            return None

        ready = []
        earliest_timeout = None
        for queue_id in candidates:
            queue = self.queues[queue_id]
            timeout_time = queue.oldest_arrival() + self.tau_ms
            if queue.length() >= self.max_batch_for(queue_id) or timeout_time <= now:
                ready.append(queue_id)
            elif earliest_timeout is None or timeout_time < earliest_timeout:
                earliest_timeout = timeout_time

        if len(ready) == 0:
            return ("wait", earliest_timeout)
        queue_id = self.oldest_queue_id(ready)
        return ("run", queue_id, self.full_batch_size(queue_id))

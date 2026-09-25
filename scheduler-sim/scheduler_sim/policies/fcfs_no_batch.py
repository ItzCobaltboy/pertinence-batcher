from __future__ import annotations

from ..policy import PolicyView, RunBatch, SchedulingPolicy, Wait, policy_registry


@policy_registry.register("fcfs_no_batch")
class FCFSNoBatch(SchedulingPolicy):
    """Reference baseline: always run the single oldest job overall,
    batch size 1."""

    def decide(self, view: PolicyView):
        if not view.accelerator_free:
            return Wait(until_time=view.now)
        best_q, best_t = None, None
        for q in view.nonempty_queues():
            t = view.queues[q].oldest_arrival_time
            if best_t is None or t < best_t:
                best_q, best_t = q, t
        if best_q is None:
            return Wait(until_time=view.now)
        return RunBatch(queue_id=best_q, batch_size=1)

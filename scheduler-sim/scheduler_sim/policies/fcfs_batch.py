from __future__ import annotations

from ..policy import PolicyView, RunBatch, SchedulingPolicy, Wait, policy_registry


@policy_registry.register("fcfs_batch")
class FCFSBatch(SchedulingPolicy):
    """Pick the queue holding the oldest job overall, then run the
    largest batch size that queue's length and its model support."""

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

        model = view.queue_to_model[best_q]
        length = view.queues[best_q].length
        profile = view.runtime_profile
        # Under both "pad" and "interpolate", running the whole queue
        # (up to the model's max supported size) is never worse than
        # running fewer jobs: pad-mode charges the same rounded-up
        # cost either way, and interpolate's cost is monotone in b.
        b = min(length, profile.max_batch_size(model))
        return RunBatch(queue_id=best_q, batch_size=b)

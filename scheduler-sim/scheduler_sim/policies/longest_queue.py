from __future__ import annotations

from ..policy import PolicyView, RunBatch, SchedulingPolicy, Wait, policy_registry


@policy_registry.register("longest_queue")
class LongestQueue(SchedulingPolicy):
    """Serve the longest queue first, largest supported batch it can
    fill. Ties broken by queue id for determinism."""

    def decide(self, view: PolicyView):
        if not view.accelerator_free:
            return Wait(until_time=view.now)
        nonempty = view.nonempty_queues()
        if not nonempty:
            return Wait(until_time=view.now)
        best_q = max(nonempty, key=lambda q: (view.queues[q].length, -q))

        model = view.queue_to_model[best_q]
        length = view.queues[best_q].length
        profile = view.runtime_profile
        b = min(length, profile.max_batch_size(model))
        return RunBatch(queue_id=best_q, batch_size=b)

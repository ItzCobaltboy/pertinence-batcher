from __future__ import annotations

from ..policy import PolicyView, RunBatch, SchedulingPolicy, Wait, policy_registry


@policy_registry.register("timeout_batch")
class TimeoutBatch(SchedulingPolicy):
    """Triton dynamic-batcher style: per queue, run as soon as it can
    fill the model's max supported batch size, or its oldest job has
    waited `tau_ms`, whichever comes first. Otherwise Wait until the
    earliest timeout across all nonempty queues.

    Ties among simultaneously-ready queues broken by earliest oldest
    arrival (most overdue first), then queue id.
    """

    def __init__(self, tau_ms: float):
        if tau_ms < 0:
            raise ValueError(f"tau_ms must be >= 0, got {tau_ms}")
        self.tau_ms = tau_ms

    def decide(self, view: PolicyView):
        if not view.accelerator_free:
            return Wait(until_time=view.now)
        nonempty = view.nonempty_queues()
        if not nonempty:
            return Wait(until_time=view.now)

        profile = view.runtime_profile
        ready = []
        earliest_timeout = None
        for q in nonempty:
            qv = view.queues[q]
            model = view.queue_to_model[q]
            max_b = profile.max_batch_size(model)
            timeout_at = qv.oldest_arrival_time + self.tau_ms
            can_fill = qv.length >= max_b
            timed_out = timeout_at <= view.now
            if can_fill or timed_out:
                ready.append(q)
            else:
                if earliest_timeout is None or timeout_at < earliest_timeout:
                    earliest_timeout = timeout_at

        if not ready:
            # nothing ready yet -- wake up at the earliest timeout
            return Wait(until_time=earliest_timeout)

        best_q = min(
            ready,
            key=lambda q: (view.queues[q].oldest_arrival_time, q),
        )
        model = view.queue_to_model[best_q]
        length = view.queues[best_q].length
        b = min(length, profile.max_batch_size(model))
        return RunBatch(queue_id=best_q, batch_size=b)

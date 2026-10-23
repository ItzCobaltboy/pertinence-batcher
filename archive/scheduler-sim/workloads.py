"""Concrete workloads: knob 1 of the simulator (how work arrives and which
queue gets it). Each class only has to implement choose_queue; the arrival
times come from the Workload base class (Poisson) unless overridden.
"""
import numpy as np

from sim import Workload


def normalise_weights(weights, number_of_queues):
    """Check the routing weights and scale them so they sum to 1."""
    if len(weights) != number_of_queues:
        raise ValueError("need one weight per queue: got " + str(len(weights))
                         + " weights for " + str(number_of_queues) + " queues")
    for weight in weights:
        if weight < 0:
            raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to more than 0")
    probabilities = []
    for weight in weights:
        probabilities.append(weight / total)
    return np.array(probabilities)


class UniformWorkload(Workload):
    """Poisson arrivals, every queue equally likely. The "no information"
    baseline: good when you have no data on which model frames need."""

    def choose_queue(self, stream_id):
        return int(self.rng.integers(0, len(self.queues)))


class WeightedWorkload(Workload):
    """Poisson arrivals, each job independently goes to queue i with
    probability weights[i]. Stand-in for real routing frequencies, e.g. the
    share of frames that need each YOLO size."""

    def __init__(self, queues, rng, num_streams, load_jobs_per_ms, weights):
        Workload.__init__(self, queues, rng, num_streams, load_jobs_per_ms)
        self.probabilities = normalise_weights(weights, len(queues))

    def choose_queue(self, stream_id):
        return int(self.rng.choice(len(self.queues), p=self.probabilities))


class StickyWorkload(Workload):
    """Poisson arrivals, but consecutive frames of a stream tend to need the
    same model: with probability stay_probability a job goes to the same queue
    as its stream's previous job, otherwise it is redrawn from weights. Mimics
    real video, where work arrives in bursts to one queue."""

    def __init__(self, queues, rng, num_streams, load_jobs_per_ms, weights, stay_probability):
        Workload.__init__(self, queues, rng, num_streams, load_jobs_per_ms)
        if stay_probability < 0 or stay_probability > 1:
            raise ValueError("stay_probability must be between 0 and 1")
        self.probabilities = normalise_weights(weights, len(queues))
        self.stay_probability = stay_probability
        self.last_queue_of_stream = {}  # stream_id -> queue of that stream's previous job

    def choose_queue(self, stream_id):
        if stream_id in self.last_queue_of_stream and self.rng.random() < self.stay_probability:
            queue_id = self.last_queue_of_stream[stream_id]
        else:
            # first job of the stream, or the stream "changed scene": redraw
            queue_id = int(self.rng.choice(len(self.queues), p=self.probabilities))
        self.last_queue_of_stream[stream_id] = queue_id
        return queue_id


class PeriodicRoutedWorkload(Workload):
    """Cameras: every stream sends a frame exactly every period, and each frame
    goes to queue i with probability weights[i], like a dispatcher reading the
    frame and picking a model. The period comes from the load:
    period_ms = num_streams / load_jobs_per_ms (8 streams at 0.25 jobs/ms is
    one frame every 32 ms per camera, about 30 fps)."""

    def __init__(self, queues, rng, num_streams, load_jobs_per_ms, weights):
        Workload.__init__(self, queues, rng, num_streams, load_jobs_per_ms)
        if load_jobs_per_ms <= 0:
            raise ValueError("load_jobs_per_ms must be positive")
        self.period_ms = num_streams / load_jobs_per_ms
        self.probabilities = normalise_weights(weights, len(queues))

    def next_arrival_time(self, now, stream_id):
        return now + self.period_ms

    def choose_queue(self, stream_id):
        return int(self.rng.choice(len(self.queues), p=self.probabilities))


class PeriodicWorkload(Workload):
    """Every stream sends a job exactly every period_ms (no randomness at all).
    Stream i goes to queue_for_stream[i] if given, else queue i % K. Mostly
    used for deterministic tests, and as an example of overriding arrivals."""

    def __init__(self, queues, rng, num_streams, period_ms, queue_for_stream=None):
        Workload.__init__(self, queues, rng, num_streams)
        if period_ms <= 0:
            raise ValueError("period_ms must be positive")
        self.period_ms = period_ms
        self.queue_for_stream = queue_for_stream

    def next_arrival_time(self, now, stream_id):
        return now + self.period_ms

    def choose_queue(self, stream_id):
        if self.queue_for_stream is not None:
            return self.queue_for_stream[stream_id]
        return stream_id % len(self.queues)

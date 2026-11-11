# Arrival lists and load maths. A schedule is a sorted list of (t_arrival_s, queue, pool_index), relative to
# the run start. The same seed gives the identical list for every policy.
import hashlib

import numpy as np

import config
from common import profile as profile_lib


def capacity(profile, weights, b_max):
    """Requests per second the board could serve if every batch were a full b_max batch:
    Lambda_cap = 1 / sum_i( w_i * T_i(b_max) / b_max )."""
    seconds_per_request = 0.0
    for q, letter in enumerate(config.QUEUES):
        t_full_s = profile_lib.service_ms(profile, letter, b_max, config.PROFILE_STAT) / 1000.0
        seconds_per_request += weights[q] * t_full_s / b_max
    return 1.0 / seconds_per_request


def queue_rates(weights, total_rate):
    return [w * total_rate for w in weights]


def build_schedule(rates, horizon_s, pool_size, seed):
    """Poisson arrivals per queue (exponential gaps), each with a random input pool index."""
    events = []
    for q, rate in enumerate(rates):
        rs = np.random.RandomState(seed * 100 + q)
        t = 0.0
        while True:
            t += rs.exponential(1.0 / rate)
            if t >= horizon_s:
                break
            events.append((t, q, int(rs.randint(pool_size))))
    events.sort()
    return events


def write_arrivals(path, schedule):
    """arrivals.csv so any run can be audited. Returns the sha1 of the file text (same hash = same job set)."""
    lines = ["job_id,t_arrival_s,queue,pool_index"]
    for job_id, (t, q, pool_index) in enumerate(schedule):
        lines.append("%d,%.6f,%d,%d" % (job_id, t, q, pool_index))
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return hashlib.sha1(text.encode("ascii")).hexdigest()

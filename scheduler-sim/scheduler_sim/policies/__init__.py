"""Policy implementations. Importing this package registers all of
them into `scheduler_sim.policy.policy_registry`."""
from . import fcfs_no_batch, fcfs_batch, longest_queue, timeout_batch  # noqa: F401

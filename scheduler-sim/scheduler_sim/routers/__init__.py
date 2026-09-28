"""Router implementations. Importing this package registers all of
them into `scheduler_sim.router.router_registry`."""
from . import uniform_random, weighted_random, trace, sticky  # noqa: F401

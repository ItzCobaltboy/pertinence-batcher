import sys
from pathlib import Path

# Make `scheduler_sim` importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scheduler_sim.job import reset_job_id_counter


@pytest.fixture(autouse=True)
def _reset_job_ids():
    reset_job_id_counter()
    yield

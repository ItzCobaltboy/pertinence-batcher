"""Padding logic for RuntimeProfile, and its pad/interpolate modes."""
import tempfile

import pytest

from scheduler_sim.runtime_profile import RuntimeProfile, UnsupportedBatchSize


def _write_profile(mode="pad", switch_cost=0.0):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write("model,1,4,8\nmodelA,5.0,6.0,9.0\nmodelB,7.0,,20.0\n")
    f.close()
    return RuntimeProfile.from_csv(f.name, partial_batch_mode=mode, switch_cost_ms=switch_cost)


def test_exact_batch_sizes_read_directly():
    profile = _write_profile()
    assert profile.runtime("modelA", 1) == 5.0
    assert profile.runtime("modelA", 4) == 6.0
    assert profile.runtime("modelA", 8) == 9.0


def test_unsupported_cell_is_empty():
    profile = _write_profile()
    assert not profile.is_supported("modelB", 4)
    assert profile.supported_batch_sizes("modelB") == [1, 8]


def test_pad_mode_rounds_up_to_next_supported_size():
    profile = _write_profile(mode="pad")
    # batch of 3 jobs -> padded to the cost of batch size 4
    assert profile.runtime("modelA", 3) == 6.0
    # batch of 5 jobs -> padded to the cost of batch size 8
    assert profile.runtime("modelA", 5) == 9.0


def test_pad_mode_raises_past_max_supported_size():
    profile = _write_profile(mode="pad")
    with pytest.raises(UnsupportedBatchSize):
        profile.runtime("modelA", 9)


def test_interpolate_mode_linear_between_bracketing_sizes():
    profile = _write_profile(mode="interpolate")
    # halfway between b=4 (6.0) and b=8 (9.0) -> b=6
    got = profile.runtime("modelA", 6)
    assert got == pytest.approx(7.5)


def test_interpolate_mode_flat_extrapolation_past_max():
    profile = _write_profile(mode="interpolate")
    assert profile.runtime("modelA", 100) == 9.0


def test_switch_cost_applied_only_on_model_change():
    profile = _write_profile(switch_cost=2.0)
    assert profile.cost(None, "modelA", 1) == 5.0
    assert profile.cost("modelA", "modelA", 1) == 5.0
    assert profile.cost("modelB", "modelA", 1) == 7.0

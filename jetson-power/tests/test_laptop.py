# Laptop checks: tegrastats parser on real lines, and energy integration on a hand-made series.
# Run from jetson-power/:  python -m pytest tests
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import tegrastats

# real tegrastats output from the AGX Xavier (as root, then without root)
LINE_ROOT = ("10-23-2026 18:33:47 RAM 1766/31003MB (lfb 7001x4MB) SWAP 0/15502MB (cached 0MB) "
             "CPU [1%@1180,0%@1180,1%@1178,0%@1177,off,off,off,off] EMC_FREQ 0%@800 GR3D_FREQ 0%@318 "
             "VIC_FREQ 1036 APE 150 AUX@26.5C CPU@27.5C thermal@27.3C Tboard@29C AO@27.5C GPU@27.5C "
             "Tdiode@31.75C PMIC@50C GPU 0mW/0mW CPU 466mW/466mW SOC 933mW/933mW CV 0mW/0mW "
             "VDDRQ 310mW/310mW SYS5V 2363mW/2363mW")
LINE_NO_ROOT = LINE_ROOT.split(" GPU 0mW/0mW")[0]
RAILS = ["GPU", "CPU", "SOC", "CV", "VDDRQ", "SYS5V"]


def test_rails_parsed_with_current_and_average():
    p = tegrastats.parse_line(LINE_ROOT)
    assert p["rails"] == {"GPU": (0, 0), "CPU": (466, 466), "SOC": (933, 933), "CV": (0, 0),
                          "VDDRQ": (310, 310), "SYS5V": (2363, 2363)}


def test_other_fields_and_unknown_tokens():
    p = tegrastats.parse_line(LINE_ROOT)
    assert p["gr3d_load"] == 0 and p["gr3d_mhz"] == 318
    assert p["emc_mhz"] == 800
    assert p["cpu"][0] == "1%@1180" and p["cpu"][4] == "off" and len(p["cpu"]) == 8
    assert p["temps"]["thermal"] == 27.3 and p["temps"]["GPU"] == 27.5 and p["temps"]["PMIC"] == 50.0
    assert p["ram_used_mb"] == 1766
    assert "VIC_FREQ 1036" in p["raw"]          # the raw line is kept, unknown tokens do not crash


def test_totals():
    pw = tegrastats.powers(tegrastats.parse_line(LINE_ROOT), RAILS)
    assert pw["total"] == 466 + 933 + 310 + 2363
    assert pw["total_excl_sys5v"] == 466 + 933 + 310
    assert pw["gpu"] == 0


def test_no_root_line_has_no_rails():
    p = tegrastats.parse_line(LINE_NO_ROOT)
    assert p["rails"] == {}
    assert tegrastats.powers(p, RAILS) is None
    assert p["temps"]["thermal"] == 27.3        # the rest still parses


def test_energy_constant_power():
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    mw = [1000.0] * 5
    assert abs(tegrastats.energy_mj(t, mw, 1.0, 3.0) - 2000.0) < 1e-9


def test_energy_ramp_with_interpolated_edges():
    # power = 1000 * t mW, so the integral from 0.5 to 2.5 is 500 * (2.5^2 - 0.5^2) = 3000 mJ
    t = [0.0, 1.0, 2.0, 3.0]
    mw = [0.0, 1000.0, 2000.0, 3000.0]
    assert abs(tegrastats.energy_mj(t, mw, 0.5, 2.5) - 3000.0) < 1e-9


def test_energy_ignores_samples_outside_window():
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    mw = [9999.0, 1000.0, 1000.0, 1000.0, 9999.0]
    assert abs(tegrastats.energy_mj(t, mw, 1.0, 3.0) - 2000.0) < 1e-9


def test_mean_power_is_time_weighted():
    samples = [(0.0, None, {"total": 0.0}), (2.0, None, {"total": 2000.0})]
    assert abs(tegrastats.mean_power(samples, "total", 0.0, 2.0) - 1000.0) < 1e-9

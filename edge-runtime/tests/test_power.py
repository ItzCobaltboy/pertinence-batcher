import math
import time

import numpy as np

import power

# Written from NVIDIA's documented tegrastats format (Jetson Linux developer guide).
ORIN_NANO = ("10-21-2026 10:00:00 RAM 2345/7620MB (lfb 2x4MB) SWAP 0/3810MB (cached 0MB) "
             "CPU [12%@1510,5%@1510,3%@1510,4%@1510,off,off] EMC_FREQ 0%@2133 GR3D_FREQ 37%@[624] "
             "NVENC off NVDEC off NVJPG off NVJPG1 off VIC off OFA off APE 200 cpu@48.5C soc2@47.2C "
             "soc0@46.9C gpu@47.5C tj@48.5C soc1@46.8C VDD_IN 7140mW/5200mW "
             "VDD_CPU_GPU_CV 2100mW/1500mW VDD_SOC 1435mW/1435mW")
AGX_ORIN = ("RAM 3049/30536MB (lfb 6527x4MB) SWAP 0/15268MB (cached 0MB) CPU [1%@729,0%@729,2%@729,"
            "0%@729,off,off,off,off,off,off,off,off] EMC_FREQ 0%@2133 GR3D_FREQ 0%@[0,0] VIC_FREQ 729 "
            "APE 174 CV0@-256C CPU@45.75C SOC2@42.343C SOC0@43.125C CV1@-256C GPU@-256C tj@45.75C "
            "SOC1@42.75C CV2@-256C VDD_GPU_SOC 2391mW/2391mW VDD_CPU_CV 398mW/398mW "
            "VIN_SYS_5V0 2523mW/2523mW VDDQ_VDD2_1V8AO 302mW/302mW")
OLD_NANO = ("RAM 1500/3964MB (lfb 100x4MB) CPU [5%@1479,2%@1479,1%@1479,0%@1479] EMC_FREQ 3%@1600 "
            "GR3D_FREQ 0%@76 PLL@31C CPU@33.5C GPU@31.5C thermal@32.25C POM_5V_IN 1820/1820 "
            "POM_5V_GPU 0/0 POM_5V_CPU 286/286")


def test_orin_nano_line():
    row = power.parse_tegrastats_line(ORIN_NANO)
    assert row["power_source"] == "VDD_IN"
    assert row["power_w"] == 7.14
    assert row["rail_VDD_CPU_GPU_CV_w"] == 2.1
    assert row["rail_VDD_SOC_w"] == 1.435
    assert row["gpu_util_pct"] == 37 and row["gpu_freq_mhz"] == 624
    assert row["emc_freq_mhz"] == 2133
    assert abs(row["cpu_load_pct"] - 6.0) < 1e-9 and row["cpu_cores_on"] == 4
    assert row["temp_gpu_c"] == 47.5 and row["temp_c"] == 47.5
    assert row["ram_used_mb"] == 2345
    assert "NVENC" in row["unknown"]          # unknown tokens kept as text


def test_agx_orin_line_sums_rails_without_vdd_in():
    row = power.parse_tegrastats_line(AGX_ORIN)
    assert row["power_source"].startswith("sum:")
    assert abs(row["power_w"] - (2.391 + 0.398 + 2.523 + 0.302)) < 1e-9
    assert row["gpu_freq_mhz"] == 0
    assert "temp_gpu_c" not in row            # -256C means the sensor is off
    assert row["temp_c"] == 45.75             # falls back to tj
    assert row["cpu_cores_on"] == 4


def test_old_format_without_units():
    row = power.parse_tegrastats_line(OLD_NANO)
    assert row["power_source"] == "POM_5V_IN"
    assert row["power_w"] == 1.82
    assert row["rail_POM_5V_GPU_w"] == 0.0
    assert row["gpu_freq_mhz"] == 76


def test_missing_rails_and_garbage():
    row = power.parse_tegrastats_line("RAM 1/2MB GR3D_FREQ 5%@100 weird_token ??? 12/34")
    assert row["power_source"] == "none" and math.isnan(row["power_w"])
    assert "weird_token" in row["unknown"]
    assert power.parse_tegrastats_line("")["power_source"] == "none"


def test_energy_integration_matches_closed_form(tmp_path):
    t0 = time.time()
    # power = 5 + 2 sin(2 pi (t - t0) / 1.5): known integral
    fn = lambda t: 5.0 + 2.0 * math.sin(2 * math.pi * (t - t0) / 1.5)
    sampler = power.FakeSampler(fn, sample_ms=20)
    path = str(tmp_path / "power.csv")
    sampler.start(path)
    time.sleep(2.2)
    sampler.stop()
    df = power.read(path)
    a, b = t0 + 0.3, t0 + 1.9
    w = 2 * math.pi / 1.5
    exact = 5.0 * (b - a) - 2.0 / w * (math.cos(w * (b - t0)) - math.cos(w * (a - t0)))
    got = power.energy_j(df["t"], df["power_w"], a, b)
    assert abs(got - exact) / exact < 0.01


def test_energy_constant_and_edges():
    t = np.arange(0, 10, 0.05)
    p = np.full_like(t, 4.0)
    assert abs(power.energy_j(t, p, 1.03, 3.51) - 4.0 * 2.48) < 1e-9
    assert math.isnan(power.energy_j(t, p, 20, 30))

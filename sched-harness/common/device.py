# Device state for a session: optionally set the nvpmodel mode, lock clocks, and always put both back.
# Used by 01_calibrate.py, 02_run.py (standalone) and 03_plan.py (once for the whole plan).
import os
import time

import config
from common import powermode


def resolve_mode(wanted):
    """Mode name or id -> id, checked against /etc/nvpmodel.conf."""
    modes = powermode.list_modes(config.NVPMODEL_CONF)
    for mode_id, name in modes:
        if wanted == mode_id or wanted == name:
            return mode_id
    raise RuntimeError("SESSION_MODE %r is not in %s: %s" % (wanted, config.NVPMODEL_CONF, modes))


def mode_name(mode_id):
    for i, name in powermode.list_modes(config.NVPMODEL_CONF):
        if i == mode_id:
            return name
    return "?"


def start_session():
    """Returns a state dict for end_session. Needs root."""
    if os.geteuid() != 0:
        raise RuntimeError("run with sudo: tegrastats shows no power rails without root, and nvpmodel needs it")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    state = {"original_mode": powermode.current_mode(), "changed_mode": False, "locked": False}
    # An existing backup means an earlier session died while clocks were locked; keep it, it holds the real original.
    state["clocks_backup"] = os.path.join(config.RESULTS_DIR, "jetson_clocks_original.conf")
    if not os.path.exists(state["clocks_backup"]):
        powermode.store_clocks(state["clocks_backup"])
    if config.SESSION_MODE is not None:
        wanted = resolve_mode(config.SESSION_MODE)
        if wanted != state["original_mode"]:
            powermode.set_mode(wanted)
            state["changed_mode"] = True
            time.sleep(config.MODE_SETTLE_S)
    if config.LOCK_CLOCKS:
        powermode.lock_clocks()
        state["locked"] = True
    state["mode"] = powermode.current_mode()
    print("power mode: %d %s | clocks %s" % (state["mode"], mode_name(state["mode"]),
                                               "locked" if state["locked"] else "not locked (DVFS)"))
    return state


def end_session(state):
    """Mode first (switching modes resets clocks), then the saved clock state on top."""
    print("restoring power mode %d and clocks" % state["original_mode"])
    if state["changed_mode"]:
        try:
            powermode.set_mode(state["original_mode"])
        except RuntimeError as e:
            print("COULD NOT RESTORE MODE:", e, "-> run: sudo nvpmodel -m %d" % state["original_mode"])
    try:
        powermode.restore_clocks(state["clocks_backup"])
        os.remove(state["clocks_backup"])
    except RuntimeError as e:
        print("COULD NOT RESTORE CLOCKS:", e, "-> run: sudo jetson_clocks --restore " + state["clocks_backup"])
    powermode.chown_to_invoker(config.RESULTS_DIR)
    powermode.chown_to_invoker(config.MODELS_DIR)


def describe(state):
    return {"mode_id": state["mode"], "mode_name": mode_name(state["mode"]),
            "original_mode": state["original_mode"], "clocks_locked": state["locked"]}

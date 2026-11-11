# Jetson power modes and clocks: list them from nvpmodel.conf, read/set the mode,
# lock clocks with jetson_clocks, and save/restore the original state.
# Needs root for anything that changes state.
import os
import re
import subprocess


def run(cmd):
    """Run a command, return (exit code, stdout+stderr text)."""
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def list_modes(conf_path):
    """Parse '< POWER_MODEL ID=7 NAME=MODE_15W_DESKTOP >' lines -> [(id, name)]."""
    modes = []
    with open(conf_path) as f:
        for line in f:
            m = re.match(r"\s*<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", line)
            if m:
                modes.append((int(m.group(1)), m.group(2)))
    return modes


def current_mode():
    """`nvpmodel -q` prints 'NV Power Mode: NAME' then the id on the last line."""
    code, text = run(["nvpmodel", "-q"])
    if code != 0:
        raise RuntimeError("nvpmodel -q failed: " + text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines[-1].isdigit():
        raise RuntimeError("could not read the mode id from nvpmodel -q output:\n" + text)
    return int(lines[-1])


def set_mode(mode_id):
    code, text = run(["nvpmodel", "-m", str(mode_id)])
    if code != 0:
        raise RuntimeError("nvpmodel -m %d failed: %s" % (mode_id, text.strip()))
    if current_mode() != mode_id:
        raise RuntimeError("nvpmodel -m %d returned 0 but the mode is %d" % (mode_id, current_mode()))


def store_clocks(path):
    code, text = run(["jetson_clocks", "--store", path])
    if code != 0:
        raise RuntimeError("jetson_clocks --store failed: " + text.strip())


def restore_clocks(path):
    code, text = run(["jetson_clocks", "--restore", path])
    if code != 0:
        raise RuntimeError("jetson_clocks --restore failed: " + text.strip())


def lock_clocks():
    code, text = run(["jetson_clocks"])
    if code != 0:
        raise RuntimeError("jetson_clocks failed: " + text.strip())


def show_clocks():
    return run(["jetson_clocks", "--show"])[1]


def chown_to_invoker(path):
    """After running under sudo, give files back to the user who called sudo."""
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None or not os.path.exists(path):
        return
    os.chown(path, int(uid), int(gid))
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                os.chown(os.path.join(root, name), int(uid), int(gid))

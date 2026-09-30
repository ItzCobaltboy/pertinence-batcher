"""
Raw file naming, atomic writes and the resume index. No torch import.

Layout under results/batch_sweep/raw/<session>/:
    image_pool.json                         the sampled image list, fixed per session
    compile/<variant>__<model>__bs<b>__<timestamp>__<status>.json
    runs/<variant>/<model>/bs<b>__run<k>__<timestamp>__<status>.json

status is one of ok / failed / skipped. Files are only ever added, never
overwritten: every attempt gets its own timestamp, so a failed attempt stays
on disk next to the later successful one.
"""

import glob
import json
import os
import time
from datetime import datetime, timezone

import settings

STATUSES = ("ok", "failed", "skipped")


def timestamp():
    # microseconds keep two attempts in the same second apart
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def session_dir(session):
    return os.path.join(settings.RAW_DIR, session)


def pool_manifest_path(session):
    return os.path.join(session_dir(session), "image_pool.json")


def run_dir(session, variant, model):
    return os.path.join(session_dir(session), "runs", variant, settings.model_stem(model))


def run_base(session, variant, model, batch_size, run_index, ts=None):
    """Path prefix for one attempt; the worker appends __<status>.json."""
    ts = ts or timestamp()
    return os.path.join(run_dir(session, variant, model), f"bs{batch_size}__run{run_index}__{ts}")


def compile_base(session, variant, model, batch_size, ts=None):
    ts = ts or timestamp()
    return os.path.join(session_dir(session), "compile",
                        f"{variant}__{settings.model_stem(model)}__bs{batch_size}__{ts}")


def write_json_atomic(path, data):
    """Write to a temp file, then rename. A job killed mid-write never leaves a
    half file that the resume logic would mistake for a finished run."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def status_of(path):
    name = os.path.basename(path)
    if not name.endswith(".json"):
        return None
    status = name[:-len(".json")].rsplit("__", 1)[-1]
    return status if status in STATUSES else None


def attempt_files(base_without_ts_glob):
    return sorted(p for p in glob.glob(base_without_ts_glob + "__*.json") if status_of(p) is not None)


def run_attempts(session, variant, model, batch_size, run_index):
    pattern = os.path.join(run_dir(session, variant, model), f"bs{batch_size}__run{run_index}__*")
    return attempt_files(pattern)


def compile_attempts(session, variant, model, batch_size):
    pattern = os.path.join(session_dir(session), "compile",
                           f"{variant}__{settings.model_stem(model)}__bs{batch_size}__*")
    return attempt_files(pattern)


def job_state(attempts, max_attempts):
    """'done' if any attempt is ok, 'gave_up' after max_attempts failures, else 'todo'."""
    statuses = [status_of(p) for p in attempts]
    if "ok" in statuses:
        return "done"
    if statuses.count("failed") >= max_attempts:
        return "gave_up"
    return "todo"


def all_run_records(session=None):
    """Every run record (any status) for one session, or every session."""
    sessions = [session] if session else sorted(os.listdir(settings.RAW_DIR)) if os.path.isdir(settings.RAW_DIR) else []
    paths = []
    for s in sessions:
        paths += glob.glob(os.path.join(session_dir(s), "runs", "*", "*", "*.json"))
    return sorted(p for p in paths if status_of(p) is not None)


def all_compile_records(session=None):
    sessions = [session] if session else sorted(os.listdir(settings.RAW_DIR)) if os.path.isdir(settings.RAW_DIR) else []
    paths = []
    for s in sessions:
        paths += glob.glob(os.path.join(session_dir(s), "compile", "*.json"))
    return sorted(p for p in paths if status_of(p) is not None)


# ---------------------------------------------------------------------------
# Engine cache (compiled TRT modules). Outside raw/: large, git-ignored,
# rebuildable, and never copied back.
# ---------------------------------------------------------------------------
def engine_paths(engine_dir, variant, model, batch_size, img_size):
    stem = f"{settings.model_stem(model)}_bs{batch_size}_img{img_size}"
    folder = os.path.join(engine_dir, variant)
    return os.path.join(folder, stem + ".ep"), os.path.join(folder, stem + ".json")


def engine_ready(engine_dir, variant, model, batch_size, img_size):
    engine, sidecar = engine_paths(engine_dir, variant, model, batch_size, img_size)
    return os.path.exists(engine) and os.path.exists(sidecar)


class Log:
    """print() that also appends to the session log file (tmux scrollback is not enough)."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

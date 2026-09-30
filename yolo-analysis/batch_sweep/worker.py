"""
One job per process. sweep.py starts `python sweep.py _worker '<job json>'`
for every compile and every timed run, so:
  - a crash that Python cannot catch (a C++ terminate() inside TensorRT, a
    segfault, the OOM killer) only kills this job, and sweep.py records it;
  - every run starts from a fresh process, fresh CUDA context, fresh model
    load and its own warmup, so the 5 runs of a config are independent.

The worker writes exactly one record: <base>__ok.json, <base>__failed.json
or <base>__skipped.json. If it dies before writing, sweep.py writes the
failed record itself.
"""

import json
import os
import platform
import socket
import sys
import time
import traceback
import zlib

import numpy as np

import gpu_log
import images
import models
import settings
import store


# ---------------------------------------------------------------------------
# Test hooks (CPU tests only). BATCH_SWEEP_TEST_FAIL / BATCH_SWEEP_TEST_CRASH =
# "<variant>:<model stem>:<batch size>" makes that job raise / abort().
# ---------------------------------------------------------------------------
def _test_hooks(job):
    key = f"{job['variant']}:{settings.model_stem(job['model'])}:{job['batch_size']}"
    if os.environ.get("BATCH_SWEEP_TEST_CRASH") == key:
        os.abort()
    if os.environ.get("BATCH_SWEEP_TEST_FAIL") == key:
        raise RuntimeError(f"injected test failure for {key}")


def versions():
    import torch
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
    }
    for name in ("torchvision", "ultralytics", "torch_tensorrt", "tensorrt", "cv2"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception as e:  # torch_tensorrt is absent on CPU test boxes
            info[name] = f"unavailable ({type(e).__name__})"
    return info


def device_info(device):
    import torch
    info = {"device": device, "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(device))
        info.update({"gpu_name": props.name, "compute_capability": f"{props.major}.{props.minor}",
                     "gpu_total_memory_mib": props.total_memory // (1024 * 1024)})
    info["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    info["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    return info


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
            "n": int(a.size)}


def rng_key(job):
    """Stable per-run seed: different runs get different images, reruns the same."""
    return [job["seed"], zlib.crc32(job["variant"].encode()),
            zlib.crc32(settings.model_stem(job["model"]).encode()), job["batch_size"], job["run_index"]]


def apply_torch_flags(job):
    import torch
    if job.get("no_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def base_record(job, kind):
    return {
        "kind": kind,
        "session": job["session"],
        "variant": job["variant"],
        "model": settings.model_stem(job["model"]),
        "model_file": job["model"],
        "batch_size": job["batch_size"],
        "run_index": job.get("run_index"),
        "attempt_timestamp": os.path.basename(job["base"]).rsplit("__", 1)[-1],
        "started_at": store.now_iso(),
        "img_size": job["img_size"],
        "precision": {"eager_fp32": "fp32 (torch defaults)", "trt_fp32": "fp32 (TensorRT)",
                      "trt_fp16": "fp16 enabled (TensorRT)"}[job["variant"]],
        "forward_definition": "ForwardOnly(DetectionModel), fused, decoded (B,84,8400) output, NMS excluded",
        "git_commit": job.get("git_commit"),
        "pid": os.getpid(),
        "argv": sys.argv[:2],
        "job": {k: v for k, v in job.items() if k != "base"},
    }


def finish(job, record, status):
    record["status"] = status
    record["finished_at"] = store.now_iso()
    store.write_json_atomic(job["base"] + f"__{status}.json", record)


# ---------------------------------------------------------------------------
# Probe: what does this machine have? (used by preflight and by sweep.py)
# ---------------------------------------------------------------------------
def probe(_job):
    import torch
    out = {"cuda_available": torch.cuda.is_available(), "versions": versions()}
    if out["cuda_available"]:
        out["device"] = device_info("cuda:0")
    try:
        import torch_tensorrt  # noqa: F401
        out["torch_tensorrt_importable"] = True
    except Exception as e:
        out["torch_tensorrt_importable"] = False
        out["torch_tensorrt_error"] = repr(e)
    print("PROBE_JSON " + json.dumps(out), flush=True)


# ---------------------------------------------------------------------------
# Compile: build one TRT engine for (variant, model, batch size), cache it.
# ---------------------------------------------------------------------------
def compile_job(job):
    import torch

    record = base_record(job, "compile")
    record["gpu_start"] = gpu_log.snapshot()
    try:
        _test_hooks(job)
        apply_torch_flags(job)
        device = job["device"]
        record["versions"] = versions()
        record["device_info"] = device_info(device)

        manifest = store.read_json(store.pool_manifest_path(job["session"]))
        batch_idx = images.batches_for_run(manifest, job["batch_size"], rng_key(job), 1)
        example = images.device_batches(manifest, batch_idx, job["img_size"], device, job["dataset_root"])[0]

        module, load_info = models.load_eager(job["model"], device)
        record["model_info"] = load_info
        models.prime_head(module, example)
        with torch.no_grad():
            reference = module(example).float()

        compiled, compile_s = models.compile_trt(module, example, job["variant"])
        record["compile_s"] = compile_s
        record["subgraphs"] = models.subgraph_report(compiled)

        with torch.no_grad():
            trt_out = compiled(example)
            if isinstance(trt_out, (tuple, list)):
                trt_out = trt_out[0]
            trt_out = trt_out.float()
        diff = (trt_out - reference).abs()
        record["check_vs_eager"] = {
            "output_shape": list(trt_out.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "reference_abs_max": float(reference.abs().max()),
            "all_finite": bool(torch.isfinite(trt_out).all()),
        }
        nms = models.nms_function()
        ref_dets = [int(d.shape[0]) for d in models.run_nms(nms, reference)]
        trt_dets = [int(d.shape[0]) for d in models.run_nms(nms, trt_out)]
        record["check_vs_eager"]["detections_eager"] = ref_dets
        record["check_vs_eager"]["detections_trt"] = trt_dets

        engine_path, sidecar_path = store.engine_paths(job["engine_dir"], job["variant"], job["model"],
                                                       job["batch_size"], job["img_size"])
        t0 = time.perf_counter()
        models.save_trt(compiled, engine_path, example)
        record["save_s"] = time.perf_counter() - t0
        record["engine_path"] = engine_path
        record["engine_size_mib"] = os.path.getsize(engine_path) / (1024 * 1024)
        record["gpu_end"] = gpu_log.snapshot()
        sidecar = {k: record[k] for k in ("variant", "model", "batch_size", "img_size", "compile_s", "save_s",
                                          "subgraphs", "check_vs_eager", "engine_size_mib", "versions",
                                          "model_info")}
        sidecar["compiled_at"] = store.now_iso()
        sidecar["compile_record"] = job["base"] + "__ok.json"
        store.write_json_atomic(sidecar_path, sidecar)
        finish(job, record, "ok")
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        record["gpu_end"] = gpu_log.snapshot()
        finish(job, record, "failed")
        raise


# ---------------------------------------------------------------------------
# Timed run: warmup, then per-batch synchronized forward timings.
# ---------------------------------------------------------------------------
def run_job(job):
    import torch

    record = base_record(job, "run")
    record["gpu_start"] = gpu_log.snapshot()
    try:
        _test_hooks(job)
        apply_torch_flags(job)
        device = job["device"]
        on_cuda = device.startswith("cuda")
        record["versions"] = versions()
        record["device_info"] = device_info(device)

        def sync():
            if on_cuda:
                torch.cuda.synchronize()

        # 1. Data: pick, load, letterbox, copy to device. All before timing.
        manifest = store.read_json(store.pool_manifest_path(job["session"]))
        batch_idx = images.batches_for_run(manifest, job["batch_size"], rng_key(job), job["max_batches"])
        t0 = time.perf_counter()
        batches = images.device_batches(manifest, batch_idx, job["img_size"], device, job["dataset_root"])
        sync()
        record["data"] = {"n_distinct_batches": len(batches), "load_s": time.perf_counter() - t0,
                          "images": [[manifest["images"][i]["split"] + "/" + manifest["images"][i]["file"]
                                      for i in b] for b in batch_idx]}

        # 2. Model: fresh eager load, or fresh load of the cached TRT engine.
        if job["variant"] == "eager_fp32":
            module, load_info = models.load_eager(job["model"], device)
            models.prime_head(module, batches[0])
            record["model_info"] = load_info
            record["engine"] = None
        else:
            engine_path, sidecar_path = store.engine_paths(job["engine_dir"], job["variant"], job["model"],
                                                           job["batch_size"], job["img_size"])
            module, load_s = models.load_trt(engine_path)
            sidecar = store.read_json(sidecar_path)
            record["model_info"] = sidecar.get("model_info")
            record["engine"] = {"path": engine_path, "load_s": load_s, "compile_s": sidecar.get("compile_s"),
                                "subgraphs": sidecar.get("subgraphs"),
                                "check_vs_eager": sidecar.get("check_vs_eager"),
                                "engine_size_mib": sidecar.get("engine_size_mib"),
                                "compiled_at": sidecar.get("compiled_at")}

        nms = models.nms_function() if job["nms"] else None
        n = len(batches)
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            # 3. Warmup, untimed.
            t0 = time.perf_counter()
            for i in range(job["warmup"]):
                out = module(batches[i % n])
            sync()
            record["warmup_s"] = time.perf_counter() - t0

            # 4. Timed: synchronize before and after every forward (benchmark.py convention,
            #    per batch). Forward+NMS is its own column, measured from the same t0.
            forward_ms, forward_nms_ms, detections = [], [], []
            for i in range(job["repeats"]):
                x = batches[i % n]
                sync()
                t0 = time.perf_counter()
                out = module(x)
                sync()
                t1 = time.perf_counter()
                forward_ms.append((t1 - t0) * 1000.0)
                if nms is not None:
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    dets = models.run_nms(nms, out)
                    sync()
                    t2 = time.perf_counter()
                    forward_nms_ms.append((t2 - t0) * 1000.0)
                    detections.append(float(np.mean([d.shape[0] for d in dets])))

            # 5. Loop-average, the older src/benchmark.py style: one sync around
            #    `repeats` back-to-back forwards. Extra column only.
            sync()
            t0 = time.perf_counter()
            for i in range(job["repeats"]):
                out = module(batches[i % n])
            sync()
            loop_mean_ms = (time.perf_counter() - t0) / job["repeats"] * 1000.0

        if isinstance(out, (tuple, list)):
            out = out[0]
        record["gpu_end"] = gpu_log.snapshot()
        record["output"] = {"shape": list(out.shape), "dtype": str(out.dtype),
                            "all_finite": bool(torch.isfinite(out).all())}
        record["peak_memory_mib"] = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if on_cuda else None
        record["other_gpu_processes_start"] = gpu_log.other_processes(record["gpu_start"])
        record["other_gpu_processes_end"] = gpu_log.other_processes(record["gpu_end"])
        record["timings"] = {
            "forward_ms": forward_ms,
            "forward_nms_ms": forward_nms_ms,
            "mean_detections_per_image": detections,
            "loop_mean_ms": loop_mean_ms,
        }
        record["summary"] = {"forward_ms": stats(forward_ms), "forward_nms_ms": stats(forward_nms_ms),
                             "loop_mean_ms": loop_mean_ms}
        finish(job, record, "ok")
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        record["gpu_end"] = gpu_log.snapshot()
        finish(job, record, "failed")
        raise


def main(job_json):
    job = json.loads(job_json)
    {"probe": probe, "compile": compile_job, "run": run_job}[job["kind"]](job)

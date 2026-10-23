# The four YOLOv8 models, run eagerly in FP32 (no TensorRT, see CLAUDE.md "TRT
# engines broken"). Weights are found and loaded with yolo-dispatcher's own lookup
# (WEIGHT_SEARCH_DIRS, yolo_backbone.load_detection_model: fused, eval).
#
# Runner.run_batch(model, payloads) stacks the per-job tensors into one batch,
# synchronizes, runs the forward, synchronizes, runs NMS, synchronizes, and returns
# the three times. FakeRunner does the same interface by sleeping T(b) from a
# profile, so the runtime can be tested on a CPU without torch or weights.
# torch is imported lazily: FakeRunner needs none of it.

import importlib.util
import os
import random
import sys
import time

import config

_YD = None


def yolo_dispatcher():
    """yolo-dispatcher/config.py, imported under another name (ours is config.py too).
    Its folder is appended to sys.path for its letterbox / yolo_backbone imports."""
    global _YD
    if _YD is None:
        if config.YOLO_DISPATCHER_DIR not in sys.path:
            sys.path.append(config.YOLO_DISPATCHER_DIR)
        path = os.path.join(config.YOLO_DISPATCHER_DIR, "config.py")
        spec = importlib.util.spec_from_file_location("yolo_dispatcher_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _YD = module
    return _YD


def find_weights(model_name):
    """Path of <model>.pt in yolo-dispatcher's search dirs, or None."""
    yd = yolo_dispatcher()
    for d in yd.WEIGHT_SEARCH_DIRS:
        path = os.path.join(d, model_name + ".pt")
        if os.path.exists(path):
            return path
    return None


def load_model(model_name, device):
    yd = yolo_dispatcher()
    from yolo_backbone import load_detection_model
    net = load_detection_model(model_name, yd.WEIGHT_SEARCH_DIRS, yd.MODELS_DIR)
    return net.to(device).eval()


def nms_function():
    try:
        from ultralytics.utils.ops import non_max_suppression
    except ImportError:
        from ultralytics.utils.nms import non_max_suppression
    return non_max_suppression


class Runner:
    def __init__(self, models, nms=True, device=None, log=print):
        import torch
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda = self.device.startswith("cuda")
        self.do_nms = nms
        self.nms_fn = nms_function() if nms else None
        self.nets = {}
        self.load_info = {}
        for m in models:
            before = torch.cuda.memory_allocated() if self.cuda else 0
            t0 = time.time()
            self.nets[m] = load_model(m, self.device)
            self.sync()
            self.load_info[m] = {"load_s": time.time() - t0,
                                 "weights_mb": ((torch.cuda.memory_allocated() - before) / 2 ** 20) if self.cuda else None}
            log("loaded %s on %s in %.1f s" % (m, self.device, self.load_info[m]["load_s"]))

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()

    def forward(self, model, batch):
        """(raw output, ms) with a sync before and after."""
        with self.torch.no_grad():
            self.sync()
            t0 = time.perf_counter()
            out = self.nets[model](batch)
            self.sync()
            return out, (time.perf_counter() - t0) * 1000.0

    def nms(self, out):
        if not self.do_nms:
            return 0.0
        t0 = time.perf_counter()
        self.nms_fn(out, conf_thres=config.NMS_CONF, iou_thres=config.NMS_IOU)
        self.sync()
        return (time.perf_counter() - t0) * 1000.0

    def run_batch(self, model, payloads):
        t0 = time.perf_counter()
        batch = self.torch.stack(payloads) if len(payloads) > 1 else payloads[0].unsqueeze(0)
        self.sync()
        assemble_ms = (time.perf_counter() - t0) * 1000.0
        out, forward_ms = self.forward(model, batch)
        nms_ms = self.nms(out)
        return {"assemble_ms": assemble_ms, "forward_ms": forward_ms, "nms_ms": nms_ms}

    def warmup(self, inputs, batch_sizes, b_max, repeats=1):
        """A forward per (model, profiled size) so the lazy CUDA/cuDNN setup for every
        shape is done before the trace starts."""
        for m in self.nets:
            for b in batch_sizes:
                if b > b_max:
                    continue
                batch = inputs.sample_batch(b)
                for _ in range(repeats):
                    out, _ = self.forward(m, batch)
                    self.nms(out)

    def info(self):
        torch = self.torch
        info = {"device": self.device, "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if self.cuda else None,
                "load": self.load_info}
        if self.cuda:
            props = torch.cuda.get_device_properties(0)
            info.update({"gpu_name": props.name, "gpu_total_mb": props.total_memory / 2 ** 20,
                         "max_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20})
        return info


class FakeRunner:
    """Sleeps the profiled T(b) (plus optional relative jitter) instead of running a model."""

    def __init__(self, profile, jitter=0.0, seed=0):
        self.profile = profile
        self.jitter = jitter
        self.rng = random.Random(seed)
        self.device = "cpu"

    def run_batch(self, model, payloads):
        ms = self.profile.t(model, len(payloads))
        if self.jitter:
            ms *= 1.0 + self.jitter * self.rng.uniform(-1.0, 1.0)
        t0 = time.perf_counter()
        time.sleep(ms / 1000.0)
        return {"assemble_ms": 0.0, "forward_ms": (time.perf_counter() - t0) * 1000.0, "nms_ms": 0.0}

    def warmup(self, inputs, batch_sizes, b_max):
        pass

    def info(self):
        return {"device": "fake", "jitter": self.jitter}

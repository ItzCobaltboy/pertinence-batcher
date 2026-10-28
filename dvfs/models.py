# Copied from sched-harness/common/models.py; only the "import config" line changed (-> dvfs.settings).
# Load the four YOLOv8 nets (eager FP16, fused) and run one batch through the whole serving pipeline:
# stack + H2D + convert, forward, NMS. Every stage is CUDA-synchronised so its time is real.
import os
import time

import torch

from dvfs import settings as config


def load_net(letter):
    """Stock COCO weights -> fused FP16 eval network on the GPU (as jetson-power bench.load_net)."""
    from ultralytics import YOLO
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, config.MODELS[letter])
    net = YOLO(path).model      # downloads into models/ when missing
    net = net.float().fuse()
    net = net.half().eval().cuda()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


NMS_BACKEND = "not checked"


def nms_plain_torch(boxes, scores, iou_threshold):
    """Greedy NMS with plain torch ops (no torchvision C++ ops). boxes are (n, 4) xyxy. Returns the indices of
    the kept boxes, highest score first, like torchvision.ops.nms. Pairwise IoU on the boxes' device; the greedy
    pass runs on the CPU over the (small) boolean matrix."""
    n = boxes.shape[0]
    order = scores.argsort(descending=True)
    if n == 0:
        return order
    b = boxes[order].float()
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    inter_w = (torch.min(x2[:, None], x2[None, :]) - torch.max(x1[:, None], x1[None, :])).clamp(min=0)
    inter_h = (torch.min(y2[:, None], y2[None, :]) - torch.max(y1[:, None], y1[None, :])).clamp(min=0)
    inter = inter_w * inter_h
    iou = inter / (areas[:, None] + areas[None, :] - inter)
    overlaps = (iou > iou_threshold).cpu().numpy()
    removed = [False] * n
    keep = []
    for i in range(n):
        if removed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, n):
            if overlaps[i, j]:
                removed[j] = True
    return order[torch.tensor(keep, dtype=torch.long, device=order.device)]


def setup_nms():
    """ultralytics NMS calls torchvision.ops.nms on CUDA tensors. On the board torchvision is a PyPI wheel that
    does not match NVIDIA's torch, so its C++ ops do not load ("Couldn't load custom C++ ops"). Probe once; if
    the real nms fails for any reason, replace it by nms_plain_torch. The text is saved in run.json."""
    global NMS_BACKEND
    import torchvision
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    try:
        torchvision.ops.nms(boxes, scores, 0.5)
        NMS_BACKEND = "torchvision nms"
        return
    except Exception as e:
        reason = str(e).split(".")[0]
    torchvision.ops.nms = nms_plain_torch
    NMS_BACKEND = "plain torch nms (torchvision ops unusable: %s)" % reason
    print("NOTICE: " + NMS_BACKEND)


def load_nets():
    """All four stay resident for the whole run. Returns {letter: net}."""
    setup_nms()
    return {letter: load_net(letter) for letter in config.QUEUES}


def nms(out):
    from ultralytics.utils import ops
    return ops.non_max_suppression(out, conf_thres=config.NMS_CONF, iou_thres=config.NMS_IOU)


def run_batch(net, buffer, arrays):
    """Returns (h2d_ms, forward_ms, nms_ms, n_boxes). The detections are discarded after timing."""
    t0 = time.perf_counter()
    x = buffer.to_gpu(arrays)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = net(x)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    detections = nms(out)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    n_boxes = sum(len(d) for d in detections)
    return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, n_boxes


def warm_up(nets, buffers, pool):
    """Run every model at every warmup batch size twice so the first real batch is not a cold one."""
    for letter in config.QUEUES:
        for b in config.WARMUP_BATCH_SIZES:
            for _ in range(2):
                run_batch(nets[letter], buffers[letter], pool[:b])
    torch.cuda.synchronize()


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()

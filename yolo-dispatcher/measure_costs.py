"""
Measures every cost the fitness function needs, in GFLOPs at 640x640 (MACs
doubled, the paper's THOP convention), and writes config.MODEL_COSTS_JSON:

  - model_gflops:      each fused YOLOv8 detection model (full forward)
  - extractor_gflops:  the yolov8n backbone embedder (layers 0-9)
  - fc_gflops:         the Linear(EMBEDDING_DIM -> NUM_CLASSES) head

Pool numbers are cross-checked against ultralytics' published ones.

Run: python measure_costs.py   (CPU is enough; downloads weights if missing)
"""

import json
import os

import config
from yolo_backbone import BackboneEmbedder, count_macs, load_detection_model

TOLERANCE = 0.05


def to_gflops(macs):
    return 2.0 * macs / 1e9


def measure():
    shape = (1, 3, config.IMG_SIZE, config.IMG_SIZE)
    model_gflops = {}
    for name in config.MODEL_NAMES:
        model = load_detection_model(name, config.WEIGHT_SEARCH_DIRS, config.MODELS_DIR)
        model_gflops[name] = to_gflops(count_macs(model, shape))
        published = config.PUBLISHED_GFLOPS[name]
        flag = "" if abs(model_gflops[name] - published) / published <= TOLERANCE else "   <-- differs from published"
        print(f"{name}: {model_gflops[name]:.2f} GFLOPs (ultralytics publishes {published}){flag}")

    extractor_model = load_detection_model(config.EXTRACTOR_MODEL, config.WEIGHT_SEARCH_DIRS, config.MODELS_DIR)
    embedder = BackboneEmbedder(extractor_model, config.EXTRACTOR_TAP_LAYERS).eval()
    extractor_gflops = to_gflops(count_macs(embedder, shape))
    fc_gflops = to_gflops(config.EMBEDDING_DIM * config.NUM_CLASSES)
    print(f"extractor ({config.EXTRACTOR_MODEL} layers 0-{max(config.EXTRACTOR_TAP_LAYERS)}): "
          f"{extractor_gflops:.3f} GFLOPs, FC head: {fc_gflops:.2e} GFLOPs")

    costs = {
        "unit": "GFLOPs (2 x conv/linear MACs, fused, 640x640, batch 1)",
        "model_gflops": model_gflops,
        "extractor_gflops": extractor_gflops,
        "fc_gflops": fc_gflops,
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.MODEL_COSTS_JSON, "w") as f:
        json.dump(costs, f, indent=2)
    print(f"Written to {config.MODEL_COSTS_JSON}")


if __name__ == "__main__":
    measure()

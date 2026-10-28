# Every constant for cityflow-dispatcher in one place (scenarios, thresholds, classes, stride).
# Change a number here, never inside a script.

# CityFlowV2 Google Drive zip (id, name and size taken from archive/cityflow-routing/dataset.py)
DATASET_FILE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"
ZIP_NAME = "AICity22_Track1_MTMC_Tracking.zip"
ZIP_SIZE = 16_912_529_116

# Splits are by scenario, never mixed
SPLITS = {
    "train": ["S01", "S03", "S04"],
    "val": ["S05"],
    "test": ["S02"],
}
STREAM_ONLY_SCENARIOS = ["S06"]   # no GT, only used for the stream analysis in Part B

FRAME_STRIDE = 5          # every 5th frame per camera, for all splits

# Models, cheapest first. Only n, s, m are fine-tuned; l is teacher and correctness reference.
MODEL_LETTERS = ["n", "s", "m", "l"]
FINETUNED_LETTERS = ["n", "s", "m"]
REFERENCE_LETTER = "l"

# COCO class ids kept for labels and matching
ROAD_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

IMG_SIZE = 640
DETECT_CONF = 0.25
MATCH_IOU = 0.5
RECALL_THRESHOLD = 0.80   # frame is correct for a model if box recall vs yolov8l >= this

JPEG_QUALITY = 95
DETECT_BATCH = 16

# Fine-tuning defaults (all overridable on the command line)
EPOCHS = 10
LR0 = 1e-4
BATCH = 16
WORKERS = 2
SEED = 0

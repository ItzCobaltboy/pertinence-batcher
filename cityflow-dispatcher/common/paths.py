# Where everything lives. All of it is inside the cityflow-dispatcher folder on D:.
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CF_TAG (environment variable) names a second experiment, e.g. "_v2": its fine-tuned weights, detection
# caches, pool tables and result folders sit next to the first set instead of overwriting it.
TAG = os.environ.get("CF_TAG", "")

DATA = os.path.join(ROOT, "data")
CITYFLOW = os.path.join(DATA, "cityflow")          # zip + extracted dataset
SMOKE = os.path.join(DATA, "smoke")                # --smoke runs write a small copy of the dataset here

MODELS = os.path.join(ROOT, "models")
FINETUNED = os.path.join(MODELS, "finetuned" + TAG)
RESULTS = os.path.join(ROOT, "results")


def data_root(smoke=False):
    return SMOKE if smoke else DATA


# ultralytics finds labels by swapping "/images/" for "/labels/" in the image path, so the
# frames folder has to be called images/ (the spec said frames/).
def images_dir(split, smoke=False):
    return os.path.join(data_root(smoke), "images", split)


def labels_dir(split, smoke=False):
    return os.path.join(data_root(smoke), "labels", split)


def manifest_path(smoke=False):
    return os.path.join(data_root(smoke), "manifest.csv")


def dataset_yaml(smoke=False):
    return os.path.join(data_root(smoke), "cityflow.yaml")


def gt_test_boxes(smoke=False):
    return os.path.join(data_root(smoke), "gt_test_boxes.csv")


def weights_path(letter):
    """Original COCO weights, ultralytics downloads them here."""
    return os.path.join(MODELS, f"yolov8{letter}.pt")


def finetuned_path(letter, smoke=False):
    folder = os.path.join(FINETUNED, "smoke") if smoke else FINETUNED
    return os.path.join(folder, f"yolov8{letter}_cityflow.pt")


def result_dir(step, smoke=False):
    path = os.path.join(RESULTS, step + TAG + ("_smoke" if smoke else ""))
    os.makedirs(path, exist_ok=True)
    return path


def extracted_dir():
    return os.path.join(CITYFLOW, "AIC22_Track1")

# Per-frame tables for the fine-tuned pool (ft n, s, m + stock l): recall vs yolov8l, correct flags,
# ideal label, and the cached 448-d embeddings. Used by 05 and 06.
import os

import numpy as np
import pandas as pd

import config
from common import detect, extractor, paths
from common.boxes import box_recall
from common.labels import ideal_letter


def pool_table(split, smoke=False):
    """One row per frame of the split (manifest order: camera, frame_idx) with recall_<letter> and
    ideal for the fine-tuned pool. Cached in data/dispatch/<split>_pool.csv."""
    cache = os.path.join(paths.data_root(smoke), "dispatch" + paths.TAG, f"{split}_pool.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    manifest = pd.read_csv(paths.manifest_path(smoke))
    m = manifest[manifest["split"] == split].sort_values(["camera", "frame_idx"]).reset_index(drop=True)
    images = list(m["path"])
    ref = detect.cached_detections("orig", "l", split, images, smoke)
    for letter in config.FINETUNED_LETTERS:
        dets = detect.cached_detections("ft", letter, split, images, smoke)
        m[f"recall_{letter}"] = [box_recall(r, d, config.MATCH_IOU) for r, d in zip(ref, dets)]
    m["recall_l"] = 1.0
    m["ideal"] = [ideal_letter({l: row[f"recall_{l}"] for l in config.MODEL_LETTERS}) for _, row in m.iterrows()]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    m.to_csv(cache, index=False)
    return m


def split_embeddings(table, embedder, smoke=False):
    """Embeddings aligned with the rows of `table`, cached per camera in data/embeddings/."""
    emb_dir = os.path.join(paths.data_root(smoke), "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    parts = []
    for (split, scenario, camera), rows in table.groupby(["split", "scenario", "camera"], sort=False):
        cache = os.path.join(emb_dir, f"{split}_{scenario}_{camera}.npz")
        if os.path.exists(cache):
            parts.append(np.load(cache)["embeddings"])
            continue
        emb = extractor.embed_files(embedder, list(rows["path"]))
        np.savez(cache, embeddings=emb, frame_idx=rows["frame_idx"].to_numpy())
        parts.append(emb)
    return np.concatenate(parts)

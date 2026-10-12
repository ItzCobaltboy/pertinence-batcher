# cityflow-routing

How the PERTINENCE dispatcher routes real traffic video, frame by frame: the routing split, how it
drifts over time and how sticky it is. The dispatcher is `../yolo-dispatcher/` (YOLOv8 n/s/m/l pool,
trained on COCO, NSGA-II Pareto front of FC heads on a frozen yolov8n backbone). Nothing is retrained
here: every frame of AI City Challenge 2022 Track 1 (CityFlowV2) goes through the same extractor and
every Pareto head, and the routing is analysed.

## What it does

- `route.py` runs the extractor once per frame (batches of 64) and then applies **every** Pareto FC
  head (each a `Linear(448, 4)`) to that embedding, so one pass gives the routing for every
  operating point. The dispatcher runs on every frame (D = 1), so stickiness is the dispatcher's own
  behaviour; coarser D or a change detector can be simulated later from the raw data.
- `analyze.py` turns the raw data into plots and tables, from `raw/` alone (no torch).

Preprocessing is yolo-dispatcher's, imported rather than copied: OpenCV decodes BGR, the frame is
converted to RGB and to a PIL image (training used PIL's `.convert("RGB")`), then
`config.IMAGE_TRANSFORM` (Letterbox 640: PIL bilinear, centred, pad 114, /255, no mean/std) and
`config.build_feature_extractor()` (fused yolov8n, GAP at layers 4/6/9, 448-d). A test checks the
two paths give the same tensor and embedding. Head index i means `MODEL_NAMES[i]` = yolov8n, s, m, l.

## Commands on the A100

Copy this folder next to `yolo-dispatcher/` (the code imports `../yolo-dispatcher/config.py`).

```bash
cd cityflow-routing
python download.py                          # 16.9 GB zip into data/ (needs gdown, see below)
python extract.py --list                    # scenario table from the zip
python extract.py --scenarios S01           # only S01's 5 videos + timing files, 0.70 GB
python route.py --scenarios S01 --run-name s01
python analyze.py --run results/s01
```

- `download.py` needs `gdown`; if it is missing it prints `pip install --user gdown` and stops. It
  skips a complete zip and resumes a partial one.
- `route.py` reads the Pareto front from `../yolo-dispatcher/results/` by default
  (`--yolo-results PATH` to change). It is resumable per camera: run it again and finished cameras
  are skipped. Add scenarios to an existing run with `--scenarios S01 S02 --run-name s01`. If the
  Pareto heads changed since the run started it refuses and asks for a new `--run-name`.
- `analyze.py --point ID` picks the operating point for the plots; the default is the point with the
  highest final_val alpha_sys (it prints which).
- `extract.py --delete-zip-after` frees the 16.9 GB once you have every scenario you want.
- `--data-dir DIR` (download, extract, route) puts the zip and videos somewhere else.

Tests (CPU, any machine with torch + ultralytics): `python -m pytest tests/ -q`.

## Runtime and disk per scenario

| scenario | split | cams | frames | video GB | route.py (estimate) |
|---|---|---|---|---|---|
| S01 | train | 5 | 10,281 | 0.70 | 5-10 min |
| S02 | validation | 4 | 8,109 | 1.02 | 4-8 min |
| S03 | train | 6 | 13,517 | 1.20 | 7-12 min |
| S04 | train | 25 | 10,780 | 1.38 | 5-10 min |
| S05 | validation | 19 | 73,845 | 11.61 | 40-70 min |
| S06 | test | 6 | 12,000 | 0.66 | 6-11 min |

- The run is CPU-bound: preprocessing a 1920x1080 frame (colour conversion, PIL resize, tensor) took
  20-50 ms single-threaded on the CPU I tested on; the GPU part is small. Not measured on the A100.
- Disk: the zip is 16.9 GB; extracted videos as above. Raw output is about 0.9 KB/frame of
  embeddings plus up to 0.5 KB/frame of routing CSV for 50 Pareto points: about 15 MB for S01,
  about 180 MB for all 128,532 frames. Analysis output is a few MB.

## Output layout

```
results/<run>/
  route.log
  raw/                                   keep everything; analyze.py needs only this folder
    manifest.json                        dataset file id, scenarios, cameras (fps, frames decoded vs
                                         header vs cam_framenum, offset), heads + sha256, versions,
                                         GPU, git commit, preprocessing
    pareto_points.csv                    point_id, test/final_val alpha_sys and cost, weights used
    embeddings_<S>_<cam>.npz             embeddings float16 [n_frames, 448], frame_idx
    routing_<S>_<cam>.csv.gz             one row per frame: scenario, camera, frame_idx, fps,
                                         video_time_s, synced_time_s, then per point p:
                                         model_p<id>, prob_n/s/m/l_p<id>
  analysis/
    summary.csv                          one row per (Pareto point, scenario)
    summary.md                           main point's numbers + the 90:10 line
    point_<id>/                          plots and CSVs for the main point
```

- `frame_idx` starts at 1, like CityFlow's own gt/det files. `video_time_s = (frame_idx - 1) / fps`,
  `synced_time_s = video_time_s + offset` from `cam_timestamp/<S>.txt`. FPS comes from each video.
- Weights per point: `results/eval/model_cache/individual_<id>.npz` (the ones the test/final_val
  numbers were computed with), else `results/nsga2/models/individual_<id>.npz`.

## Analysis definitions

Runs and transitions are computed per camera and never cross from one camera to another.

| output | definition |
|---|---|
| `timeline_<S>_<cam>.png` | 4 rows (n/s/m/l), one column per frame, drawn as an image: long runs are solid bands, switches are breaks |
| `timeline_<S>.png` | one strip per camera, stacked, shared synced-time axis |
| `run_lengths.png/.csv` | a run is a maximal block of consecutive frames on one model; histograms per model in frames and seconds; mean/median per model per camera (`ALL` pools a scenario) |
| `split.png` | fraction of frames per model, per camera and per scenario |
| `drift_<S>.png` | split in 10 s windows of video time, stacked area per camera |
| `transitions_<S>.png/.csv` | P(next model given current model) from consecutive frames |
| `arrival_rates_<S>.png/.csv` | the scenario's cameras merged on synced time (what one shared box would see), frames per model queue per 1 s window |
| `summary.csv` | per point and scenario: share per model, mean run length per model (frames), switches per minute (changes / camera-minutes), max 10 s share (largest share of any one model in any 10 s synced window of the merged stream; windows with fewer than half the median frame count, i.e. partial edges, are left out), heaviest model and share, `above_90_10` |

## Known limits

- **No correctness on CityFlow.** The dispatcher was trained on COCO labels and CityFlow has no
  all-class labels, so this measures routing behaviour, not accuracy.
- **Domain shift** between COCO photos and fixed traffic cameras: the routing may not mean on
  CityFlow what it meant on COCO.
- **Resolution**: CityFlow frames are 1920x1080 or 1280x960; they are letterboxed to 640 like the
  training images.
- **Synced time is approximate**: the dataset's ReadMe says some frames are skipped or stuck in
  transmission, so cameras are not perfectly aligned. `manifest.json` records decoded frames vs
  the official counts.

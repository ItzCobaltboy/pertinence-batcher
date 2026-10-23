# cityflow-dispatcher

Fine-tunes yolo-dispatcher's COCO-trained PERTINENCE heads on CCTV frames from CityFlowV2 (AI City
Challenge 2022 Track 1). On CityFlow, the COCO heads never routed frames to yolov8n or yolov8s:
COCO photos and fixed traffic cameras look nothing alike. This track relabels CityFlow frames with a
CCTV-appropriate correctness criterion and nudges each Pareto head toward it.

A thin track like yolo-dispatcher: it reuses yolo-dispatcher's config (pool, costs, extractor,
letterbox, chromosome layout), the shared `../dispatcher_analysis/` loss and weighting code, and
`../cityflow-routing/`'s dataset reader and Pareto loader.

## What it does

1. **`label.py`**: for every 5th frame of each selected camera:
   - runs the dispatcher's extractor (frozen yolov8n backbone, same preprocessing as
     `cityflow-routing/route.py`, checked by a test) to get a 448-d embedding;
   - runs yolov8n/s/m/l (road classes only: person, bicycle, car, motorcycle, bus, truck;
     conf 0.25, 640);
   - scores each model by **box recall against yolov8l**: same class, IoU >= 0.5, greedy by
     confidence;
   - labels the frame with the cheapest model reaching recall >= 0.80.
2. **`finetune.py`**: for every COCO Pareto point:
   - starts from its trained head;
   - keeps its chromosome (penalty matrix + weighting scheme; class weights recomputed on the
     CityFlow labels);
   - trains 10 epochs at lr 1e-4 with the shared penalized loss on the train scenarios;
   - evaluates COCO heads vs fine-tuned heads on the held-out scenarios.

**Data split (never mixed):**

| role | scenarios | used for |
|---|---|---|
| train | S01, S03, S04 (CityFlow train split) | fine-tuning |
| eval | S02 (CityFlow validation split) | every reported number, and the routing run |

S05 (validation, 11.6 GB of video) is left out by default to save disk. Add it to `EVAL_SCENARIOS`
in `cf_config.py` if there is room.

## Commands on the A100

Needs `../yolo-dispatcher/results/` (the COCO Pareto front) and `../cityflow-routing/` with the zip
downloaded. Videos are shared with cityflow-routing, so nothing is extracted twice.

```bash
cd cityflow-routing
python extract.py --scenarios S01 S02 S03 S04      # ~4.3 GB of video (S02 may already be there)

cd ../cityflow-dispatcher
python label.py                                     # all four scenarios, resumable per camera
python finetune.py                                  # minutes
cat results/eval/summary.md

cd ../cityflow-routing                              # route S02 with the fine-tuned heads
python route.py --scenarios S02 --run-name s02_ft --yolo-results ../cityflow-dispatcher/results
python analyze.py --run results/s02_ft
```

- `label.py --scenarios S01` labels a subset. Rerunning skips finished cameras. To relabel a
  camera, delete its `data/labels/<S>_<cam>.csv.gz`.
- `finetune.py --epochs N --lr X` to change the fine-tuning budget. `--coco-results PATH` if the
  COCO front lives elsewhere.
- Tests (CPU, no weights needed): `python -m pytest tests/ -q`.

**Rough cost:** about 7,000 train frames and 1,600 eval frames at stride 5, so roughly 8,700
frames x (extractor + 4 detectors). Minutes on the A100; the bottleneck is decoding 1080p video.
Embeddings add ~1.8 KB per frame (float32), about 16 MB in total.

## Outputs

```
data/                                   CityFlow training/eval data, separate from routing runs
  labels/<S>_<cam>.csv.gz               split, scenario, camera, role (train/eval), fps, frame_idx,
                                        n_ref (yolov8l boxes), boxes_<n|s|m|l>, recall_<n|s|m|l>,
                                        ideal_label, ideal_model
  embeddings/<S>_<cam>.npz              embeddings float32 [n, 448], frame_idx (same order)
  manifest.json, label.log              settings and per-camera label split
results/                                same layout as yolo-dispatcher/results/
  nsga2/pareto_front.csv                copy of the COCO front (same chromosomes)
  eval/model_cache/individual_<id>.npz  fine-tuned W, b, chromosome, scheme, init_weights, init_sha256
  eval/final_val_summary.csv            fine-tuned heads on EVAL (held out); `nondominated` column
  eval/test_summary.csv                 fine-tuned heads on TRAIN (in-sample, for overfit checks)
  eval/coco_heads_eval_summary.csv      COCO heads unchanged on EVAL (the "before")
  eval/baselines_eval.csv               always-n/s/m/l (no dispatcher overhead) and oracle on EVAL
  eval/label_distribution.csv           ideal-label split per scenario
  eval/plots/front_eval.png             alpha_sys vs GFLOPs: COCO vs fine-tuned vs baselines
  eval/plots/label_split.png
  eval/summary.md, finetune_manifest.json
```

Summary columns: `alpha_sys` (fraction of frames where the dispatched model is correct),
`avg_model_cost` (GFLOPs/frame including extractor + FC, yolo-dispatcher's Eq. 4 costs),
`routing_accuracy` (dispatched == ideal), `underestimate_rate`, `overestimate_rate`, and
`frac_<model>` (routing split).

`cityflow-routing/analyze.py` picks the point with the highest `final_val` alpha_sys by default.
With yolov8l as the reference, that is often the most conservative head. Use `--point ID` for a
cheaper operating point from `final_val_summary.csv`.

## Known limits

- **yolov8l is the reference, not ground truth.** It is correct by definition, so labels can only
  say "a smaller model finds what yolov8l finds", never "yolov8l missed something".
- **Recall only.** Extra boxes from a smaller model are not penalised.
- **The criterion decides the split.** In a smoke test on ultralytics' bus.jpg, n/s/m recovered 5
  of yolov8l's 7 boxes (0.71), so the frame was labelled yolov8l. On crowded traffic frames with
  many small objects, a 0.80 box-recall bar can push labels toward the large models. If the label
  split comes out l-heavy, `RECALL_THRESHOLD`, `MATCH_IOU` and `DETECT_CLASSES` in `cf_config.py`
  are the levers.
- **Frames with no reference boxes** (empty road) are labelled yolov8n (nothing to miss).
  `n_ref == 0` marks them, so they can be filtered.
- **Not a new Pareto front.** The fine-tuned set reuses the COCO chromosomes; there is no NSGA-II
  re-search on CityFlow. The `nondominated` column shows which points are still non-dominated on
  held-out data.

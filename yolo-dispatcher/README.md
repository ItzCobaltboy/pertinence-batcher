# yolo-dispatcher

PERTINENCE input dispatcher over the YOLOv8 n/s/m/l detection pool on COCO. It picks
one detector per image to trade accuracy against GFLOPs, and finds the trade-off
front with NSGA-II, the same way the paper does. This folder is a thin track like
`archive/cifar-100/fig9c/`: a config plus entry points that reuse the shared
`../dispatcher/` (search) and `../dispatcher_analysis/` (evaluation) code.

## Run

```bash
git clone <repo> && cd pertinence-batcher/yolo-dispatcher
./setup.sh                 # checks deps + inputs, installs nothing (--install to pip install)
python run_all.py --smoke  # whole pipeline on a tiny GA budget -> results_smoke/ (minutes)
python run_all.py          # paper budget: pop 50 x 50 generations -> results/
```

`python run_all.py --from <stage>` resumes after a failure; `--only <stage>` runs one stage.
COCO images default to `../yolo-analysis/dataset/` (already there on a machine that ran the
benchmark); set `PERTINENCE_COCO_DIR` to use another folder.

| stage | script | does | rough time (A100) |
|---|---|---|---|
| label | `label_data.py` | ground-truth CSVs from `../yolo-analysis/results/*/coco_class_recall_benchmark.csv` | seconds |
| images | `download_images.py` | downloads only missing images (25,000 total, ~3.9 GB) | minutes, or nothing |
| costs | `measure_costs.py` | GFLOPs of each model + extractor -> `data/model_costs.json` | seconds |
| search | `run_dispatcher.py` | caches embeddings, runs NSGA-II, saves Pareto FC weights | hours (2,550 FC trainings) |
| analysis | `run_dispatcher_analysis.py` | test + final_val evaluation, CSVs, plots | minutes |

## Design

- **Correctness**: a model is correct on an image if its class-set recall is >= 0.80 (Week4
  `[DECISION]`). The routing label is the cheapest correct model. Images no model gets right
  (~16% of train, ~20% of val2017) are kept and labelled `yolov8l`, the CIFAR-100 convention.
- **Splits**:
  - train: the 20,000-image train2017 subset
  - test (70% of val2017, stratified by label): NSGA-II fitness, the paper's "test set"
  - final_val (the other 30%): reported once, the paper's "validation set"

  YOLOv8 was trained on train2017, but it doesn't saturate there the way the CIFAR classifiers did
  (yolov8n is correct on 47% of train vs 53% of val2017), so the train labels are usable.
- **Feature extractor**: frozen yolov8n backbone (layers 0-9, fused), images letterboxed to 640,
  global-average-pooled at P3/P4/P5 and concatenated (64 + 128 + 256 = 448 dims). Embeddings are
  cached once, so each NSGA-II individual trains only `Linear(448, 4)`.
- **Cost**: GFLOPs at 640x640, counted as 2 x conv/linear MACs on the fused models (the paper's
  THOP convention), checked against ultralytics' published 8.7 / 28.6 / 78.9 / 165.2.
  `avg_model_cost` includes the extractor + FC on every image (paper Eq. 4). Set
  `REUSE_EXTRACTOR_FOR_SMALLEST = True` in `config.py` to count only yolov8n's neck + head when it
  is picked, since its backbone already ran.
- **Search**: 12 penalty genes + 1 INS/ISNS/ENS gene, penalties in [0, 100], pop 50, 50
  generations, SBX (eta 20, p 0.9), PM (eta 25), 20 FC epochs: the paper's values.

## Outputs

- `data/`: `{train,test,final_val}_ground_truth.csv` (also per-model recall), `model_costs.json`
- `results/nsga2/`: `pareto_front.csv` (non-dominated over every evaluated individual),
  `all_evaluated.npz`, checkpoints, per-individual FC weights
- `results/eval/`: `final_val_summary.csv` (report this one), `test_summary.csv`,
  predictions, plots (`accuracy_vs_gflops_*.png` in the paper's Fig. 6-10 style, Pareto scatter, confusion matrices)

## Known limits

- Correctness is class-set recall only: no boxes, no penalty for extra classes (recall vs F1 is
  still open).
- Cost is counted on square 640x640 inputs; ultralytics' rect inference on COCO is cheaper in
  absolute terms, but it is the same for every model.
- The 20k train subset is a bit harder than val2017 (stratified toward rare categories).

# pertinence-batcher

Re-implementation and batching extension of **PERTINENCE** — a runtime method for opportunistic dynamic execution of neural networks, selecting the lightest model that correctly handles each input.

**Supervisors**: Prof. Gayathri Ananthanarayanan (IIT Dharwad) · Dr. Marcello Traiola (INRIA)

---

## What this is

The original PERTINENCE paper assumes single-image inference. This project extends it to **batched mixed-complexity inference**: a dispatcher routes each image in a batch to the most efficient model that can handle it, runs sub-batches per model, and reassembles results.

**Model pool**: FP32 ResNet18 / ResNet34 / ResNet50 / ResNet152, evaluated on ImageNette.  
Latency spread: ~1.7–8.5 ms · Accuracy spread: 78–91% Top-1.

## Repo layout

```
code/
  Model Analysis/   pipeline scripts (labeler, dispatcher, metrics, etc.)
  dataset/          ImageNette (10-class ImageNet subset, 224px)
  ResnetModels/     PyTorch .pth checkpoints
  OnnxModels/       ONNX FP32 exports (CUDAExecutionProvider)
  results/          output CSVs and plots
Journel/            work session logs (narrative "why" record)
```

## Status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Model pool benchmarking + quantization exploration | ✅ Done |
| 1 | Dispatcher — labeling, training, EDA, batching | 🔄 In progress |
| 2 | NSGA-II Pareto search over dispatcher configurations | ⬜ Planned |

## Key finding so far

Quantization (torchao weight-only, torchao dynamic, ONNX INT8 + CUDA) showed no meaningful Pareto improvement for ResNet CNN inference — dropped. Pool is FP32-only.

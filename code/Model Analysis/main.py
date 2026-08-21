import urllib.request
import tarfile
import os
import torch
import torchvision
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from quantizer_onnx import quantize_model_onnx
from metrics_onnx import model_metrics_onnx
from metrics import model_metrics  # still used to get FLOPs from the PyTorch FP32 model

path = "./../dataset/"
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
tgz_path = os.path.join(path, "imagenette2-320.tgz")
modelPath = "./../ResnetModels/"
onnxModelPath = "./../OnnxModels/"


def download():
    if not os.path.exists(os.path.join(path, "imagenette2-320")):
        print("Downloading ImageNette...")
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path) as f:
            f.extractall(path)
        print("Done!")


def pull_models():
    return {
        "resnet18":  torchvision.models.resnet18(weights='DEFAULT'),
        "resnet34":  torchvision.models.resnet34(weights='DEFAULT'),
        "resnet50":  torchvision.models.resnet50(weights='DEFAULT'),
        "resnet152": torchvision.models.resnet152(weights='DEFAULT'),
    }


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # download()  # run once to get the dataset

    os.makedirs(onnxModelPath, exist_ok=True)

    models = pull_models()
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")

        # FLOPs don't change between PyTorch and ONNX exports of the same architecture,
        # so grab it once from the PyTorch model rather than recomputing per format
        flops_metrics = model_metrics(model, path, device,
                                       pth_path=os.path.join(modelPath, f"{name}_float32.pth"))
        flops = flops_metrics["flops"]

        # export to ONNX + produce static INT8 quantized version
        onnx_paths = quantize_model_onnx(model, name, onnxModelPath, path, device)

        for precision, onnx_path in [("fp32", onnx_paths["fp32_path"]),
                                       ("int8", onnx_paths["int8_path"])]:
            print(f"Evaluating {name}_{precision} (ONNX)...")

            metrics = model_metrics_onnx(onnx_path, path, use_tensorrt=False)

            results.append({
                "model":         name,
                "precision":     precision,
                "flops":         flops,
                "flops_G":       round(flops / 1e9, 3),
                "model_size_mb": metrics["model_size_mb"],
                "accuracy":      metrics["accuracy"],
                "latency_ms":    metrics["latency_ms"],
                "provider":      metrics["provider"]
            })

            print(f"  Acc: {metrics['accuracy']:.2f}% | Size: {metrics['model_size_mb']}MB | "
                  f"Latency: {metrics['latency_ms']}ms | Provider: {metrics['provider']}")

    os.makedirs("./../results/", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("./../results/step0_metrics_onnx.csv", index=False)
    print("\nResults saved → results/step0_metrics_onnx.csv")
    print(df.to_string())
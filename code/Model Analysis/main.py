import urllib.request
import tarfile
import os
import torch
import torchvision
import pandas as pd
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Float8WeightOnlyConfig
from quantizer import quantize_model
from metrics import model_metrics
import warnings
warnings.filterwarnings("ignore")

path = "./../dataset/"
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
tgz_path = os.path.join(path, "imagenette2-320.tgz")
modelPath = "./../ResnetModels/"


def download():
    if not os.path.exists(os.path.join(path, "imagenette2-320")):
        print("Downloading ImageNette...")
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path) as f:
            f.extractall(path)
        print("Done!")


def pull_and_save_models():
    res18  = torchvision.models.resnet18(weights='DEFAULT')
    res34  = torchvision.models.resnet34(weights='DEFAULT')
    res50  = torchvision.models.resnet50(weights='DEFAULT')
    res152 = torchvision.models.resnet152(weights='DEFAULT')

    os.makedirs(modelPath, exist_ok=True)

    models      = [res18, res34, res50, res152]
    model_names = ["resnet18", "resnet34", "resnet50", "resnet152"]

    for model, name in zip(models, model_names):
        torch.save(model.state_dict(), modelPath + name + "_float32.pth")
        quantize_model(model, name, modelPath, device)


def load_model(model_fn, precision, pth_path, device):
    model = model_fn(weights=None)

    # Quantized models need the quantization applied to the architecture
    # before loading weights — you can't load a quantized state dict
    # into a plain FP32 model
    if precision == "int8":
        quantize_(model, Int8WeightOnlyConfig())
    elif precision == "float8":
        quantize_(model, Float8WeightOnlyConfig())

    model.load_state_dict(torch.load(pth_path, map_location=device))
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # download()          # run once to get the dataset
    # pull_and_save_models()  # run once to generate all .pth files

    model_configs = [
        ("resnet18",  torchvision.models.resnet18,  "float32"),
        ("resnet18",  torchvision.models.resnet18,  "int8"),
        ("resnet18",  torchvision.models.resnet18,  "float8"),
        ("resnet34",  torchvision.models.resnet34,  "float32"),
        ("resnet34",  torchvision.models.resnet34,  "int8"),
        ("resnet34",  torchvision.models.resnet34,  "float8"),
        ("resnet50",  torchvision.models.resnet50,  "float32"),
        ("resnet50",  torchvision.models.resnet50,  "int8"),
        ("resnet50",  torchvision.models.resnet50,  "float8"),
        ("resnet152", torchvision.models.resnet152, "float32"),
        ("resnet152", torchvision.models.resnet152, "int8"),
        ("resnet152", torchvision.models.resnet152, "float8"),
    ]

    results = []

    for name, model_fn, precision in model_configs:
        pth_path = os.path.join(modelPath, f"{name}_{precision}.pth")

        if not os.path.exists(pth_path):
            print(f"Skipping {name}_{precision} — .pth not found")
            continue

        print(f"Evaluating {name}_{precision}...")
        model = load_model(model_fn, precision, pth_path, device)

        model_size_mb = os.path.getsize(pth_path) / (1024 * 1024)

        metrics = model_metrics(model, path, device, pth_path)
        results.append({
            "model":     name,
            "precision": precision,
            "flops":     metrics["flops"],
            "flops_G":   round(metrics["flops"] / 1e9, 3),
            "model_size_mb": round(model_size_mb, 2),
            "accuracy":  metrics["accuracy"],
            "latency_ms": metrics["latency_ms"]
        })
        print(f"  FLOPs: {metrics['flops']/1e9:.2f}G | Acc: {metrics['accuracy']:.2f}% | Size: {model_size_mb}MB")

    os.makedirs("./../results/", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("./../results/step0_metrics.csv", index=False)
    print("\nResults saved → results/step0_metrics.csv")
    print(df.to_string())
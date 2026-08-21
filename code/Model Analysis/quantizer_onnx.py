import os
import torch
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat


def export_to_onnx(model: torch.nn.Module, model_name: str, save_path: str, device: torch.device) -> str:
    """
    Exports a PyTorch model to ONNX format (FP32).

    Args:
        model:      PyTorch model, already loaded with weights
        model_name: name used for the output file
        save_path:  directory to save the .onnx file
        device:     torch device (export happens on CPU internally but weights come from here)

    Returns:
        Path to the exported .onnx file
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = os.path.join(save_path, f"{model_name}_fp32.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17
    )

    print(f"Exported {model_name} to {onnx_path}")
    return onnx_path


class ImageNetteCalibrationReader(CalibrationDataReader):
    """
    Feeds calibration images to onnxruntime's static quantizer.
    Static PTQ needs real data to compute activation ranges, unlike weight-only
    quantization which only looks at the weights themselves.
    """

    def __init__(self, dataset_path: str, num_samples: int = 200, batch_size: int = 1):
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset = ImageFolder(root=os.path.join(dataset_path, "imagenette2-320/val"), transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.data = []
        for i, (images, _) in enumerate(loader):
            if i >= num_samples:
                break
            self.data.append({"input": images.numpy().astype(np.float32)})

        self.enum_data = iter(self.data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter(self.data)


def quantize_onnx_model(fp32_onnx_path: str, model_name: str, save_path: str, dataset_path: str, num_calib_samples: int = 200) -> str:
    """
    Applies static INT8 post-training quantization to an ONNX model.
    Static quantization calibrates activation ranges using real data (unlike dynamic
    quantization which estimates ranges on the fly at inference time) — generally
    gives better accuracy and is what TensorRT expects for INT8 execution.

    Args:
        fp32_onnx_path:    path to the FP32 .onnx file
        model_name:        name used for the output file
        save_path:         directory to save the quantized .onnx file
        dataset_path:       path to dataset root (for calibration data)
        num_calib_samples: number of images used to calibrate activation ranges

    Returns:
        Path to the quantized .onnx file
    """
    calibration_reader = ImageNetteCalibrationReader(dataset_path, num_samples=num_calib_samples)
    int8_path = os.path.join(save_path, f"{model_name}_int8.onnx")

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,   # QDQ = QuantizeLinear/DequantizeLinear nodes, what TensorRT EP expects
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    print(f"Quantized {model_name} saved to {int8_path}")
    return int8_path


def quantize_model_onnx(model: torch.nn.Module, model_name: str, save_path: str, dataset_path: str, device: torch.device) -> dict:
    """
    Full pipeline: export FP32 model to ONNX, then produce a static INT8 quantized version.

    Returns:
        dict with keys: 'fp32_path', 'int8_path'
    """
    os.makedirs(save_path, exist_ok=True)

    fp32_path = export_to_onnx(model, model_name, save_path, device)
    int8_path = quantize_onnx_model(fp32_path, model_name, save_path, dataset_path)

    return {
        "fp32_path": fp32_path,
        "int8_path": int8_path
    }
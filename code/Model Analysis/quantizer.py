import copy

import torch
import torchvision
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig, Float8DynamicActivationInt4WeightConfig

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using Device : ")
# print(device)

def quantize_model(model, model_name : str, savePath : str, device : torch.device):
    print("Quantizing Model : " + model_name + "...")

    model.to(device)

    # to int4 & int8
    model_int4 = copy.deepcopy(model)
    model_int8 = copy.deepcopy(model)

    quantize_(model_int4, Float8DynamicActivationInt4WeightConfig())
    quantize_(model_int8, Int8DynamicActivationInt8WeightConfig())
    print("Quantized Model : " + model_name + "Successfully Quantized!")

    torch.save(model_int4.state_dict(), savePath + model_name + "_float8.pth")
    torch.save(model_int8.state_dict(), savePath + model_name + "_int8.pth")
    print("Quantized Model : " + model_name + "Saved!")

    return

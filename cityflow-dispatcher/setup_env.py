# Creates cityflow-dispatcher/.venv, installs a torch build that supports the RTX 5070 Ti
# (Blackwell, sm_120 -> CUDA 12.8+), installs the rest, proves CUDA works, runs the preflight.
# Run with any python:  python setup_env.py
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
VENV_PY = os.path.join(VENV, "Scripts", "python.exe")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
OTHER_PACKAGES = ["ultralytics", "opencv-python", "numpy", "pandas", "matplotlib", "pyyaml", "gdown", "tqdm"]


def find_python():
    """Returns the command (list) of a Python 3.10 to 3.12 via the py launcher."""
    for version in ["3.12", "3.11", "3.10"]:
        command = ["py", f"-{version}"]
        try:
            subprocess.run(command + ["-c", "pass"], check=True, capture_output=True)
            return command
        except (OSError, subprocess.CalledProcessError):
            continue
    sys.exit("No Python 3.10-3.12 found (checked with `py -3.x`). Install one by hand and rerun.")


def run(command):
    print(">", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    if not os.path.exists(VENV_PY):
        run(find_python() + ["-m", "venv", VENV])
    run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"])
    run([VENV_PY, "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX])
    run([VENV_PY, "-m", "pip", "install"] + OTHER_PACKAGES)

    # prove the GPU works: sm_120 in the arch list and a real matmul on cuda
    check = (
        "import torch; assert torch.cuda.is_available(), 'CUDA not available';"
        "arch = torch.cuda.get_arch_list(); print('arch list', arch);"
        "assert 'sm_120' in arch, 'torch build has no sm_120 kernels';"
        "a = torch.randn(2048, 2048, device='cuda'); b = (a @ a).sum().item();"
        "print('cuda matmul ok on', torch.cuda.get_device_name(0), b)"
    )
    run([VENV_PY, "-c", check])

    freeze = subprocess.run([VENV_PY, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    with open(os.path.join(HERE, "requirements.lock.txt"), "w") as f:
        f.write(freeze.stdout)
    with open(os.path.join(HERE, "requirements.txt"), "w") as f:
        f.write(f"# torch from {TORCH_INDEX} (install with --index-url), the rest from PyPI\ntorch\ntorchvision\n")
        f.write("\n".join(OTHER_PACKAGES) + "\n")

    run([VENV_PY, os.path.join(HERE, "00_preflight.py")])


if __name__ == "__main__":
    main()

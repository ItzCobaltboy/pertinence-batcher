#!/usr/bin/env bash
# Checks the environment for yolo-dispatcher. Installs NOTHING unless asked:
#   ./setup.sh            check only
#   ./setup.sh --install  pip install -r requirements.txt into the active env
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

if [[ "${1:-}" == "--install" ]]; then
  "$PY" -m pip install -r requirements.txt
fi

"$PY" - <<'PYEOF'
import importlib, os, sys
ok = True
for mod in ["numpy", "pandas", "PIL", "matplotlib", "torch", "ultralytics", "pymoo"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:<12} {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  MISSING {mod:<9} ({e.__class__.__name__})")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -- search will run on CPU (slow)"))
except Exception:
    pass
for split in ["train2017", "val2017"]:
    path = os.path.join("..", "yolo-analysis", "results", split, "coco_class_recall_benchmark.csv")
    present = os.path.exists(path)
    ok &= present
    print(f"  {'ok  ' if present else 'MISSING'} {path}")
print("\nReady: python run_all.py --smoke   (then python run_all.py)" if ok
      else "\nFix the MISSING items above (./setup.sh --install installs the python ones).")
sys.exit(0 if ok else 1)
PYEOF

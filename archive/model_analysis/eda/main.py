"""
eda — analysis of model_analysis's benchmark results, single entry point.

Reads model_analysis's two output CSVs (live, not copied) and produces
comparison plots. Does not benchmark anything itself and does not touch
code/dispatcher's own EDA.

Folder layout:
  main.py           <- you are here, run this file
  src/
    constants.py       paths (reads from ../model_analysis/results/)
    eda.py                plots driver

  results/            accuracy/latency/size grouped bar charts +
                       accuracy-vs-latency scatter (with eager baseline)
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

from eda import run_eda


if __name__ == "__main__":
    run_eda()
    print("\nDone.")

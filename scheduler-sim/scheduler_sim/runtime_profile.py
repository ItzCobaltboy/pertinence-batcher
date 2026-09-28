"""RuntimeProfile: the T_i(b) matrix -- rows = models, columns = batch
sizes, cells = runtime in ms. Loaded from a CSV of the form:

    model,1,4,8,16,32
    modelA,5.06,6.30,9.24,16.2,33.43
    modelB,7.45,13.04,27.25,64.94,149.61

An empty cell means that batch size is not supported by that model.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


class UnsupportedBatchSize(Exception):
    pass


@dataclass
class RuntimeProfile:
    models: List[str]
    batch_sizes: List[int]
    matrix: Dict[str, Dict[int, float]]  # model -> {batch_size: ms}, only supported sizes present
    partial_batch_mode: str = "pad"  # "pad" or "interpolate"
    switch_cost_ms: float = 0.0  # extra cost when the accelerator changes model between batches

    def __post_init__(self):
        if self.partial_batch_mode not in ("pad", "interpolate"):
            raise ValueError(
                f"partial_batch_mode must be 'pad' or 'interpolate', got {self.partial_batch_mode!r}"
            )

    @classmethod
    def from_csv(
        cls,
        path: str,
        partial_batch_mode: str = "pad",
        switch_cost_ms: float = 0.0,
    ) -> "RuntimeProfile":
        with open(path, newline="") as f:
            lines = (line for line in f if not line.lstrip().startswith("#"))
            reader = csv.reader(lines)
            header = next(reader)
            if header[0].strip().lower() != "model":
                raise ValueError(f"{path}: expected first column 'model', got {header[0]!r}")
            batch_sizes = [int(c) for c in header[1:]]
            models: List[str] = []
            matrix: Dict[str, Dict[int, float]] = {}
            for row in reader:
                if not row or not row[0].strip():
                    continue
                name = row[0].strip()
                models.append(name)
                cells: Dict[int, float] = {}
                for b, cell in zip(batch_sizes, row[1:]):
                    cell = cell.strip() if cell is not None else ""
                    if cell == "" or cell.lower() == "nan":
                        continue
                    cells[b] = float(cell)
                if not cells:
                    raise ValueError(f"{path}: model '{name}' has no supported batch sizes")
                matrix[name] = cells
        return cls(
            models=models,
            batch_sizes=batch_sizes,
            matrix=matrix,
            partial_batch_mode=partial_batch_mode,
            switch_cost_ms=switch_cost_ms,
        )

    def supported_batch_sizes(self, model: str) -> List[int]:
        return sorted(self.matrix[model].keys())

    def max_batch_size(self, model: str) -> int:
        return max(self.supported_batch_sizes(model))

    def is_supported(self, model: str, b: int) -> bool:
        return b in self.matrix[model]

    def largest_supported_at_most(self, model: str, b: int) -> Optional[int]:
        """Largest supported batch size <= b, or None if b is smaller
        than every supported size."""
        candidates = [s for s in self.matrix[model] if s <= b]
        return max(candidates) if candidates else None

    def smallest_supported_at_least(self, model: str, b: int) -> Optional[int]:
        candidates = [s for s in self.matrix[model] if s >= b]
        return min(candidates) if candidates else None

    def runtime(self, model: str, b: int) -> float:
        """Runtime in ms to run `b` jobs of `model` as one batch.

        If `b` is exactly a supported size, returns that cell directly.
        Otherwise falls back to `partial_batch_mode`:
          - "pad": run at the cost of the smallest supported size >= b
            (matches a fixed-shape TensorRT engine padding the batch
            up). If b exceeds the model's max supported size, this is
            an error -- callers must split into multiple batches.
          - "interpolate": linearly interpolate between the two
            bracketing supported sizes (extrapolates flat at the ends).
        """
        if model not in self.matrix:
            raise KeyError(f"Unknown model '{model}'")
        if b <= 0:
            raise ValueError(f"batch size must be positive, got {b}")
        cells = self.matrix[model]
        if b in cells:
            return cells[b]

        sizes = sorted(cells.keys())
        if self.partial_batch_mode == "pad":
            target = self.smallest_supported_at_least(model, b)
            if target is None:
                raise UnsupportedBatchSize(
                    f"model '{model}' batch {b} exceeds max supported size {sizes[-1]}"
                )
            return cells[target]

        # interpolate
        if b <= sizes[0]:
            return cells[sizes[0]]
        if b >= sizes[-1]:
            # flat extrapolation past the largest measured point, since
            # we have no data on how the curve continues.
            return cells[sizes[-1]]
        lo = max(s for s in sizes if s < b)
        hi = min(s for s in sizes if s > b)
        frac = (b - lo) / (hi - lo)
        return cells[lo] + frac * (cells[hi] - cells[lo])

    def cost(self, prev_model: Optional[str], model: str, b: int) -> float:
        """Total ms including the model-switch cost, if any."""
        base = self.runtime(model, b)
        if prev_model is not None and prev_model != model:
            base += self.switch_cost_ms
        return base

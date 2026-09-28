"""Logging setup: standard Python logging with levels, plus an
optional full event trace to JSONL (off by default -- it gets large).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

LOGGER_NAME = "scheduler_sim"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


class EventTraceWriter:
    """Writes one JSON object per line per simulator event. Off by
    default; pass a path to enable."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._fh: Optional[TextIO] = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w")

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def write(self, record: Dict[str, Any]) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

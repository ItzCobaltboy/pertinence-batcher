"""Tiny name -> class registry, shared by routers and policies so a YAML
or python config can pick an implementation by string."""
from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._classes: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._classes:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._classes[name] = cls
            return cls

        return deco

    def get(self, name: str) -> Type:
        try:
            return self._classes[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {sorted(self._classes)}"
            )

    def create(self, name: str, /, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self):
        return sorted(self._classes)

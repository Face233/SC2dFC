from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if overrides:
        config = _merge(config, overrides)
    root = Path(config["paths"].get("root", "."))
    if not root.is_absolute():
        root = (path.parent.parent / root).resolve()
    config["paths"]["root"] = str(root)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    value = config["paths"][key]
    path = Path(value)
    return path if path.is_absolute() else Path(config["paths"]["root"]) / path


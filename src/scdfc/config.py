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


def _load_with_base(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular config inheritance involving {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_reference = current.pop("base", None)
    if base_reference is None:
        current["_config_root_anchor"] = str(path.parent.parent)
        return current
    base_path = Path(base_reference)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge(_load_with_base(base_path, seen), current)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _load_with_base(path)
    if overrides:
        config = _merge(config, overrides)
    root_anchor = Path(config.pop("_config_root_anchor", path.parent.parent))
    root = Path(config["paths"].get("root", "."))
    if not root.is_absolute():
        root = (root_anchor / root).resolve()
    config["paths"]["root"] = str(root)
    config["_config_source"] = str(path)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    value = config["paths"][key]
    path = Path(value)
    return path if path.is_absolute() else Path(config["paths"]["root"]) / path

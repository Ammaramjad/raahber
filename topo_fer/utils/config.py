from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from omegaconf import OmegaConf


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = OmegaConf.create(base)
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override))
    return OmegaConf.to_container(cfg, resolve=True)


def parse_overrides(overrides: Sequence[str] | None) -> Dict[str, Any]:
    """Parse KEY=VALUE override strings into a nested dictionary."""
    if not overrides:
        return {}
    result: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected format KEY=VALUE.")
        key, value = override.split("=", 1)
        parts = key.split(".")
        ref: Dict[str, Any] = result
        for part in parts[:-1]:
            ref = ref.setdefault(part, {})
        try:
            value_loaded = json.loads(value)
        except json.JSONDecodeError:
            value_loaded = value
        ref[parts[-1]] = value_loaded
    return result


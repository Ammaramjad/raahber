from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = OmegaConf.create(base)
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override))
    return OmegaConf.to_container(cfg, resolve=True)


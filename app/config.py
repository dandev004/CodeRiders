"""Încărcarea configurației din config/config.yaml (cu override prin variabile de mediu SECURE_MOM_*)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("SECURE_MOM_CONFIG", ROOT / "config" / "config.yaml"))


class Cfg(dict):
    """Dict cu acces prin atribut: cfg.asr.backend."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(v) if isinstance(v, dict) else v


def _apply_env_overrides(data: dict) -> None:
    # SECURE_MOM_LLM__MODEL=qwen3.5:9b -> data["llm"]["model"]
    for key, value in os.environ.items():
        if not key.startswith("SECURE_MOM_") or key == "SECURE_MOM_CONFIG":
            continue
        path = key[len("SECURE_MOM_"):].lower().split("__")
        node = data
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = yaml.safe_load(value)


@lru_cache(maxsize=1)
def get_config() -> Cfg:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _apply_env_overrides(data)
    return Cfg(data)


def resolve(path: str | Path) -> Path:
    """Căi relative din config sunt relative la rădăcina proiectului."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

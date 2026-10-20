"""YAML config loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "experiments" / "configs" / "base.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or BASE_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

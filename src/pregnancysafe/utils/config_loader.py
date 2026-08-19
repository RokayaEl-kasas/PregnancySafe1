"""Loads config/config.yaml and data/medication_safety.json into validated
pydantic models, so every other module gets typed, pre-checked data instead
of re-parsing YAML/JSON and re-discovering malformed entries at random points
in the pipeline.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pregnancysafe.schemas import Disease, DiseaseMedications, DocumentSource, RedFlag

# Repo root = three levels up from this file (src/pregnancysafe/utils/config_loader.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
DEFAULT_MED_SAFETY_PATH = REPO_ROOT / "data" / "medication_safety.json"


@lru_cache(maxsize=1)
def load_raw_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_diseases(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Disease]:
    """Parse the `disease_sources` block of config.yaml into Disease models."""
    raw = load_raw_config(config_path)
    diseases: dict[str, Disease] = {}
    for disease_id, block in raw["disease_sources"].items():
        sources = [DocumentSource(**s) for s in block.get("sources", [])]
        diseases[disease_id] = Disease(
            id=disease_id,
            label_ar=block.get("label_ar", disease_id),
            label_en=block.get("label_en", disease_id.replace("_", " ").title()),
            folder=block["folder"],
            sources=sources,
        )
    return diseases


@lru_cache(maxsize=1)
def load_medication_safety(
    path: Path = DEFAULT_MED_SAFETY_PATH,
) -> dict[str, DiseaseMedications]:
    """Parse data/medication_safety.json into validated DiseaseMedications models.

    Raises a pydantic.ValidationError immediately (with the offending field
    named) if any medication entry is malformed — e.g. an unrecognized tier
    string — rather than letting a bad record propagate into the agent.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        disease_id: DiseaseMedications(**block)
        for disease_id, block in raw["diseases"].items()
    }


@lru_cache(maxsize=1)
def load_red_flags(path: Path = DEFAULT_MED_SAFETY_PATH) -> list[RedFlag]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [RedFlag(**rf) for rf in raw.get("red_flags", [])]

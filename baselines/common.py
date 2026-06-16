"""Helpers partagés par les baselines : normalisation unit/direction/type + I/O JSONL."""

import json
import os

from synonym_lookup import normalize

_UNIT_BPS = {"bps", "bp", "points de base", "pdb", "point de base"}
_UNIT_PCT = {"%", "pct", "pourcent", "pourcentage", "percent"}

_DIR_UP = {"increase", "hausse", "augmente", "augmentation", "augmenter", "+", "monte", "hausser"}
_DIR_DOWN = {"decrease", "baisse", "baisser", "diminue", "diminution", "reduire", "-", "baisser"}


def normalize_unit(unit: str):
    n = normalize(unit)
    if n in _UNIT_BPS:
        return "bps"
    if n in _UNIT_PCT or "%" in (unit or ""):
        return "%"
    return None


def normalize_direction(direction: str):
    n = normalize(direction)
    if n in _DIR_UP:
        return "increase"
    if n in _DIR_DOWN:
        return "decrease"
    return None


def type_from_unit(unit: str):
    """bps → absolute ; % → relative."""
    if unit == "bps":
        return "absolute"
    if unit == "%":
        return "relative"
    return None


def to_float(value):
    try:
        return float(str(value).replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


def empty_prediction():
    """Squelette de prédiction baseline : champs hors-portée laissés à null."""
    return {
        "intent": None, "mode": None, "product_id": None,
        "drivers": [], "targets": [],
        "lever": None, "goal": None, "target": None,
        "unknown_terms": [],
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

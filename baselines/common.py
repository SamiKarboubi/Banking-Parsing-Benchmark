"""Helpers partagés par les baselines : normalisation unit/direction/type + I/O JSONL.

finalize_baseline() est l'ASSEMBLEUR rule-based : à partir de ce qui a été extrait, il dérive
intent/mode (les baselines ne sont pas des classifieurs de tâche) pour pouvoir matcher les
exemples forward et other. Le mode inverse reste hors-portée (pas de lever/goal extraits).
"""

import json
import os

from closed_world import DRIVERS, KPIS, SOURCES
from synonym_lookup import normalize

LEVER_KEYS = set(SOURCES) | set(DRIVERS)
KPI_KEYS = set(KPIS)

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


def finalize_baseline(pred):
    """Assemble la prédiction baseline en sortie canonique.

    - value des variations -> magnitude (|value|) ;
    - garde-fous : un driver doit avoir une clé de levier valide, une target une clé de KPI ;
    - dérive intent/mode : contenu de simulation -> forward ; sinon -> other (rien extrait).
    """
    drivers = []
    for d in pred.get("drivers") or []:
        if d.get("entity_key") in LEVER_KEYS:
            ch = d.get("change") or {}
            v = ch.get("value")
            if isinstance(v, (int, float)):
                ch = dict(ch, value=abs(v))
            drivers.append({"entity_key": d["entity_key"], "change": ch})
    pred["drivers"] = drivers
    pred["targets"] = sorted({t for t in (pred.get("targets") or []) if t in KPI_KEYS})

    has_sim = bool(pred.get("product_id") or pred["drivers"] or pred["targets"])
    if has_sim:
        pred["intent"], pred["mode"] = "simulation", "forward"
    else:
        # rien d'exploitable -> on parie sur "other" (sortie canonique vide)
        return {"intent": "other", "mode": None, "product_id": None,
                "drivers": [], "targets": [], "lever": None, "goal": None,
                "target": None, "unknown_terms": []}
    return pred


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

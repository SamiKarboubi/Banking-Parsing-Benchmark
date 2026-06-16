"""Schéma de sortie Pydantic + dérivations en code (entity_type, needs_clarification).

On expose deux choses :
  - les modèles de l'union discriminée (FORWARD / INVERSE / OTHER) pour la validation post-génération ;
  - un schéma PLAT (FlatOutput) utilisé pour le décodage contraint vLLM, car beaucoup de
    backends gèrent mal une union discriminée sur DEUX champs (intent + mode). Voir README.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from closed_world import SOURCES, type_of

ProductId = Literal["CRED_IMMO", "CRED_CONSO", "CRED_PME"]
LeverKey = Literal[
    "DAV", "DAT", "MARCHE", "FP",
    "taux_client", "cout_risque", "cout_operationnel", "cout_capital", "commissions",
]
KpiKey = Literal[
    "marge_interet", "pnb", "cout_risque_total", "marge_nette", "raroc", "coef_exploitation",
]


class Change(BaseModel):
    type: Literal["absolute", "relative"]
    value: float
    unit: Literal["bps", "%"]
    direction: Literal["increase", "decrease"]


class Driver(BaseModel):
    entity_key: LeverKey
    change: Change


class Lever(BaseModel):
    entity_key: LeverKey


# --- Union discriminée (validation "stricte") -------------------------------

class ForwardOutput(BaseModel):
    intent: Literal["simulation"]
    mode: Literal["forward"]
    product_id: Optional[ProductId] = None
    drivers: List[Driver]
    targets: List[KpiKey]
    unknown_terms: List[str]


class InverseOutput(BaseModel):
    intent: Literal["simulation"]
    mode: Literal["inverse"]
    product_id: Optional[ProductId] = None
    lever: Lever
    goal: Change
    target: KpiKey
    unknown_terms: List[str]


class OtherOutput(BaseModel):
    intent: Literal["other"]
    unknown_terms: List[str] = Field(default_factory=list)


Output = Union[ForwardOutput, InverseOutput, OtherOutput]


# --- Schéma plat (décodage contraint) ---------------------------------------

class FlatOutput(BaseModel):
    """Tous les champs des 3 branches, en nullable. On reconstruit/valide ensuite."""
    intent: Literal["simulation", "other"]
    mode: Optional[Literal["forward", "inverse"]] = None
    product_id: Optional[ProductId] = None
    drivers: Optional[List[Driver]] = None
    targets: Optional[List[KpiKey]] = None
    lever: Optional[Lever] = None
    goal: Optional[Change] = None
    target: Optional[KpiKey] = None
    unknown_terms: List[str] = Field(default_factory=list)


def flat_to_canonical(d: dict) -> dict:
    """Réduit un dict (issu du schéma plat) à la branche canonique attendue."""
    if d.get("intent") == "other":
        return {"intent": "other", "unknown_terms": [t.lower() for t in d.get("unknown_terms") or []]}
    out = {
        "intent": "simulation",
        "mode": d.get("mode"),
        "product_id": d.get("product_id"),
        "unknown_terms": [t.lower() for t in d.get("unknown_terms") or []],
    }
    if d.get("mode") == "forward":
        out["drivers"] = d.get("drivers") or []
        out["targets"] = d.get("targets") or []
    elif d.get("mode") == "inverse":
        out["lever"] = d.get("lever")
        out["goal"] = d.get("goal")
        out["target"] = d.get("target")
    return out


def validate(d: dict) -> Optional[dict]:
    """Valide un dict contre l'union discriminée. Retourne le dict normalisé ou None."""
    try:
        if d.get("intent") == "other":
            return OtherOutput(**d).model_dump()
        if d.get("mode") == "forward":
            return ForwardOutput(**d).model_dump()
        if d.get("mode") == "inverse":
            return InverseOutput(**d).model_dump()
    except Exception:
        return None
    return None


# --- Dérivations métier (hors schéma) ---------------------------------------

def entity_type_of(key: str) -> str:
    return type_of(key)


def needs_clarification(d: dict) -> bool:
    """Dérivé en code, jamais stocké dans le JSON."""
    if d.get("intent") != "simulation":
        return False
    if d.get("mode") == "forward":
        return d.get("product_id") is None or not d.get("drivers")
    if d.get("mode") == "inverse":
        return (
            d.get("product_id") is None
            or not d.get("lever")
            or not d.get("goal")
            or not d.get("target")
        )
    return True

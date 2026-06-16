"""Baseline GLiNER (NER zero-shot) : spans étiquetés + regroupement en drivers.

Champs intent/mode/lever/goal/target laissés null (hors-portée d'un NER).
"""

import gc
import time

import torch

from common import (empty_prediction, finalize_baseline, normalize_direction,
                    normalize_unit, to_float, type_from_unit, write_jsonl)
from synonym_lookup import match

MODEL_ID = "urchade/gliner_multi-v2.1"
THRESHOLD = 0.4

LABELS = [
    "produit_credit", "source_financement", "driver_levier",
    "kpi_cible", "valeur_numerique", "unite_variation",
    "direction_variation", "terme_inconnu",
]


def _group_drivers(lever_spans, value_spans, unit_spans, dir_spans):
    """Associe à chaque levier la valeur/unité/direction la plus proche dans la phrase."""
    drivers = []
    for lev in lever_spans:
        key = match(lev["text"])
        center = (lev["start"] + lev["end"]) / 2

        def nearest(spans):
            if not spans:
                return None
            return min(spans, key=lambda s: abs((s["start"] + s["end"]) / 2 - center))

        v = nearest(value_spans)
        u = nearest(unit_spans)
        dr = nearest(dir_spans)
        unit = normalize_unit(u["text"]) if u else None
        drivers.append({
            "entity_key": key,
            "change": {
                "type": type_from_unit(unit),
                "value": to_float(v["text"]) if v else None,
                "unit": unit,
                "direction": normalize_direction(dr["text"]) if dr else None,
            },
            "_unmatched": key is None,
            "_raw": lev["text"],
        })
    return drivers


def postprocess(text: str, spans: list) -> dict:
    """spans : [{text, label, start, end, score}]."""
    pred = empty_prediction()
    by_label = {lab: [s for s in spans if s["label"] == lab] for lab in LABELS}
    unknown = []

    for s in by_label["produit_credit"]:
        canon = match(s["text"])
        if canon and pred["product_id"] is None:
            pred["product_id"] = canon
        elif not canon:
            unknown.append(s["text"].lower())

    lever_spans = by_label["source_financement"] + by_label["driver_levier"]
    drivers = _group_drivers(
        lever_spans, by_label["valeur_numerique"],
        by_label["unite_variation"], by_label["direction_variation"],
    )
    for d in drivers:
        if d.pop("_unmatched"):
            unknown.append(d.pop("_raw").lower())
        else:
            d.pop("_raw", None)
            pred["drivers"].append(d)

    for s in by_label["kpi_cible"]:
        canon = match(s["text"])
        if canon:
            pred["targets"].append(canon)
        else:
            unknown.append(s["text"].lower())

    for s in by_label["terme_inconnu"]:
        if s["score"] > THRESHOLD and match(s["text"]) is None:
            unknown.append(s["text"].lower())

    pred["targets"] = sorted(set(pred["targets"]))
    pred["unknown_terms"] = sorted(set(unknown))
    return finalize_baseline(pred)


def _load():
    from gliner import GLiNER
    model = GLiNER.from_pretrained(MODEL_ID)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def run(rows, out_path):
    model = _load()
    try:
        results = []
        for r in rows:
            t0 = time.perf_counter()
            try:
                spans = model.predict_entities(r["text"], LABELS, threshold=THRESHOLD)
                pred = postprocess(r["text"], spans)
                valid, latency = True, time.perf_counter() - t0
            except Exception:
                pred, valid, latency = None, False, time.perf_counter() - t0
            results.append({
                "row_id": r["row_id"], "text": r["text"], "gold": r["gold"],
                "prediction": pred, "valid_json": valid,
                "latency_s": latency if pred is not None else 0.0,
            })
        write_jsonl(out_path, results)
        print(f"écrit -> {out_path}")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

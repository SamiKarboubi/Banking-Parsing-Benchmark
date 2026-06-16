"""Baseline NuExtract 2.0 (texte seul) : extraction par template à enums + post-processing.

API réelle NuExtract 2.0 (base Qwen2-VL) : AutoProcessor + apply_chat_template(messages, template=...).
Le template encode le MONDE FERMÉ via des enums (NuExtract : ["a","b"] = choix exclusif,
[["a","b"]] = multi-label, "verbatim-string" = texte libre) → on « donne la liste » au modèle.
"""

import gc
import json
import time

import torch

from closed_world import KPIS, PRODUITS
from common import (empty_prediction, finalize_baseline, normalize_direction,
                    normalize_unit, to_float, type_from_unit, write_jsonl)
from synonym_lookup import match

MODEL_ID = "numind/NuExtract-2.0-2B"

# Enums = monde fermé donné au modèle. entity reste libre ("verbatim-string") pour capter les
# termes hors-périmètre (→ unknown_terms) ; value est typée "number" (sinon NuExtract renvoie null).
TEMPLATE = {
    "product": PRODUITS,                                  # enum exclusif
    "drivers": [{
        "entity": "verbatim-string",
        "value": "number",
        "unit": ["bps", "%"],                             # enum exclusif
        "direction": ["increase", "decrease"],            # enum exclusif
    }],
    "targets": [KPIS],                                    # multi-label enum
}
_TEMPLATE_STR = json.dumps(TEMPLATE, ensure_ascii=False)


def postprocess(extracted: dict) -> dict:
    """Mappe la sortie flat NuExtract vers le schéma du benchmark."""
    pred = empty_prediction()
    unknown = []

    prod_raw = (extracted.get("product") or "")
    prod_raw = prod_raw.strip() if isinstance(prod_raw, str) else ""
    if prod_raw:
        canon = match(prod_raw)
        if canon:
            pred["product_id"] = canon
        else:
            unknown.append(prod_raw.lower())

    for d in extracted.get("drivers") or []:
        ent_raw = (d.get("entity") or "").strip()
        if not ent_raw:
            continue
        key = match(ent_raw)
        if key is None:
            unknown.append(ent_raw.lower())
            continue
        unit = normalize_unit(d.get("unit"))
        pred["drivers"].append({
            "entity_key": key,
            "change": {
                "type": type_from_unit(unit),
                "value": to_float(d.get("value")),
                "unit": unit,
                "direction": normalize_direction(d.get("direction")),
            },
        })

    for t in extracted.get("targets") or []:
        raw = t if isinstance(t, str) else t.get("entity", "")
        key = match(raw)
        if key:
            pred["targets"].append(key)
        elif raw:
            unknown.append(str(raw).lower())

    pred["targets"] = sorted(set(pred["targets"]))
    pred["unknown_terms"] = sorted(set(unknown))
    return finalize_baseline(pred)


def _load():
    """NuExtract 2.0 a une base Qwen2-VL : on tente les classes image-text puis CausalLM."""
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    last_err = None
    for cls_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText",
                     "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                MODEL_ID, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto",
            )
            return processor, model
        except Exception as e:  # config non reconnue par cette classe → suivante
            last_err = e
    raise RuntimeError(f"Chargement NuExtract impossible : {last_err}")


def _generate(processor, model, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, template=_TEMPLATE_STR, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor.tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run(rows, out_path):
    processor, model = _load()
    try:
        results = []
        for r in rows:
            t0 = time.perf_counter()
            try:
                raw = _generate(processor, model, r["text"])
                extracted = json.loads(raw)
                pred = postprocess(extracted)
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

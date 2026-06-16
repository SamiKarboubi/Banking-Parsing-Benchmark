"""Smoke test : 3 exemples × (Qwen3-0.6B contraint, NuExtract, GLiNER).

Vérifie que chaque pipeline produit une prédiction exploitable. Imprime PASS/FAIL.
Doit passer à 100% avant de considérer le projet prêt.
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

from schema import validate  # noqa: E402

DATA = os.path.join(ROOT, "data", "dataset.csv")
N = 3


def load_rows():
    df = pd.read_csv(DATA, dtype=str).head(N)
    return [
        {"row_id": int(i), "text": r["text"], "gold": json.loads(r["json"])}
        for i, r in df.iterrows()
    ]


def check_llm(rows):
    """Pipeline contraint complet sur un petit modèle Qwen3."""
    import inference
    from prompt_builder import build_messages
    from models import MODELS

    llm = inference.load_model(MODELS["Qwen3-0.6B"])
    try:
        results = inference.generate(llm, [build_messages(r["text"]) for r in rows])
    finally:
        inference.free_model(llm)
    # valid_json => conforme à l'union discriminée
    return all(valid and pred is not None for pred, valid, _ in results), results


def check_baseline(module_name, out_name, rows):
    import importlib
    mod = importlib.import_module(module_name)
    out_path = os.path.join(ROOT, "output", out_name)
    mod.run(rows, out_path)
    preds = [json.loads(l)["prediction"] for l in open(out_path, encoding="utf-8")]
    # Baselines : intent/mode null par construction → on valide juste la structure flat exploitable.
    ok = all(p is not None and isinstance(p.get("drivers"), list) for p in preds)
    return ok, preds


def main():
    rows = load_rows()
    report = {}

    print("\n[1/3] Qwen3-0.6B (décodage contraint)…")
    try:
        ok, _ = check_llm(rows)
        report["Qwen3-0.6B"] = ok
    except Exception as e:
        report["Qwen3-0.6B"] = False
        print(f"  erreur: {e}")

    print("\n[2/3] NuExtract-2.0…")
    try:
        ok, _ = check_baseline("adapter_nuextract", "NuExtract-2.0.jsonl", rows)
        report["NuExtract-2.0"] = ok
    except Exception as e:
        report["NuExtract-2.0"] = False
        print(f"  erreur: {e}")

    print("\n[3/3] GLiNER-large…")
    try:
        ok, _ = check_baseline("adapter_gliner", "GLiNER-large.jsonl", rows)
        report["GLiNER-large"] = ok
    except Exception as e:
        report["GLiNER-large"] = False
        print(f"  erreur: {e}")

    print("\n=== SMOKE TEST ===")
    for name, ok in report.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    allok = all(report.values())
    print(f"\nRÉSULTAT GLOBAL : {'PASS' if allok else 'FAIL'}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()

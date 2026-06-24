"""Lance chaque LLM généraliste sur le dataset, écrit output/<model>.jsonl."""

import argparse
import json
import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models import MODELS                          # noqa: E402
from prompt_builder import build_messages          # noqa: E402
import inference                                    # noqa: E402

DATA = os.path.join(ROOT, "data", "dataset.csv")
OUT_DIR = os.path.join(ROOT, "output")


def gpu_check():
    assert torch.cuda.is_available(), "GPU non détecté — arrêt."
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {torch.cuda.get_device_name(0)}, VRAM : {props.total_memory / 1e9:.1f} GB")


def load_dataset():
    df = pd.read_csv(DATA, dtype=str)
    rows = []
    for i, row in df.iterrows():
        rows.append({"row_id": int(i), "text": row["text"], "gold": json.loads(row["json"])})
    return rows


def _purge_hf_cache(hf_id):
    """Supprime les poids d'un modèle du cache HF (si BENCH_PURGE_CACHE=1).

    Permet de tenir sur un petit disque : le pic ≈ le plus gros modèle au lieu de la somme.
    """
    if os.environ.get("BENCH_PURGE_CACHE") != "1":
        return
    import shutil
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    path = os.path.join(HF_HUB_CACHE, "models--" + hf_id.replace("/", "--"))
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f"cache purgé : {path}")


def run_model(display_name, hf_id, rows):
    print(f"\n=== {display_name} ({hf_id}) ===")
    llm = inference.load_model(hf_id)
    try:
        messages_list = [build_messages(r["text"]) for r in rows]
        results = inference.generate(llm, messages_list)
    finally:
        inference.free_model(llm)
        _purge_hf_cache(hf_id)

    out_path = os.path.join(OUT_DIR, f"{display_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r, (pred, valid, latency) in zip(rows, results):
            f.write(json.dumps({
                "row_id": r["row_id"], "text": r["text"], "gold": r["gold"],
                "prediction": pred, "valid_json": valid,
                "latency_s": latency if pred is not None else 0.0,
            }, ensure_ascii=False) + "\n")
    print(f"écrit -> {out_path}")


def run(selected=None):
    """Lance les LLM (tous par défaut). Réutilisé par scripts/run_all.py."""
    gpu_check()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_dataset()
    for name in (selected or list(MODELS)):
        run_model(name, MODELS[name], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="nom d'un seul modèle (défaut : tous)")
    args = ap.parse_args()
    run([args.model] if args.model else None)


if __name__ == "__main__":
    main()

"""Exécute UN seul run (un modèle + une précision) puis se termine.

Lancé en sous-processus par run_all.py. À la sortie du process, la VRAM est intégralement
rendue par l'OS (filet de sécurité au-delà du free_model explicite). Écrit
benchmark_quant/output/<run>.jsonl.

Usage : python benchmark_quant/worker.py <nom_du_run>
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from prompt_builder import build_messages   # noqa: E402  (src partagé)
from runs import BY_NAME                     # noqa: E402
import backends                              # noqa: E402

DATA = os.path.join(ROOT, "data", "dataset.csv")
OUT_DIR = os.path.join(HERE, "output")


def load_rows():
    df = pd.read_csv(DATA, dtype=str)
    return [{"row_id": int(i), "text": r["text"], "gold": json.loads(r["json"])}
            for i, r in df.iterrows()]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BY_NAME:
        sys.exit(f"usage: worker.py <run>  (parmi: {', '.join(BY_NAME)})")
    run = BY_NAME[sys.argv[1]]

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    print(f"=== {run.name} [{run.precision}/{run.backend}] sur {len(rows)} exemples ===", flush=True)

    messages_list = [build_messages(r["text"]) for r in rows]
    results = backends.run_generation(run, messages_list)

    out_path = os.path.join(OUT_DIR, f"{run.name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r, (pred, valid, latency) in zip(rows, results):
            f.write(json.dumps({
                "row_id": r["row_id"], "text": r["text"], "gold": r["gold"],
                "prediction": pred, "valid_json": valid,
                "latency_s": latency if pred is not None else 0.0,
                "precision": run.precision, "backend": run.backend,
            }, ensure_ascii=False) + "\n")
    print(f"écrit -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

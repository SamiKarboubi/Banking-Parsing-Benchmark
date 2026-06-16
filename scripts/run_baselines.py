"""Lance les baselines (NuExtract puis GLiNER), GPU libéré entre les deux.

Sortie identique à run_benchmark.py : output/<model>.jsonl.
"""

import argparse
import gc
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

import adapter_gliner       # noqa: E402
import adapter_nuextract    # noqa: E402

DATA = os.path.join(ROOT, "data", "dataset.csv")
OUT_DIR = os.path.join(ROOT, "output")


def load_dataset():
    df = pd.read_csv(DATA, dtype=str)
    return [
        {"row_id": int(i), "text": row["text"], "gold": json.loads(row["json"])}
        for i, row in df.iterrows()
    ]


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run(which="all"):
    """Lance les baselines, GPU libéré entre les deux. Réutilisé par scripts/run_all.py."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_dataset()

    if which in ("nuextract", "all"):
        print("\n=== NuExtract-2.0 ===")
        adapter_nuextract.run(rows, os.path.join(OUT_DIR, "NuExtract-2.0.jsonl"))
        _free_gpu()

    if which in ("gliner", "all"):
        print("\n=== GLiNER-large ===")
        adapter_gliner.run(rows, os.path.join(OUT_DIR, "GLiNER-large.jsonl"))
        _free_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["nuextract", "gliner", "all"], default="all")
    args = ap.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()

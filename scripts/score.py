"""Lit output/*.jsonl, calcule les métriques agrégées + médiane/p95 latence."""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scoring import score_example  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output")
RES_DIR = os.path.join(ROOT, "results")

# Métriques numériques moyennées (None ignorés).
MEAN_FIELDS = [
    "intent_correct", "other_recall", "mode_correct", "product_id_correct",
    "drivers_precision", "drivers_recall", "drivers_change_accuracy",
    "targets_precision", "targets_recall",
    "lever_correct", "goal_correct", "target_correct",
    "unknown_terms_precision", "unknown_terms_recall",
]


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def model_type_of(name):
    low = name.lower()
    if "nuextract" in low:
        return "nuextract"
    if "gliner" in low:
        return "gliner"
    return "llm_generalist"


def score_model(path):
    name = os.path.splitext(os.path.basename(path))[0]
    per_example, valids, latencies = [], [], []
    details = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            s = score_example(rec["prediction"], rec["gold"])
            s["schema_valid"] = bool(rec["valid_json"]) and rec["prediction"] is not None
            per_example.append(s)
            valids.append(s["schema_valid"])
            latencies.append(rec["latency_s"])
            details.append({"row_id": rec["row_id"], "text": rec["text"], **s})

    with open(os.path.join(RES_DIR, f"details_{name}.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    summary = {"model": name, "model_type": model_type_of(name)}
    for field in MEAN_FIELDS:
        summary[field] = _mean([s.get(field) for s in per_example])
    summary["exact_match_rate"] = _mean([s["exact_match"] for s in per_example])
    summary["schema_valid_rate"] = _mean(valids)
    summary["latency_median_s"] = float(np.median(latencies)) if latencies else None
    summary["latency_p95_s"] = float(np.percentile(latencies, 95)) if latencies else None
    return summary


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl")))
    if not paths:
        print("Aucun output trouvé. Lance d'abord run_benchmark.py.")
        return

    rows = [score_model(p) for p in paths]
    df = pd.DataFrame(rows).sort_values("exact_match_rate", ascending=False)
    df.to_csv(os.path.join(RES_DIR, "summary.csv"), index=False)

    cols = ["model", "exact_match_rate", "schema_valid_rate", "intent_correct",
            "mode_correct", "product_id_correct", "drivers_recall", "targets_recall",
            "latency_median_s", "latency_p95_s"]
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)

    llm = df[df["model_type"] == "llm_generalist"]
    base = df[df["model_type"] != "llm_generalist"]

    print("\n=== LLM GÉNÉRALISTES (trié par exact_match_rate) ===")
    print(llm[cols].to_string(index=False, float_format=fmt))

    if not base.empty:
        print("\n=== BASELINES (extraction spécialisée) ===")
        print("NB : intent & mode sont HORS-PORTEE des baselines (toujours null) -> ces colonnes")
        print("     sont structurellement a 0 ; comparer surtout product_id / drivers / targets.")
        print(base[cols].to_string(index=False, float_format=fmt))

    print(f"\nDétails complets -> {os.path.join(RES_DIR, 'summary.csv')}")


if __name__ == "__main__":
    main()

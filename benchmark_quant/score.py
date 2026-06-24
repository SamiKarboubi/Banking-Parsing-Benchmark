"""Score benchmark_quant/output/*.jsonl -> results/summary.csv + details + rapport_quant.md.

Le rapport est généré automatiquement à partir des VRAIES sorties, des labels et de la data :
classement, impact de la quantification (BF16 vs Q8 vs Q4 par modèle de base), et une fiche
par run (forces/faiblesses chiffrées + échantillon d'erreurs concrètes). Latence médiane incluse.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from scoring import score_example      # noqa: E402  (src partagé)
from runs import RUNS, BY_NAME         # noqa: E402

OUT_DIR = os.path.join(HERE, "output")
RES_DIR = os.path.join(HERE, "results")

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


def _type_of(gold):
    if gold.get("intent") == "other":
        return "other"
    return gold.get("mode")           # "forward" | "inverse"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_model(path):
    name = os.path.splitext(os.path.basename(path))[0]
    run = BY_NAME.get(name)
    per_example, valids, latencies, details = [], [], [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            s = score_example(rec["prediction"], rec["gold"])
            s["schema_valid"] = bool(rec["valid_json"]) and rec["prediction"] is not None
            per_example.append(s)
            valids.append(s["schema_valid"])
            latencies.append(rec["latency_s"])
            details.append({
                "row_id": rec["row_id"], "text": rec["text"], "type": _type_of(rec["gold"]),
                "gold": rec["gold"], "prediction": rec["prediction"],
                "exact_match": s["exact_match"], "schema_valid": s["schema_valid"],
            })

    with open(os.path.join(RES_DIR, f"details_{name}.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    by_type = {}
    for t in ("forward", "inverse", "other"):
        rows = [d for d in details if d["type"] == t]
        hits = sum(d["exact_match"] for d in rows)
        by_type[t] = (hits, len(rows))

    summary = {
        "model": name,
        "precision": run.precision if run else "?",
        "backend": run.backend if run else "?",
    }
    for field in MEAN_FIELDS:
        summary[field] = _mean([s.get(field) for s in per_example])
    summary["exact_match_rate"] = _mean([s["exact_match"] for s in per_example])
    summary["schema_valid_rate"] = _mean(valids)
    summary["latency_median_s"] = float(np.median(latencies)) if latencies else None
    summary["latency_p95_s"] = float(np.percentile(latencies, 95)) if latencies else None
    summary["n"] = len(details)
    for t, (h, tot) in by_type.items():
        summary[f"em_{t}"] = f"{h}/{tot}"
    return summary, details


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else ("-" if x is None else str(x))


def _diff_fields(gold, pred):
    """Renvoie les aspects de haut niveau qui diffèrent (gold vs pred), en compact."""
    pred = pred or {}
    out = []

    def add(label, g, p):
        if g != p:
            out.append(f"{label}: attendu={json.dumps(g, ensure_ascii=False)} / "
                       f"obtenu={json.dumps(p, ensure_ascii=False)}")

    if gold.get("intent") != pred.get("intent"):
        out.append(f"intent: attendu={gold.get('intent')} / obtenu={pred.get('intent')}")
        return out
    if gold.get("intent") == "other":
        return out
    add("mode", gold.get("mode"), pred.get("mode"))
    add("product_id", gold.get("product_id"), pred.get("product_id"))
    if gold.get("mode") == "forward":
        gd = {d["entity_key"]: d["change"] for d in gold.get("drivers") or []}
        pd_ = {d["entity_key"]: d.get("change") for d in (pred.get("drivers") or [])}
        if gd != pd_:
            add("drivers", gd, pd_)
        add("targets", sorted(gold.get("targets") or []), sorted(pred.get("targets") or []))
    if gold.get("mode") == "inverse":
        add("lever", (gold.get("lever") or {}).get("entity_key"),
            (pred.get("lever") or {}).get("entity_key"))
        add("goal", gold.get("goal"), pred.get("goal"))
        add("target", gold.get("target"), pred.get("target"))
    add("unknown_terms", sorted(gold.get("unknown_terms") or []),
        sorted(pred.get("unknown_terms") or []))
    return out


def _error_samples(details, k=4):
    samples = []
    for d in details:
        if d["exact_match"]:
            continue
        diffs = _diff_fields(d["gold"], d["prediction"])
        if diffs:
            samples.append((d["text"], diffs))
        if len(samples) >= k:
            break
    return samples


def _quant_impact(summaries):
    """Regroupe par modèle de base pour comparer les précisions disponibles."""
    groups = {
        "gemma-3-12b-it": ["gemma-3-12b-it", "gemma-3-12b-it-Q8"],
        "gemma-4-12B-it": ["gemma-4-12B-it", "gemma-4-12B-it-Q8"],
        "Mistral-Small-24B": ["Mistral-Small-24B-Q8", "Mistral-Small-24B-Q4"],
        "gemma-4-26B-A4B-it": ["gemma-4-26B-A4B-it-Q8", "gemma-4-26B-A4B-it-Q4"],
        "Qwen3.6-35B-A3B": ["Qwen3.6-35B-A3B-Q8", "Qwen3.6-35B-A3B-Q4"],
    }
    by_name = {s["model"]: s for s in summaries}
    lines = []
    for base, names in groups.items():
        present = [by_name[n] for n in names if n in by_name]
        if len(present) < 2:
            continue
        lines.append(f"### {base}")
        lines.append("")
        lines.append("| Variante | Précision | Exact match | Latence méd. |")
        lines.append("|---|:---:|:---:|:---:|")
        for s in present:
            lines.append(f"| {s['model']} | {s['precision']} | {_fmt(s['exact_match_rate'])} | "
                         f"{_fmt(s['latency_median_s'])} s |")
        ems = [s["exact_match_rate"] for s in present if s["exact_match_rate"] is not None]
        if len(ems) >= 2:
            delta = (ems[-1] - ems[0]) * 100   # variation dans le sens de la flèche (négatif = perte)
            lines.append("")
            lines.append(f"Variation exact match {present[0]['precision']} -> {present[-1]['precision']} : "
                         f"{delta:+.1f} points.")
        lines.append("")
    return lines


def build_report(summaries, details_by_name, df):
    L = []
    L.append("# Rapport — Benchmark de quantification (BF16 vs Q8 vs Q4)")
    L.append("")
    L.append(f"Évaluation de {len(summaries)} runs (modèle x précision) sur la tâche de semantic parsing bancaire "
             "(phrase FR -> JSON, monde fermé), sur le dataset augmenté.")
    L.append("")
    n = df["n"].iloc[0] if "n" in df and len(df) else "?"
    L.append(f"- **Exemples** : {n} (data partagée avec le benchmark principal, augmentée).")
    L.append("- **Métrique** : exact_match (égalité profonde ; listes comparées comme ensembles).")
    L.append("- **Moteurs** : BF16 -> vLLM ; Q8/Q4 (GGUF) -> llama.cpp. Décodage JSON contraint, "
             "température 0, prompt identique partout.")
    L.append("- **Latence** : médiane par requête. Note : vLLM (BF16) et llama.cpp (GGUF) sont des "
             "moteurs différents ; la latence reflète le couple modèle+moteur de déploiement réel, "
             "pas seulement la précision.")
    L.append("")

    # --- classement ---
    L.append("## 1. Classement général")
    L.append("")
    L.append("| Rang | Run | Précision | Exact match | Forward | Inverse | Other | Schéma valide | Latence méd. |")
    L.append("|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    ranked = df.sort_values("exact_match_rate", ascending=False).reset_index(drop=True)
    for i, r in ranked.iterrows():
        L.append(f"| {i + 1} | {r['model']} | {r['precision']} | {_fmt(r['exact_match_rate'])} | "
                 f"{r['em_forward']} | {r['em_inverse']} | {r['em_other']} | "
                 f"{_fmt(r['schema_valid_rate'])} | {_fmt(r['latency_median_s'])} s |")
    L.append("")

    # --- impact quantification ---
    L.append("## 2. Impact de la quantification (par modèle de base)")
    L.append("")
    impact = _quant_impact(summaries)
    if impact:
        L.extend(impact)
    else:
        L.append("Données insuffisantes pour comparer les précisions (runs manquants).")
        L.append("")

    # --- fiches par run ---
    L.append("## 3. Fiche par run")
    L.append("")
    by_name = {s["model"]: s for s in summaries}
    for r in ranked.itertuples():
        s = by_name[r.model]
        L.append(f"### {s['model']} — {s['precision']} ({s['backend']}) — exact_match {_fmt(s['exact_match_rate'])}")
        L.append("")
        L.append(f"- Par type : forward {s['em_forward']}, inverse {s['em_inverse']}, other {s['em_other']}.")
        L.append(f"- Schéma valide {_fmt(s['schema_valid_rate'])} ; latence médiane {_fmt(s['latency_median_s'])} s "
                 f"(p95 {_fmt(s['latency_p95_s'])} s).")
        L.append(f"- product_id {_fmt(s['product_id_correct'])} ; drivers P/R "
                 f"{_fmt(s['drivers_precision'])}/{_fmt(s['drivers_recall'])} ; "
                 f"change_acc {_fmt(s['drivers_change_accuracy'])} ; "
                 f"targets P/R {_fmt(s['targets_precision'])}/{_fmt(s['targets_recall'])}.")
        L.append(f"- Inverse : lever {_fmt(s['lever_correct'])}, goal {_fmt(s['goal_correct'])}, "
                 f"target {_fmt(s['target_correct'])}. unknown_terms P/R "
                 f"{_fmt(s['unknown_terms_precision'])}/{_fmt(s['unknown_terms_recall'])}.")
        samples = _error_samples(details_by_name[s["model"]])
        if samples:
            L.append("- Échantillon d'erreurs :")
            for text, diffs in samples:
                L.append(f"  - « {text} »")
                for d in diffs:
                    L.append(f"    - {d}")
        else:
            L.append("- Aucune erreur (exact_match parfait).")
        L.append("")

    return "\n".join(L)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl")))
    if not paths:
        print("Aucun output. Lance d'abord run_all.py.")
        return

    summaries, details_by_name = [], {}
    for p in paths:
        s, details = score_model(p)
        summaries.append(s)
        details_by_name[s["model"]] = details

    df = pd.DataFrame(summaries).sort_values("exact_match_rate", ascending=False)
    df.to_csv(os.path.join(RES_DIR, "summary.csv"), index=False)

    cols = ["model", "precision", "exact_match_rate", "em_forward", "em_inverse", "em_other",
            "schema_valid_rate", "latency_median_s"]
    print("\n=== CLASSEMENT (exact_match_rate) ===")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    report = build_report(summaries, details_by_name, df)
    report_path = os.path.join(RES_DIR, "rapport_quant.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nsummary.csv + rapport_quant.md -> {RES_DIR}")


if __name__ == "__main__":
    main()

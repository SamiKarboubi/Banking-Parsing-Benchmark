"""Orchestrateur du benchmark de quantification (14 runs).

Pour chaque run : lance un SOUS-PROCESSUS dédié (worker.py), attend sa fin, puis SUPPRIME
les poids du disque. Un process par modèle garantit :
  - libération totale de la VRAM entre chaque modèle (le process meurt) ;
  - pas de cohabitation vLLM / llama.cpp dans le même interpréteur.
Pic disque = le plus gros modèle (pas la somme). Lance le scoring à la fin.

Usage :
  python benchmark_quant/run_all.py                 # les 14 runs + scoring
  python benchmark_quant/run_all.py --only Qwen3-8B Qwen3.6-35B-A3B-Q4
  KEEP_WEIGHTS=1 python benchmark_quant/run_all.py   # ne pas purger le disque (debug)
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runs import RUNS, BY_NAME      # noqa: E402

WORKER = os.path.join(HERE, "worker.py")


def _cache_root():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return HF_HUB_CACHE
    except Exception:
        home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        return os.path.join(home, "hub")


def purge_weights(run):
    """Supprime du cache HF les poids du run (libère le disque pour le suivant)."""
    if os.environ.get("KEEP_WEIGHTS") == "1":
        return
    root = _cache_root()
    for repo in run.weight_repos():
        path = os.path.join(root, "models--" + repo.replace("/", "--"))
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"  disque purgé : {path}", flush=True)


def run_one(run):
    print(f"\n########## {run.name} ##########", flush=True)
    proc = subprocess.run([sys.executable, WORKER, run.name], env=os.environ.copy())
    ok = proc.returncode == 0
    if not ok:
        print(f"  ECHEC ({run.name}) code={proc.returncode} — on continue.", flush=True)
    purge_weights(run)               # purge même en cas d'échec (téléchargement partiel)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="RUN", help="sous-ensemble de runs à lancer")
    ap.add_argument("--no-score", action="store_true", help="ne pas lancer le scoring à la fin")
    args = ap.parse_args()

    selected = RUNS
    if args.only:
        unknown = [n for n in args.only if n not in BY_NAME]
        if unknown:
            sys.exit(f"runs inconnus : {unknown}\ndisponibles : {', '.join(BY_NAME)}")
        selected = [BY_NAME[n] for n in args.only]

    results = {r.name: run_one(r) for r in selected}

    ok = [n for n, v in results.items() if v]
    ko = [n for n, v in results.items() if not v]
    print(f"\n=== Terminé : {len(ok)}/{len(results)} runs OK ===", flush=True)
    if ko:
        print(f"Echecs : {', '.join(ko)}", flush=True)

    if not args.no_score:
        print("\n=== Scoring ===", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "score.py")], env=os.environ.copy())


if __name__ == "__main__":
    main()

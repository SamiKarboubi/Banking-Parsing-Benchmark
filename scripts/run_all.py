"""Point d'entrée UNIQUE : lance TOUTES les inférences (LLM + baselines).

    python scripts/run_all.py

Chaque modèle est chargé un par un, le GPU est libéré entre chaque. Les prédictions
sont écrites dans output/<modèle>.jsonl. Le scoring se fait ensuite avec scripts/score.py.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_baselines   # noqa: E402
import run_benchmark   # noqa: E402


def main():
    print("########## LLM GÉNÉRALISTES ##########")
    run_benchmark.run()
    print("\n########## BASELINES ##########")
    run_baselines.run("all")
    print("\nTerminé. Lance le scoring :  python scripts/score.py")


if __name__ == "__main__":
    main()

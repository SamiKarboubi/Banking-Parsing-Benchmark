"""Les 15 runs du benchmark de quantification (modèle x précision).

NB : la consigne annonçait « 14 runs » mais en listait 15 (7 BF16 + 5 Q8 + 3 Q4) ; on
benchmarke les 15 explicitement cités.

IDs vérifiés sur HuggingFace le 2026-06-24. Deux familles de moteurs :
  - BF16 natif      -> vLLM (décodage structuré mûr, optimal pour les poids HF pleine précision)
  - Q8 / Q4 (GGUF)  -> llama.cpp (support GGUF + MoE de référence, grammaire JSON contrainte)

Chaque run tourne dans un SOUS-PROCESSUS dédié (cf. run_all.py) : à sa fin, le process meurt
=> toute la VRAM est rendue, puis les poids sont supprimés du disque. Le pic disque/VRAM = le
plus gros modèle, jamais la somme.

GGUF non précisés par l'utilisateur (Mistral-Small-24B, gemma-4-26B-A4B) : dépôts unsloth,
convention `unsloth/<model>-GGUF`, existence confirmée. `quant` est la sous-chaîne du nom de
fichier .gguf à télécharger (gère aussi les variantes unsloth dynamiques, ex. `UD-Q4_K_M`).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Run:
    name: str                       # nom d'affichage (= nom du fichier de sortie)
    backend: str                    # "vllm" (BF16) | "gguf" (llama.cpp)
    precision: str                  # "BF16" | "Q8" | "Q4"
    model: str                      # repo HF : poids (vllm) OU tokenizer/chat-template (gguf)
    gguf_repo: Optional[str] = None  # dépôt GGUF (gguf uniquement)
    quant: Optional[str] = None      # sous-chaîne du fichier .gguf (ex. "Q8_0", "Q4_K_M")
    gated: bool = False              # licence à accepter + token HF

    def weight_repos(self) -> List[str]:
        """Dépôts dont les fichiers doivent être purgés du disque après le run."""
        return [self.gguf_repo, self.model] if self.backend == "gguf" else [self.model]


RUNS: List[Run] = [
    # ---- BF16 natifs (vLLM) ----
    Run("gemma-4-E2B-it",            "vllm", "BF16", "google/gemma-4-E2B-it", gated=True),
    Run("gemma-4-E4B-it",            "vllm", "BF16", "google/gemma-4-E4B-it", gated=True),
    Run("gemma-4-12B-it",            "vllm", "BF16", "google/gemma-4-12B-it", gated=True),
    Run("gemma-3-12b-it",            "vllm", "BF16", "google/gemma-3-12b-it", gated=True),
    Run("Qwen3-4B",                  "vllm", "BF16", "Qwen/Qwen3-4B"),
    Run("Qwen3-8B",                  "vllm", "BF16", "Qwen/Qwen3-8B"),
    Run("Ministral-8B",              "vllm", "BF16", "mistralai/Ministral-8B-Instruct-2410", gated=True),

    # ---- Q8 (GGUF, llama.cpp) ----
    Run("gemma-3-12b-it-Q8",         "gguf", "Q8", "google/gemma-3-12b-it",
        gguf_repo="unsloth/gemma-3-12b-it-GGUF", quant="Q8_0", gated=True),
    Run("gemma-4-12B-it-Q8",         "gguf", "Q8", "google/gemma-4-12B-it",
        gguf_repo="unsloth/gemma-4-12B-it-GGUF", quant="Q8_0", gated=True),
    Run("Mistral-Small-24B-Q8",      "gguf", "Q8", "mistralai/Mistral-Small-24B-Instruct-2501",
        gguf_repo="unsloth/Mistral-Small-24B-Instruct-2501-GGUF", quant="Q8_0"),
    Run("gemma-4-26B-A4B-it-Q8",     "gguf", "Q8", "google/gemma-4-26B-A4B-it",
        gguf_repo="unsloth/gemma-4-26B-A4B-it-GGUF", quant="Q8_0", gated=True),
    Run("Qwen3.6-35B-A3B-Q8",        "gguf", "Q8", "Qwen/Qwen3.6-35B-A3B",
        gguf_repo="unsloth/Qwen3.6-35B-A3B-GGUF", quant="Q8_0"),

    # ---- Q4 (GGUF, llama.cpp) ----
    Run("Mistral-Small-24B-Q4",      "gguf", "Q4", "mistralai/Mistral-Small-24B-Instruct-2501",
        gguf_repo="unsloth/Mistral-Small-24B-Instruct-2501-GGUF", quant="Q4_K_M"),
    Run("gemma-4-26B-A4B-it-Q4",     "gguf", "Q4", "google/gemma-4-26B-A4B-it",
        gguf_repo="unsloth/gemma-4-26B-A4B-it-GGUF", quant="Q4_K_M", gated=True),
    Run("Qwen3.6-35B-A3B-Q4",        "gguf", "Q4", "Qwen/Qwen3.6-35B-A3B",
        gguf_repo="unsloth/Qwen3.6-35B-A3B-GGUF", quant="Q4_K_M"),
]

BY_NAME = {r.name: r for r in RUNS}

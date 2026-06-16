"""Configuration des modèles à évaluer.

IDs vérifiés sur https://huggingface.co le 2026-06-15 (existence + accès public/gated).
NB : la série "Qwen3.5" n'existe pas sur HF — la bonne famille est Qwen3 (mise à jour 2507).
     Ministral-3B n'est pas publié sur HF (API only) → on retient Ministral-8B-Instruct-2410.
"""

# LLM généralistes (génération contrainte vLLM). IDs vérifiés sur HF le 2026-06-16.
MODELS = {
    # display_name      : hf_model_id                                  # accès | taille
    "Qwen3-8B":          "Qwen/Qwen3-8B",                              # public | 8.2B
    "Qwen3-4B":          "Qwen/Qwen3-4B",                              # public | 4B
    "Qwen3-1.7B":        "Qwen/Qwen3-1.7B",                            # public | 1.7B
    "Qwen3-0.6B":        "Qwen/Qwen3-0.6B",                            # public | 0.6B (borne basse)
    "Gemma3-12B":        "google/gemma-3-12b-it",                      # gated (licence Google) | 12B
    "Gemma3-4B":         "google/gemma-3-4b-it",                       # gated (licence Google) | 4B
    "Phi-4-mini":        "microsoft/Phi-4-mini-instruct",             # public | 3.8B
    "Ministral-8B":      "mistralai/Ministral-8B-Instruct-2410",      # gated (Mistral Research) | 8B
    "Mistral-Small-24B": "mistralai/Mistral-Small-24B-Instruct-2501",  # public | 24B (plafond)
    "SmolLM3-3B":        "HuggingFaceTB/SmolLM3-3B",                  # public | 3B
}

# Baselines spécialisées (extraction), lancées par scripts/run_baselines.py.
BASELINES = {
    # NuExtract 2.0 est une famille multimodale (base Qwen2-VL) ; on l'utilise en TEXTE seul.
    "NuExtract-2.0": "numind/NuExtract-2.0-2B",                      # public | 2B (texte ici)
    # GLiNER multilingue (français supporté), NER zero-shot.
    "GLiNER-large":  "urchade/gliner_multi-v2.1",                    # public | 0.2B
}

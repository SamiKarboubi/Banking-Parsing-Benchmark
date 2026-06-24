# Benchmark de quantification — BF16 vs Q8 vs Q4

Banc **séparé** du benchmark principal (`scripts/`, `output/`, `results/` à la racine). Il
compare **15 runs** (modèle x précision — la consigne disait « 14 » mais en listait 15) sur la
**même tâche** de semantic parsing bancaire
(phrase FR -> JSON, monde fermé) et sur le **dataset augmenté** partagé (`../data/dataset.csv`).
Il réutilise le monde fermé, le prompt, les few-shot et le scoring de `../src/` : seuls les
poids (précision) et le moteur changent.

## Les 15 runs

| Précision | Runs |
|---|---|
| **BF16** (vLLM) | gemma-4-E2B-it, gemma-4-E4B-it, gemma-4-12B-it, gemma-3-12b-it, Qwen3-4B, Qwen3-8B, Ministral-8B |
| **Q8** (GGUF / llama.cpp) | gemma-3-12b-it, gemma-4-12B-it, Mistral-Small-24B, gemma-4-26B-A4B-it, Qwen3.6-35B-A3B |
| **Q4** (GGUF / llama.cpp) | Mistral-Small-24B, gemma-4-26B-A4B-it, Qwen3.6-35B-A3B |

## Choix du moteur (pourquoi deux)

C'est le choix **optimal** pour ce banc mixte, et il est délibéré :

- **BF16 -> vLLM.** Décodage structuré mûr (le banc principal l'utilise déjà), débit élevé sur
  poids HF pleine précision. Idéal pour les modèles servis en BF16.
- **Q8 / Q4 (GGUF) -> llama.cpp.** Les poids demandés sont des GGUF unsloth. Le support GGUF de
  vLLM est expérimental et fragile sur les architectures **MoE** (gemma-4-26B-A4B, Qwen3.6-35B-A3B,
  tous deux à experts) ; llama.cpp a le support GGUF + MoE de référence et tourne en plein offload
  GPU. La génération JSON reste **contrainte** (grammaire GBNF dérivée du même JSON Schema
  `FlatOutput`), donc un JSON toujours valide, comme côté vLLM.

Conséquence assumée : le **prompt, la grammaire JSON, la température (0) et le post-traitement
sont identiques** dans les deux moteurs ; seuls la précision des poids et le moteur de service
diffèrent — ce qui correspond au déploiement réel (on sert du BF16 en vLLM et du GGUF en llama.cpp).
La latence est donc à lire comme « modèle + moteur », pas seulement « précision » (noté dans le rapport).

## Isolation : VRAM rendue + disque purgé entre chaque modèle

`run_all.py` lance **un sous-processus par run** (`worker.py`). À la fin de chaque run :

1. le sous-processus se termine -> l'OS **rend toute la VRAM** (filet de sécurité au-delà du
   `free_model` / `llm.close()` explicite ; évite aussi toute cohabitation vLLM / llama.cpp) ;
2. les **poids sont supprimés du disque** (cache HF du dépôt, et du dépôt GGUF).

Donc **pic VRAM = pic disque = le plus gros modèle**, jamais la somme.

## Lancer sur RunPod

```bash
cd /workspace
# (cloner / uploader le projet, puis :)
cd "Benchmark 3"

pip install -r benchmark_quant/requirements.txt
# Roue CUDA pour llama.cpp (adapter cuXXX à la CUDA du pod) :
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Modèles gated (Gemma 3/4, Ministral, + tokenizers Mistral) : se connecter
huggingface-cli login        # ou: export HF_TOKEN=...
export HF_HUB_ENABLE_HF_TRANSFER=1   # téléchargements rapides

# Les 14 runs (VRAM + disque libérés entre chaque) PUIS le scoring + le rapport :
python benchmark_quant/run_all.py
```

Sous-ensemble / debug :

```bash
python benchmark_quant/run_all.py --only Qwen3-8B Qwen3.6-35B-A3B-Q4   # quelques runs
python benchmark_quant/worker.py gemma-4-12B-it                        # un seul run
python benchmark_quant/score.py                                        # (re)scorer
KEEP_WEIGHTS=1 python benchmark_quant/run_all.py                       # ne pas purger le disque
```

Récupérer les résultats :

```bash
zip -r resultats_quant.zip benchmark_quant/output benchmark_quant/results
```

## Matériel conseillé

- **VRAM : 80 Go** (H100/A100 80G) recommandé — fixé par le plus gros run, Qwen3.6-35B-A3B en Q8
  (~37 Go de poids + cache). Sur 48 Go (L40S/A6000) tout passe **sauf** les plus gros Q8 (offload
  partiel CPU possible mais lent). Les BF16 (max gemma-4-12B ~24 Go) passent partout.
- **Disque : ~80–100 Go** suffisent grâce à la purge entre runs (pic = plus gros modèle + cache HF
  + environnement). Sans purge il faudrait plusieurs centaines de Go : ne pas mettre `KEEP_WEIGHTS=1`
  sur petit disque.

## Sorties

- `benchmark_quant/output/<run>.jsonl` — 1 ligne/exemple (prédiction, gold, latence, précision).
- `benchmark_quant/results/summary.csv` — métriques par run (dont `latency_median_s`).
- `benchmark_quant/results/details_<run>.jsonl` — exact_match ligne par ligne.
- `benchmark_quant/results/rapport_quant.md` — **rapport auto** : classement, impact de la
  quantification par modèle de base (BF16/Q8/Q4 + deltas), fiche par run (forces/faiblesses
  chiffrées + échantillon d'erreurs concrètes), latence médiane.

## Notes

- IDs HuggingFace vérifiés le 2026-06-24. Dépôts GGUF non précisés (Mistral-Small-24B,
  gemma-4-26B-A4B) : convention unsloth `unsloth/<model>-GGUF`, existence confirmée. Le champ
  `quant` (`runs.py`) sélectionne le fichier `.gguf` par sous-chaîne (gère les variantes unsloth
  dynamiques type `UD-Q4_K_M`) ; ajuster si un dépôt n'expose pas exactement `Q8_0` / `Q4_K_M`.
- `enable_thinking` est neutralisé et, sous grammaire JSON, aucun bloc `<think>` ne peut être émis.

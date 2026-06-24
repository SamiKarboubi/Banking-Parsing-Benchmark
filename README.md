# Benchmark — Semantic Parsing bancaire (FR → JSON)

Évalue 8 petits LLM généralistes — plus 2 baselines d'extraction spécialisée (NuExtract 2.0,
GLiNER) — sur une tâche de *semantic parsing* : transformer une phrase en français en un JSON
structuré conforme à un **monde fermé** (produits, leviers, KPIs) d'un simulateur de rentabilité
bancaire. Les LLM génèrent sous **décodage contraint** (vLLM guided decoding) ; les baselines
extraient puis sont mappées vers le schéma via une table de synonymes.

## Structure

```
├── data/dataset.csv        # fourni (colonnes: text, json) — NE PAS régénérer
├── src/
│   ├── schema.py           # modèles Pydantic + dérivations (entity_type, needs_clarification)
│   ├── closed_world.py     # bloc de grounding + enums Python
│   ├── few_shot.py         # 9 exemples in-context (hors dataset)
│   ├── prompt_builder.py   # grounding + few-shot + question
│   ├── models.py           # IDs HuggingFace (LLM + baselines), vérifiés 2026-06-15
│   ├── inference.py        # vLLM + guided_json (schéma plat) + latence
│   └── scoring.py          # scoring champ par champ
├── baselines/
│   ├── synonym_lookup.py   # synonyme normalisé → clé canonique (tout le monde fermé)
│   ├── common.py           # normalisation unit/direction/type + I/O
│   ├── adapter_nuextract.py# NuExtract 2.0 (texte) : template + post-processing
│   └── adapter_gliner.py   # GLiNER : NER zero-shot + regroupement en drivers
├── scripts/
│   ├── run_benchmark.py    # LLM → output/<model>.jsonl
│   ├── run_baselines.py    # NuExtract + GLiNER → output/<model>.jsonl
│   ├── run_all.py          # POINT D'ENTRÉE UNIQUE : LLM + baselines
│   ├── score.py            # agrège -> results/summary.csv + tables (LLM / baselines)
│   └── smoke_test.py       # vérif rapide bout-en-bout (3 exemples)
├── output/                 # 1 .jsonl par modèle
└── results/                # summary.csv + details_<model>.jsonl
```

## Schéma de sortie

3 branches, union discriminée par `intent` + `mode` : **forward**, **inverse**, **other**.

### Conventions (dérivées en CODE, pas dans le JSON)

- **`entity_type`** n'est pas stocké : `schema.entity_type_of(key)` →
  `"source_financement"` si la clé ∈ {DAV, DAT, MARCHE, FP}, sinon `"driver"`.
- **Couplage variation de levier** : `unit "bps"` ⇒ `type "absolute"` ; `unit "%"` ⇒ `type "relative"`.
  ⚠ NON imposé au `goal` inverse, qui peut être un **niveau cible** en `%` (`type "absolute"`).
- **`targets`** = KPIs **explicitement** cités. Aucun ⇒ `[]` (les défauts métier sont hors LLM).
- **`product_id`** = `null` si aucun produit mentionné.
- **`unknown_terms`** = termes ressemblant à un levier/produit/KPI mais **absents** du monde fermé,
  stockés en minuscules ; jamais forcés sur une clé valide.
- **`needs_clarification`** dérivé en code (`schema.needs_clarification`) :
  `product_id == null` OU `drivers == []` (forward) OU champs inverse manquants.

## Décodage contraint — choix retenu

L'union discriminée sur **deux** champs (`intent` + `mode`) est mal supportée par les backends
de guided decoding. On contraint donc la génération avec le **schéma plat** `FlatOutput`
(tous les champs nullable), puis on **valide en post-traitement** contre l'union discriminée
(`schema.flat_to_canonical` + `schema.validate`). `valid_json` reflète cette double validation
(JSON parsable **ET** conforme Pydantic).

Paramètres : `temperature=0` (greedy, reproductible), reasoning/thinking désactivé
(`enable_thinking=False` pour Qwen3/SmolLM3 ; et sous décodage contraint JSON aucun bloc
`<think>` n'est de toute façon émis), `max_tokens=512`. Les modèles sont chargés **un par un**
(GPU libéré entre chaque via `gc` + `torch.cuda.empty_cache()`). Latence par requête
(`time.perf_counter`).

L'appel de décodage structuré est compatible **ancien et nouveau vLLM** : `StructuredOutputsParams`
(vLLM ≥ 0.10.2) avec repli sur `GuidedDecodingParams` (vLLM 0.8–0.10). Le rôle `system` est
fusionné dans le 1er message utilisateur pour les templates qui le refusent (Gemma).

## Lancer (2 commandes)

```bash
pip install -r requirements.txt

# 1) TOUTES les inférences (8 LLM + 2 baselines), modèles chargés un par un
python scripts/run_all.py

# 2) Scoring de tout output/*.jsonl
python scripts/score.py
```

Commandes granulaires optionnelles : `python scripts/smoke_test.py` (sanity 3 exemples),
`python scripts/run_benchmark.py [--model <nom>]` (LLM seuls / un seul),
`python scripts/run_baselines.py [--model nuextract|gliner|all]`.

## Lancer sur RunPod

```bash
# Dans le pod (template PyTorch + CUDA ; GPU 80 Go conseillé, voir plus bas)
cd /workspace
git clone <repo> bench && cd bench         # ou upload du dossier
pip install -r requirements.txt
huggingface-cli login                       # modèles gated (Gemma3, Ministral, Mistral-Small)

# Disque limité ? Purge le cache HF après chaque modèle (pic ≈ plus gros modèle, pas la somme)
export BENCH_PURGE_CACHE=1

python scripts/run_all.py                   # ~toutes les inférences
python scripts/score.py                     # résultats

# Récupérer les résultats en local : on zippe output/ + results/
zip -r resultats.zip output results
# puis téléchargement via l'onglet "Files" de RunPod (ou: runpodctl send resultats.zip)
```

**Stockage du container (disque) : prévoir ≥ 200 Go** (10 LLM + 2 baselines).
Poids des modèles en bf16 (~2 octets/param) : Mistral-Small-24B ≈ 48 Go, Gemma3-12B ≈ 24 Go,
Qwen3-8B / Ministral-8B ≈ 16 Go, Qwen3-4B / Gemma3-4B / Phi-4-mini ≈ 8 Go, SmolLM3-3B ≈ 6 Go,
NuExtract-2B ≈ 4 Go, Qwen3-1.7B ≈ 3,5 Go, Qwen3-0.6B ≈ 1,2 Go, GLiNER ≈ 0,5 Go
→ **~145 Go de poids**, + cache HF, + environnement (torch/vLLM/CUDA ~15 Go).

**VRAM GPU : minimum 80 Go (A100 80G / H100 80G).** Les modèles sont chargés **un seul à la
fois**, donc c'est le plus gros (Mistral-Small-24B ≈ 48 Go de poids + cache KV à
`gpu_memory_utilization=0.90`) qui fixe la borne ; Gemma3-12B et Qwen3-8B passent largement.
Sur un GPU 48 Go (A6000/L40S), tout passe **sauf** Mistral-Small-24B en bf16 : soit on le retire
de `MODELS`, soit on le charge quantifié. **`Gemma3-12B` est gated** (même licence que Gemma3-4B).

### Modèles (IDs vérifiés sur HuggingFace, 2026-06-16)

LLM généralistes (10) : `Qwen3-8B`, `Qwen3-4B`, `Qwen3-1.7B`, `Qwen3-0.6B`, `Gemma3-12B`,
`Gemma3-4B`, `Phi-4-mini`, `Ministral-8B`, `Mistral-Small-24B`, `SmolLM3-3B`.
Baselines : `NuExtract-2.0`, `GLiNER-large`.

Corrections notables : la série « Qwen3.5 » **n'existe pas** (bonne famille = Qwen3) ;
`Ministral-3B` n'est **pas publié** sur HF → on retient `Ministral-8B-Instruct-2410`.
Les deux **Gemma3** (`-4b-it`, `-12b-it`) et les modèles **Mistral** sont **gated**
(accepter la licence + token HF avec accès aux repos gated).

## Baselines — portée limitée

NuExtract et GLiNER ne produisent **pas** `intent`/`mode`/`lever`/`goal`/`target` (laissés `null`) :
ce ne sont pas des classifieurs de tâche. Ils sont évalués honnêtement sur `product_id`, `drivers`,
`targets`, `unknown_terms`. Le tableau de scoring affiche une **section BASELINES** séparée et la
colonne `model_type` (`llm_generalist` | `nuextract` | `gliner`) dans `summary.csv`.

**Mapping vers les clés canoniques** (`baselines/synonym_lookup.py`) : les spans/valeurs extraits
sont rapprochés des clés du monde fermé en 3 passes — lookup exact, puis synonyme connu présent
comme phrase dans le span (frontières de mots), puis correspondance approchée (difflib, seuil 0.9).
NuExtract reçoit en plus le monde fermé **directement** via un template à enums
(`product`/`unit`/`direction` = listes de valeurs, `targets` = multi-label) ; `entity` reste libre
pour pouvoir détecter les `unknown_terms`. GLiNER reste zero-shot pur (labels de type d'entité).

## Optimisations appliquées (v2 — relancer pour en bénéficier)

Suite à l'analyse des erreurs (`results/analyse.md`, run v1), corrections visant l'`exact_match` :

**Déterministes (code, gain garanti tous modèles)** — dans `schema.flat_to_canonical` :
- `value` des variations ⇒ **magnitude `|value|`** (corrige le « double signe » `-15`+`decrease` :
  Ministral, Phi-4, Qwen3-4B…).
- `intent=other` ⇒ **`unknown_terms=[]`** forcé (corrige les fuites sur « other » : Mistral ×3,
  Phi-4 ×4, Qwen ×1, Ministral ×1).

**Prompt (`prompt_builder.RULES`)** — 7 règles critiques ciblant : signe de `value`, couplage
`bps→absolute` / `%→relative` (+ exception niveau-cible inverse), `targets` = KPIs cités uniquement
(pas d'ajout de `marge_interet`), marge_nette vs marge_interet, terme inconnu jamais forcé sur une
clé, inverse = lever+goal+target obligatoires, `other` n'extrait rien.

**Few-shot** — 7 exemples (2 ajoutés) : un inverse à `goal` *relatif* (vs FS2 niveau absolu) et un
« other » piège citant un KPI (question de définition).

**Baselines** — assembleur `common.finalize_baseline` : dérive `intent`/`mode`
(contenu ⇒ forward, vide ⇒ other) → les baselines matchent désormais **forward + other** (0 → 6+
sur les seuls « other », vérifié) ; garde-fous (driver = clé de levier valide, target = clé KPI) ;
`value` ⇒ magnitude. NuExtract : template `value` typé **`number`** (corrige le `value:null`
systématique). Le mode **inverse** reste hors-portée des baselines.

> Ces changements n'affectent pas la validité : dataset 56 ex. 100 % valides, 7 few-shot valides,
> tout compile. **Relance `run_all.py` puis `score.py`** pour obtenir les scores v2.

## Métriques (par modèle, dans `results/summary.csv`)

`model_type`, `intent_correct`, rappel de `other`, `mode_correct` (si simulation),
`product_id_correct`, drivers (precision/recall + accuracy des sous-champs `change`),
targets (precision/recall), inverse (`lever`/`goal`/`target`), `unknown_terms` (precision/recall),
`exact_match_rate`, `schema_valid_rate`, `latency_median_s`, `latency_p95_s`.
Le détail ligne par ligne est dans `results/details_<model>.jsonl`.

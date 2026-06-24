"""Deux moteurs d'inférence derrière une interface unique.

run_generation(run, messages_list) -> [(prediction|None, valid_json, latency_s)]

Le post-traitement est IDENTIQUE aux deux moteurs (et au benchmark principal) :
texte brut -> json.loads -> schema.flat_to_canonical -> schema.validate. Seuls les poids
(BF16 vs Q8/Q4) et le moteur (vLLM vs llama.cpp) changent ; le prompt et la grammaire JSON
sont les mêmes, pour une comparaison de quantification équitable.

Imports paresseux : un sous-processus "gguf" n'importe jamais vLLM, et inversement.
"""

import json
import os
import re
import time

from schema import FlatOutput, flat_to_canonical, validate

FLAT_SCHEMA = FlatOutput.model_json_schema()
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_TOKENS = 512
STOP = ["</s>", "<|im_end|>", "<|end|>", "<end_of_turn>"]


def _finalize(raw_text):
    """Texte généré -> (prediction|None, valid_json). Logique commune aux 2 moteurs."""
    try:
        raw = _THINK_RE.sub("", raw_text).strip()
        canonical = flat_to_canonical(json.loads(raw))
    except Exception:
        return None, False
    validated = validate(canonical)
    if validated is not None:
        return validated, True
    return canonical, False           # JSON parsable mais non conforme à l'union


# ---------------------------------------------------------------------------
# BF16 -> vLLM (réutilise le moteur du benchmark principal, src/inference.py)
# ---------------------------------------------------------------------------

def _run_vllm(run, messages_list):
    import inference                  # importe vLLM (uniquement dans un run BF16)
    llm = inference.load_model(run.model)
    try:
        return inference.generate(llm, messages_list)
    finally:
        inference.free_model(llm)


# ---------------------------------------------------------------------------
# Q8 / Q4 (GGUF) -> llama.cpp
# ---------------------------------------------------------------------------

def _merge_system(messages):
    """Fusionne le rôle system dans le 1er message user (templates qui le refusent, ex. Gemma)."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_content, rest = messages[0]["content"], messages[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            merged = dict(m, content=f"{sys_content}\n\n{m['content']}")
            return rest[:i] + [merged] + rest[i + 1:]
    return rest


def _apply_template(tok, messages):
    """apply_chat_template avec enable_thinking=False si connu, et repli sans rôle system."""
    def render(msgs):
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        return render(messages)
    except Exception:
        return render(_merge_system(messages))


def _download_gguf(gguf_repo, quant):
    """Télécharge le(s) fichier(s) .gguf correspondant à `quant`. Retourne le chemin local
    du (premier) shard ; llama.cpp recolle automatiquement les shards d'un même dossier."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(gguf_repo)
             if f.endswith(".gguf") and quant in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"Aucun fichier *{quant}*.gguf dans {gguf_repo}")

    # Shards: '<name>-00001-of-00003.gguf'. On télécharge tous les shards du quant choisi.
    chosen = sorted(files)[0]
    shard = re.search(r"-(\d+)-of-(\d+)\.gguf$", chosen)
    if shard:
        stem = chosen[: shard.start()]
        files = sorted(f for f in files if f.startswith(stem))
    else:
        files = [chosen]

    local = [hf_hub_download(gguf_repo, filename=f) for f in files]
    return sorted(local)[0]


def _run_gguf(run, messages_list):
    from llama_cpp import Llama, LlamaGrammar
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(run.model, trust_remote_code=True)
    prompts = [_apply_template(tok, m) for m in messages_list]

    model_path = _download_gguf(run.gguf_repo, run.quant)
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    grammar = LlamaGrammar.from_json_schema(json.dumps(FLAT_SCHEMA))

    results = []
    try:
        for prompt in prompts:
            t0 = time.perf_counter()
            pred, valid = None, False
            try:
                out = llm.create_completion(
                    prompt, max_tokens=MAX_TOKENS, temperature=0.0, grammar=grammar, stop=STOP)
                latency = time.perf_counter() - t0
                pred, valid = _finalize(out["choices"][0]["text"])
            except Exception:
                latency = time.perf_counter() - t0
            results.append((pred, valid, latency))
    finally:
        _free_llama(llm)
    return results


def _free_llama(llm):
    """Libère le contexte llama.cpp (le process se termine juste après : filet de sécurité)."""
    import gc
    try:
        llm.close()                   # bindings récents (context manager)
    except Exception:
        pass
    del llm
    gc.collect()


def run_generation(run, messages_list):
    if run.backend == "vllm":
        return _run_vllm(run, messages_list)
    if run.backend == "gguf":
        return _run_gguf(run, messages_list)
    raise ValueError(f"backend inconnu : {run.backend}")

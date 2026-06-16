"""Inférence vLLM avec décodage contraint (guided_json) sur le schéma plat.

Choix retenu : on contraint la génération avec le JSON Schema de FlatOutput (tous les champs
nullable). L'union discriminée sur 2 champs (intent+mode) est mal gérée par les backends de
guided decoding ; on valide donc la branche correcte EN POST-TRAITEMENT (schema.validate).
"""

import gc
import json
import re
import time

import torch
import vllm.sampling_params as _sp
from vllm import LLM, SamplingParams

from schema import FlatOutput, flat_to_canonical, validate

FLAT_SCHEMA = FlatOutput.model_json_schema()
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _structured_kwargs(schema):
    """Décodage JSON contraint, compatible vLLM ancien et récent.

    vLLM >= 0.10.2 : structured_outputs=StructuredOutputsParams(json=...)
    vLLM 0.8–0.10  : guided_decoding=GuidedDecodingParams(json=...)
    """
    if hasattr(_sp, "StructuredOutputsParams"):
        return {"structured_outputs": _sp.StructuredOutputsParams(json=schema)}
    if hasattr(_sp, "GuidedDecodingParams"):
        return {"guided_decoding": _sp.GuidedDecodingParams(json=schema)}
    raise RuntimeError("Cette version de vLLM n'expose pas d'API de décodage structuré.")


def load_model(hf_model_id: str) -> LLM:
    """Charge un modèle en bfloat16, repli float16 si non supporté par le GPU."""
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return LLM(
        model=hf_model_id,
        dtype=dtype,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )


def free_model(llm: LLM):
    """Libère le GPU entre deux modèles (un seul en mémoire à la fois)."""
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,                       # greedy, reproductible (thinking neutralisé)
        max_tokens=512,
        stop=["</s>", "<|im_end|>", "<|end|>"],
        **_structured_kwargs(FLAT_SCHEMA),
    )


def _merge_system(messages):
    """Fusionne le message system dans le 1er message user (templates qui refusent 'system')."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_content = messages[0]["content"]
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            merged = dict(m, content=f"{sys_content}\n\n{m['content']}")
            return rest[:i] + [merged] + rest[i + 1:]
    return rest


def _apply(tok, messages):
    """apply_chat_template avec enable_thinking=False si le template le connaît."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _to_prompt(llm: LLM, messages):
    """Applique le chat template du modèle.

    enable_thinking=False couvre Qwen3/SmolLM3 (ignoré ailleurs). Sous décodage contraint
    JSON, aucun bloc <think> ne peut de toute façon être émis. Repli si le template refuse
    le rôle 'system' (ex. Gemma) : on le fusionne dans le premier message utilisateur.
    """
    tok = llm.get_tokenizer()
    try:
        return _apply(tok, messages)
    except Exception:
        return _apply(tok, _merge_system(messages))


def generate(llm: LLM, messages_list):
    """Génère pour une liste de conversations. Renvoie [(pred|None, valid_json, latency_s)]."""
    params = make_sampling_params()
    prompts = [_to_prompt(llm, m) for m in messages_list]

    results = []
    for prompt in prompts:
        t0 = time.perf_counter()
        pred, valid = None, False
        try:
            out = llm.generate([prompt], params)
            latency = time.perf_counter() - t0
            raw = _THINK_RE.sub("", out[0].outputs[0].text).strip()
            flat = json.loads(raw)
            canonical = flat_to_canonical(flat)
            validated = validate(canonical)
            if validated is not None:
                pred, valid = validated, True
            else:
                pred = canonical  # JSON parsable mais non conforme à l'union
        except Exception:
            latency = time.perf_counter() - t0
        results.append((pred, valid, latency))
    return results

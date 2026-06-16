"""Construction du prompt : grounding (monde fermé) + few-shot + question."""

import json

from closed_world import GROUNDING
from few_shot import FEW_SHOTS

SYSTEM = (
    "Tu es un parseur sémantique pour un simulateur de rentabilité bancaire. "
    "Tu convertis une phrase en français en un objet JSON STRICTEMENT conforme au monde fermé "
    "ci-dessous. Réponds UNIQUEMENT par le JSON, sans texte autour.\n\n" + GROUNDING
)


def build_messages(question: str):
    """Renvoie une liste de messages chat (system + few-shot + user)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for text, gold in FEW_SHOTS:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)})
    messages.append({"role": "user", "content": question})
    return messages

"""Construction du prompt : grounding (monde fermé) + few-shot + question."""

import json

from closed_world import GROUNDING
from few_shot import FEW_SHOTS

RULES = """
# RÈGLES CRITIQUES (à respecter impérativement)
1. value = nombre POSITIF (la magnitude). Le sens est porté UNIQUEMENT par "direction".
   Jamais de valeur négative. Ex: "baisse de 10 %" -> value=10, direction="decrease".
2. Unité d'une VARIATION : "bps" -> type="absolute" ; "%" de variation -> type="relative".
   EXCEPTION (mode inverse) : un NIVEAU CIBLE ("atteindre/porter à X %", "une RAROC de 15 %")
   -> type="absolute". Un CHANGE cible ("en hausse de 10 %") -> type="relative".
3. targets = UNIQUEMENT les KPIs explicitement cités. N'ajoute jamais un KPI non mentionné
   (en particulier n'ajoute pas "marge_interet" par défaut). Aucun KPI cité -> [].
4. "marge nette" -> marge_nette ; "marge d'intérêt"/MNI/NIM -> marge_interet. Ne confonds pas.
   "marge" seule, sans qualificatif -> ne devine pas (laisse hors targets).
5. Un terme qui ressemble à un levier/produit/KPI mais ABSENT du monde fermé va dans
   unknown_terms (en minuscules). Ne le force JAMAIS sur une clé valide. Ex: "livret A",
   "assurance emprunteur", "taux de la BCE", "taux de la Fed", "ratio CET1" -> unknown_terms.
   Si le SEUL élément moteur de la phrase est un tel terme inconnu, alors drivers=[] (forward)
   et le terme va dans unknown_terms (le produit et les targets cités restent, eux, renseignés).
6. mode "inverse" : remplis TOUJOURS lever, goal ET target (jamais null).
7. intent "other" (salutation, question générale, définition, hors-simulation) :
   sortie EXACTEMENT {"intent":"other","unknown_terms":[]}. N'extrais rien, même si la phrase
   cite un KPI ou la BCE.
"""

SYSTEM = (
    "Tu es un parseur sémantique pour un simulateur de rentabilité bancaire. "
    "Tu convertis une phrase en français en un objet JSON STRICTEMENT conforme au monde fermé "
    "ci-dessous. Réponds UNIQUEMENT par le JSON, sans texte autour.\n\n"
    + GROUNDING + "\n" + RULES
)


def build_messages(question: str):
    """Renvoie une liste de messages chat (system + few-shot + user)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for text, gold in FEW_SHOTS:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)})
    messages.append({"role": "user", "content": question})
    return messages

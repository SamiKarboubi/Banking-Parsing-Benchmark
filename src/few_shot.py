"""Exemples few-shot pour l'in-context learning. DISTINCTS du dataset d'évaluation."""

FEW_SHOTS = [
    (
        "Pour le crédit immobilier, augmente le coût du risque de 5 % et baisse les commissions "
        "de 10 bps. Quel impact sur la marge nette et la RAROC ?",
        {"intent": "simulation", "mode": "forward", "product_id": "CRED_IMMO",
         "drivers": [
             {"entity_key": "cout_risque", "change": {"type": "relative", "value": 5, "unit": "%", "direction": "increase"}},
             {"entity_key": "commissions", "change": {"type": "absolute", "value": 10, "unit": "bps", "direction": "decrease"}}],
         "targets": ["marge_nette", "raroc"], "unknown_terms": []},
    ),
    (
        "De combien baisser les coûts opérationnels du crédit PME pour atteindre un coefficient "
        "d'exploitation de 50 % ?",
        {"intent": "simulation", "mode": "inverse", "product_id": "CRED_PME",
         "lever": {"entity_key": "cout_operationnel"},
         "goal": {"type": "absolute", "value": 50, "unit": "%", "direction": "decrease"},
         "target": "coef_exploitation", "unknown_terms": []},
    ),
    (
        "Crédit conso : augmente le taux client de 25 bps et le taux du PEL de 10 bps. Impact sur le PNB ?",
        {"intent": "simulation", "mode": "forward", "product_id": "CRED_CONSO",
         "drivers": [
             {"entity_key": "taux_client", "change": {"type": "absolute", "value": 25, "unit": "bps", "direction": "increase"}}],
         "targets": ["pnb"], "unknown_terms": ["pel"]},
    ),
    (
        "Augmente le taux des dépôts à terme de 30 bps.",
        {"intent": "simulation", "mode": "forward", "product_id": None,
         "drivers": [
             {"entity_key": "DAT", "change": {"type": "absolute", "value": 30, "unit": "bps", "direction": "increase"}}],
         "targets": [], "unknown_terms": []},
    ),
    (
        "Bonjour, peux-tu m'aider ?",
        {"intent": "other", "unknown_terms": []},
    ),
    # Inverse avec goal RELATIF (un CHANGE cible "en hausse de X %") — à opposer à FS2 (niveau cible absolu).
    (
        "De combien relever les commissions du crédit immobilier pour un PNB en hausse de 8 % ?",
        {"intent": "simulation", "mode": "inverse", "product_id": "CRED_IMMO",
         "lever": {"entity_key": "commissions"},
         "goal": {"type": "relative", "value": 8, "unit": "%", "direction": "increase"},
         "target": "pnb", "unknown_terms": []},
    ),
    # "other" piège : la phrase cite un KPI mais c'est une question de définition → rien n'est extrait.
    (
        "Peux-tu m'expliquer ce qu'est le PNB ?",
        {"intent": "other", "unknown_terms": []},
    ),
]

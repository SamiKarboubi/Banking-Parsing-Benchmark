"""Scoring champ par champ d'une prédiction vs ground truth."""


def _prf(pred_set, gold_set):
    """Precision / recall sur deux ensembles.

    Conventions ensembles vides : pred vide → precision 1.0 (aucun faux positif) ;
    gold vide → recall 1.0 (rien à rappeler). Les faux positifs pénalisent la precision.
    """
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 1.0
    recall = tp / len(gold_set) if gold_set else 1.0
    return precision, recall


def _drivers_by_key(drivers):
    return {d["entity_key"]: d.get("change", {}) for d in (drivers or [])}


def _norm_terms(terms):
    return {t.lower() for t in (terms or [])}


def _canon(d):
    """Représentation canonique (listes -> ensembles) pour l'exact match profond."""
    if not isinstance(d, dict):
        return d
    intent = d.get("intent")
    if intent == "other":
        return ("other", frozenset(_norm_terms(d.get("unknown_terms"))))
    mode = d.get("mode")
    base = (intent, mode, d.get("product_id"), frozenset(_norm_terms(d.get("unknown_terms"))))
    if mode == "forward":
        drv = frozenset(
            (k, tuple(sorted(v.items()))) for k, v in _drivers_by_key(d.get("drivers")).items()
        )
        tgt = frozenset(d.get("targets") or [])
        return base + (drv, tgt)
    if mode == "inverse":
        goal = tuple(sorted((d.get("goal") or {}).items()))
        lever = (d.get("lever") or {}).get("entity_key")
        return base + (lever, goal, d.get("target"))
    return base


def score_example(pred, gold):
    """Renvoie un dict de métriques pour un exemple. pred peut être None."""
    s = {}
    if pred is None:
        pred = {}

    gold_intent = gold.get("intent")
    pred_intent = pred.get("intent")
    s["intent_correct"] = pred_intent == gold_intent
    # rappel spécifique de la classe "other"
    s["other_recall"] = (pred_intent == "other") if gold_intent == "other" else None

    if gold_intent == "simulation":
        s["mode_correct"] = pred.get("mode") == gold.get("mode")
    else:
        s["mode_correct"] = None

    s["product_id_correct"] = pred.get("product_id") == gold.get("product_id")

    # --- forward : drivers + targets ---
    if gold.get("mode") == "forward":
        gp = _drivers_by_key(pred.get("drivers"))
        gg = _drivers_by_key(gold.get("drivers"))
        p, r = _prf(set(gp), set(gg))
        s["drivers_precision"], s["drivers_recall"] = p, r

        matched = set(gp) & set(gg)
        subfields = ["type", "value", "unit", "direction"]
        if matched:
            accs = []
            for k in matched:
                accs.append(sum(gp[k].get(f) == gg[k].get(f) for f in subfields) / len(subfields))
            s["drivers_change_accuracy"] = sum(accs) / len(accs)
        else:
            s["drivers_change_accuracy"] = None

        p, r = _prf(set(pred.get("targets") or []), set(gold.get("targets") or []))
        s["targets_precision"], s["targets_recall"] = p, r

    # --- inverse : lever / goal / target ---
    if gold.get("mode") == "inverse":
        s["lever_correct"] = (pred.get("lever") or {}).get("entity_key") == \
            (gold.get("lever") or {}).get("entity_key")
        s["goal_correct"] = (pred.get("goal") or {}) == (gold.get("goal") or {})
        s["target_correct"] = pred.get("target") == gold.get("target")

    p, r = _prf(_norm_terms(pred.get("unknown_terms")), _norm_terms(gold.get("unknown_terms")))
    s["unknown_terms_precision"], s["unknown_terms_recall"] = p, r

    s["exact_match"] = _canon(pred) == _canon(gold)
    return s

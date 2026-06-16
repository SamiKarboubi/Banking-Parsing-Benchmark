"""Table synonyme normalisé → clé canonique, couvrant tout le monde fermé.

Utilisée par les baselines (NuExtract, GLiNER) pour mapper les spans extraits vers les clés.
Normalisation : strip, lower, suppression des accents.

match() fait d'abord un lookup EXACT, puis (fuzzy=True) un repli tolérant aux frontières de
span imparfaites : un synonyme connu présent comme phrase (frontières de mots) dans le span,
sinon une correspondance approchée (fautes/variantes morphologiques) à seuil élevé.
"""

import difflib
import re
import unicodedata


def normalize(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.strip().lower()


# {synonyme (forme libre) -> clé canonique}. Les clés du dict sont normalisées au chargement.
_RAW = {
    # --- PRODUITS ---
    "cred_immo": "CRED_IMMO", "immo": "CRED_IMMO", "credit immo": "CRED_IMMO",
    "credit immobilier": "CRED_IMMO", "pret immobilier": "CRED_IMMO", "pret habitat": "CRED_IMMO",
    "credit logement": "CRED_IMMO", "pret maison": "CRED_IMMO", "financement immobilier": "CRED_IMMO",
    "cred_conso": "CRED_CONSO", "conso": "CRED_CONSO", "credit conso": "CRED_CONSO",
    "pret perso": "CRED_CONSO", "pret personnel": "CRED_CONSO", "credit consommation": "CRED_CONSO",
    "credit a la consommation": "CRED_CONSO",
    "cred_pme": "CRED_PME", "pme": "CRED_PME", "credit pro": "CRED_PME",
    "credit professionnel": "CRED_PME", "credit entreprise": "CRED_PME", "pret pro": "CRED_PME",
    "pret entreprise": "CRED_PME", "credit d'investissement pme": "CRED_PME",

    # --- SOURCES DE FINANCEMENT ---
    "dav": "DAV", "depots a vue": "DAV", "comptes courants": "DAV", "compte courant": "DAV",
    "dat": "DAT", "depots a terme": "DAT", "comptes a terme": "DAT", "cat": "DAT",
    "placements a terme": "DAT",
    "marche": "MARCHE", "refinancement de marche": "MARCHE", "interbancaire": "MARCHE",
    "marche monetaire": "MARCHE",
    "fp": "FP", "fonds propres": "FP", "capital": "FP", "capitaux propres": "FP", "equity": "FP",

    # --- DRIVERS ---
    "taux_client": "taux_client", "taux client": "taux_client", "taux du credit": "taux_client",
    "taux du pret": "taux_client", "teg": "taux_client", "taeg": "taux_client",
    "taux nominal": "taux_client", "prix du credit": "taux_client",
    "cout_risque": "cout_risque", "cout du risque": "cout_risque", "risque": "cout_risque",
    "provisionnement": "cout_risque", "provisions": "cout_risque", "cor": "cout_risque",
    "cout_operationnel": "cout_operationnel", "couts operationnels": "cout_operationnel",
    "charges": "cout_operationnel", "frais de gestion": "cout_operationnel",
    "opex": "cout_operationnel", "frais generaux": "cout_operationnel",
    "cout_capital": "cout_capital", "cout du capital": "cout_capital",
    "charge en capital": "cout_capital", "cout des fonds propres": "cout_capital",
    "commissions": "commissions", "frais de dossier": "commissions",
    "frais de montage": "commissions", "frais": "commissions",

    # --- KPIs ---
    "marge_interet": "marge_interet", "marge nette d'interet": "marge_interet",
    "marge d'interet": "marge_interet", "mni": "marge_interet", "nim": "marge_interet",
    "pnb": "pnb", "produit net bancaire": "pnb", "revenu net bancaire": "pnb",
    "cout_risque_total": "cout_risque_total", "cout du risque total": "cout_risque_total",
    "cout du risque resultant": "cout_risque_total", "cor total": "cout_risque_total",
    "marge_nette": "marge_nette", "marge nette": "marge_nette",
    "marge nette de rentabilite": "marge_nette", "rentabilite nette": "marge_nette",
    "raroc": "raroc", "rentabilite ajustee du risque": "raroc",
    "return on risk-adjusted capital": "raroc",
    "coef_exploitation": "coef_exploitation", "coefficient d'exploitation": "coef_exploitation",
    "coef d'exploitation": "coef_exploitation", "cost/income": "coef_exploitation",
    "cost income ratio": "coef_exploitation",
}

LOOKUP = {normalize(k): v for k, v in _RAW.items()}

# Synonymes longs d'abord : on préfère la correspondance la plus spécifique.
_SYN_BY_LEN = sorted(LOOKUP, key=len, reverse=True)


def _phrase_in(span_norm: str, syn: str) -> bool:
    """syn présent dans span_norm sur des frontières de mots (évite 'cat' dans 'catégorie')."""
    return re.search(r"\b" + re.escape(syn) + r"\b", span_norm) is not None


def match(text: str, fuzzy: bool = True):
    """Renvoie la clé canonique pour un span, ou None si aucun synonyme ne correspond.

    fuzzy=True ajoute deux replis utiles aux baselines (spans non « propres ») :
      1. un synonyme connu apparaît comme phrase dans le span (le plus long gagne) ;
      2. correspondance approchée (difflib) à seuil élevé pour fautes/variantes.
    """
    n = normalize(text)
    if not n:
        return None
    if n in LOOKUP:
        return LOOKUP[n]
    if not fuzzy:
        return None
    # 1) synonyme contenu dans le span (ou span contenu dans un synonyme), le plus long d'abord
    for syn in _SYN_BY_LEN:
        if _phrase_in(n, syn) or (len(n) >= 4 and n in syn):
            return LOOKUP[syn]
    # 2) correspondance approchée (variantes morphologiques, fautes de frappe)
    close = difflib.get_close_matches(n, _SYN_BY_LEN, n=1, cutoff=0.9)
    return LOOKUP[close[0]] if close else None

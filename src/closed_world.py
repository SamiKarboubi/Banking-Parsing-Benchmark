"""Monde fermé : énumérations canoniques + bloc de grounding réinjecté dans chaque prompt."""

PRODUITS = ["CRED_IMMO", "CRED_CONSO", "CRED_PME"]
SOURCES = ["DAV", "DAT", "MARCHE", "FP"]
DRIVERS = ["taux_client", "cout_risque", "cout_operationnel", "cout_capital", "commissions"]
KPIS = ["marge_interet", "pnb", "cout_risque_total", "marge_nette", "raroc", "coef_exploitation"]

# Clés acceptées comme entity_key d'un levier (source de financement OU driver produit).
LEVER_KEYS = SOURCES + DRIVERS


def type_of(key):
    """Dérive entity_type (non stocké dans le JSON) à partir de la clé."""
    return "source_financement" if key in SOURCES else "driver"


GROUNDING = """\
# MONDE FERMÉ — n'utilise QUE les clés ci-dessous.
# Toute formulation de l'utilisateur doit être mappée vers une de ces clés.
# Un terme qui ne correspond à AUCUNE clé → unknown_terms (jamais inventé).

## PRODUITS  (champ: product_id)
- CRED_IMMO   : Crédit immobilier taux fixe 20 ans. Prêt long terme pour l'achat d'un logement.
                Synonymes: immo, crédit immo, prêt immobilier, prêt habitat, crédit logement, prêt maison, financement immobilier.
- CRED_CONSO  : Crédit à la consommation. Prêt court/moyen terme pour des achats personnels.
                Synonymes: conso, crédit conso, prêt perso, prêt personnel, crédit consommation.
- CRED_PME    : Crédit d'investissement PME. Financement professionnel pour une entreprise.
                Synonymes: pme, crédit pro, crédit professionnel, crédit entreprise, prêt pro, prêt entreprise.

## LEVIERS — TYPE 1 : SOURCES DE FINANCEMENT
# Composantes du COÛT des ressources de la banque. On fait varier LEUR TAUX.
# Repère: l'utilisateur parle du taux/coût D'UNE SOURCE (ex. "le taux DES dépôts à terme").
- DAV     : Dépôts à vue. Argent des comptes courants, ressource la moins chère.
            Synonymes: dépôts à vue, DAV, comptes courants, compte courant.
- DAT     : Dépôts à terme. Épargne bloquée et rémunérée, plus chère que les DAV.
            Synonymes: dépôts à terme, DAT, comptes à terme, CAT, placements à terme.
- MARCHE  : Refinancement de marché. Emprunts de la banque sur les marchés/interbancaire.
            Synonymes: refinancement de marché, marché, interbancaire, marché monétaire.
- FP      : Fonds propres. Capital de la banque, ressource la plus coûteuse.
            Synonymes: fonds propres, FP, capital, capitaux propres, equity.

## LEVIERS — TYPE 2 : DRIVERS
# Postes de rentabilité réglables, directement attachés au produit.
- taux_client        : Taux client (TEG). Taux d'intérêt PAYÉ PAR LE CLIENT sur le crédit.
                       (≠ taux d'une source de financement ci-dessus)
                       Synonymes: taux client, taux du crédit, taux du prêt, TEG, TAEG, taux nominal, prix du crédit.
- cout_risque        : Coût du risque — LEVIER D'ENTRÉE qu'on FAIT VARIER (niveau de provisionnement choisi).
                       (≠ cout_risque_total, qui est l'indicateur mesuré)
                       Synonymes: coût du risque, risque, provisionnement, provisions, CoR.
- cout_operationnel  : Coûts opérationnels. Frais de fonctionnement du crédit.
                       Synonymes: coûts opérationnels, charges, frais de gestion, opex, frais généraux.
- cout_capital       : Coût du capital. Charge liée à la rémunération des fonds propres mobilisés.
                       Synonymes: coût du capital, charge en capital, coût des fonds propres.
- commissions        : Commissions et frais. Revenus annexes (frais de dossier, etc.).
                       Synonymes: commissions, frais de dossier, frais de montage, frais.

## INDICATEURS / KPIs  (champ: targets en forward, target en inverse)
# Résultats qu'on MESURE (jamais des leviers — un KPI ne se "fait pas varier").
- marge_interet      : Marge nette d'intérêt. Écart entre taux client et coût des ressources.
                       Synonymes: marge nette d'intérêt, marge d'intérêt, MNI, NIM.
- pnb                : Produit net bancaire. Revenu total généré par le crédit.
                       Synonymes: PNB, produit net bancaire, revenu net bancaire.
- cout_risque_total  : Coût du risque — INDICATEUR DE SORTIE qu'on MESURE (résultat, pas un levier).
                       (≠ cout_risque, le levier d'entrée)
                       Synonymes: coût du risque total, coût du risque résultant, CoR total.
- marge_nette        : Marge nette de rentabilité. Rentabilité finale après tous les coûts.
                       Synonymes: marge nette, marge nette de rentabilité, rentabilité nette.
- raroc              : RAROC. Rentabilité ajustée du risque (rendement rapporté au capital risqué).
                       Synonymes: RAROC, rentabilité ajustée du risque, return on risk-adjusted capital.
- coef_exploitation  : Coefficient d'exploitation. Ratio coûts / revenus (cost/income).
                       Synonymes: coefficient d'exploitation, coef d'exploitation, cost/income, cost income ratio.

## RÈGLES DE DÉSAMBIGUÏSATION
1. "coût du risque" : cout_risque (driver) si on le FAIT VARIER ; cout_risque_total (KPI) si on le MESURE. Le rôle tranche.
2. "taux" : "taux du crédit/client" → taux_client ; "taux DES dépôts à terme / d'une source" → la source. Le complément indique la source.
3. "marge" sans qualificatif est ambigu (marge_interet vs marge_nette) : ne devine pas.
"""

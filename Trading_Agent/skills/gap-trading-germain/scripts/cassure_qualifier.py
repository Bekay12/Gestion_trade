#!/usr/bin/env python3
"""
cassure_qualifier.py - classe une cassure intraday SANS gap, sans appel reseau.

Pourquoi un module separe de gap_qualifier. Les deux configurations ne se
jugent pas avec les memes mesures:

  * Un gap se lit a l'ouverture, se normalise par l'ATR et se juge sur une
    probabilite de comblement. Le tableau COMBLEMENT_PAR_ATR n'a aucun sens
    pour une cassure, qui n'a pas d'ecart d'ouverture a combler.
  * Une cassure se lit sur la seance entiere: elle est confirmee par l'endroit
    ou la cloture se situe dans le range du jour, pas par la tenue a 30 min.

Ce que ce module ajoute a la methode. Le 22.09.2026, huit des dix plus fortes
hausses de la seance americaine ont ouvert a plat ou en baisse puis sont montees
toute la journee. INDP, la plus forte (+28,7 % de l'ouverture a la cloture), a
gappe de +0,3 %. Aucune n'etait visible du screener de gaps, par construction:
le filtre Gap les exclut toutes. Detail et mesures dans
docs/methode-gaps-et-cassures.md, section 4.

Les seuils partages (RVOL, liquidite, signaux de manipulation) sont importes de
gap_qualifier et ne sont pas redefinis ici: une seule source par seuil.
"""
from dataclasses import dataclass, field
from typing import Optional

from gap_qualifier import (RVOL_FORT, RVOL_EXPLOSIF, RVOL_SURVEILLER,
                           SHORT_INTEREST_SQUEEZE,
                           VOLUME_JOUR_MINIMAL, VOLUME_MOYEN_PLANCHER,
                           FORMULAIRES_DILUTIFS, FLOAT_BAS, CAP_NANO,
                           RATIO_TENDANCE_ETABLIE, STOP_VENDEUR_PCT, SSR_SEUIL)

# Direction de chaque classe de cassure. CASSURE est la seule configuration
# haussiere du module; EPUISEMENT decrit un mouvement deja distribue, donc
# vendeur. A_SURVEILLER ne promet rien et ne porte aucune direction.
#
# Ces directions ne sont PAS mesurees: le backtest du 26.09.2026 ne couvre que
# les gaps, faute d'historique intrajournalier permettant de reconstruire une
# cassure. Elles sont deduites de ce que chaque classe promet.
DIRECTION_PAR_CLASSE = {
    "CASSURE":      "acheteuse",
    "EPUISEMENT":   "vendeuse",
    "A_SURVEILLER": None,
    "PUMP_RISK":    "vendeuse",
    "INSUFFISANT":  None,
}

# Position de la cloture dans le range du jour, de 0 (au plus bas) a 1 (au plus
# haut). Une cassure vendue en fin de seance cloture dans le bas de son range;
# une cassure tenue cloture dans le haut.
#
# Calibration du depot, pas un chiffre de la formation. Mesure du 22.09.2026 sur
# les deux cassures completes de la seance: INDP 0,94 et MAZE 0,79. Le seuil est
# place sous les deux avec une marge, au tiers superieur du range.
CLOTURE_HAUTE = 0.70
# Sous ce niveau, la seance a rendu l'essentiel de son avance: le mouvement a
# ete distribue, quelle que soit la performance affichee en cloture.
CLOTURE_VENDUE = 0.40


@dataclass
class Cassure:
    """Mesures d'une cassure intraday. Tout champ inconnu reste None."""
    ticker: str
    gap_pct: Optional[float] = None           # ecart d'ouverture, proche de 0 ici
    variation_seance_pct: Optional[float] = None   # ouverture -> cloture
    rvol: Optional[float] = None
    rvol_pre_tendance: Optional[float] = None
    prix: Optional[float] = None
    vwap: Optional[float] = None
    haut_jour: Optional[float] = None
    bas_jour: Optional[float] = None
    cloture_veille: Optional[float] = None    # cloture de la seance precedente
    volume_jour: Optional[float] = None
    volume_moyen: Optional[float] = None
    float_actions: Optional[float] = None
    market_cap: Optional[float] = None
    short_interest_pct: Optional[float] = None
    catalyseur: Optional[str] = None
    formulaire_sec: Optional[str] = None
    seances_de_hausse: Optional[int] = None   # longueur de la tendance en cours


@dataclass
class VerdictCassure:
    ticker: str
    classe: str          # CASSURE | A_SURVEILLER | EPUISEMENT | PUMP_RISK | INSUFFISANT
    horizon: str
    sortie: str
    motifs: list = field(default_factory=list)
    alertes: list = field(default_factory=list)
    inconnues: list = field(default_factory=list)
    position_cloture: Optional[float] = None
    direction: Optional[str] = None
    stop_pct: Optional[float] = None


def position_cloture(c: Cassure) -> Optional[float]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Situe la cloture dans le range de la seance, de 0 (au plus bas) a 1 (au
        plus haut). C'est l'arbitre propre a la cassure: une cassure achetee
        jusqu'au coup de cloche termine haut dans son range, une cassure
        distribuee termine bas, et les deux peuvent afficher la meme performance
        par rapport a la veille.

    Inputs:
        c (Cassure): mesures collectees

    Outputs:
        position (float | None): ratio dans [0, 1], None si le range est inconnu
                                 ou nul
    --------------------------------------------------------------------------
    """
    if c.prix is None or c.haut_jour is None or c.bas_jour is None:
        return None
    etendue = c.haut_jour - c.bas_jour
    if etendue <= 0:
        return None
    return max(0.0, min(1.0, (c.prix - c.bas_jour) / etendue))


def gap_efface(c: Cassure) -> Optional[bool]:
    """Vrai quand le titre est repasse sous la cloture de la veille apres avoir
    monte dans la seance.

    Cas de reference, revue Academy Germain du 24.09.2026 sur VBIO: +78 % en
    deux minutes, puis retour sous la cloture de la veille en un quart d'heure,
    les acheteurs du spike a -45 %. C'est la configuration inverse d'une
    cassure tenue, et elle prime sur la performance affichee.

    La condition d'avoir monte se lit sur la variation de seance, pas sur le
    gap: une cassure n'a par definition pas de gap."""
    if c.prix is None or c.cloture_veille is None or c.cloture_veille <= 0:
        return None
    monte = (c.variation_seance_pct or 0) > 0 or (c.haut_jour or 0) > c.cloture_veille
    if not monte:
        return None
    return c.prix < c.cloture_veille


def rvol_effectif(c: Cassure) -> Optional[float]:
    """Le plus grand des deux RVOL connus. Voir gap_qualifier.rvol_effectif pour
    le motif: la fenetre glissante est gonflee par la tendance en cours, et une
    cassure survient precisement au bout d'une tendance."""
    mesures = [r for r in (c.rvol, c.rvol_pre_tendance) if r is not None]
    return max(mesures) if mesures else None


def tendance_etablie(c: Cassure) -> Optional[bool]:
    """Vrai quand le regime de volume avait deja change avant la seance."""
    if c.rvol is None or c.rvol_pre_tendance is None or c.rvol <= 0:
        return None
    return (c.rvol_pre_tendance / c.rvol) >= RATIO_TENDANCE_ETABLIE


def signaux_pump(c: Cassure) -> list:
    """Les signaux de manipulation mesurables sur une cassure.

    Meme grille que pour un gap (P7 Ch.03), moins les deux signaux qui n'ont pas
    de sens ici (pic de volume au sommet du gap, comblement), plus un signal
    propre a la cassure: une avance rendue en fin de seance.
    """
    alertes = []
    rvol = rvol_effectif(c)
    if c.catalyseur is None and (rvol or 0) >= RVOL_EXPLOSIF:
        alertes.append("montee rapide sans catalyseur identifie (RVOL eleve)")
    if c.market_cap is not None and c.market_cap < CAP_NANO:
        alertes.append(f"nano cap ({c.market_cap/1e6:.0f} M$), facile a manipuler")
    if c.float_actions is not None and c.float_actions < FLOAT_BAS:
        alertes.append(f"low float ({c.float_actions/1e6:.1f} M), mouvements violents")
    if c.prix is not None and c.vwap is not None and c.prix < c.vwap:
        ecart = (c.prix / c.vwap - 1) * 100
        alertes.append(f"cours sous le VWAP ({ecart:+.1f} %), pression vendeuse")
    pos = position_cloture(c)
    if pos is not None and pos < CLOTURE_VENDUE:
        alertes.append(f"cloture a {pos:.0%} du range: avance rendue en seance")
    if (c.prix is not None and c.cloture_veille is not None
            and c.cloture_veille > 0 and c.prix <= c.cloture_veille * SSR_SEUIL):
        alertes.append(f"Rule 201 probablement declenchee (cours a "
                       f"{(c.prix / c.cloture_veille - 1) * 100:+.1f} % de la veille)")
    if gap_efface(c):
        ecart = (c.prix / c.cloture_veille - 1) * 100
        alertes.append(f"repasse sous la cloture de la veille ({ecart:+.1f} %): "
                       f"tout acheteur du jour est en perte")
    if c.formulaire_sec and any(f in c.formulaire_sec.upper() for f in FORMULAIRES_DILUTIFS):
        alertes.append(f"depot {c.formulaire_sec} = dilution, catalyseur baissier deguise")
    return alertes


def _inconnues(c: Cassure) -> list:
    manquant = []
    if c.catalyseur is None and c.formulaire_sec is None:
        manquant.append("catalyseur non verifie (EDGAR / news)")
    if c.vwap is None:
        manquant.append("VWAP non mesure")
    if position_cloture(c) is None:
        manquant.append("range de la seance inconnu: position de cloture non mesurable")
    if rvol_effectif(c) is None:
        manquant.append("RVOL non mesurable")
    if c.cloture_veille is None:
        manquant.append("cloture de la veille non mesuree")
    manquant.append("disponibilite et cout du borrow inconnus (a relever chez le courtier)")
    if c.volume_jour is None:
        manquant.append("volume du jour non mesure (plancher de liquidite non teste)")
    if c.short_interest_pct is None:
        manquant.append("short interest inconnu")
    return manquant


def qualifier_cassure(c: Cassure) -> VerdictCassure:
    """Classe une cassure, puis lui attache sa direction et son stop.

    La direction est appliquee en un point unique, apres le classement. Un
    verdict vendeur reste conditionnel a la disponibilite du borrow."""
    v = _classer_cassure(c)
    v.direction = DIRECTION_PAR_CLASSE.get(v.classe)
    if v.direction == "vendeuse":
        v.stop_pct = STOP_VENDEUR_PCT
        v.sortie = (f"racheter sur stop a {STOP_VENDEUR_PCT:.0f} % au-dessus de "
                    f"l'entree, ou en fin d'horizon. Sous reserve du borrow.")
    elif v.direction == "acheteuse":
        v.sortie = f"{v.sortie} | ACHETEUSE"
    return v


def _classer_cassure(c: Cassure) -> VerdictCassure:
    """
    --------------------------------------------------------------------------
    Purpose:
        Classe une cassure intraday en CASSURE / A_SURVEILLER / EPUISEMENT /
        PUMP_RISK / INSUFFISANT, et en deduit l'horizon et la sortie. Comme pour
        les gaps, le risque de manipulation prime sur toute lecture haussiere.

    Inputs:
        c (Cassure): mesures collectees, champs inconnus a None

    Outputs:
        verdict (VerdictCassure): classe, horizon, sortie, motifs, alertes,
                                  inconnues, position de cloture
    --------------------------------------------------------------------------
    """
    alertes = signaux_pump(c)
    inconnues = _inconnues(c)
    pos = position_cloture(c)
    rvol = rvol_effectif(c)
    rvol_connu = rvol is not None
    en_tendance = tendance_etablie(c)
    motifs = []

    catalyseur_haussier = bool(c.catalyseur) and not (
        c.formulaire_sec and any(f in c.formulaire_sec.upper() for f in FORMULAIRES_DILUTIFS))
    au_dessus_vwap = (c.prix is not None and c.vwap is not None and c.prix >= c.vwap)

    # 1. Liquidite, meme regle que pour un gap: le volume du jour decide de
    #    l'execution, le volume moyen ne sert que de plancher structurel.
    if c.volume_jour is not None and c.volume_jour < VOLUME_JOUR_MINIMAL:
        return VerdictCassure(c.ticker, "INSUFFISANT", horizon="aucun",
                              sortie="ne pas entrer",
                              motifs=[f"volume du jour {c.volume_jour:,.0f} < "
                                      f"{VOLUME_JOUR_MINIMAL:,}"],
                              alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)
    if c.volume_moyen is not None and c.volume_moyen < VOLUME_MOYEN_PLANCHER:
        return VerdictCassure(c.ticker, "INSUFFISANT", horizon="aucun",
                              sortie="ne pas entrer",
                              motifs=[f"volume moyen {c.volume_moyen:,.0f} < "
                                      f"{VOLUME_MOYEN_PLANCHER:,} (plancher structurel)"],
                              alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)

    # 2. Manipulation.
    if (c.catalyseur is None and rvol_connu and rvol >= RVOL_EXPLOSIF) or len(alertes) >= 3:
        return VerdictCassure(c.ticker, "PUMP_RISK", horizon="aucun",
                              sortie="ne pas entrer; si deja en position, sortir",
                              motifs=[f"{len(alertes)} signal(aux) de manipulation presents"],
                              alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)

    # 3. Epuisement: la cassure a eu lieu puis a ete vendue. Le test passe AVANT
    #    la confirmation, parce qu'une cloture dans le bas du range invalide la
    #    cassure quelle que soit la performance affichee par rapport a la veille.
    #    Miroir mesure le 22.09.2026: LXEO a gappe de +5,6 % puis rendu -5,6 % de
    #    l'ouverture a la cloture, le schema inverse d'INDP le meme jour.
    if gap_efface(c):
        motifs.append(f"repasse sous la cloture de la veille "
                      f"({(c.prix / c.cloture_veille - 1) * 100:+.1f} %)")
        if pos is not None:
            motifs.append(f"cloture a {pos:.0%} du range de la seance")
        return VerdictCassure(c.ticker, "EPUISEMENT",
                              horizon="aucun; le gain du jour a disparu",
                              sortie="ne pas entrer; les acheteurs du jour sont en perte",
                              motifs=motifs, alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)
    if pos is not None and pos < CLOTURE_VENDUE:
        motifs.append(f"cloture a {pos:.0%} du range de la seance")
        if c.variation_seance_pct is not None:
            motifs.append(f"variation de seance {c.variation_seance_pct:+.1f} %")
        return VerdictCassure(c.ticker, "EPUISEMENT",
                              horizon="aucun; la seance a deja distribue",
                              sortie="ne pas entrer a la cloture; reexaminer a l'ouverture",
                              motifs=motifs, alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)

    # 4. Cassure confirmee: catalyseur, volume fort, au-dessus du VWAP et
    #    cloture dans le haut du range.
    if (catalyseur_haussier and rvol_connu and rvol >= RVOL_FORT
            and au_dessus_vwap and pos is not None and pos >= CLOTURE_HAUTE
            and not gap_efface(c)):
        motifs.append(f"catalyseur haussier: {c.catalyseur}")
        motifs.append(f"RVOL {rvol:.1f} >= {RVOL_FORT:.0f}")
        if en_tendance:
            motifs.append(f"RVOL lu sur base pre-tendance ({c.rvol_pre_tendance:.1f}) "
                          f"et non sur la fenetre glissante ({c.rvol:.1f})")
        motifs.append("cloture au-dessus du VWAP")
        motifs.append(f"cloture a {pos:.0%} du range: la cassure a tenu jusqu'au soir")
        if c.short_interest_pct is not None and c.short_interest_pct >= SHORT_INTEREST_SQUEEZE:
            motifs.append(f"short interest {c.short_interest_pct:.1f} %: carburant "
                          f"supplementaire, sans etre requis")
        if c.seances_de_hausse:
            motifs.append(f"{c.seances_de_hausse} seances de hausse consecutives: "
                          f"tendance etablie, risque de fin de mouvement")
        # Une cassure en fin de tendance longue se joue plus court qu'une cassure
        # en debut de mouvement: la premiere peut etre le sommet.
        horizon = ("jour, prolongeable a la semaine" if not c.seances_de_hausse
                   or c.seances_de_hausse < 8 else "jour; tendance deja longue")
        return VerdictCassure(c.ticker, "CASSURE", horizon=horizon,
                              sortie="cloture sous le VWAP, ou sous le plus bas de la "
                                     "seance de cassure",
                              motifs=motifs, alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)

    # 5. A surveiller: la structure est la mais une mesure decisive manque.
    structure_favorable = (pos is not None and pos >= CLOTURE_HAUTE) or au_dessus_vwap
    mesure_manquante = (not rvol_connu) or c.vwap is None or pos is None
    if structure_favorable and (catalyseur_haussier or mesure_manquante):
        if catalyseur_haussier:
            motifs.append(f"catalyseur haussier: {c.catalyseur}")
        if pos is not None:
            motifs.append(f"cloture a {pos:.0%} du range")
        if rvol_connu and rvol < RVOL_FORT:
            motifs.append(f"RVOL {rvol:.1f} < {RVOL_FORT:.0f}: volume insuffisant "
                          f"pour confirmer")
        if mesure_manquante:
            motifs.append("une mesure decisive manque (RVOL, VWAP ou range)")
        return VerdictCassure(c.ticker, "A_SURVEILLER",
                              horizon="indetermine tant que la mesure manque",
                              sortie="requalifier a la seance suivante avant toute entree",
                              motifs=motifs, alertes=alertes, inconnues=inconnues,
                              position_cloture=pos)

    # 6. Par defaut: la hausse n'est pas adossee.
    if not catalyseur_haussier:
        motifs.append("pas de catalyseur haussier verifie")
    if rvol_connu and rvol < RVOL_SURVEILLER:
        motifs.append(f"RVOL {rvol:.1f} < {RVOL_SURVEILLER:.0f}: seance ordinaire")
    if pos is not None:
        motifs.append(f"cloture a {pos:.0%} du range")
    if c.prix is not None and c.vwap is not None and not au_dessus_vwap:
        motifs.append("cloture sous le VWAP")
    return VerdictCassure(c.ticker, "A_SURVEILLER",
                          horizon="aucun a ce stade",
                          sortie="pas d'entree; le titre reste sur la liste d'observation",
                          motifs=motifs, alertes=alertes, inconnues=inconnues,
                          position_cloture=pos)


def rendre(v: VerdictCassure) -> str:
    """Rendu texte d'un verdict de cassure, aligne sur gap_qualifier.rendre."""
    tete = f"### {v.ticker} — {v.classe}"
    if v.direction:
        tete += f"  [{v.direction.upper()}]"
    lignes = [tete,
              f"HORIZON    {v.horizon}",
              f"SORTIE     {v.sortie}"]
    if v.stop_pct:
        lignes.append(f"STOP       {v.stop_pct:.0f} % au-dessus de l'entree")
    if v.position_cloture is not None:
        lignes.append(f"CLOTURE    {v.position_cloture:.0%} du range de la seance")
    for m in v.motifs:
        lignes.append(f"  + {m}")
    for a in v.alertes:
        lignes.append(f"  ! {a}")
    for i in v.inconnues:
        lignes.append(f"  ? {i}")
    return "\n".join(lignes)


if __name__ == "__main__":
    # INDP, 22.09.2026: plus forte hausse de la seance, sans gap.
    indp = Cassure(ticker="INDP", gap_pct=0.32, variation_seance_pct=28.71,
                   rvol=1.99, rvol_pre_tendance=11.34, prix=3.99,
                   vwap=3.70, haut_jour=4.06, bas_jour=2.95,
                   volume_jour=1_072_003, volume_moyen=539_473,
                   float_actions=40_000_000, market_cap=120_000_000,
                   short_interest_pct=4.0,
                   catalyseur="8-K du 22.09.2026: resultats cliniques",
                   formulaire_sec="8-K", seances_de_hausse=10)
    print(rendre(qualifier_cassure(indp)))

#!/usr/bin/env python3
"""
gap_qualifier.py - classe un gap selon la methode Germain, sans appel reseau.

Toutes les fonctions sont pures: elles prennent un dictionnaire de mesures deja
collectees et rendent un verdict. C'est ce qui les rend testables hors ligne
(scripts/Test/test_gap_qualifier.py, stdlib seule) - le scan reseau vit dans
gap_scan.py et n'entre jamais ici.

Les quatre verdicts et ce qui les separe:

  SQUEEZE       short interest eleve AVANT le mouvement + volume progressif +
                catalyseur reel -> le mouvement peut durer plusieurs jours
  CONTINUATION  catalyseur haussier + RVOL >= 3 + cours au-dessus du VWAP
  FADE          petit gap rapporte a l'ATR, RVOL faible, pas de catalyseur
  PUMP_RISK     signaux de manipulation presents -> ne pas jouer

Sources des seuils: Academy Germain P7 Ch.01 (echelle RVOL, VWAP), P7 Ch.03
(discrimination squeeze/pump, sept signaux d'alarme), P12 Ch.01 (filtres de
base); probabilites de comblement par taille de gap: statistiques externes
citees dans references/methode-germain.md.

Aucun seuil n'est invente: chacun porte en commentaire son origine.
"""
from dataclasses import dataclass, field
from typing import Optional

# --- Seuils, avec leur origine -------------------------------------------
RVOL_SURVEILLER = 2.0    # P7 Ch.01: "RVOL > 2 = activite anormale, premier filtre"
RVOL_FORT       = 3.0    # stats externes: RVOL >= 3 + catalyseur -> continuation
RVOL_ANORMAL    = 5.0    # P7 Ch.01: "volume anormal, news ? manipulation ?"
RVOL_EXPLOSIF   = 10.0   # P7 Ch.01: "verifier IMMEDIATEMENT"
SHORT_INTEREST_SQUEEZE = 10.0  # P12 Ch.01: filtre "Float Short > 10%"
VOLUME_MOYEN_MINIMAL   = 500_000  # P7 Ch.01: "ADV < 500K = risque pour le trading actif"
FLOAT_BAS              = 20_000_000  # P7 Ch.01: float 9,37M qualifie de "LOW FLOAT"
CAP_NANO               = 50_000_000  # nano cap: "facile a manipuler"

# Taille du gap en multiples d'ATR -> probabilite de comblement le jour meme.
# Source externe (references/methode-germain.md), pas la formation Germain.
COMBLEMENT_PAR_ATR = (
    (0.3, 78),   # < 0,3 ATR
    (0.7, 42),
    (1.2, 25),
    (float("inf"), 8),
)

# Formulaires SEC qui annoncent une dilution: un gap haussier par-dessus est une
# alerte, pas une confirmation (P9 Ch.03).
FORMULAIRES_DILUTIFS = ("S-3", "424B", "S-1")


@dataclass
class Gap:
    """Mesures d'un gap. Tout champ inconnu reste None et n'est jamais devine."""
    ticker: str
    gap_pct: Optional[float] = None          # taille du gap en %
    atr_pct: Optional[float] = None          # ATR(14) en % du cours
    rvol: Optional[float] = None
    prix: Optional[float] = None
    vwap: Optional[float] = None
    volume_moyen: Optional[float] = None
    float_actions: Optional[float] = None
    market_cap: Optional[float] = None
    short_interest_pct: Optional[float] = None   # mesure AVANT le mouvement
    catalyseur: Optional[str] = None             # description libre, None = introuvable
    formulaire_sec: Optional[str] = None         # "8-K", "S-3", "424B"...
    volume_pic_au_sommet: Optional[bool] = None  # pic de volume exactement au plus haut
    chute_depuis_sommet_pct: Optional[float] = None
    gap_tenu_30min: Optional[bool] = None        # connu seulement apres l'ouverture


@dataclass
class Verdict:
    ticker: str
    classe: str                       # SQUEEZE | CONTINUATION | FADE | PUMP_RISK | INSUFFISANT
    horizon: str
    sortie: str
    motifs: list = field(default_factory=list)
    alertes: list = field(default_factory=list)
    inconnues: list = field(default_factory=list)
    comblement_attendu_pct: Optional[int] = None


def gap_en_atr(gap_pct: Optional[float], atr_pct: Optional[float]) -> Optional[float]:
    """Taille du gap normalisee par l'ATR. Un gap de 5% ne vaut pas la meme chose
    sur un titre qui bouge de 2% par jour que sur un qui en bouge de 15%."""
    if gap_pct is None or not atr_pct:
        return None
    return abs(gap_pct) / atr_pct


def probabilite_comblement(gap_atr: Optional[float]) -> Optional[int]:
    if gap_atr is None:
        return None
    for seuil, proba in COMBLEMENT_PAR_ATR:
        if gap_atr < seuil:
            return proba
    return COMBLEMENT_PAR_ATR[-1][1]


def signaux_pump(g: Gap) -> list:
    """Les sept signaux d'alarme de P7 Ch.03, ceux qui sont mesurables ici.

    Ne rend que ce qui est CONSTATE: un champ inconnu n'est pas un signal absent,
    il est simplement muet (et ressort dans Verdict.inconnues).
    """
    alertes = []
    if g.catalyseur is None and (g.rvol or 0) >= RVOL_ANORMAL:
        alertes.append("montee rapide sans catalyseur identifie (RVOL eleve)")
    if g.volume_pic_au_sommet:
        alertes.append("pic de volume au plus haut = distribution")
    if g.market_cap is not None and g.market_cap < CAP_NANO:
        alertes.append(f"nano cap ({g.market_cap/1e6:.0f} M$), facile a manipuler")
    if g.float_actions is not None and g.float_actions < FLOAT_BAS:
        alertes.append(f"low float ({g.float_actions/1e6:.1f} M), mouvements violents")
    if g.chute_depuis_sommet_pct is not None and g.chute_depuis_sommet_pct <= -20:
        alertes.append(f"chute de {g.chute_depuis_sommet_pct:.0f} % depuis le sommet")
    if g.prix is not None and g.vwap is not None and g.prix < g.vwap:
        ecart = (g.prix / g.vwap - 1) * 100
        alertes.append(f"cours sous le VWAP ({ecart:+.1f} %), pression vendeuse")
    if g.formulaire_sec and any(f in g.formulaire_sec.upper() for f in FORMULAIRES_DILUTIFS):
        alertes.append(f"depot {g.formulaire_sec} = dilution, catalyseur baissier deguise")
    return alertes


def _inconnues(g: Gap) -> list:
    manquant = []
    if g.catalyseur is None and g.formulaire_sec is None:
        manquant.append("catalyseur non verifie (EDGAR / news)")
    if g.vwap is None:
        manquant.append("VWAP non mesure")
    if g.short_interest_pct is None:
        manquant.append("short interest inconnu")
    if g.atr_pct is None:
        manquant.append("ATR inconnu: taille du gap non normalisable")
    if g.rvol is None:
        # Cas mesure le 22.09.2026: yfinance ne rend aucun volume pre-marche sur
        # les barres 1 min. Un RVOL absent n'est PAS un RVOL nul; le confondre
        # revient a lire "personne ne traite" la ou on ne sait simplement pas.
        manquant.append("RVOL non mesurable (pas de volume pre-marche disponible)")
    if g.gap_tenu_30min is None:
        manquant.append("tenue a 30 min inconnue (mesurable seulement apres l'ouverture)")
    return manquant


def qualifier(g: Gap) -> Verdict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Classe un gap en SQUEEZE / CONTINUATION / FADE / PUMP_RISK et en deduit
        l'horizon et le declencheur de sortie. L'ordre des tests est significatif:
        le risque de manipulation prime sur toute lecture haussiere.

    Inputs:
        g (Gap): mesures collectees, champs inconnus a None

    Outputs:
        verdict (Verdict): classe, horizon, sortie, motifs, alertes, inconnues
    --------------------------------------------------------------------------
    """
    alertes = signaux_pump(g)
    inconnues = _inconnues(g)
    gap_atr = gap_en_atr(g.gap_pct, g.atr_pct)
    comblement = probabilite_comblement(gap_atr)
    motifs = []

    au_dessus_vwap = (g.prix is not None and g.vwap is not None and g.prix >= g.vwap)
    catalyseur_haussier = bool(g.catalyseur) and not (
        g.formulaire_sec and any(f in g.formulaire_sec.upper() for f in FORMULAIRES_DILUTIFS))
    # RVOL absent et RVOL nul ne se confondent pas: le premier est une ignorance,
    # le second une mesure. `g.rvol or 0.0` melangeait les deux et rendait FADE
    # sur un titre dont on n'avait simplement pas le volume (LXEO, 22.09.2026).
    rvol = g.rvol
    rvol_connu = rvol is not None

    # 1. Liquidite: en dessous, la formation dit de ne pas trader activement.
    if g.volume_moyen is not None and g.volume_moyen < VOLUME_MOYEN_MINIMAL:
        return Verdict(g.ticker, "INSUFFISANT",
                       horizon="aucun",
                       sortie="ne pas entrer",
                       motifs=[f"volume moyen {g.volume_moyen:,.0f} < {VOLUME_MOYEN_MINIMAL:,} "
                               f"(P7 Ch.01: risque pour le trading actif)"],
                       alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)

    # 2. Manipulation: trois alertes ou plus, ou l'absence de catalyseur sur
    #    volume explosif, suffisent a ecarter le titre.
    pump_decisif = ((g.catalyseur is None and rvol_connu and rvol >= RVOL_EXPLOSIF)
                    or len(alertes) >= 3)
    if pump_decisif:
        return Verdict(g.ticker, "PUMP_RISK",
                       horizon="aucun",
                       sortie="ne pas entrer; si deja en position, sortir",
                       motifs=[f"{len(alertes)} signal(aux) de manipulation presents"],
                       alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)

    # 3. Squeeze: la configuration explicite de la formation,
    #    "Short Interest eleve + Low Float + Catalyseur".
    # Le short interest et le catalyseur ne suffisent pas: un titre sous son VWAP
    # est en distribution, pas en couverture de shorts (P7 Ch.01, et non-negociable
    # 4 du SKILL.md). Mesure du 22.09.2026 sur LXEO: SI 29,1 %, catalyseur reel,
    # RVOL 5,0 - et pourtant gap comble, cloture sous le VWAP et sous la cloture
    # de la veille. Sans ces deux gardes, le code prononcait SQUEEZE sur une
    # seance de distribution et annoncait un horizon de plusieurs jours.
    sous_vwap = g.vwap is not None and g.prix is not None and g.prix < g.vwap
    squeeze = (g.short_interest_pct is not None
               and g.short_interest_pct >= SHORT_INTEREST_SQUEEZE
               and catalyseur_haussier
               and rvol_connu and rvol >= RVOL_SURVEILLER
               and not g.volume_pic_au_sommet
               and not sous_vwap
               and g.gap_tenu_30min is not False)
    if squeeze:
        motifs.append(f"short interest {g.short_interest_pct:.1f} % avant le mouvement "
                      f"(seuil {SHORT_INTEREST_SQUEEZE:.0f} %)")
        motifs.append(f"catalyseur: {g.catalyseur}")
        motifs.append(f"RVOL {rvol:.1f}, volume sans pic terminal")
        if g.float_actions is not None and g.float_actions < FLOAT_BAS:
            motifs.append(f"low float {g.float_actions/1e6:.1f} M: amplifie le mouvement")
        return Verdict(g.ticker, "SQUEEZE",
                       horizon="plusieurs jours (3 a 5 seances)",
                       sortie="retour durable sous le VWAP, ou RVOL < 1,5 (epuisement)",
                       motifs=motifs, alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)

    # 4. Continuation: catalyseur + volume fort + au-dessus du VWAP.
    if catalyseur_haussier and rvol_connu and rvol >= RVOL_FORT and au_dessus_vwap:
        motifs.append(f"catalyseur haussier: {g.catalyseur}")
        motifs.append(f"RVOL {rvol:.1f} >= {RVOL_FORT:.0f}")
        motifs.append("cours au-dessus du VWAP")
        if gap_atr is not None:
            motifs.append(f"gap {gap_atr:.1f} x ATR, comblement attendu {comblement} %")
        horizon = "jour" if not g.gap_tenu_30min else "jour, prolongeable a la semaine"
        return Verdict(g.ticker, "CONTINUATION",
                       horizon=horizon,
                       sortie="retour sous le VWAP, ou cloture sous le plus bas des 30 min",
                       motifs=motifs, alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)

    # 5. A confirmer: le catalyseur est la et la structure est favorable, mais
    #    les deux filtres qui trancheraient (RVOL, VWAP) ne sont pas mesurables.
    #    Cas rencontre en production le 22.09.2026: aucun volume pre-marche
    #    disponible chez yfinance. Degrader en FADE reviendrait a conclure d'une
    #    ignorance; la methode dit d'attendre l'ouverture et la regle des 30 min.
    volume_indecidable = not rvol_connu or g.vwap is None
    if catalyseur_haussier and volume_indecidable:
        motifs.append(f"catalyseur haussier: {g.catalyseur}")
        if g.short_interest_pct is not None and g.short_interest_pct >= SHORT_INTEREST_SQUEEZE:
            motifs.append(f"short interest {g.short_interest_pct:.1f} %: carburant de squeeze")
        if gap_atr is not None:
            motifs.append(f"gap {gap_atr:.2f} x ATR, comblement attendu {comblement} %")
        motifs.append("RVOL et/ou VWAP non mesurables avant l'ouverture")
        return Verdict(g.ticker, "A_CONFIRMER",
                       horizon="indetermine tant que l'ouverture n'a pas eu lieu",
                       sortie="requalifier a 10h00 ET (regle des 30 min) avant toute entree",
                       motifs=motifs, alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)

    # 6. Fade par defaut: ce qui reste est un gap sans adossement.
    if not catalyseur_haussier:
        motifs.append("pas de catalyseur haussier verifie")
    if rvol_connu and rvol < RVOL_FORT:
        motifs.append(f"RVOL {rvol:.1f} < {RVOL_FORT:.0f}")
    elif not rvol_connu:
        motifs.append("RVOL non mesure")
    if g.prix is not None and g.vwap is not None and not au_dessus_vwap:
        motifs.append("cours sous le VWAP")
    if comblement is not None:
        motifs.append(f"comblement attendu {comblement} % le jour meme")
    return Verdict(g.ticker, "FADE",
                   horizon="jour, souvent la premiere heure",
                   sortie="comblement du gap, ou fin de la premiere heure",
                   motifs=motifs, alertes=alertes, inconnues=inconnues,
                   comblement_attendu_pct=comblement)


def rendre(v: Verdict) -> str:
    """Rendu texte d'un verdict, format du rapport-template."""
    lignes = [f"### {v.ticker} — {v.classe}",
              f"HORIZON    {v.horizon}",
              f"SORTIE     {v.sortie}"]
    if v.comblement_attendu_pct is not None:
        lignes.append(f"COMBLEMENT {v.comblement_attendu_pct} % attendu le jour meme")
    for m in v.motifs:
        lignes.append(f"  + {m}")
    for a in v.alertes:
        lignes.append(f"  ! {a}")
    for i in v.inconnues:
        lignes.append(f"  ? {i}")
    return "\n".join(lignes)


if __name__ == "__main__":
    exemple = Gap(ticker="DEMO", gap_pct=12.4, atr_pct=6.8, rvol=6.2, prix=4.85,
                  vwap=4.60, volume_moyen=660_000, float_actions=9_370_000,
                  market_cap=310_000_000, short_interest_pct=18.3,
                  catalyseur="8-K du 21.09.2026: contrat de distribution",
                  formulaire_sec="8-K", volume_pic_au_sommet=False)
    print(rendre(qualifier(exemple)))

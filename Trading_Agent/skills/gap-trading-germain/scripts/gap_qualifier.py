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

Deux mesures de volume portent des roles distincts depuis la revision du
22.09.2026 (voir docs/methode-gaps-et-cassures.md, section 2):

  volume_jour        decide de l execution -> plancher de 500 K
  volume_moyen       plancher structurel bas -> 100 K
  rvol               volume du jour / mediane des 20 seances precedentes
  rvol_pre_tendance  meme volume / mediane de la base [-60:-20]

Le classement retient le plus grand des deux RVOL, parce que la fenetre
glissante est gonflee par la tendance qu elle est censee detecter.
"""
from dataclasses import dataclass, field
from typing import Optional

# --- Seuils, avec leur origine -------------------------------------------
RVOL_SURVEILLER = 2.0    # P7 Ch.01: "RVOL > 2 = activite anormale, premier filtre"
RVOL_FORT       = 3.0    # stats externes: RVOL >= 3 + catalyseur -> continuation
RVOL_ANORMAL    = 5.0    # P7 Ch.01: "volume anormal, news ? manipulation ?"
RVOL_EXPLOSIF   = 10.0   # P7 Ch.01: "verifier IMMEDIATEMENT"
SHORT_INTEREST_SQUEEZE = 10.0  # P12 Ch.01: filtre "Float Short > 10%"

# Liquidite. P7 Ch.01 enonce "ADV < 500K = risque pour le trading actif". Le
# seuil est repris tel quel, mais applique au volume DU JOUR et non a la
# moyenne. Motif, mesure le 22.09.2026: MAZE a un ADV de 290 127 titres et a
# echange 2 726 280 titres le jour du gap (RVOL 13,4). Filtrer sur la moyenne
# ecartait le seul gap-and-go complet de la seance, alors que la liquidite
# reellement disponible pour executer etait de 2,7 M. Inversement STFS a
# echange 45 K titres ce jour-la et doit etre ecarte: c est bien le volume du
# jour qui decide de l execution, pas l historique.
VOLUME_JOUR_MINIMAL    = 500_000
# Plancher structurel, distinct du precedent: un titre dont l activite
# ordinaire est quasi nulle garde un spread punitif le lendemain, meme apres
# une seance active. Ce seuil ne vient pas de la formation; il borne le risque
# de sortie et il est volontairement bas pour ne pas recreer le filtre que la
# ligne precedente corrige (MAZE, ADV 290 K, passe).
VOLUME_MOYEN_PLANCHER  = 100_000
# Au-dela de ce rapport entre le RVOL mesure sur base pre-tendance et le RVOL
# sur fenetre glissante, le titre est deja installe dans un regime de volume
# eleve. Mesure du 22.09.2026: INDP 11,34 / 1,99 = 5,7 (dix seances de hausse
# continue avant la cassure); MAZE 14,38 / 13,45 = 1,07 (aucune tendance
# prealable). Le seuil separe ces deux regimes avec une large marge.
RATIO_TENDANCE_ETABLIE = 2.0
FLOAT_BAS              = 20_000_000  # P7 Ch.01: float 9,37M qualifie de "LOW FLOAT"
CAP_NANO               = 50_000_000  # nano cap: "facile a manipuler"

# Stop d'une position vendeuse, en pourcentage au-dessus du cours d'entree.
# Origine: backtest du 26.09.2026 sur 226 verdicts (Trading_Agent/gaps/backtests).
# Sans stop, la vente a decouvert des PUMP_RISK rend -9,57 % en moyenne malgre
# 68 % de reussite: 109 gagnants a +21,6 % contre 52 perdants a -74,8 %, pire cas
# -1750 %. Avec un stop declenche sur le plus haut de seance:
#   stop 10 % -> +9,09 %   15 % -> +10,29 %   20 % -> +11,83 %   30 % -> +11,05 %
# La zone 20 a 30 % est plate, donc le reglage n'est pas ajuste au bruit.
STOP_VENDEUR_PCT = 20.0

# Seuil de declenchement de la Rule 201 (SSR): la vente a decouvert n'est plus
# possible qu'au cours acheteur. Source unique du seuil: R7 dans
# Trading_Agent/agent/rules.py, fonction ssr_triggered. Mesure du 26.09.2026:
# 65 des 226 candidats du backtest cloturent au moins 10 % sous leur ouverture,
# soit pres d'un tiers des seances concernees.
SSR_SEUIL = 0.90

# Direction que porte chaque classe, etablie par le backtest du 26.09.2026.
# FADE et PUMP_RISK sont deux configurations vendeuses de profils opposes:
#   FADE       +4,65 % avec stop, +4,75 % sans: le stop ne change presque rien,
#              donc le titre revient rarement contre la position. 95 % de justesse.
#   PUMP_RISK  +11,83 % avec stop mais -9,57 % sans: tout depend du stop.
# Les classes haussieres exigent un catalyseur verifie, jamais renseigne
# automatiquement: elles n'apparaissent pas dans le backtest et leur direction
# n'est donc PAS mesuree, seulement deduite de ce qu'elles promettent.
DIRECTION_PAR_CLASSE = {
    "FADE":         "vendeuse",
    "PUMP_RISK":    "vendeuse",
    "CONTINUATION": "acheteuse",
    "SQUEEZE":      "acheteuse",
    "A_CONFIRMER":  None,      # ne promet rien
    "INSUFFISANT":  None,      # ecarte
}

# Taille du gap en multiples d'ATR -> probabilite de comblement le jour meme.
# Source externe (references/methode-germain.md), pas la formation Germain.
COMBLEMENT_PAR_ATR = (
    (0.3, 78),   # < 0,3 ATR
    (0.7, 42),
    (1.2, 25),
    (float("inf"), 8),
)

# Formulaires et items SEC qui annoncent une dilution: un gap haussier
# par-dessus est une alerte, pas une confirmation (P9 Ch.03). Le test est une
# recherche de sous-chaine dans `formulaire_sec`, renseigne par edgar_depots.
#
# Etendu le 26.09.2026 apres deux journees de verification manuelle:
#   F-1, F-3  equivalents etrangers du S-1 et du S-3, rencontres sur WETO.
#   2.03      item de 8-K creant une obligation financiere directe. Cas PFSA du
#             16.09.2026: convertible payable en actions, plancher 1,07 $, titre
#             a -14,04 % le 24.09. edgar_depots rend alors "8-K/2.03".
#
# L'item 5.07 reste DELIBEREMENT absent. Il a autorise un regroupement d'actions
# chez PFSA le 21.09.2026, mais une assemblee generale ordinaire porte le meme
# code: le type seul ne permet pas de conclure. Ces depots sortent en "a lire".
FORMULAIRES_DILUTIFS = ("S-3", "424B", "S-1", "F-3", "F-1", "2.03")


@dataclass
class Gap:
    """Mesures d'un gap. Tout champ inconnu reste None et n'est jamais devine."""
    ticker: str
    gap_pct: Optional[float] = None          # taille du gap en %
    atr_pct: Optional[float] = None          # ATR(14) en % du cours
    rvol: Optional[float] = None            # volume du jour / mediane 20 seances
    rvol_pre_tendance: Optional[float] = None   # meme volume / base [-60:-20]
    prix: Optional[float] = None
    vwap: Optional[float] = None
    ouverture: Optional[float] = None        # premier cours de la seance
    cloture_veille: Optional[float] = None   # cloture de la seance precedente
    volume_jour: Optional[float] = None      # volume echange la seance meme
    volume_moyen: Optional[float] = None     # mediane des 20 seances precedentes
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
    direction: Optional[str] = None   # vendeuse | acheteuse | None
    stop_pct: Optional[float] = None  # stop de la position vendeuse, en %
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


def retour_sous_cloture_veille(g: Gap) -> Optional[bool]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Vrai quand un titre qui avait gappe a la hausse repasse sous la cloture
        de la veille. C'est le signal le plus net de la journee perdue: tout le
        gain a disparu, et chaque acheteur de la seance est en perte.

        Source: revue Academy Germain du 24.09.2026 sur VBIO, ingeree par
        germain_revues.py. Le titre bondit de 2,72 a 4,84 $ (+78 %) en deux
        minutes, puis repasse sous 2,72 $ en un quart d'heure; les acheteurs du
        spike perdent jusqu'a 45 %. Sa formulation: "quand un titre qui vient de
        bondir repasse sous la cloture de la veille, tout le gain de la journee
        a disparu - les acheteurs du matin sont tous en perte, et beaucoup
        vendent pour limiter la casse."

        Le gap haussier est une condition: un titre qui n'a jamais gappe n'a
        pas de gain a rendre. Cette condition se lit sur l'OUVERTURE comparee a
        la cloture de la veille, jamais sur gap_pct.

        Piege evite le 25.09.2026: dans gap_scan, gap_pct porte la variation
        COURANTE rendue par Finviz, pas l'ecart d'ouverture. Une variation
        courante positive implique deja un prix au-dessus de la cloture de la
        veille, donc la condition et sa consequence se contredisaient et le
        signal n'aurait jamais pu se declencher.

    Inputs:
        g (Gap): mesures collectees

    Outputs:
        retour (bool | None): None tant qu'une mesure necessaire manque
    --------------------------------------------------------------------------
    """
    if g.prix is None or g.cloture_veille is None or g.cloture_veille <= 0:
        return None
    if g.ouverture is None or g.ouverture <= g.cloture_veille:
        return None          # n'a pas ouvert en hausse: rien a rendre
    return g.prix < g.cloture_veille


def rvol_effectif(g: Gap) -> Optional[float]:
    """
    --------------------------------------------------------------------------
    Purpose:
        RVOL retenu pour la classification: le plus grand des deux RVOL connus,
        celui mesure sur les 20 seances precedentes et celui mesure sur une base
        anterieure a la tendance en cours.

        Motif. Le RVOL sur fenetre glissante compare le volume du jour a un
        denominateur que la tendance elle-meme a gonfle. Un titre qui monte
        depuis dix seances a deja des volumes eleves dans sa fenetre, donc un
        RVOL faible, precisement le jour ou il explose. Mesure du 22.09.2026 sur
        INDP, plus forte hausse de la seance (+28,7 % de l ouverture a la
        cloture): RVOL 1,99 sur les 20 dernieres seances, sous le seuil de 2 du
        screener, et 11,34 sur la base [-60:-20] anterieure a la tendance.

        Prendre le maximum ne peut jamais abaisser un verdict: sur un titre sans
        tendance prealable les deux mesures coincident (MAZE le meme jour: 13,45
        et 14,38).

    Inputs:
        g (Gap): mesures collectees

    Outputs:
        rvol (float | None): RVOL retenu, None si aucune des deux mesures
    --------------------------------------------------------------------------
    """
    mesures = [r for r in (g.rvol, g.rvol_pre_tendance) if r is not None]
    return max(mesures) if mesures else None


def tendance_etablie(g: Gap) -> Optional[bool]:
    """Vrai quand le regime de volume a deja change avant la seance observee.

    Se lit sur l ecart entre les deux RVOL: si la base pre-tendance donne un
    RVOL nettement plus eleve que la fenetre glissante, c est que la fenetre est
    deja saturee par le mouvement en cours. Rend None tant que les deux mesures
    ne sont pas disponibles, jamais False par defaut.
    """
    if g.rvol is None or g.rvol_pre_tendance is None or g.rvol <= 0:
        return None
    return (g.rvol_pre_tendance / g.rvol) >= RATIO_TENDANCE_ETABLIE


def signaux_pump(g: Gap) -> list:
    """Les sept signaux d'alarme de P7 Ch.03, ceux qui sont mesurables ici.

    Ne rend que ce qui est CONSTATE: un champ inconnu n'est pas un signal absent,
    il est simplement muet (et ressort dans Verdict.inconnues).
    """
    alertes = []
    rvol = rvol_effectif(g)
    if g.catalyseur is None and (rvol or 0) >= RVOL_ANORMAL:
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
    if (g.prix is not None and g.cloture_veille is not None
            and g.cloture_veille > 0 and g.prix <= g.cloture_veille * SSR_SEUIL):
        alertes.append(f"Rule 201 probablement declenchee (cours a "
                       f"{(g.prix / g.cloture_veille - 1) * 100:+.1f} % de la veille): "
                       f"vente a decouvert restreinte au cours acheteur")
    if retour_sous_cloture_veille(g):
        ecart = (g.prix / g.cloture_veille - 1) * 100
        alertes.append(f"repasse sous la cloture de la veille ({ecart:+.1f} %): "
                       f"le gap est efface, tout acheteur du jour est en perte")
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
    # Aucune source publique ne diffuse la disponibilite du borrow (convention
    # du depot, Trading_Agent/docu/portefeuille/watchlist.json). Un verdict
    # vendeur reste donc conditionnel, toujours.
    manquant.append("disponibilite et cout du borrow inconnus (a relever chez le courtier)")
    if g.cloture_veille is None:
        manquant.append("cloture de la veille non mesuree: gap efface non detectable")
    if g.volume_jour is None:
        # Le plancher de liquidite porte sur cette mesure: ne pas l avoir
        # signifie que le titre n a PAS ete teste, pas qu il a passe le test.
        manquant.append("volume du jour non mesure (plancher de liquidite non teste)")
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
        Classe un gap, puis lui attache sa direction et son stop.

        La direction est appliquee en un point unique, apres le classement, pour
        qu'aucune des six sorties de _classer ne puisse l'oublier.

        Un verdict vendeur est TOUJOURS conditionnel a la disponibilite du
        borrow, qu'aucune source publique ne donne. Le texte de sortie le dit, et
        l'inconnue correspondante figure dans chaque verdict.

    Inputs:
        g (Gap): mesures collectees, champs inconnus a None

    Outputs:
        verdict (Verdict): classe, direction, stop, horizon, sortie, motifs
    --------------------------------------------------------------------------
    """
    v = _classer(g)
    v.direction = DIRECTION_PAR_CLASSE.get(v.classe)
    if v.direction == "vendeuse":
        v.stop_pct = STOP_VENDEUR_PCT
        # Le texte est REMPLACE, pas complete. Le libelle d'origine de PUMP_RISK
        # disait "ne pas entrer", ce qui contredisait la direction vendeuse une
        # fois celle-ci attachee (constate le 26.09.2026 sur PFSA). Une classe
        # vendeuse decrit la sortie d'un short, pas une abstention.
        v.sortie = (f"racheter sur stop a {STOP_VENDEUR_PCT:.0f} % au-dessus de "
                    f"l'entree, ou en fin d'horizon. Sous reserve du borrow.")
    elif v.direction == "acheteuse":
        v.sortie = f"{v.sortie} | ACHETEUSE"
    return v


def _classer(g: Gap) -> Verdict:
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
    # Le RVOL retenu est le plus grand des deux mesures (voir rvol_effectif):
    # la fenetre glissante seule aveugle le classement sur un titre deja en
    # tendance (INDP, 22.09.2026).
    rvol = rvol_effectif(g)
    rvol_connu = rvol is not None
    en_tendance = tendance_etablie(g)

    # 1. Liquidite, en deux tests qui ne mesurent pas la meme chose.
    #
    #    a) Le volume DU JOUR decide de l execution: c est lui qui dit si la
    #       position peut etre prise et rendue sans payer le spread. Le seuil de
    #       500 K de P7 Ch.01 est conserve, la colonne change.
    #    b) Le volume moyen garde un role, mais seulement comme plancher
    #       structurel tres bas: un titre habituellement mort reste difficile a
    #       revendre le lendemain de la seance active.
    #
    #    Ce que ce decoupage corrige, mesure le 22.09.2026: MAZE (ADV 290 K,
    #    2,73 M echanges le jour du gap, +26,8 % au total) etait ecarte par un
    #    filtre sur la moyenne, alors que STFS (45 K echanges le jour meme) doit
    #    l etre et l est toujours.
    #
    #    Un volume du jour inconnu ne vaut pas un volume insuffisant: il ressort
    #    en inconnue et n entraine aucun verdict, conformement a la convention
    #    du depot sur les donnees absentes.
    if g.volume_jour is not None and g.volume_jour < VOLUME_JOUR_MINIMAL:
        return Verdict(g.ticker, "INSUFFISANT",
                       horizon="aucun",
                       sortie="ne pas entrer",
                       motifs=[f"volume du jour {g.volume_jour:,.0f} < {VOLUME_JOUR_MINIMAL:,} "
                               f"(P7 Ch.01: risque pour le trading actif)"],
                       alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)
    if g.volume_moyen is not None and g.volume_moyen < VOLUME_MOYEN_PLANCHER:
        return Verdict(g.ticker, "INSUFFISANT",
                       horizon="aucun",
                       sortie="ne pas entrer",
                       motifs=[f"volume moyen {g.volume_moyen:,.0f} < {VOLUME_MOYEN_PLANCHER:,} "
                               f"(plancher structurel: sortie de position incertaine)"],
                       alertes=alertes, inconnues=inconnues,
                       comblement_attendu_pct=comblement)

    # 2. Manipulation: trois alertes ou plus, ou l'absence de catalyseur sur
    #    volume explosif, suffisent a ecarter le titre.
    pump_decisif = ((g.catalyseur is None and rvol_connu and rvol >= RVOL_EXPLOSIF)
                    or len(alertes) >= 3)
    if pump_decisif:
        return Verdict(g.ticker, "PUMP_RISK",
                       # Horizon de la position VENDEUSE: le backtest du
                       # 26.09.2026 mesure sur trois seances.
                       horizon="jour a trois seances, en vendeuse",
                       sortie="ne pas acheter",
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
    # Un gap efface interdit toute lecture haussiere, au meme titre qu'un cours
    # sous le VWAP: il n'y a plus de gap en cours a jouer.
    gap_efface = retour_sous_cloture_veille(g) is True
    squeeze = (g.short_interest_pct is not None
               and g.short_interest_pct >= SHORT_INTEREST_SQUEEZE
               and catalyseur_haussier
               and rvol_connu and rvol >= RVOL_SURVEILLER
               and not g.volume_pic_au_sommet
               and not sous_vwap
               and not gap_efface
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
    if (catalyseur_haussier and rvol_connu and rvol >= RVOL_FORT
            and au_dessus_vwap and not gap_efface):
        motifs.append(f"catalyseur haussier: {g.catalyseur}")
        motifs.append(f"RVOL {rvol:.1f} >= {RVOL_FORT:.0f}")
        if en_tendance:
            motifs.append(f"RVOL lu sur base pre-tendance ({g.rvol_pre_tendance:.1f}) "
                          f"et non sur la fenetre glissante ({g.rvol:.1f}): "
                          f"tendance deja installee")
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
    tete = f"### {v.ticker} — {v.classe}"
    if v.direction:
        tete += f"  [{v.direction.upper()}]"
    lignes = [tete,
              f"HORIZON    {v.horizon}",
              f"SORTIE     {v.sortie}"]
    if v.stop_pct:
        lignes.append(f"STOP       {v.stop_pct:.0f} % au-dessus de l'entree "
                      f"(backtest 26.09.2026)")
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
                  vwap=4.60, volume_jour=1_400_000, volume_moyen=660_000,
                  float_actions=9_370_000,
                  market_cap=310_000_000, short_interest_pct=18.3,
                  catalyseur="8-K du 21.09.2026: contrat de distribution",
                  formulaire_sec="8-K", volume_pic_au_sommet=False)
    print(rendre(qualifier(exemple)))

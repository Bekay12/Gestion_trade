"""
autonomy - Garde-fous de l'execution autonome (bot eToro).

Exception consciente a l'invariant du moteur ("le moteur ne decide jamais
d'entrer", voir CLAUDE.md) : decidee avec l'utilisateur le 09.10.2026 pour le
bot eToro uniquement. gate.py et evaluate() restent inchanges et continuent a
qualifier-et-refuser pour IBKR ; ce module est une couche separee, au-dessus
du moteur de qualification, qui porte les limites explicitement acceptees
pour laisser une boucle logicielle ouvrir et fermer des positions sans
confirmation humaine par trade :

    - stop-loss obligatoire sur toute position (impose cote eToro de toute
      facon pour un sellShort ou un levier > 1, voir sources/etoro.py) ;
    - coupe-circuit : perte journaliere max en fraction du capital total ;
    - plafond de positions simultanees ;
    - palier de levier qui monte apres des semaines beneficiaires d'affilee
      et redescend d'un cran apres une seule semaine perdante ;
    - sortie forcee a la fin de l'horizon promis par le verdict si le stop
      n'a pas ete touche avant ;
    - interrupteur d'urgence sur fichier, verifie avant chaque action.

Toutes les fonctions sont pures sauf kill_switch_active(), qui lit le
systeme de fichiers par construction (c'est son role).

Reference : accord du 09.10.2026 (session), pas encore reporte dans
docu/methode/01-regles.md - a faire si ce module se stabilise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Paliers de levier CFD, du plus prudent au plus permissif. Point de depart
# impose : indice 0 (1:1, aucun levier reel).
LEVERAGE_LADDER: tuple[int, ...] = (1, 2, 5)

# Semaines beneficiaires d'affilee necessaires pour monter d'un palier.
PROMOTION_STREAK = 4

# Perte journaliere maximale avant arret automatique, en fraction du capital
# TOTAL du compte (pas une enveloppe separee - choix explicite de
# l'utilisateur le 09.10.2026).
DAILY_BREAKER_PCT = 0.10

# Positions ouvertes simultanement, maximum.
MAX_OPEN_POSITIONS = 3


def leverage_tier_after_week(
    current_tier: int, consecutive_profitable_weeks: int, week_pnl: float
) -> tuple[int, int]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Fait evoluer le palier de levier apres une semaine de resultats.
        Monte d'un palier apres PROMOTION_STREAK semaines beneficiaires
        d'affilee ; redescend d'un palier des la premiere semaine perdante
        (asymetrique par choix explicite : la sanction est immediate, la
        recompense ne l'est pas).

    Inputs:
        current_tier (int): indice courant dans LEVERAGE_LADDER
        consecutive_profitable_weeks (int): serie en cours avant cette semaine
        week_pnl (float): resultat net de la semaine qui vient de se terminer

    Outputs:
        (nouveau_tier, nouvelle_serie) (tuple[int, int])
    --------------------------------------------------------------------------
    """
    if not 0 <= current_tier < len(LEVERAGE_LADDER):
        raise ValueError(f"palier hors bornes : {current_tier}")
    if week_pnl < 0:
        return max(0, current_tier - 1), 0
    streak = consecutive_profitable_weeks + 1
    if streak >= PROMOTION_STREAK and current_tier < len(LEVERAGE_LADDER) - 1:
        return current_tier + 1, 0
    return current_tier, streak


def current_leverage(tier: int) -> int:
    """Multiplicateur de levier pour un indice de palier donne."""
    if not 0 <= tier < len(LEVERAGE_LADDER):
        raise ValueError(f"palier hors bornes : {tier}")
    return LEVERAGE_LADDER[tier]


def daily_breaker_triggered(
    cumulative_pnl_today: float, capital_total: float, max_loss_pct: float = DAILY_BREAKER_PCT
) -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Coupe-circuit journalier. Le bot doit arreter toute nouvelle action
        pour la journee si la perte cumulee depasse la fraction autorisee du
        capital total du compte.

    Inputs:
        cumulative_pnl_today (float): resultat net cumule depuis l'ouverture
            de la seance (negatif si perte)
        capital_total (float): capital total du compte eToro
        max_loss_pct (float): fraction maximale tolere, defaut DAILY_BREAKER_PCT

    Outputs:
        triggered (bool): vrai si le coupe-circuit doit se declencher
    --------------------------------------------------------------------------
    """
    if capital_total <= 0:
        raise ValueError("capital total doit etre strictement positif")
    if not 0 < max_loss_pct <= 1:
        raise ValueError("max_loss_pct hors bornes (0, 1]")
    return cumulative_pnl_today < 0 and abs(cumulative_pnl_today) >= capital_total * max_loss_pct


def can_open_new_position(open_positions: int, max_positions: int = MAX_OPEN_POSITIONS) -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Plafond de positions simultanees. Une nouvelle position ne peut
        s'ouvrir que si le nombre de positions deja ouvertes est strictement
        sous le plafond.

    Inputs:
        open_positions (int): positions actuellement ouvertes par le bot
        max_positions (int): plafond, defaut MAX_OPEN_POSITIONS

    Outputs:
        allowed (bool): vrai si une nouvelle position peut s'ouvrir
    --------------------------------------------------------------------------
    """
    if open_positions < 0:
        raise ValueError("open_positions negatif")
    return open_positions < max_positions


def sessions_elapsed(opened_on: date, today: date) -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        Nombre de seances ouvrables (lundi-vendredi) ecoulees entre la date
        d'ouverture d'une position et aujourd'hui, le jour d'ouverture compte
        comme la seance 0. Approximation : les jours feries US ne sont PAS
        retires, parce qu'aucun calendrier ferie n'est cable ici - une
        sortie a l'horizon peut donc se declencher un jour ferie. A corriger
        si un calendrier de seances fiable est ajoute au depot.

    Inputs:
        opened_on (date): date d'ouverture de la position
        today (date): date du jour

    Outputs:
        n (int): nombre de seances ouvrables ecoulees (0 le jour meme)
    --------------------------------------------------------------------------
    """
    if today < opened_on:
        raise ValueError("today est avant opened_on")
    n = 0
    cursor = opened_on
    while cursor < today:
        cursor = date.fromordinal(cursor.toordinal() + 1)
        if cursor.weekday() < 5:  # 0=lundi ... 4=vendredi
            n += 1
    return n


def horizon_exit_due(opened_on: date, horizon_sessions: int, today: date) -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Dit si une position doit etre fermee parce que l'horizon promis par
        le verdict qui l'a ouverte est ecoule, independamment du stop. Un
        verdict PUMP_RISK promet "jour a trois seances", pas "jusqu'a ce que
        ca tourne" - au-dela, la position n'est plus celle que le signal a
        justifiee (regle actee le 09.10.2026).

    Inputs:
        opened_on (date): date d'ouverture de la position
        horizon_sessions (int): horizon promis, en seances, LA SEANCE
            D'OUVERTURE COMPTANT COMME LA PREMIERE (1 pour FADE/jour meme,
            3 pour PUMP_RISK : jour d'ouverture + 2 seances)
        today (date): date du jour

    Outputs:
        due (bool): vrai si la position doit etre fermee a la cloture du jour
    --------------------------------------------------------------------------
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions doit etre strictement positif")
    return sessions_elapsed(opened_on, today) >= horizon_sessions - 1


def kill_switch_active(flag_path: Path) -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        Interrupteur d'urgence. Verifie avant CHAQUE action du bot (ouverture,
        fermeture, changement de palier de levier) : si le fichier existe, le
        bot n'entreprend plus rien de nouveau. Seule fonction de ce module a
        toucher le systeme de fichiers - c'est son role, pas un effet de bord
        accidentel.

    Inputs:
        flag_path (Path): chemin du fichier interrupteur

    Outputs:
        active (bool): vrai si l'interrupteur est arme (fichier present)
    --------------------------------------------------------------------------
    """
    return flag_path.exists()


@dataclass(frozen=True)
class AutonomyState:
    """Etat persistant de l'enveloppe autonome, independant d'une position."""

    leverage_tier: int = 0
    consecutive_profitable_weeks: int = 0
    open_positions: int = 0
    cumulative_pnl_today: float = 0.0

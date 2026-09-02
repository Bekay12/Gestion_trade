"""
gate - Le crible de decision : R3, R6, R9, R10, R11, R12.

Ce module decide si une operation peut etre armee. Sa regle de conception tient
en une phrase : un motif de refus BLOQUE, il n'avertit pas. La seance perdante
documentee dans le corpus n'a pas eu lieu par ignorance de la regle mais parce
qu'au moment de decider, la conviction l'a emporte sur une regle connue. Un
moteur qui se contente d'afficher un avertissement reproduit cette situation
avec une etape de plus.

Corollaire : une donnee absente est traitee comme un refus, jamais comme un
feu vert. Voir models.Snapshot.

Reference : docu/methode/02-logique-decision.md
"""

from __future__ import annotations

import logging
from datetime import time

from .models import RiskProfile, Snapshot, TradePlan, Verdict
from .rules import (
    ROTATION_BANDS,
    SHORT_INTEREST_BANDS,
    band,
    drawdown_scale,
    float_rotation,
    position_size,
    relative_volume,
    risk_amount,
    risk_reward,
    short_interest_pct,
)

logger = logging.getLogger(__name__)

# R6 - bornes du terrain de jeu de la methode.
MIN_PRICE, MAX_PRICE = 0.50, 20.0
MIN_AVG_VOLUME = 500_000
MIN_RVOL = 2.0
MAX_MARKET_CAP = 300_000_000
LOW_FLOAT_CEILING = 20_000_000

# Garde-fou derive, absent du corpus : celui-ci signale que les ecarts de
# pre-marche sont trois a cinq fois plus larges qu'en seance, sans donner de
# seuil. Un ecart large est un cout d'aller-retour paye avant tout mouvement,
# et sur un flottant reduit il depasse couramment ce que vise la these.
SPREAD_WARN = 0.02
SPREAD_BLOCK = 0.05

# R9 - seuils de la configuration d'exclusion.
R9_FLOAT_CEILING = 20_000_000
R9_SHORT_INTEREST = 0.15

# R10 - fenetres horaires, heure de l'Est.
OPEN_TIME = time(9, 30)
NO_TRADE_UNTIL = time(9, 35)
PRIME_START, PRIME_END = time(10, 0), time(11, 30)
LUNCH_START, LUNCH_END = time(11, 30), time(14, 0)
CLOSE_TIME = time(16, 0)


def check_account(profile: RiskProfile, account, verdict: Verdict) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R3, R4, R12 - bornes portant sur le compte, pas sur le titre. Elles
        s'evaluent en premier : aucune qualite de setup ne rouvre une seance
        close par la perte journaliere.

    Inputs:
        profile (RiskProfile): parametres de risque
        account: portefeuille exposant daily_pnl, drawdown, consecutive_losses
        verdict (Verdict): accumulateur de motifs

    Outputs:
        scale (float): coefficient de taille issu du drawdown (R4)
    --------------------------------------------------------------------------
    """
    limit = -abs(profile.capital * profile.daily_max_loss_pct)
    if account.daily_pnl <= limit:
        verdict.blocks.append(
            f"R3 : perte journaliere atteinte ({account.daily_pnl:.2f} <= {limit:.2f})"
        )

    if account.consecutive_losses >= profile.max_consecutive_losses:
        verdict.blocks.append(
            f"R12 : {account.consecutive_losses} pertes consecutives, seance close"
        )

    scale = drawdown_scale(account.drawdown)
    if scale == 0.0:
        verdict.blocks.append(
            f"R4 : drawdown {account.drawdown:.1%} — simulation uniquement"
        )
    elif scale < 1.0:
        verdict.warnings.append(
            f"R4 : drawdown {account.drawdown:.1%} — taille reduite a {scale:.0%}"
        )
    return scale


def check_filters(snap: Snapshot, verdict: Verdict) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        R6 - filtres de selection du titre.

    Inputs:
        snap (Snapshot): etat marche
        verdict (Verdict): accumulateur de motifs

    Outputs:
        None (verdict enrichi sur place)
    --------------------------------------------------------------------------
    """
    if snap.halted:
        verdict.blocks.append("R6 : titre suspendu (halt)")

    if not MIN_PRICE <= snap.price <= MAX_PRICE:
        verdict.blocks.append(
            f"R6 : prix {snap.price:.2f} hors bornes [{MIN_PRICE}, {MAX_PRICE}]"
        )

    if snap.average_volume < MIN_AVG_VOLUME:
        verdict.blocks.append(
            f"R6 : volume moyen {snap.average_volume:,.0f} < {MIN_AVG_VOLUME:,}"
        )
    else:
        rvol = relative_volume(snap.volume, snap.average_volume)
        if rvol < MIN_RVOL:
            verdict.blocks.append(f"R6 : volume relatif {rvol:.2f} < {MIN_RVOL}")

    if snap.market_cap is not None and snap.market_cap > MAX_MARKET_CAP:
        verdict.warnings.append(
            f"R6 : capitalisation {snap.market_cap:,.0f} au-dela du terrain habituel"
        )

    spread = snap.spread_pct
    if spread is not None:
        if spread >= SPREAD_BLOCK:
            verdict.blocks.append(
                f"R6 : ecart acheteur-vendeur {spread:.1%} — l'aller-retour coute "
                "plus que ce que vise la these"
            )
        elif spread >= SPREAD_WARN:
            verdict.warnings.append(f"R6 : ecart {spread:.1%}, execution couteuse")

    if snap.quote_stale:
        verdict.warnings.append(
            "Donnee : dernier prix repris de la cloture, aucune transaction dans la seance"
        )

    if snap.free_float is None:
        verdict.warnings.append("R6 : flottant inconnu — verification manuelle requise")
    elif snap.free_float < LOW_FLOAT_CEILING:
        rot = float_rotation(snap.volume, snap.free_float)
        verdict.warnings.append(
            f"R6 : flottant reduit ({snap.free_float:,.0f}), rotation "
            f"{rot:.2f}x ({band(rot, ROTATION_BANDS, 'suspecte')})"
        )


def check_short_constraints(snap: Snapshot, verdict: Verdict) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        R7 et R9 - contraintes propres a la vente a decouvert.

        R9 est le coeur du module. La configuration « restriction active +
        titre difficile a emprunter + flottant reduit + interet vendeur eleve »
        est celle qui ressemble le plus a une opportunite et qui coute le plus
        cher : les vendeurs en place paient chaque jour, ne peuvent plus
        frapper le bid, et la moindre poussee les force a racheter.

        Les donnees d'emprunt et de restriction n'etant pas diffusees
        gratuitement, leur absence bloque : le moteur refuse plutot que de
        supposer favorable ce qu'il ignore.

    Inputs:
        snap (Snapshot): etat marche
        verdict (Verdict): accumulateur de motifs

    Outputs:
        None (verdict enrichi sur place)
    --------------------------------------------------------------------------
    """
    if snap.borrow is None:
        verdict.blocks.append(
            "R7 : statut d'emprunt inconnu — renseigner avant toute position vendeuse"
        )
    elif snap.borrow == "none":
        verdict.blocks.append("R7 : titre indisponible a l'emprunt, non jouable")

    if snap.ssr_active is None:
        verdict.blocks.append(
            "R9 : statut Rule 201 inconnu — la regle d'exclusion n'est pas calculable"
        )

    if snap.borrow_rate is not None and snap.borrow_rate >= 0.50:
        verdict.warnings.append(
            f"R7 : cout d'emprunt {snap.borrow_rate:.0%}/an — trade necessairement court"
        )

    # R9 : le cumul, evalue seulement si les quatre donnees sont disponibles.
    conditions: list[str] = []
    if snap.ssr_active:
        conditions.append("Rule 201 active")
    if snap.borrow == "hard":
        conditions.append("difficile a emprunter")
    if snap.free_float is not None and snap.free_float < R9_FLOAT_CEILING:
        conditions.append("flottant reduit")
    if snap.shares_short is not None and snap.free_float:
        si = short_interest_pct(snap.shares_short, snap.free_float)
        if si >= R9_SHORT_INTEREST:
            conditions.append(
                f"interet vendeur {si:.1%} ({band(si, SHORT_INTEREST_BANDS, 'extreme')})"
            )

    if len(conditions) >= 3:
        verdict.blocks.append(
            "R9 : configuration piege pour les vendeurs — " + ", ".join(conditions)
        )
    elif len(conditions) == 2:
        verdict.warnings.append("R9 : deux conditions reunies — " + ", ".join(conditions))


def check_timing(now: time | None, verdict: Verdict) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        R10 - fenetres horaires. Le pre-marche sert a analyser, pas a operer.

    Inputs:
        now (time | None): heure de l'Est, None pour ignorer le controle
        verdict (Verdict): accumulateur de motifs

    Outputs:
        None (verdict enrichi sur place)
    --------------------------------------------------------------------------
    """
    if now is None:
        return
    if now < OPEN_TIME:
        verdict.blocks.append("R10 : pre-marche — analyse seulement")
    elif now < NO_TRADE_UNTIL:
        verdict.blocks.append("R10 : cinq premieres minutes — laisser le marche se fixer")
    elif now >= CLOSE_TIME:
        verdict.blocks.append("R10 : seance close")
    elif LUNCH_START <= now < LUNCH_END:
        verdict.warnings.append("R10 : creux de milieu de journee, signaux moins fiables")
    elif not (PRIME_START <= now < PRIME_END):
        verdict.warnings.append("R10 : hors fenetre principale (10h00-11h30)")


def evaluate(
    plan: TradePlan,
    snap: Snapshot,
    profile: RiskProfile,
    account,
    now: time | None = None,
) -> Verdict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Passer un plan de trade au crible complet et calculer sa taille.

    Inputs:
        plan (TradePlan): plan redige avant l'ouverture
        snap (Snapshot): etat marche du titre
        profile (RiskProfile): parametres de risque
        account: portefeuille papier
        now (time | None): heure de l'Est pour le controle R10

    Outputs:
        verdict (Verdict): autorisation, motifs et taille calculee
    --------------------------------------------------------------------------
    """
    verdict = Verdict()

    if plan.symbol != snap.symbol:
        verdict.blocks.append("Coherence : le plan et l'etat marche portent sur deux titres")
        return verdict

    if not plan.catalyst.strip():
        verdict.blocks.append("R11 : aucun catalyseur enonce — le titre sort de la liste")

    # Coherence du sens : le stop est du cote ou la these est invalidee.
    if plan.direction == "long" and plan.stop >= plan.entry:
        verdict.blocks.append("R5 : stop au-dessus de l'entree sur une position acheteuse")
    if plan.direction == "short" and plan.stop <= plan.entry:
        verdict.blocks.append("R5 : stop sous l'entree sur une position vendeuse")

    scale = check_account(profile, account, verdict)
    check_timing(now, verdict)
    check_filters(snap, verdict)
    if plan.direction == "short":
        check_short_constraints(snap, verdict)

    # R2 : le rapport gain/risque se mesure sur le premier objectif.
    if plan.targets:
        try:
            ratio = risk_reward(plan.entry, plan.stop, plan.targets[0])
            verdict.risk_reward = ratio
            if ratio < profile.min_risk_reward:
                verdict.blocks.append(
                    f"R2 : rapport {ratio:.2f} < {profile.min_risk_reward} requis"
                )
        except ValueError as error:
            verdict.blocks.append(f"R2 : {error}")
    else:
        verdict.blocks.append("R11 : aucun objectif defini")

    # R1 : la taille n'est calculee que si tout le reste passe.
    if verdict.allowed:
        try:
            amount = risk_amount(profile.capital, profile.risk_pct) * scale
            verdict.size = position_size(amount, plan.entry, plan.stop)
            if verdict.size <= 0:
                verdict.blocks.append("R1 : taille calculee nulle, ecart entree/stop trop large")
        except ValueError as error:
            verdict.blocks.append(f"R1 : {error}")

    logger.info(
        "[GATE] %s %s -> %s",
        plan.symbol,
        plan.direction,
        "arme" if verdict.allowed else "refuse: " + "; ".join(verdict.blocks),
    )
    return verdict

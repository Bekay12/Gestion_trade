"""
eastern - Heure de l'Est et etat du marche americain.

Un seul endroit sait convertir l'heure locale en heure de marche. Le decalage
etait ecrit deux fois — dans live.py et dans le preflight — et une constante
horaire dupliquee finit toujours par diverger.

Les bornes de seance viennent de gate.py (R10), pas d'une copie : la fenetre
qui autorise a operer et celle qui decide si une donnee est plausible sont la
meme fenetre.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .gate import CLOSE_TIME, OPEN_TIME

# Le marche americain vit en heure de l'Est ; l'ecart a UTC est de 4 heures en
# heure d'ete, periode qui couvre l'usage courant. Un decalage d'une heure hors
# saison ne fait que deplacer les bornes R10, il ne fausse aucun calcul.
EASTERN_OFFSET = timedelta(hours=-4)

JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def eastern_now() -> datetime:
    """Heure de l'Est approchee depuis UTC, sans fuseau attache."""
    return (datetime.now(timezone.utc) + EASTERN_OFFSET).replace(tzinfo=None)


def market_open(now: datetime | None = None) -> tuple[bool, str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Dire si le marche est ouvert, et le formuler pour un humain.

        Sans ce controle, un diagnostic impute a un droit manquant ce qui n'est
        qu'un week-end : hors seance il n'y a NI carnet NI transaction. Le
        2026-08-23, un preflight lance un dimanche a conclu a un abonnement
        absent et a failli faire souscrire pour rien.

    Inputs:
        now (datetime | None): instant de reference, heure de l'Est

    Outputs:
        (ouvert, libelle) (tuple[bool, str]): etat et texte affichable
    --------------------------------------------------------------------------
    """
    est = now or eastern_now()
    horodatage = f"{JOURS[est.weekday()]} {est:%H:%M} (Est)"
    if est.weekday() >= 5:
        return False, f"FERME — {horodatage}, week-end"
    if not (OPEN_TIME <= est.time() < CLOSE_TIME):
        return False, f"FERME — {horodatage}, hors seance"
    return True, f"OUVERT — {horodatage}"

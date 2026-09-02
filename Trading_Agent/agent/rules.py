"""
rules - Calculs purs des regles R1, R2, R4, R7 et R8.

Aucun effet de bord, aucune donnee externe : ces fonctions sont testables hors
ligne et constituent le socle verifiable du moteur. Les regles de decision qui
combinent ces calculs vivent dans gate.py.

Reference : docu/methode/01-regles.md
"""

from __future__ import annotations

import math

# R1 - taux de risque par trade selon l'experience, en fraction du capital.
SIZING_LADDER: tuple[tuple[int, float], ...] = (
    (6, 0.005),    # 0-6 mois
    (18, 0.01),    # 6-18 mois
    (36, 0.02),    # 18 mois et plus
)
PRO_RISK_PCT = 0.03

# R4 - paliers de drawdown : seuil atteint -> coefficient applique a la taille.
DRAWDOWN_LADDER: tuple[tuple[float, float], ...] = (
    (0.05, 1.00),
    (0.10, 0.75),
    (0.15, 0.50),
    (0.20, 0.00),   # simulation uniquement
)

# R8 - lecture de l'interet vendeur, en fraction du flottant.
SHORT_INTEREST_BANDS: tuple[tuple[float, str], ...] = (
    (0.05, "neutre"),
    (0.15, "modere"),
    (0.25, "eleve"),
    (0.40, "tres eleve"),
)

# R8 - rotation du flottant : volume rapporte au flottant.
ROTATION_BANDS: tuple[tuple[float, str], ...] = (
    (0.5, "calme"),
    (1.0, "moderee"),
    (3.0, "active"),
    (7.0, "exceptionnelle"),
)


def risk_amount(capital: float, risk_pct: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R1 - montant maximal risque sur une operation.

    Inputs:
        capital (float): capital total
        risk_pct (float): fraction risquee, ex. 0.01

    Outputs:
        amount (float): montant en devise du compte
    --------------------------------------------------------------------------
    """
    if capital <= 0:
        raise ValueError("capital doit etre strictement positif")
    if not 0 < risk_pct <= 0.05:
        raise ValueError("risk_pct hors bornes raisonnables (0 < r <= 5 %)")
    return capital * risk_pct


def position_size(risk: float, entry: float, stop: float) -> int:
    """
    --------------------------------------------------------------------------
    Purpose:
        R1 - taille de position. La taille est une consequence du risque
        accepte et du niveau d'invalidation, jamais un choix premier.

    Inputs:
        risk (float): montant risque
        entry (float): prix d'entree
        stop (float): niveau d'invalidation

    Outputs:
        size (int): nombre d'actions, arrondi a l'inferieur

    Raises:
        ValueError: si l'ecart entree/stop est nul
    --------------------------------------------------------------------------
    """
    # Les prix sont des decimaux exprimes en binaire : 5.00 - 4.80 vaut
    # 0.20000000000000018, et une division entiere brute rend 499 au lieu de
    # 500. L'erreur est systematique et toujours dans le meme sens, donc elle
    # sous-dimensionne chaque position. On arrondit au millionieme, sous la
    # granularite de toute cotation, avant de plancher.
    per_share = round(abs(entry - stop), 6)
    if per_share <= 0:
        raise ValueError("entree et stop confondus : risque par action nul")
    return int(math.floor(risk / per_share + 1e-9))


def risk_reward(entry: float, stop: float, target: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R2 - rapport gain/risque. Fonctionne dans les deux sens : sur une
        position vendeuse, la cible est sous l'entree et le stop au-dessus.

    Inputs:
        entry (float): prix d'entree
        stop (float): niveau d'invalidation
        target (float): objectif

    Outputs:
        ratio (float): gain potentiel divise par la perte maximale
    --------------------------------------------------------------------------
    """
    loss = abs(entry - stop)
    if loss <= 0:
        raise ValueError("entree et stop confondus : perte maximale nulle")
    return abs(target - entry) / loss


def breakeven_win_rate(ratio: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R2 - taux de reussite au-dela duquel un rapport donne devient
        profitable. C'est ce qui rend la rentabilite independante du besoin
        d'avoir raison.

    Inputs:
        ratio (float): rapport gain/risque

    Outputs:
        rate (float): fraction de trades gagnants necessaire
    --------------------------------------------------------------------------
    """
    if ratio <= 0:
        raise ValueError("ratio doit etre strictement positif")
    return 1.0 / (1.0 + ratio)


def sizing_pct(months_experience: int, has_journal: bool = False) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R1 - taux de risque autorise selon l'anciennete. Le palier
        professionnel exige un journal tenu.

    Inputs:
        months_experience (int): mois de pratique
        has_journal (bool): journal de trading tenu

    Outputs:
        pct (float): fraction du capital risquable par trade
    --------------------------------------------------------------------------
    """
    if months_experience < 0:
        raise ValueError("anciennete negative")
    for ceiling, pct in SIZING_LADDER:
        if months_experience < ceiling:
            return pct
    return PRO_RISK_PCT if has_journal else SIZING_LADDER[-1][1]


def drawdown_scale(drawdown: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R4 - coefficient applique a la taille de position selon la perte
        depuis le sommet de capital. Zero signifie arret des positions reelles.

    Inputs:
        drawdown (float): fraction perdue depuis le sommet, ex. 0.12

    Outputs:
        scale (float): multiplicateur de taille entre 0 et 1
    --------------------------------------------------------------------------
    """
    if drawdown < 0:
        raise ValueError("drawdown negatif")
    for ceiling, scale in DRAWDOWN_LADDER:
        if drawdown < ceiling:
            return scale
    return 0.0


def borrow_cost(position_value: float, annual_rate: float, days: int = 1) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R7 - cout d'emprunt d'une position vendeuse. Ce cout court meme si le
        cours ne bouge pas : il s'integre au rapport gain/risque avant l'entree.

    Inputs:
        position_value (float): valeur de la position
        annual_rate (float): taux annualise, ex. 0.8 pour 80 %/an
        days (int): nombre de jours de detention

    Outputs:
        cost (float): cout total en devise du compte
    --------------------------------------------------------------------------
    """
    if position_value < 0 or annual_rate < 0 or days < 0:
        raise ValueError("valeurs negatives interdites")
    return position_value * annual_rate / 365.0 * days


def ssr_triggered(price: float, previous_close: float) -> bool:
    """
    --------------------------------------------------------------------------
    Purpose:
        R7 - la restriction Rule 201 se declenche a -10 % sous la cloture de la
        veille. Ne dit PAS si elle est active : une restriction declenchee hier
        court encore aujourd'hui, ce que ce calcul ne peut pas savoir.

    Inputs:
        price (float): cours courant
        previous_close (float): cloture de la veille

    Outputs:
        triggered (bool): le seuil de declenchement est franchi aujourd'hui
    --------------------------------------------------------------------------
    """
    if previous_close <= 0:
        raise ValueError("cloture precedente invalide")
    return price <= previous_close * 0.90


def short_interest_pct(shares_short: float, free_float: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R8 - interet vendeur rapporte au FLOTTANT, jamais aux actions emises.

    Inputs:
        shares_short (float): actions vendues a decouvert
        free_float (float): flottant reel

    Outputs:
        pct (float): fraction du flottant vendue a decouvert
    --------------------------------------------------------------------------
    """
    if free_float <= 0:
        raise ValueError("flottant invalide")
    return shares_short / free_float


def days_to_cover(shares_short: float, average_volume: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R8 - jours de volume moyen necessaires aux vendeurs pour deboucler.

    Inputs:
        shares_short (float): actions vendues a decouvert
        average_volume (float): volume quotidien moyen

    Outputs:
        days (float): nombre de jours
    --------------------------------------------------------------------------
    """
    if average_volume <= 0:
        raise ValueError("volume moyen invalide")
    return shares_short / average_volume


def float_rotation(volume: float, free_float: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R8 - nombre de fois que le flottant a change de mains dans la seance.

    Inputs:
        volume (float): volume de la seance
        free_float (float): flottant reel

    Outputs:
        rotation (float): multiple du flottant
    --------------------------------------------------------------------------
    """
    if free_float <= 0:
        raise ValueError("flottant invalide")
    return volume / free_float


def relative_volume(volume: float, average_volume: float) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        R6 - volume relatif, le filtre decisif : il detecte l'anomalie du jour
        plutot que la taille de l'entreprise.

    Inputs:
        volume (float): volume de la seance
        average_volume (float): volume quotidien moyen

    Outputs:
        rvol (float): multiple du volume habituel
    --------------------------------------------------------------------------
    """
    if average_volume <= 0:
        raise ValueError("volume moyen invalide")
    return volume / average_volume


def band(value: float, bands: tuple[tuple[float, str], ...], above: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Nommer la plage dans laquelle tombe une mesure (R8).

    Inputs:
        value (float): mesure
        bands (tuple): couples (plafond, libelle) ordonnes
        above (str): libelle au-dela du dernier plafond

    Outputs:
        label (str): libelle de la plage
    --------------------------------------------------------------------------
    """
    for ceiling, label in bands:
        if value < ceiling:
            return label
    return above

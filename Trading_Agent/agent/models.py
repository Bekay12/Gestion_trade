"""
models - Types partages du moteur de decision.

Un principe gouverne ces structures : une donnee absente vaut None, jamais une
valeur par defaut. Le moteur traite None comme un refus, pas comme un feu vert.
Les donnees d'emprunt et de restriction Rule 201 ne sont pas diffusees
gratuitement ; les modeliser comme optionnelles est ce qui empeche le moteur de
supposer favorable ce qu'il ignore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

Direction = Literal["long", "short"]
BorrowStatus = Literal["easy", "hard", "none"]


@dataclass(frozen=True)
class RiskProfile:
    """
    --------------------------------------------------------------------------
    Purpose:
        Parametres de risque de l'operateur (R1, R3, R12).

    Inputs:
        capital (float): capital total du compte
        risk_pct (float): fraction risquee par trade, ex. 0.01 pour 1 %
        daily_max_loss_pct (float): perte journaliere maximale, ex. 0.03
        max_consecutive_losses (int): pertes consecutives avant arret
        min_risk_reward (float): rapport gain/risque minimal admis

    Outputs:
        (dataclass immuable)
    --------------------------------------------------------------------------
    """

    capital: float
    risk_pct: float = 0.01
    daily_max_loss_pct: float = 0.03
    max_consecutive_losses: int = 3
    min_risk_reward: float = 2.0


@dataclass(frozen=True)
class Snapshot:
    """
    --------------------------------------------------------------------------
    Purpose:
        Etat marche d'un titre a l'instant de la decision.

    Inputs:
        symbol (str): ticker
        price (float): dernier cours
        previous_close (float): cloture de la veille, pour la Rule 201
        volume (float): volume cumule de la seance
        average_volume (float): volume quotidien moyen
        market_cap (float | None): capitalisation
        free_float (float | None): flottant reel, pas les actions emises
        shares_short (float | None): actions vendues a decouvert
        borrow (BorrowStatus | None): disponibilite a l'emprunt, None si inconnue
        borrow_rate (float | None): taux annualise, ex. 0.8 pour 80 %/an
        ssr_active (bool | None): restriction Rule 201, None si inconnue
        vwap (float | None): prix moyen pondere de la seance
        halted (bool): titre suspendu

    Outputs:
        (dataclass immuable)
    --------------------------------------------------------------------------
    """

    symbol: str
    price: float
    previous_close: float
    volume: float
    average_volume: float
    market_cap: float | None = None
    free_float: float | None = None
    shares_short: float | None = None
    borrow: BorrowStatus | None = None
    borrow_rate: float | None = None
    ssr_active: bool | None = None
    vwap: float | None = None
    halted: bool = False
    asof: datetime | None = None
    # Haut du carnet. Fourni par un courtier, absent des sources publiques.
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    # Qualite de la donnee, pour que le moteur sache sur quoi il decide.
    realtime: bool = False
    quote_stale: bool = False

    @property
    def spread_pct(self) -> float | None:
        """
        Ecart acheteur-vendeur rapporte au milieu. None si le carnet est
        indisponible. Sur un flottant reduit en pre-marche, cet ecart depasse
        couramment 10 %, ce qui rend l'execution plus couteuse que la these.
        """
        if not self.bid or not self.ask or self.ask <= 0:
            return None
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid if mid > 0 else None


@dataclass(frozen=True)
class TradePlan:
    """
    --------------------------------------------------------------------------
    Purpose:
        Plan de trade au sens de R11 : redige avant l'ouverture, il porte le
        niveau d'invalidation et la condition d'annulation.

    Inputs:
        symbol (str): ticker
        direction (Direction): sens de la position
        catalyst (str): le catalyseur en une phrase (test d'admission R11)
        entry (float): prix d'entree vise
        stop (float): niveau d'invalidation
        targets (list[float]): objectifs echelonnes
        invalidation (str): evenement qui annule le plan

    Outputs:
        (dataclass immuable)
    --------------------------------------------------------------------------
    """

    symbol: str
    direction: Direction
    catalyst: str
    entry: float
    stop: float
    targets: list[float] = field(default_factory=list)
    invalidation: str = ""


@dataclass
class Verdict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Resultat du passage au crible. `blocks` non vide signifie refus : le
        moteur n'arme pas l'ordre, il ne se contente pas d'avertir.

    Inputs:
        blocks (list[str]): motifs de refus, regle citee
        warnings (list[str]): points d'attention sans effet bloquant
        size (int): taille de position calculee, 0 si refus
        risk_reward (float | None): rapport calcule

    Outputs:
        (dataclass mutable, construit par accumulation)
    --------------------------------------------------------------------------
    """

    blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    size: int = 0
    risk_reward: float | None = None

    @property
    def allowed(self) -> bool:
        """True seulement si aucun motif de refus n'a ete accumule."""
        return not self.blocks


@dataclass
class Position:
    """Position ouverte dans le portefeuille papier."""

    symbol: str
    direction: Direction
    size: int
    entry: float
    stop: float
    opened: datetime
    borrow_rate: float | None = None
    catalyst: str = ""

    def unrealised(self, price: float) -> float:
        """P&L latent, signe selon le sens de la position."""
        delta = price - self.entry
        return delta * self.size if self.direction == "long" else -delta * self.size


@dataclass
class ClosedTrade:
    """Operation cloturee, unite du journal."""

    symbol: str
    direction: Direction
    size: int
    entry: float
    exit_price: float
    opened: datetime
    closed: datetime
    pnl: float
    fees: float
    reason: str
    catalyst: str = ""

    @property
    def day(self) -> date:
        return self.closed.date()

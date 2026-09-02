"""
portfolio - Portefeuille papier : positions, executions simulees, journal.

Tient les grandeurs dont le crible a besoin pour appliquer R3, R4 et R12 :
resultat du jour, drawdown depuis le sommet de capital, pertes consecutives.
Aucune connexion a un courtier : les executions sont simulees au prix fourni,
avec un glissement configurable.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from .models import ClosedTrade, Direction, Position
from .rules import borrow_cost

logger = logging.getLogger(__name__)

# Glissement par defaut : les petites capitalisations a flottant reduit ne se
# traitent pas au prix affiche, surtout dans un mouvement rapide.
DEFAULT_SLIPPAGE = 0.002
DEFAULT_FEE = 1.0


class PaperPortfolio:
    """
    Portefeuille simule. Toute mutation passe par open_position ou
    close_position, de sorte que le journal soit toujours complet.
    """

    def __init__(
        self,
        capital: float,
        slippage: float = DEFAULT_SLIPPAGE,
        fee_per_trade: float = DEFAULT_FEE,
    ) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Initialiser un compte papier.

        Inputs:
            capital (float): capital de depart
            slippage (float): glissement applique a chaque execution
            fee_per_trade (float): frais fixes par execution

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        if capital <= 0:
            raise ValueError("capital initial invalide")
        self.initial = capital
        self.cash = capital
        self.slippage = slippage
        self.fee = fee_per_trade
        self.positions: dict[str, Position] = {}
        self.closed: list[ClosedTrade] = []
        self.peak_equity = capital
        self._today: date | None = None
        self._day_start_equity = capital

    # ---------------------------------------------------------------- etat

    @property
    def realised(self) -> float:
        """Somme des resultats nets deja encaisses."""
        return self.cash - self.initial

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """
        ----------------------------------------------------------------------
        Purpose:
            Capital total, latent inclus.

        Inputs:
            prices (dict | None): derniers cours par ticker

        Outputs:
            equity (float): cash augmente du latent des positions ouvertes
        ----------------------------------------------------------------------
        """
        prices = prices or {}
        latent = sum(
            pos.unrealised(prices[sym]) for sym, pos in self.positions.items() if sym in prices
        )
        return self.cash + latent

    @property
    def drawdown(self) -> float:
        """Fraction perdue depuis le sommet de capital (R4)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.cash) / self.peak_equity)

    @property
    def daily_pnl(self) -> float:
        """Resultat encaisse depuis le debut de la journee courante (R3)."""
        return self.cash - self._day_start_equity

    @property
    def consecutive_losses(self) -> int:
        """Pertes consecutives les plus recentes (R12)."""
        count = 0
        for trade in reversed(self.closed):
            if trade.pnl < 0:
                count += 1
            else:
                break
        return count

    def start_day(self, day: date) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Ouvrir une journee : fige le capital de reference pour R3.

        Inputs:
            day (date): jour de bourse

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        self._today = day
        self._day_start_equity = self.cash
        logger.info("[PF] journee %s ouverte, capital %.2f", day, self.cash)

    # ------------------------------------------------------------ mutations

    def open_position(
        self,
        symbol: str,
        direction: Direction,
        size: int,
        price: float,
        stop: float,
        when: datetime,
        borrow_rate: float | None = None,
        catalyst: str = "",
    ) -> Position:
        """
        ----------------------------------------------------------------------
        Purpose:
            Ouvrir une position au prix fourni, glissement defavorable applique.

        Inputs:
            symbol (str): ticker
            direction (Direction): sens
            size (int): nombre d'actions
            price (float): prix de reference
            stop (float): niveau d'invalidation
            when (datetime): horodatage
            borrow_rate (float | None): taux d'emprunt annualise si vendeuse
            catalyst (str): catalyseur enonce

        Outputs:
            position (Position): la position ouverte

        Raises:
            ValueError: taille nulle ou position deja ouverte sur ce titre
        ----------------------------------------------------------------------
        """
        if size <= 0:
            raise ValueError("taille de position nulle")
        if symbol in self.positions:
            raise ValueError(f"position deja ouverte sur {symbol}")

        fill = price * (1 + self.slippage) if direction == "long" else price * (1 - self.slippage)
        self.cash -= self.fee
        position = Position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry=fill,
            stop=stop,
            opened=when,
            borrow_rate=borrow_rate,
            catalyst=catalyst,
        )
        self.positions[symbol] = position
        logger.info("[PF] ouverture %s %s %d @ %.4f", direction, symbol, size, fill)
        return position

    def close_position(
        self, symbol: str, price: float, when: datetime, reason: str = "manuelle"
    ) -> ClosedTrade:
        """
        ----------------------------------------------------------------------
        Purpose:
            Cloturer une position et l'inscrire au journal. Le cout d'emprunt
            d'une position vendeuse est deduit ici : il court par jour de
            detention, meme si le cours n'a pas bouge (R7).

        Inputs:
            symbol (str): ticker
            price (float): prix de reference
            when (datetime): horodatage
            reason (str): motif de sortie

        Outputs:
            trade (ClosedTrade): l'operation cloturee
        ----------------------------------------------------------------------
        """
        if symbol not in self.positions:
            raise ValueError(f"aucune position ouverte sur {symbol}")
        pos = self.positions.pop(symbol)

        fill = price * (1 - self.slippage) if pos.direction == "long" else price * (1 + self.slippage)
        gross = pos.unrealised(fill)

        fees = self.fee
        if pos.direction == "short" and pos.borrow_rate:
            days = max(1, (when.date() - pos.opened.date()).days)
            fees += borrow_cost(abs(pos.size * pos.entry), pos.borrow_rate, days)

        net = gross - fees
        self.cash += net
        self.peak_equity = max(self.peak_equity, self.cash)

        trade = ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            size=pos.size,
            entry=pos.entry,
            exit_price=fill,
            opened=pos.opened,
            closed=when,
            pnl=net,
            fees=fees,
            reason=reason,
            catalyst=pos.catalyst,
        )
        self.closed.append(trade)
        logger.info("[PF] cloture %s @ %.4f -> %+.2f (%s)", symbol, fill, net, reason)
        return trade

    def apply_stops(self, prices: dict[str, float], when: datetime) -> list[ClosedTrade]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Declencher les stops atteints. Le stop ne se deplace jamais dans le
            sens defavorable (R5) : cette methode ne fait que constater.

        Inputs:
            prices (dict): derniers cours par ticker
            when (datetime): horodatage

        Outputs:
            closed (list[ClosedTrade]): operations cloturees par stop
        ----------------------------------------------------------------------
        """
        hit: list[ClosedTrade] = []
        for symbol, pos in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            touched = price <= pos.stop if pos.direction == "long" else price >= pos.stop
            if touched:
                hit.append(self.close_position(symbol, pos.stop, when, reason="stop"))
        return hit

    # ------------------------------------------------------------- journal

    def stats(self) -> dict:
        """
        ----------------------------------------------------------------------
        Purpose:
            Agreger le journal en indicateurs de suivi.

        Inputs:
            None

        Outputs:
            stats (dict): nombre d'operations, taux de reussite, resultats
        ----------------------------------------------------------------------
        """
        wins = [t for t in self.closed if t.pnl > 0]
        losses = [t for t in self.closed if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        return {
            "trades": len(self.closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.closed) if self.closed else 0.0,
            "net": self.realised,
            "avg_win": gross_win / len(wins) if wins else 0.0,
            "avg_loss": -gross_loss / len(losses) if losses else 0.0,
            "profit_factor": gross_win / gross_loss if gross_loss else None,
            "drawdown": self.drawdown,
            "equity": self.cash,
        }

    def save(self, path: Path) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Ecrire le journal et l'etat sur disque, de facon atomique.

        Inputs:
            path (Path): fichier JSON cible

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        payload = {
            "initial": self.initial,
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "stats": self.stats(),
            "open": [
                {
                    "symbol": p.symbol, "direction": p.direction, "size": p.size,
                    "entry": p.entry, "stop": p.stop, "opened": p.opened.isoformat(),
                    "catalyst": p.catalyst,
                }
                for p in self.positions.values()
            ],
            "closed": [
                {
                    "symbol": t.symbol, "direction": t.direction, "size": t.size,
                    "entry": t.entry, "exit": t.exit_price,
                    "opened": t.opened.isoformat(), "closed": t.closed.isoformat(),
                    "pnl": round(t.pnl, 2), "fees": round(t.fees, 2),
                    "reason": t.reason, "catalyst": t.catalyst,
                }
                for t in self.closed
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        capital: float,
        slippage: float = DEFAULT_SLIPPAGE,
        fee_per_trade: float = DEFAULT_FEE,
    ) -> "PaperPortfolio":
        """
        ----------------------------------------------------------------------
        Purpose:
            Reprendre un portefeuille sauvegarde, ou en creer un neuf si aucun
            fichier n'existe. C'est ce qui permet a une session papier de durer
            plusieurs jours : sans reprise d'etat, le drawdown et la serie de
            pertes repartiraient de zero a chaque lancement, et R4 comme R12
            n'auraient plus de sens.

        Inputs:
            path (Path): fichier JSON ecrit par save()
            capital (float): capital initial si le fichier n'existe pas
            slippage (float): glissement applique aux executions
            fee_per_trade (float): frais fixes par execution

        Outputs:
            portfolio (PaperPortfolio): etat restaure
        ----------------------------------------------------------------------
        """
        if not path.exists():
            return cls(capital, slippage, fee_per_trade)

        data = json.loads(path.read_text(encoding="utf-8"))
        pf = cls(data.get("initial", capital), slippage, fee_per_trade)
        pf.cash = data.get("cash", pf.initial)
        pf.peak_equity = data.get("peak_equity", pf.cash)
        pf._day_start_equity = pf.cash

        for row in data.get("open", []):
            pf.positions[row["symbol"]] = Position(
                symbol=row["symbol"], direction=row["direction"], size=row["size"],
                entry=row["entry"], stop=row["stop"],
                opened=datetime.fromisoformat(row["opened"]),
                borrow_rate=row.get("borrow_rate"), catalyst=row.get("catalyst", ""),
            )
        for row in data.get("closed", []):
            pf.closed.append(ClosedTrade(
                symbol=row["symbol"], direction=row["direction"], size=row["size"],
                entry=row["entry"], exit_price=row["exit"],
                opened=datetime.fromisoformat(row["opened"]),
                closed=datetime.fromisoformat(row["closed"]),
                pnl=row["pnl"], fees=row["fees"], reason=row["reason"],
                catalyst=row.get("catalyst", ""),
            ))
        logger.info(
            "[PF] reprise : %.2f de capital, %d position(s), %d operation(s)",
            pf.cash, len(pf.positions), len(pf.closed),
        )
        return pf

"""
broker - Pont vers un instantane courtier depose par un connecteur externe.

Raison d'etre : le connecteur IBKR vit dans une session d'assistant, pas dans
ce processus Python. Un script ne peut donc pas l'appeler directement. Ce
module lit un fichier JSON qu'un producteur externe alimente — l'assistant
aujourd'hui, une connexion TWS/IB Gateway demain — de sorte que le moteur
ignore d'ou vient la donnee.

Ce que le courtier apporte et qu'aucune source publique ne donne :

    - le drapeau de suspension de cotation (halted), que R6 doit pouvoir
      rejeter et que le code mettait jusqu'ici en dur a False ;
    - le haut du carnet (bid/ask et tailles), donc l'ecart reel ;
    - un volume qui compte la seance de pre-marche ;
    - la distinction entre cotation vivante et cloture perimee.

La fraicheur est verifiee, jamais supposee : un instantane trop vieux est
ignore plutot que servi, parce qu'un prix perime est plus dangereux qu'un prix
absent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "docu" / "portefeuille" / "broker_snapshot.json"
# Au-dela, un instantane de courtier ne decrit plus la seance en cours.
DEFAULT_MAX_AGE = timedelta(minutes=15)


@dataclass(frozen=True)
class BrokerQuote:
    """Cotation courtier pour un titre."""

    symbol: str
    price: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    prior_close: float | None = None
    halted: bool = False
    realtime: bool = False
    quote_stale: bool = False
    asof: datetime | None = None
    # Emprunt : absent du connecteur MCP, fourni par une connexion TWS.
    # Voir sources/ibkr.py. None reste None, jamais "easy" par defaut.
    borrow: str | None = None
    borrow_rate: float | None = None
    shortable_shares: float | None = None

    def age(self, now: datetime | None = None) -> timedelta | None:
        if self.asof is None:
            return None
        return (now or datetime.now()) - self.asof


class BrokerBridge:
    """Lecture et ecriture du fichier d'instantanes."""

    def __init__(self, path: Path | None = None, max_age: timedelta = DEFAULT_MAX_AGE) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Preparer le pont.

        Inputs:
            path (Path | None): fichier JSON d'instantanes
            max_age (timedelta): age au-dela duquel une cotation est ignoree

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        self.path = path or DEFAULT_PATH
        self.max_age = max_age

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("[BROKER] instantane illisible : %s", error)
            return {}

    def quote(self, symbol: str, now: datetime | None = None) -> BrokerQuote | None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Lire la cotation d'un titre si elle est assez fraiche.

        Inputs:
            symbol (str): ticker
            now (datetime | None): instant de reference, pour les tests

        Outputs:
            quote (BrokerQuote | None): None si absente ou perimee
        ----------------------------------------------------------------------
        """
        row = (self._read().get("quotes") or {}).get(symbol.strip().upper())
        if not row:
            return None

        asof = None
        if row.get("asof"):
            try:
                asof = datetime.fromisoformat(row["asof"])
            except ValueError:
                asof = None

        quote = BrokerQuote(
            symbol=symbol.strip().upper(),
            price=row.get("price"), bid=row.get("bid"), ask=row.get("ask"),
            bid_size=row.get("bid_size"), ask_size=row.get("ask_size"),
            volume=row.get("volume"), prior_close=row.get("prior_close"),
            halted=bool(row.get("halted", False)),
            realtime=bool(row.get("realtime", False)),
            quote_stale=bool(row.get("quote_stale", False)),
            asof=asof,
            borrow=row.get("borrow"),
            borrow_rate=row.get("borrow_rate"),
            shortable_shares=row.get("shortable_shares"),
        )

        age = quote.age(now)
        if age is not None and age > self.max_age:
            logger.info(
                "[BROKER] %s ignore : instantane vieux de %d min",
                quote.symbol, age.total_seconds() // 60,
            )
            return None
        return quote

    def write(self, quotes: dict[str, dict], now: datetime | None = None) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Deposer un lot de cotations. Chaque entree recoit son horodatage si
            elle n'en porte pas, de sorte que la fraicheur reste verifiable.

        Inputs:
            quotes (dict): {symbole: champs}
            now (datetime | None): horodatage applique par defaut

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        stamp = (now or datetime.now()).isoformat(timespec="seconds")
        payload = self._read()
        payload.setdefault("quotes", {})
        for symbol, fields in quotes.items():
            row = dict(fields)
            row.setdefault("asof", stamp)
            payload["quotes"][symbol.strip().upper()] = row
        payload["written"] = stamp

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        logger.info("[BROKER] %d cotation(s) deposee(s)", len(quotes))


def from_ibkr_snapshot(symbol: str, payload: dict) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Traduire une reponse du connecteur IBKR en entree du pont.

        Deux details du format meritent l'attention : les cles de reponse
        utilisent des tirets la ou les champs demandes utilisent des
        soulignes, et le drapeau `is_close` sur `last` signale que le prix est
        une cloture reprise, non une transaction de la seance.

    Inputs:
        symbol (str): ticker
        payload (dict): reponse brute de get_price_snapshot

    Outputs:
        row (dict): champs prets pour BrokerBridge.write
    --------------------------------------------------------------------------
    """
    last = payload.get("last") or {}
    quote = payload.get("bid-ask") or {}
    status = (payload.get("top-status") or {}).get("status", "")

    return {
        "price": last.get("price"),
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "bid_size": quote.get("bid_size"),
        "ask_size": quote.get("ask_size"),
        "volume": (payload.get("volume") or {}).get("volume"),
        "prior_close": (payload.get("prior-close") or {}).get("priorClose"),
        "halted": bool(last.get("halted", False)),
        "realtime": status.upper() == "REALTIME",
        "quote_stale": bool(last.get("is_close", False)),
    }

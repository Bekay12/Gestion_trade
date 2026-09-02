"""
market - Construction d'un Snapshot a partir de donnees de marche publiques.

Couvre F1 (prix, volume, volume relatif) et F5 (flottant, interet vendeur) de
docu/methode/03-outils.md. Les cours sont differes : suffisant pour tenir un
journal papier, insuffisant pour une execution reelle — la limite est enoncee
ici plutot que decouverte plus tard.

Le flottant retourne est bien le flottant disponible, distinct des actions
emises. La confusion entre les deux fausse R6, R8 et R9 ; c'est la raison pour
laquelle ce module ne se rabat jamais sur les actions emises quand le flottant
manque : il laisse None.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from ..models import BorrowStatus, Snapshot
from .broker import BrokerBridge
from .ssr import SsrCalendar

logger = logging.getLogger(__name__)


class MarketUnavailable(RuntimeError):
    """Le fournisseur n'a rien retourne d'exploitable pour ce titre."""


def _first(info: dict, *keys: str) -> float | None:
    """Premiere valeur numerique exploitable parmi plusieurs cles possibles."""
    for key in keys:
        value = info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


class MarketSource:
    """
    Fournisseur de donnees de marche. Isole yfinance derriere une interface
    etroite, de sorte qu'un autre fournisseur puisse le remplacer sans toucher
    au moteur.
    """

    def __init__(
        self, ssr: SsrCalendar | None = None, broker: BrokerBridge | None = None
    ) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Preparer la source : donnees publiques, calendrier de restrictions,
            et pont courtier optionnel qui prime sur le public quand il est
            frais.

        Inputs:
            ssr (SsrCalendar | None): calendrier Rule 201, cree par defaut
            broker (BrokerBridge | None): pont courtier, cree par defaut

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        self.ssr = ssr if ssr is not None else SsrCalendar()
        self.broker = broker if broker is not None else BrokerBridge()
        self._ticker_factory = None

    def _ticker(self, symbol: str):
        """Import differe : yfinance est lourd et inutile aux tests hors ligne."""
        if self._ticker_factory is None:
            import yfinance
            self._ticker_factory = yfinance.Ticker
        return self._ticker_factory(symbol)

    def snapshot(
        self,
        symbol: str,
        borrow: BorrowStatus | None = None,
        borrow_rate: float | None = None,
        day: date | None = None,
    ) -> Snapshot:
        """
        ----------------------------------------------------------------------
        Purpose:
            Assembler l'etat marche d'un titre.

            Le statut d'emprunt reste un parametre : aucune source publique ne
            le diffuse, il vient du courtier. Le laisser a None fait refuser
            toute position vendeuse par le moteur, ce qui est le comportement
            voulu.

        Inputs:
            symbol (str): ticker
            borrow (BorrowStatus | None): disponibilite a l'emprunt, saisie
            borrow_rate (float | None): taux annualise, saisi
            day (date | None): journee pour la restriction Rule 201

        Outputs:
            snapshot (Snapshot): etat marche, champs inconnus a None

        Raises:
            MarketUnavailable: si prix ou cloture precedente sont introuvables
        ----------------------------------------------------------------------
        """
        symbol = symbol.strip().upper()
        try:
            info = self._ticker(symbol).info or {}
        except Exception as error:                     # yfinance leve large
            raise MarketUnavailable(f"{symbol} : {error}") from error

        price = _first(info, "currentPrice", "regularMarketPrice", "previousClose")
        previous_close = _first(info, "previousClose", "regularMarketPreviousClose")
        if price is None or previous_close is None:
            raise MarketUnavailable(f"{symbol} : prix ou cloture precedente absents")

        volume = _first(info, "regularMarketVolume", "volume") or 0.0
        average = _first(info, "averageVolume", "averageVolume10days")
        if average is None:
            raise MarketUnavailable(f"{symbol} : volume moyen absent, R6 incalculable")

        # Le flottant reel, jamais remplace par les actions emises.
        free_float = _first(info, "floatShares")
        shares_short = _first(info, "sharesShort")

        ssr_active = self.ssr.is_active(symbol, day)
        if ssr_active is None:
            logger.warning("[MARKET] %s : statut Rule 201 indeterminable", symbol)

        # Le courtier prime sur le public quand il est frais : prix temps reel,
        # volume incluant le pre-marche, drapeau de suspension et haut du
        # carnet, qu'aucune source publique ne donne.
        bid = ask = bid_size = ask_size = None
        halted = False
        realtime = quote_stale = False

        quote = self.broker.quote(symbol)
        if quote is not None:
            if quote.price:
                price = quote.price
            if quote.prior_close:
                previous_close = quote.prior_close
            if quote.volume is not None:
                volume = quote.volume
            bid, ask = quote.bid, quote.ask
            bid_size, ask_size = quote.bid_size, quote.ask_size
            halted = quote.halted
            realtime = quote.realtime
            quote_stale = quote.quote_stale
            # La saisie manuelle prime : elle est un arbitrage de l'operateur,
            # le courtier n'est qu'une source. Il ne comble que le silence.
            if borrow is None and quote.borrow is not None:
                borrow = quote.borrow
                logger.info("[MARKET] %s : emprunt '%s' repris du courtier", symbol, borrow)
            if borrow_rate is None and quote.borrow_rate is not None:
                borrow_rate = quote.borrow_rate
            logger.info("[MARKET] %s : instantane courtier applique", symbol)

        return Snapshot(
            symbol=symbol,
            price=price,
            previous_close=previous_close,
            volume=volume,
            average_volume=average,
            market_cap=_first(info, "marketCap"),
            free_float=free_float,
            shares_short=shares_short,
            borrow=borrow,
            borrow_rate=borrow_rate,
            ssr_active=ssr_active,
            vwap=None,                    # non fourni par ces sources
            halted=halted,
            asof=datetime.now(),
            bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
            realtime=realtime, quote_stale=quote_stale,
        )

    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Titres d'actualite recents pour un titre (F3).

            Sert le test d'admission de R11 : si le catalyseur ne se resume pas
            en une phrase, le titre sort de la liste. Un titre d'article ne
            remplace pas la lecture du communique — le corpus insiste sur ce
            point, titre et contenu divergeant regulierement — mais il indique
            ou regarder.

        Inputs:
            symbol (str): ticker
            limit (int): nombre maximal d'articles

        Outputs:
            items (list[dict]): {title, publisher, link}, vide en cas d'echec
        ----------------------------------------------------------------------
        """
        try:
            raw = self._ticker(symbol.strip().upper()).news or []
        except Exception as error:                     # yfinance leve large
            logger.warning("[MARKET] actualite %s indisponible : %s", symbol, error)
            return []

        items: list[dict] = []
        for entry in raw[:limit]:
            # Le format a change de version en version : le contenu est tantot
            # a la racine, tantot sous une cle 'content'.
            body = entry.get("content", entry) if isinstance(entry, dict) else {}
            provider = body.get("provider")
            publisher = (
                provider.get("displayName") if isinstance(provider, dict)
                else body.get("publisher")
            )
            title = body.get("title")
            if not title:
                continue
            link = body.get("canonicalUrl") or {}
            items.append({
                "title": str(title),
                "publisher": str(publisher or ""),
                "link": link.get("url") if isinstance(link, dict) else body.get("link", ""),
            })
        return items

    def screen(self, symbols: list[str], **kwargs) -> list[Snapshot]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Construire les etats marche d'une liste de titres, en ignorant
            ceux qui echouent plutot qu'en interrompant le balayage.

        Inputs:
            symbols (list[str]): tickers
            **kwargs: transmis a snapshot()

        Outputs:
            snapshots (list[Snapshot]): ceux qui ont pu etre construits
        ----------------------------------------------------------------------
        """
        out: list[Snapshot] = []
        for symbol in symbols:
            try:
                out.append(self.snapshot(symbol, **kwargs))
            except MarketUnavailable as error:
                logger.warning("[MARKET] %s ignore : %s", symbol, error)
        return out

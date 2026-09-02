"""
ibkr - Producteur d'instantanes depuis TWS ou IB Gateway.

Remplace la session d'assistant qui alimentait jusqu'ici le pont JSON. Le pont
lui-meme ne bouge pas : ce module ecrit dans le meme fichier, par le meme
BrokerBridge.write(). Le moteur ignore donc d'ou vient la donnee, et le journal
papier continue de tourner sans TWS ouvert.

Ce que cette voie apporte et que le connecteur precedent ne donnait pas :

    - le statut d'emprunt (tick 46) et le nombre d'actions empruntables
      (tick 89), tous deux derriere le tick generique 236 ;
    - le TAUX d'emprunt, par reqHistoricalData(whatToShow="FEE_RATE"), que la
      documentation du projet annoncait hors de portee. Il l'etait du connecteur
      MCP, pas d'une connexion Gateway.

Ces deux champs sont l'entree de R7 et de R9. Les obtenir automatiquement leve
le verrou qui bloquait tout le cote vendeur, c'est-a-dire la methode elle-meme.
Valides contre un Gateway reel le 2026-08-23.

La traduction des ticks vit dans ib_ticks.py ; ici ne restent que la connexion
et les requetes. Deux comportements meritent attention, tous deux decouverts en
confrontant ce module a un vrai Gateway :

    - un contrat doit etre QUALIFIE (conId renseigne) avant toute requete ;
      un Stock brut est refuse par ib_async avant meme d'atteindre le reseau ;
    - un compte sans abonnement temps reel ne recoit RIEN en direct (erreur
      10089), pas une version degradee. Le repli sur le differe est donc
      automatique, et n'est jamais presente comme du direct.

Reference : docu/methode/07-etat-automatisation.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from .broker import BrokerBridge
from .ib_ports import (  # re-exportes : point d'entree unique
    KNOWN_PORTS,
    LIVE_GATEWAY_PORT,
    LIVE_TWS_PORT,
    PAPER_GATEWAY_PORT,
    PAPER_TWS_PORT,
    detect_port,
    probe_port,
    scan_ports,
)
from .ib_ticks import (  # re-exportes : ce module reste le point d'entree
    LOT_VOLUME_MULTIPLIER,
    MARKET_DATA_DELAYED,
    MARKET_DATA_LIVE,
    MAX_PLAUSIBLE_VOLUME,
    SHARES_VOLUME_MULTIPLIER,
    SHORTABLE_TICKS,
    US_VOLUME_MULTIPLIER,
    borrow_from_shortable,
    halted_from_tick,
    num,
    session_volume,
)

logger = logging.getLogger(__name__)


# Delai laisse aux ticks pour arriver avant de publier ce qu'on a.
TICK_TIMEOUT = 6.0

# Compatibilite : ce nom etait exporte avant la separation en ib_ticks.
_num = num


@dataclass(frozen=True)
class IbkrSettings:
    """Parametres de connexion. Le compte papier est le defaut assume."""

    host: str = "127.0.0.1"
    port: int = PAPER_TWS_PORT
    client_id: int = 17
    timeout: float = 8.0
    # 1 si TWS envoie des actions (defaut moderne), 100 s'il envoie des lots.
    us_volume_multiplier: int = US_VOLUME_MULTIPLIER
    # 1 direct, 3 differe. Le differe est le repli des comptes sans abonnement.
    market_data_type: int = MARKET_DATA_LIVE
    # Repli automatique sur le differe quand le direct ne rend aucun prix.
    auto_delayed: bool = True


class IbkrUnavailable(RuntimeError):
    """TWS ou Gateway n'est pas joignable, ou la bibliotheque manque."""


class IbkrConnector:
    """
    Producteur d'instantanes. Ne detient aucun etat de marche : il interroge,
    traduit, et depose dans le pont. Toute la logique de fraicheur reste dans
    BrokerBridge, qui ignore deja un instantane trop vieux.
    """

    def __init__(self, settings: IbkrSettings | None = None, ib=None) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Preparer le connecteur. L'objet client est injectable, de sorte que
            la suite de tests tourne sans TWS ni reseau.

        Inputs:
            settings (IbkrSettings | None): parametres de connexion
            ib: client ib_async deja construit, ou None pour en creer un

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        self.settings = settings or IbkrSettings()
        self._ib = ib
        self._owns_connection = ib is None
        # Contrats qualifies, par symbole. Un contrat non qualifie n'a pas de
        # conId et ib_async refuse de l'utiliser ; la qualification coute un
        # aller-retour, on ne la refait pas a chaque cotation.
        self._contracts: dict[str, object] = {}

    def _client(self):
        """Client ib_async, importe tardivement pour rester optionnel."""
        if self._ib is not None:
            return self._ib
        try:
            from ib_async import IB
        except ImportError as error:                   # dependance optionnelle
            raise IbkrUnavailable(
                "ib_async absent : .venv\\Scripts\\python.exe -m pip install ib_async"
            ) from error
        self._ib = IB()
        return self._ib

    def connect(self):
        """
        ----------------------------------------------------------------------
        Purpose:
            Ouvrir la connexion et demander le type de flux configure. Le type
            reellement servi est relu sur chaque cotation : demander le direct
            ne garantit pas de l'obtenir.

        Inputs:
            None

        Outputs:
            ib: client connecte

        Raises:
            IbkrUnavailable: si TWS ou Gateway refuse la connexion
        ----------------------------------------------------------------------
        """
        ib = self._client()
        if getattr(ib, "isConnected", lambda: False)():
            return ib
        try:
            ib.connect(
                self.settings.host,
                self.settings.port,
                clientId=self.settings.client_id,
                timeout=self.settings.timeout,
            )
        except Exception as error:                     # ib_async leve large
            raise IbkrUnavailable(
                f"TWS/Gateway injoignable sur {self.settings.host}:{self.settings.port} "
                f"— API activee et port ouvert ? ({error})"
            ) from error
        ib.reqMarketDataType(self.settings.market_data_type)
        logger.info("[IBKR] connecte a %s:%s", self.settings.host, self.settings.port)
        return ib

    def disconnect(self) -> None:
        """Fermer si et seulement si ce connecteur a ouvert la connexion."""
        if self._ib is not None and self._owns_connection:
            try:
                self._ib.disconnect()
            except Exception as error:                 # fermeture best effort
                logger.warning("[IBKR] fermeture imparfaite : %s", error)

    def _contract(self, symbol: str):
        """
        ----------------------------------------------------------------------
        Purpose:
            Contrat QUALIFIE pour un ticker. IB exige un conId, que seul un
            aller-retour de qualification renseigne ; un Stock brut est refuse
            par ib_async avant meme d'atteindre le reseau.

            Un symbole que TWS ne reconnait pas remonte None : il sera saute et
            journalise, jamais remplace par un contrat approchant.

        Inputs:
            symbol (str): ticker

        Outputs:
            contract: contrat qualifie, ou None si TWS ne le reconnait pas
        ----------------------------------------------------------------------
        """
        symbol = symbol.strip().upper()
        if symbol in self._contracts:
            return self._contracts[symbol]

        from ib_async import Stock

        ib = self._client()
        try:
            qualified = ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
        except Exception as error:                     # symbole inconnu, reseau
            logger.warning("[IBKR] %s : qualification refusee (%s)", symbol, error)
            qualified = []

        contract = qualified[0] if qualified else None
        if contract is None:
            logger.warning("[IBKR] %s : contrat inconnu de TWS, symbole saute", symbol)
        self._contracts[symbol] = contract
        return contract

    def borrow_rate(self, symbol: str) -> float | None:
        """
        ----------------------------------------------------------------------
        Purpose:
            R7 - taux d'emprunt annualise, par une serie historique
            whatToShow="FEE_RATE". C'est la seule voie programmable vers cette
            donnee : aucune source publique ne la diffuse, et le tick de
            marche ne la porte pas.

            IB exprime ce taux en POURCENTAGE ; le moteur raisonne en fraction.
            La conversion se fait ici, une seule fois.

        Inputs:
            symbol (str): ticker

        Outputs:
            rate (float | None): taux annualise en fraction, None si absent
        ----------------------------------------------------------------------
        """
        contract = self._contract(symbol)
        if contract is None:
            return None
        ib = self._client()
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="1 day",
                whatToShow="FEE_RATE",
                useRTH=False,
            )
        except Exception as error:                     # titre sans SLB, etc.
            logger.info("[IBKR] %s : taux d'emprunt indisponible (%s)", symbol, error)
            return None

        if not bars:
            logger.info("[IBKR] %s : aucune serie FEE_RATE", symbol)
            return None
        percent = num(getattr(bars[-1], "close", None))
        return None if percent is None else percent / 100.0

    def quote_row(self, ticker, symbol: str, rate: float | None = None) -> dict:
        """
        ----------------------------------------------------------------------
        Purpose:
            Traduire un Ticker ib_async en ligne du pont. Aucun champ manquant
            n'est comble : nan devient None et le crible refusera.

        Inputs:
            ticker: objet Ticker d'ib_async
            symbol (str): ticker, pour tracabilite
            rate (float | None): taux d'emprunt deja obtenu

        Outputs:
            row (dict): champs prets pour BrokerBridge.write
        ----------------------------------------------------------------------
        """
        last = num(getattr(ticker, "last", None))
        close = num(getattr(ticker, "close", None))
        # Aucune transaction dans la seance : le prix retombe sur la cloture,
        # et on le dit. Un prix de cloture presente comme un prix de seance est
        # la meme faute qu'un prix perime servi comme frais.
        stale = last is None
        price = last if last is not None else close

        flag, known = halted_from_tick(getattr(ticker, "halted", None))
        if not known:
            logger.warning(
                "[IBKR] %s : statut de suspension inconnu, traite comme non "
                "suspendu — verification manuelle si le titre a saute",
                symbol,
            )

        data_type = getattr(ticker, "marketDataType", None)
        realtime = data_type == MARKET_DATA_LIVE
        if not realtime:
            logger.warning("[IBKR] %s : flux de type %s, pas du direct", symbol, data_type)

        return {
            "price": price,
            "bid": num(getattr(ticker, "bid", None)),
            "ask": num(getattr(ticker, "ask", None)),
            "bid_size": num(getattr(ticker, "bidSize", None)),
            "ask_size": num(getattr(ticker, "askSize", None)),
            "volume": session_volume(
                getattr(ticker, "volume", None),
                self.settings.us_volume_multiplier,
                symbol,
            ),
            "prior_close": close,
            "halted": flag,
            "realtime": realtime,
            "quote_stale": stale,
            "borrow": borrow_from_shortable(getattr(ticker, "shortable", None)),
            "borrow_rate": rate,
            "shortable_shares": num(getattr(ticker, "shortableShares", None)),
        }

    def _collect(self, ib, symbols: list[str], with_rate: bool) -> dict[str, dict]:
        """Un passage de collecte, sans repli."""
        rows: dict[str, dict] = {}
        tickers = {}
        for raw in symbols:
            symbol = raw.strip().upper()
            contract = self._contract(symbol)
            if contract is None:
                continue
            tickers[symbol] = ib.reqMktData(contract, SHORTABLE_TICKS, False, False)

        # Les ticks arrivent de facon asynchrone : on laisse le temps au flux
        # de se remplir plutot que de lire une structure encore vide.
        ib.sleep(TICK_TIMEOUT)

        for symbol, ticker in tickers.items():
            rate = self.borrow_rate(symbol) if with_rate else None
            rows[symbol] = self.quote_row(ticker, symbol, rate)
            try:
                ib.cancelMktData(self._contracts[symbol])
            except Exception as error:                 # annulation best effort
                logger.info("[IBKR] %s : annulation refusee (%s)", symbol, error)
        return rows

    def quotes(self, symbols: list[str], with_rate: bool = True) -> dict[str, dict]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Interroger une liste de titres et rendre les lignes du pont.

            Un compte sans abonnement temps reel ne recoit aucun prix en direct
            — pas un prix degrade, aucun prix. Le repli sur le differe est donc
            tente automatiquement, et journalise : mieux vaut une donnee
            annoncee differee qu'une absence prise pour une panne.

        Inputs:
            symbols (list[str]): tickers
            with_rate (bool): joindre le taux d'emprunt (une requete de plus)

        Outputs:
            rows (dict): {symbole: champs}
        ----------------------------------------------------------------------
        """
        ib = self.connect()
        rows = self._collect(ib, symbols, with_rate)

        aucun_prix = bool(rows) and all(r.get("price") is None for r in rows.values())
        if (
            aucun_prix
            and self.settings.auto_delayed
            and self.settings.market_data_type == MARKET_DATA_LIVE
        ):
            logger.warning(
                "[IBKR] aucun prix en direct (abonnement absent ?) — repli differe"
            )
            ib.reqMarketDataType(MARKET_DATA_DELAYED)
            rows = self._collect(ib, symbols, with_rate)

        logger.info("[IBKR] %d cotation(s) collectee(s)", len(rows))
        return rows

    def publish(
        self,
        symbols: list[str],
        bridge: BrokerBridge | None = None,
        now: datetime | None = None,
        with_rate: bool = True,
    ) -> dict[str, dict]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Collecter puis deposer dans le pont. C'est le point d'entree
            normal : le moteur relit ensuite le fichier comme avant, sans
            savoir qu'un producteur a change.

        Inputs:
            symbols (list[str]): tickers
            bridge (BrokerBridge | None): pont cible
            now (datetime | None): horodatage, pour les tests
            with_rate (bool): joindre le taux d'emprunt

        Outputs:
            rows (dict): ce qui a ete depose
        ----------------------------------------------------------------------
        """
        rows = self.quotes(symbols, with_rate=with_rate)
        (bridge or BrokerBridge()).write(rows, now=now)
        return rows

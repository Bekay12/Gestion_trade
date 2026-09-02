"""
sync_ibkr - Alimente le pont courtier depuis TWS ou IB Gateway.

Le chainon qui manquait : la watchlist donne les symboles, TWS donne les
cotations, le pont recoit le tout, et `live.py scan` travaille ensuite sur des
donnees temps reel sans savoir que le producteur a change.

    python agent/sync_ibkr.py                      # les symboles de la watchlist
    python agent/sync_ibkr.py --symbols CDTG,CNET  # une liste explicite
    python agent/sync_ibkr.py --gateway            # port 4002 au lieu de 7497
    python agent/sync_ibkr.py --loop --every 60    # rafraichissement continu

Prerequis cote TWS : API activee (Configuration globale > API > Parametres),
« Enable ActiveX and Socket Clients » coche, et le port qui correspond. Le
compte papier est le defaut assume : les ports d'un compte finance ne sont pas
dans les valeurs par defaut de ce module.

Ce script n'ouvre aucune position et ne passe aucun ordre. Il lit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as clock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources.broker import BrokerBridge
from agent.sources.ibkr import (
    MARKET_DATA_DELAYED,
    MARKET_DATA_LIVE,
    PAPER_GATEWAY_PORT,
    IbkrConnector,
    IbkrSettings,
    IbkrUnavailable,
    detect_port,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST = ROOT / "docu" / "portefeuille" / "watchlist.json"


def watchlist_symbols(path: Path = WATCHLIST) -> list[str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lire les symboles des plans de trade. Aucune invention : si le fichier
        manque, la liste est vide et l'appelant le saura.

    Inputs:
        path (Path): fichier de watchlist

    Outputs:
        symbols (list[str]): tickers, sans doublon, dans l'ordre du fichier
    --------------------------------------------------------------------------
    """
    if not path.exists():
        logger.warning("[SYNC] watchlist introuvable : %s", path)
        return []
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        logger.error("[SYNC] watchlist illisible : %s", error)
        return []

    seen: list[str] = []
    for plan in config.get("plans", []):
        symbol = str(plan.get("symbol", "")).strip().upper()
        if symbol and symbol not in seen:
            seen.append(symbol)
    return seen


def report(rows: dict[str, dict]) -> None:
    """Resume lisible de ce qui vient d'etre depose."""
    for symbol, row in sorted(rows.items()):
        borrow = row.get("borrow") or "inconnu"
        rate = row.get("borrow_rate")
        rate_text = f"{rate:.0%}/an" if rate is not None else "taux inconnu"
        flux = "direct" if row.get("realtime") else "DIFFERE"
        stale = " [cloture reprise]" if row.get("quote_stale") else ""
        halt = " [SUSPENDU]" if row.get("halted") else ""
        price = row.get("price")
        price_text = f"{price:.4g}" if price is not None else "prix absent"
        print(
            f"  {symbol:<6} {price_text:>10}  {flux:<7} "
            f"emprunt {borrow:<7} {rate_text}{stale}{halt}"
        )


def sync(
    connector: IbkrConnector,
    symbols: list[str],
    bridge: BrokerBridge,
    with_rate: bool = True,
) -> int:
    """Un passage. Rend le nombre de cotations deposees."""
    rows = connector.publish(symbols, bridge=bridge, with_rate=with_rate)
    report(rows)
    return len(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--symbols", help="liste separee par des virgules")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--delayed", action="store_true",
                        help="donnees differees (compte sans abonnement direct)")
    parser.add_argument("--gateway", action="store_true", help="port IB Gateway")
    parser.add_argument("--client-id", type=int, default=17)
    parser.add_argument("--no-rate", action="store_true", help="sauter FEE_RATE")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--every", type=int, default=60, help="secondes")
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else watchlist_symbols()
    )
    if not symbols:
        print("Aucun symbole : renseigner la watchlist ou passer --symbols.")
        return 2

    # Le port se detecte : imposer a l'operateur de retenir lequel des quatre
    # correspond a son installation produit exactement le meme message qu'un
    # logiciel ferme, ce qui envoie chercher la panne au mauvais endroit.
    port = args.port or (PAPER_GATEWAY_PORT if args.gateway else detect_port(args.host))
    if port is None:
        print("[SYNC] Aucun port IB n'ecoute (7497, 4002, 7496, 4001).")
        print("[SYNC] Lancer TWS ou IB Gateway, API activee. Diagnostic complet :")
        print("[SYNC]   python agent/ibkr_doctor.py")
        return 1
    connector = IbkrConnector(IbkrSettings(
        host=args.host, port=port, client_id=args.client_id,
        market_data_type=MARKET_DATA_DELAYED if args.delayed else MARKET_DATA_LIVE,
    ))
    bridge = BrokerBridge()

    try:
        while True:
            try:
                count = sync(connector, symbols, bridge, with_rate=not args.no_rate)
                print(f"[SYNC] {count} cotation(s) deposee(s) dans {bridge.path}")
            except IbkrUnavailable as error:
                # Une panne de source n'est pas une donnee : on le dit et on
                # ne depose rien, plutot que de laisser vieillir en silence.
                print(f"[SYNC] {error}")
                return 1
            if not args.loop:
                return 0
            clock.sleep(args.every)
    except KeyboardInterrupt:
        print("\n[SYNC] arret demande")
        return 0
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

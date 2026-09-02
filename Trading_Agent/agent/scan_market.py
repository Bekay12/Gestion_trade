"""
scan_market - Liste de pre-marche : decouvrir, filtrer, qualifier.

La routine de pre-marche de la methode, en une commande. Trois etapes, dans
l'ordre ou elles coutent le moins cher :

    1. DECOUVRIR  balayage TWS borne par R6            (sources/scanner.py)
    2. FILTRER    le crible R6 lui-meme, pas une copie (gate.check_filters)
    3. QUALIFIER  dilution S-3 / 424B, en option       (sources/edgar.py)

    python agent/scan_market.py                     # spike + volume
    python agent/scan_market.py --scans spike       # un seul balayage
    python agent/scan_market.py --edgar             # ajoute la dilution
    python agent/scan_market.py --rows 20 --all     # montre aussi les refuses

Ce script ne redige aucun plan de trade et n'ouvre aucune position. Il rend une
liste a examiner. Le catalyseur — le test d'admission de R11 — reste a ecrire a
la main : c'est la couche 2, qu'aucun code de ce depot ne couvre encore.

Prerequis : TWS ou Gateway ouvert, API activee. Pour --edgar, la variable
SEC_USER_AGENT.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.gate import check_filters
from agent.models import Verdict
from agent.sources.broker import BrokerBridge
from agent.sources.edgar import EdgarClient, EdgarError, assess_dilution
from agent.sources.ibkr import IbkrConnector, IbkrSettings, IbkrUnavailable
from agent.sources.market import MarketSource, MarketUnavailable
from agent.sources.scanner import (
    DEFAULT_ROWS,
    DEFAULT_SCANS,
    R9_TRAP,
    SCAN_CODES,
    MarketScanner,
    ScanFilters,
    ScanUnavailable,
)

logger = logging.getLogger(__name__)


def screen(symbols: list[str], source: MarketSource) -> dict[str, tuple]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Passer chaque symbole au filtre R6 — celui de gate.py, appele
        directement. Aucun seuil n'est recopie ici, sans quoi la liste de
        pre-marche et le crible divergeraient en silence.

    Inputs:
        symbols (list[str]): tickers issus du balayage
        source (MarketSource): fournisseur d'etats marche

    Outputs:
        results (dict): {symbole: (snapshot | None, verdict | None, erreur)}
    --------------------------------------------------------------------------
    """
    results: dict[str, tuple] = {}
    for symbol in symbols:
        try:
            snap = source.snapshot(symbol)
        except MarketUnavailable as error:
            # Donnee absente : le titre sort de la liste, il n'est pas suppose
            # conforme. C'est le meme principe que dans le crible.
            results[symbol] = (None, None, str(error))
            continue
        verdict = Verdict()
        check_filters(snap, verdict)
        results[symbol] = (snap, verdict, None)
    return results


def dilution_line(client: EdgarClient, symbol: str) -> str:
    """Une ligne de qualification F4, ou la raison de son absence."""
    try:
        report = assess_dilution(symbol, client.filings(symbol))
    except EdgarError as error:
        return f"EDGAR indisponible ({error})"
    # `risk` est le champ actionnable : une emission constatee (424B) prime
    # sur une capacite dormante (S-3). C'est la distinction centrale de F4.
    return f"risque {report.risk} — {report.summary()}"


def render(results: dict[str, tuple], origins: dict[str, list[str]],
           show_all: bool, edgar: EdgarClient | None) -> int:
    """Afficher la liste. Rend le nombre de titres retenus."""
    retenus = [s for s, (_, v, _) in results.items() if v is not None and not v.blocks]
    refuses = [s for s in results if s not in retenus]

    print(f"\n=== RETENUS PAR R6 : {len(retenus)} ===")
    if not retenus:
        print("  aucun — le balayage n'a rien remonte qui passe le filtre.")
    for symbol in retenus:
        snap, verdict, _ = results[symbol]
        source_scan = "+".join(origins.get(symbol, []))
        rvol = snap.volume / snap.average_volume if snap.average_volume else 0.0
        flux = "direct" if snap.realtime else "differe"
        print(f"\n  {symbol}  [{source_scan}]  {snap.price:.4g}  "
              f"RVOL {rvol:.1f}x  {flux}")
        for warning in verdict.warnings:
            print(f"      ! {warning}")
        if edgar is not None:
            print(f"      F4 {dilution_line(edgar, symbol)}")

    if show_all and refuses:
        print(f"\n=== ECARTES : {len(refuses)} ===")
        for symbol in refuses:
            _, verdict, error = results[symbol]
            motif = error if error else " ; ".join(verdict.blocks)
            print(f"  {symbol:<6} {motif}")

    print("\nLe catalyseur reste a etablir a la main (R11) avant tout plan.")
    return len(retenus)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(description="Liste de pre-marche")
    parser.add_argument("--scans", default=",".join(DEFAULT_SCANS),
                        help=f"separes par des virgules : {', '.join(SCAN_CODES)}")
    parser.add_argument("--low-float", action="store_true",
                        help="borner le flottant au plafond de R6")
    parser.add_argument("--shortable", action="store_true",
                        help="ecarter les titres non empruntables (R7)")
    parser.add_argument("--ssr", action="store_true",
                        help="ne garder que les titres sous Rule 201 (R9)")
    parser.add_argument("--change-above", type=float, default=None,
                        help="variation minimale, en pourcentage")
    parser.add_argument("--piege-r9", action="store_true",
                        help="balayer la configuration que R9 REFUSE : a lire "
                             "pour savoir quoi ne pas vendre a decouvert")
    parser.add_argument("--halted", action="store_true",
                        help="inclure les titres suspendus (R6 les refuse)")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--delayed", action="store_true",
                        help="donnees differees (compte sans abonnement direct)")
    parser.add_argument("--gateway", action="store_true")
    parser.add_argument("--client-id", type=int, default=18)
    parser.add_argument("--edgar", action="store_true", help="qualifier la dilution")
    parser.add_argument("--all", action="store_true", help="montrer les ecartes")
    args = parser.parse_args()

    from agent.sources.ibkr import PAPER_GATEWAY_PORT, detect_port
    # Le port se detecte : imposer a l'operateur de retenir lequel des quatre
    # correspond a son installation produit exactement le meme message qu'un
    # logiciel ferme, ce qui envoie chercher la panne au mauvais endroit.
    port = args.port or (PAPER_GATEWAY_PORT if args.gateway else detect_port(args.host))
    if port is None:
        print("[SCAN] Aucun port IB n'ecoute (7497, 4002, 7496, 4001).")
        print("[SCAN] Lancer TWS ou IB Gateway, API activee. Diagnostic complet :")
        print("[SCAN]   python agent/ibkr_doctor.py")
        return 1
    from agent.sources.ibkr import MARKET_DATA_DELAYED, MARKET_DATA_LIVE
    connector = IbkrConnector(IbkrSettings(
        host=args.host, port=port, client_id=args.client_id,
        market_data_type=MARKET_DATA_DELAYED if args.delayed else MARKET_DATA_LIVE,
    ))

    filters = R9_TRAP if args.piege_r9 else ScanFilters(
        low_float=args.low_float,
        exclude_halted=not args.halted,
        shortable_only=args.shortable,
        ssr_only=args.ssr,
        change_pct_above=args.change_above,
    )
    if args.piege_r9:
        print("[SCAN] Configuration R9 : ces titres sont a NE PAS vendre a")
        print("[SCAN] decouvert. Le crible les refusera — c'est une liste de")
        print("[SCAN] vigilance, pas une liste de candidats.")

    scans = tuple(s.strip() for s in args.scans.split(",") if s.strip())
    try:
        origins = MarketScanner(connector).sweep(scans, args.rows, filters)
    except ScanUnavailable as error:
        print(f"[SCAN] {error}")
        connector.disconnect()
        return 1

    symbols = list(origins)
    if not symbols:
        print("[SCAN] aucun symbole remonte.")
        connector.disconnect()
        return 0
    print(f"[SCAN] {len(symbols)} symbole(s) : {', '.join(symbols)}")

    # Les cotations temps reel passent par le pont, comme partout ailleurs :
    # le producteur change, le moteur ne bouge pas.
    bridge = BrokerBridge()
    try:
        connector.publish(symbols, bridge=bridge)
    except IbkrUnavailable as error:
        print(f"[SCAN] cotations indisponibles : {error}")
        connector.disconnect()
        return 1

    edgar = None
    if args.edgar:
        if not os.environ.get("SEC_USER_AGENT"):
            # L'en-tete engage l'identite de l'appelant : jamais devine.
            print("[SCAN] --edgar ignore : SEC_USER_AGENT non defini.")
        else:
            edgar = EdgarClient()

    try:
        render(screen(symbols, MarketSource(broker=bridge)), origins, args.all, edgar)
    finally:
        connector.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

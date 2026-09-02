"""
live - Session papier en temps reel.

Fait tourner le compte de demonstration sur des donnees de marche reelles, en
conservant son etat d'une execution a l'autre. Ce n'est pas un backtest : les
decisions sont horodatees au moment ou elles sont prises, et rien n'est rejoue.

    python agent/live.py scan          # etats marche + verdicts du crible
    python agent/live.py open CNET     # ouvre si le crible l'autorise
    python agent/live.py tick          # valorise et declenche les stops
    python agent/live.py close CNET
    python agent/live.py status
    python agent/live.py loop --every 300

Limite a connaitre : les cours de la source publique sont differes. Suffisant
pour tenir un journal honnete, insuffisant pour une execution reelle.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as clock
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import PaperPortfolio, RiskProfile, TradePlan, evaluate
from agent.eastern import EASTERN_OFFSET, eastern_now
from agent.sources.broker import BrokerBridge
from agent.sources.edgar import EdgarClient, EdgarError, assess_dilution, assess_ownership
from agent.sources.market import MarketSource, MarketUnavailable

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "docu" / "portefeuille" / "journal.json"
WATCHLIST = ROOT / "docu" / "portefeuille" / "watchlist.json"

# Le decalage vit dans agent/eastern.py : une seule definition.


def load_watchlist() -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lire la liste de surveillance et les plans de trade. Le fichier porte
        aussi le statut d'emprunt, seule donnee qu'aucune source publique ne
        fournit et qui doit donc etre saisie a la main depuis le courtier.

    Inputs:
        None (lit WATCHLIST)

    Outputs:
        config (dict): capital, risque et plans
    --------------------------------------------------------------------------
    """
    if not WATCHLIST.exists():
        WATCHLIST.parent.mkdir(parents=True, exist_ok=True)
        template = {
            "capital": 10000,
            "risk_pct": 0.01,
            "_aide": (
                "borrow: 'easy' | 'hard' | 'none' — a relever chez le courtier. "
                "Absent, toute position vendeuse est refusee (R7/R9)."
            ),
            "plans": [{
                "symbol": "CNET", "direction": "long",
                "catalyst": "Decrire ici le catalyseur en une phrase",
                "entry": 1.40, "stop": 1.30, "targets": [1.70],
                "invalidation": "424B depose avant l'ouverture",
                "borrow": None, "borrow_rate": None,
            }],
        }
        WATCHLIST.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Modele de liste cree : {WATCHLIST}\nRenseigne-le puis relance.")
        raise SystemExit(0)
    return json.loads(WATCHLIST.read_text(encoding="utf-8"))


def plan_from(row: dict) -> TradePlan:
    """Construire un plan de trade depuis une entree de la liste."""
    return TradePlan(
        symbol=row["symbol"].upper(), direction=row["direction"],
        catalyst=row.get("catalyst", ""), entry=float(row["entry"]),
        stop=float(row["stop"]), targets=[float(t) for t in row.get("targets", [])],
        invalidation=row.get("invalidation", ""),
    )


def open_portfolio(config: dict) -> PaperPortfolio:
    """Reprendre l'etat sauvegarde et ouvrir la journee si elle est neuve."""
    pf = PaperPortfolio.load(STATE, float(config.get("capital", 10000)))
    today = eastern_now().date()
    if pf._today != today:
        pf.start_day(today)
    return pf


def describe(snap, verdict, plan: TradePlan) -> None:
    """Rendu lisible d'un verdict, provenance des donnees comprise."""
    mark = "ARME " if verdict.allowed else "REFUS"
    rvol = snap.volume / snap.average_volume if snap.average_volume else 0
    origin = "courtier temps reel" if snap.realtime else "public differe"

    print(f"\n[{mark}] {plan.symbol:6} {plan.direction:5} "
          f"cours {snap.price:.2f}  RVOL {rvol:.1f}x", end="")
    if snap.free_float:
        print(f"  flottant {snap.free_float/1e6:.1f}M", end="")
    if snap.ssr_active is not None:
        print(f"  SSR {'OUI' if snap.ssr_active else 'non'}", end="")
    spread = snap.spread_pct
    if spread is not None:
        print(f"  ecart {spread:.1%}", end="")
    print(f"   [{origin}]")

    if verdict.allowed:
        print(f"        taille {verdict.size} actions, R/R {verdict.risk_reward:.2f}")
    for reason in verdict.blocks:
        print(f"        BLOQUE  {reason}")
    for note in verdict.warnings:
        print(f"        note    {note}")


def context(edgar: EdgarClient | None, source: MarketSource, symbol: str) -> None:
    """
    --------------------------------------------------------------------------
    Purpose:
        Afficher le contexte qui sert a qualifier le catalyseur : signaux de
        dilution et de detention tires d'EDGAR, titres d'actualite recents.

        Un titre d'article ne remplace pas la lecture du communique — le corpus
        insiste sur la divergence frequente entre les deux — mais il indique ou
        regarder.

    Inputs:
        edgar (EdgarClient | None): client EDGAR, None si indisponible
        source (MarketSource): source d'actualite
        symbol (str): ticker

    Outputs:
        None
    --------------------------------------------------------------------------
    """
    if edgar is not None:
        try:
            filings = edgar.filings(symbol)
            dilution = assess_dilution(symbol, filings)
            ownership = assess_ownership(symbol, filings)
            print(f"        dilution   {dilution.risk.upper():8} {dilution.summary()}")
            print(f"        detention  {'':8} {ownership.summary()}")
        except EdgarError as error:
            print(f"        EDGAR indisponible : {error}")

    for item in source.news(symbol, limit=3):
        publisher = f" ({item['publisher']})" if item["publisher"] else ""
        print(f"        actu       {item['title'][:78]}{publisher}")


def cmd_scan(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """Balayer la liste et afficher les verdicts, sans rien engager."""
    profile = RiskProfile(pf.cash, float(config.get("risk_pct", 0.01)))
    now = eastern_now().time()
    print(f"Heure de l'Est : {now.strftime('%H:%M')}  |  capital {pf.cash:,.2f}")

    # EDGAR est optionnel : sans en-tete nominatif, le balayage continue sans
    # le contexte reglementaire plutot que de s'interrompre.
    try:
        edgar = EdgarClient()
    except EdgarError as error:
        edgar = None
        print(f"Contexte EDGAR desactive : {error}")

    for row in config["plans"]:
        plan = plan_from(row)
        try:
            snap = source.snapshot(plan.symbol, row.get("borrow"), row.get("borrow_rate"))
        except MarketUnavailable as error:
            print(f"\n[DONNEES] {plan.symbol} indisponible : {error}")
            continue
        describe(snap, evaluate(plan, snap, profile, pf, now), plan)
        if not args.brief:
            context(edgar, source, plan.symbol)


def cmd_broker(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """
    Etat du pont courtier : quels titres portent une cotation fraiche, lesquels
    retombent sur la source publique differee.
    """
    bridge = BrokerBridge()
    symbols = [p["symbol"].upper() for p in config["plans"]] + list(pf.positions)
    print(f"Pont : {bridge.path}")
    print(f"Age maximal accepte : {int(bridge.max_age.total_seconds() // 60)} min\n")
    for symbol in dict.fromkeys(symbols):
        quote = bridge.quote(symbol)
        if quote is None:
            print(f"  {symbol:6} — aucune cotation fraiche, repli sur le public differe")
            continue
        age = quote.age()
        minutes = int(age.total_seconds() // 60) if age else 0
        flags = []
        if quote.realtime:
            flags.append("temps reel")
        if quote.quote_stale:
            flags.append("prix de cloture")
        if quote.halted:
            flags.append("SUSPENDU")
        print(f"  {quote.symbol:6} {quote.price if quote.price else '—':>8}  "
              f"il y a {minutes:>3} min  {', '.join(flags)}")


def cmd_open(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """Ouvrir une position si et seulement si le crible l'autorise."""
    row = next((p for p in config["plans"] if p["symbol"].upper() == args.symbol.upper()), None)
    if row is None:
        print(f"{args.symbol} absent de la liste de surveillance.")
        return
    plan = plan_from(row)
    snap = source.snapshot(plan.symbol, row.get("borrow"), row.get("borrow_rate"))
    verdict = evaluate(plan, snap, RiskProfile(pf.cash, float(config.get("risk_pct", 0.01))),
                       pf, eastern_now().time())
    describe(snap, verdict, plan)
    if not verdict.allowed:
        print("\n        -> rien n'est engage.")
        return
    pf.open_position(plan.symbol, plan.direction, verdict.size, snap.price, plan.stop,
                     datetime.now(), row.get("borrow_rate"), plan.catalyst)
    pf.save(STATE)
    print(f"\n        -> position ouverte, journal mis a jour.")


def cmd_tick(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """Valoriser les positions ouvertes et declencher les stops atteints."""
    if not pf.positions:
        print("Aucune position ouverte.")
        return
    prices: dict[str, float] = {}
    for symbol in list(pf.positions):
        try:
            prices[symbol] = source.snapshot(symbol).price
        except MarketUnavailable as error:
            logger.warning("[LIVE] %s : %s", symbol, error)

    for trade in pf.apply_stops(prices, datetime.now()):
        print(f"STOP  {trade.symbol} a {trade.exit_price:.2f} -> {trade.pnl:+,.2f}")

    for symbol, pos in pf.positions.items():
        price = prices.get(symbol)
        if price is not None:
            print(f"      {symbol:6} {pos.direction:5} {pos.size:>6} @ {pos.entry:.2f} "
                  f"| cours {price:.2f} | latent {pos.unrealised(price):+,.2f}")
    pf.save(STATE)


def cmd_close(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """Cloturer une position au cours courant."""
    snap = source.snapshot(args.symbol.upper())
    trade = pf.close_position(args.symbol.upper(), snap.price, datetime.now(), "manuelle")
    pf.save(STATE)
    print(f"{trade.symbol} cloture a {trade.exit_price:.2f} -> {trade.pnl:+,.2f}")


def cmd_status(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """Etat du compte et bornes de risque en vigueur."""
    s = pf.stats()
    print(f"Capital        {s['equity']:,.2f}   (depart {pf.initial:,.2f})")
    print(f"Resultat net   {s['net']:+,.2f}")
    print(f"Journee        {pf.daily_pnl:+,.2f}")
    print(f"Drawdown       {s['drawdown']:.1%}")
    print(f"Operations     {s['trades']}  ({s['wins']} gagnantes, "
          f"taux {s['win_rate']:.0%})")
    if s["profit_factor"]:
        print(f"Facteur profit {s['profit_factor']:.2f}")
    print(f"Pertes d'affilee {pf.consecutive_losses}")
    for symbol, pos in pf.positions.items():
        print(f"  ouvert : {symbol} {pos.direction} {pos.size} @ {pos.entry:.2f}")


def cmd_loop(config: dict, pf: PaperPortfolio, source: MarketSource, args) -> None:
    """Boucler sur tick pendant la seance."""
    print(f"Boucle toutes les {args.every}s. Ctrl+C pour arreter.")
    try:
        while True:
            now = eastern_now()
            print(f"\n--- {now.strftime('%Y-%m-%d %H:%M:%S')} ET ---")
            if time(9, 30) <= now.time() < time(16, 0):
                cmd_tick(config, pf, source, args)
            else:
                print("Hors seance — aucune valorisation.")
            clock.sleep(args.every)
    except KeyboardInterrupt:
        pf.save(STATE)
        print("\nArret demande, journal sauvegarde.")


COMMANDS = {
    "scan": cmd_scan, "open": cmd_open, "tick": cmd_tick,
    "close": cmd_close, "status": cmd_status, "loop": cmd_loop,
    "broker": cmd_broker,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Session papier en temps reel")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("symbol", nargs="?", help="ticker, pour open et close")
    parser.add_argument("--every", type=int, default=300, help="periode de loop, secondes")
    parser.add_argument("--brief", action="store_true",
                        help="scan sans le contexte EDGAR ni l'actualite")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s", stream=sys.stderr,
    )

    config = load_watchlist()
    pf = open_portfolio(config)
    COMMANDS[args.command](config, pf, MarketSource(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

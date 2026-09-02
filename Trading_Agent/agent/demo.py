"""
demo - Rejoue trois scenarios sur le portefeuille papier.

Le troisieme est le plus important : il reconstitue la configuration decrite
dans la seance perdante du 18 aout et montre le moteur la refuser. C'est la
raison d'etre du module — la regle etait connue de l'operateur, elle n'a pas
suffi ; un moteur qui bloque change le resultat.

    .venv\\Scripts\\python.exe agent\\demo.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, time
from pathlib import Path

# Lance directement (python agent/demo.py), le paquet parent n'est pas sur le
# chemin d'import : on l'y met avant d'importer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import PaperPortfolio, RiskProfile, Snapshot, TradePlan, evaluate

CAPITAL = 10_000.0
PRIME = time(10, 30)
DAY = datetime(2026, 8, 21, 10, 30)


def render(title: str, verdict, plan: TradePlan) -> None:
    """Afficher un verdict de facon lisible."""
    mark = "ARME " if verdict.allowed else "REFUS"
    print(f"\n[{mark}] {title}")
    print(f"        {plan.direction.upper():5} {plan.symbol}  "
          f"entree {plan.entry:.2f}  stop {plan.stop:.2f}", end="")
    if verdict.risk_reward:
        print(f"  R/R {verdict.risk_reward:.2f}", end="")
    print()
    if verdict.allowed:
        print(f"        taille calculee : {verdict.size} actions "
              f"({verdict.size * plan.entry:,.0f} $ de position)")
    for reason in verdict.blocks:
        print(f"        BLOQUE  {reason}")
    for warning in verdict.warnings:
        print(f"        note    {warning}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    profile = RiskProfile(capital=CAPITAL, risk_pct=0.01)
    pf = PaperPortfolio(CAPITAL, slippage=0.002, fee_per_trade=1.0)
    pf.start_day(DAY.date())

    print("=" * 68)
    print(f"PORTEFEUILLE PAPIER — capital {CAPITAL:,.0f} $, risque 1 % par trade")
    print("=" * 68)

    # -- 1. Position acheteuse conforme ------------------------------------
    plan_long = TradePlan(
        symbol="ABCD", direction="long",
        catalyst="Contrat gouvernemental annonce en 8-K ce matin",
        entry=5.00, stop=4.80, targets=[5.60, 6.00],
        invalidation="424B depose avant l'ouverture",
    )
    snap_long = Snapshot(
        symbol="ABCD", price=5.00, previous_close=4.40,
        volume=6_000_000, average_volume=1_000_000,
        market_cap=90_000_000, free_float=8_000_000,
        borrow="easy", borrow_rate=0.04, ssr_active=False, vwap=4.85,
    )
    v1 = evaluate(plan_long, snap_long, profile, pf, PRIME)
    render("Catalyseur haussier, donnees completes", v1, plan_long)

    if v1.allowed:
        pf.open_position("ABCD", "long", v1.size, plan_long.entry,
                         plan_long.stop, DAY, catalyst=plan_long.catalyst)
        trade = pf.close_position("ABCD", 5.60, DAY, reason="cible 1")
        print(f"        -> cloture a 5.60 : {trade.pnl:+,.2f} $")

    # -- 2. Vendeuse sur dilution, donnees d'emprunt manquantes ------------
    plan_short = TradePlan(
        symbol="EFGH", direction="short",
        catalyst="Placement prive annonce, titre en hausse de 80 %",
        entry=5.00, stop=5.40, targets=[4.00],
        invalidation="Reprise au-dessus du plus haut du jour",
    )
    snap_missing = Snapshot(
        symbol="EFGH", price=5.00, previous_close=2.80,
        volume=40_000_000, average_volume=2_000_000,
        market_cap=60_000_000, free_float=7_000_000,
        borrow=None, ssr_active=None,          # non diffuse gratuitement
    )
    render("Meme these, statut d'emprunt et Rule 201 inconnus",
           evaluate(plan_short, snap_missing, profile, pf, PRIME), plan_short)

    # -- 3. La configuration piege (R9) ------------------------------------
    snap_trap = Snapshot(
        symbol="EFGH", price=5.00, previous_close=2.80,
        volume=40_000_000, average_volume=2_000_000,
        market_cap=60_000_000, free_float=6_000_000,
        shares_short=1_400_000,                 # ~23 % du flottant
        borrow="hard", borrow_rate=0.85,
        ssr_active=True,
    )
    render("Donnees renseignees : la configuration se revele",
           evaluate(plan_short, snap_trap, profile, pf, PRIME), plan_short)

    # -- 4. Bornes de compte -----------------------------------------------
    # Trois pertes reelles au stop, pas une soustraction sur le solde : on veut
    # exercer le vrai chemin de code, journal compris.
    for i in range(3):
        pf.open_position(f"LOSS{i}", "long", 500, 5.00, 4.80, DAY)
        pf.close_position(f"LOSS{i}", 4.80, DAY, reason="stop")
    print()
    print(f"        (trois pertes au stop : journee a {pf.daily_pnl:+,.2f} $, "
          f"{pf.consecutive_losses} pertes consecutives)")
    render("Apres trois pertes consecutives",
           evaluate(plan_long, snap_long, profile, pf, PRIME), plan_long)

    print("\n" + "=" * 68)
    stats = pf.stats()
    print(f"Operations : {stats['trades']}  |  gagnantes : {stats['wins']}  "
          f"|  net : {stats['net']:+,.2f} $  |  capital : {stats['equity']:,.2f} $")

    out = Path(__file__).resolve().parents[1] / "docu" / "portefeuille" / "journal.json"
    pf.save(out)
    print(f"Journal ecrit dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

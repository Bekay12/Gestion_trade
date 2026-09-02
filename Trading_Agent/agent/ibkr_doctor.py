"""
ibkr_doctor - Preflight de la connexion TWS/Gateway.

Rien de ce connecteur n'a ete confronte a un vrai TWS : les codes de balayage,
les droits de donnees et la disponibilite du taux d'emprunt dependent du compte
et ne peuvent pas etre devines depuis le depot. Ce script les mesure au lieu de
les supposer.

    python agent/ibkr_doctor.py            # sonde les quatre ports, puis teste
    python agent/ibkr_doctor.py --symbol AAPL
    python agent/ibkr_doctor.py --port 4002

Il ne modifie rien et ne depose rien dans le pont. Il repond a une seule
question : qu'est-ce que CE compte, sur CETTE machine, sait reellement faire.

Chaque capacite est testee separement, parce qu'elles echouent separement : un
compte peut servir des cotations differees sans balayage, ou un balayage sans
taux d'emprunt. Un diagnostic global « ca marche / ca ne marche pas » ne dirait
pas quoi corriger.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources.ibkr import (
    KNOWN_PORTS,
    LOT_VOLUME_MULTIPLIER,
    MARKET_DATA_DELAYED,
    MARKET_DATA_LIVE,
    PAPER_GATEWAY_PORT,
    PAPER_TWS_PORT,
    SHORTABLE_TICKS,
    IbkrConnector,
    IbkrSettings,
    IbkrUnavailable,
    num as _num,
    borrow_from_shortable,
    halted_from_tick,
    probe_port,
    scan_ports,
    session_volume,
)
from agent.eastern import market_open
from agent.gate import CLOSE_TIME, OPEN_TIME
from agent.sources.scanner import MarketScanner, ScanUnavailable

marche_ouvert = market_open  # nom historique de ce module

OK, KO, MEH = "[ OK ]", "[ KO ]", "[ ~~ ]"


def quote_capabilities(ticker) -> dict[str, tuple[str, str]]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Dire, champ par champ, ce que la cotation recue contient reellement.
        Un champ absent n'est pas une panne du connecteur : c'est un droit de
        donnees manquant, et la difference se voit ici.

    Inputs:
        ticker: objet Ticker d'ib_async

    Outputs:
        report (dict): {capacite: (marqueur, detail)}
    --------------------------------------------------------------------------
    """
    data_type = getattr(ticker, "marketDataType", None)
    live = data_type == MARKET_DATA_LIVE

    bid = _num(getattr(ticker, "bid", None))
    ask = _num(getattr(ticker, "ask", None))
    last = _num(getattr(ticker, "last", None))
    volume = _num(getattr(ticker, "volume", None))
    retenu = session_volume(getattr(ticker, "volume", None), 1, "preflight")
    shortable = getattr(ticker, "shortable", None)
    shares = _num(getattr(ticker, "shortableShares", None))
    _, halt_known = halted_from_tick(getattr(ticker, "halted", None))

    borrow = borrow_from_shortable(shortable)

    return {
        "Flux temps reel": (
            (OK, "type 1, direct") if live
            else (MEH, f"type {data_type} — differe ou fige, R6 utilisable, "
                       "execution non")
        ),
        "Dernier prix": (OK, f"{last}") if last is not None else (KO, "absent"),
        "Carnet (bid/ask)": (
            (OK, f"{bid} / {ask}") if bid is not None and ask is not None
            else (KO, "absent — l'ecart acheteur-vendeur de R6 sera ignore")
        ),
        # L'unite depend d'un reglage de TWS que l'API ne rapporte pas. On
        # affiche les deux lectures possibles : c'est a l'operateur de dire
        # laquelle correspond au volume qu'il voit a l'ecran.
        # On montre ce que le connecteur RETIENDRAIT, garde-fou applique, et
        # non la valeur brute : un preflight qui affiche autre chose que ce que
        # le moteur utilisera ne diagnostique pas le moteur.
        "Volume de seance": (
            (OK, f"{retenu:,.0f} actions") if retenu is not None
            else (KO, f"brut {volume:,.0f} REJETE — invraisemblable, unite ou "
                      f"locale de TWS (x100 = {volume * LOT_VOLUME_MULTIPLIER:,.0f})")
            if volume is not None
            else (KO, "absent — le volume relatif de R6 est incalculable")
        ),
        "Suspension (tick 49)": (
            (OK, "renseigne") if halt_known
            else (MEH, "non renseigne — normal hors liste TWS, traite comme "
                       "non suspendu et journalise")
        ),
        "Statut d'emprunt (236)": (
            (OK, f"'{borrow}'") if borrow is not None
            else (KO, "absent — toute position vendeuse restera refusee (R7)")
        ),
        "Actions empruntables": (
            (OK, f"{shares:,.0f}") if shares is not None else (MEH, "non diffuse")
        ),
    }


def render(title: str, rows: dict[str, tuple[str, str]]) -> None:
    print(f"\n=== {title} ===")
    for name, (mark, detail) in rows.items():
        print(f"  {mark} {name:<24} {detail}")


def diagnose(connector: IbkrConnector, symbol: str) -> int:
    """Tester les capacites une par une. Rend le nombre d'echecs bloquants."""
    ib = connector.connect()
    print(f"  connecte — client {connector.settings.client_id}, "
          f"port {connector.settings.port}")

    accounts = []
    try:
        accounts = list(ib.managedAccounts() or [])
    except Exception as error:                         # non bloquant
        print(f"  comptes illisibles : {error}")
    if accounts:
        kind = "PAPIER" if any(a.startswith("D") for a in accounts) else "a verifier"
        print(f"  compte(s) : {', '.join(accounts)}  [{kind}]")

    contract = connector._contract(symbol)
    if contract is None:
        print()
        print(f"  {symbol} n'est pas reconnu par TWS — essayer --symbol AAPL")
        return 1
    print(f"  contrat qualifie : conId {getattr(contract, 'conId', '?')}")

    def lire(kind: int, label: str):
        ib.reqMarketDataType(kind)
        tick = ib.reqMktData(contract, SHORTABLE_TICKS, False, False)
        ib.sleep(7.0)
        result = quote_capabilities(tick)
        render(f"Cotation {symbol} — {label}", result)
        try:
            ib.cancelMktData(contract)
        except Exception:                              # annulation best effort
            pass
        return result

    caps = lire(MARKET_DATA_LIVE, "flux DIRECT demande")

    # Un compte sans abonnement ne recoit rien en direct (erreur 10089). Le
    # differe est offert : on le mesure au lieu de conclure a une panne.
    if caps["Dernier prix"][0] == KO:
        print()
        print("  Aucun prix en direct — nouvelle tentative en differe.")
        differe = lire(MARKET_DATA_DELAYED, "flux DIFFERE")
        if differe["Dernier prix"][0] != KO:
            print("  -> Le differe repond. Utiliser --delayed sur les scripts.")
            caps = differe

    # Taux d'emprunt : requete distincte, droits distincts.
    rate = IbkrConnector(connector.settings, ib=ib).borrow_rate(symbol)
    extra = {
        "Taux d'emprunt": (
            (OK, f"{rate:.1%}/an") if rate is not None
            else (KO, "FEE_RATE indisponible — R7 restera sans cout d'emprunt")
        )
    }

    # Balayage : le plus susceptible d'etre refuse faute de droits.
    try:
        found = MarketScanner(IbkrConnector(connector.settings, ib=ib)).scan("spike")
        extra["Balayage (scanner)"] = (
            (OK, f"{len(found)} symbole(s) : {', '.join(found[:5])}") if found
            else (MEH, "accepte mais vide — marche calme, ou bornes R6 trop strictes")
        )
    except ScanUnavailable as error:
        extra["Balayage (scanner)"] = (KO, str(error)[:90])

    render("Capacites du compte", extra)

    print()
    print("=== UNITE DU VOLUME — a trancher a l'oeil ===")
    print("  Comparer la ligne « Volume de seance » au volume affiche par")
    print("  Gateway pour le meme titre. La lecture qui correspond donne le")
    print("  reglage : ACTIONS -> us_volume_multiplier = 1 (defaut), lots -> 100.")
    print("  Ce reglage n'est pas lisible par l'API. Se tromper fausse le")
    print("  volume relatif de R6 d'un facteur cent.")

    ouvert, _ = marche_ouvert()
    bloquants = [n for n, (m, _) in {**caps, **extra}.items() if m == KO]
    # Hors seance, ces absences sont attendues : il n'y a pas de carnet sans
    # marche. Les imputer a un abonnement enverrait payer pour rien.
    attendus = {"Carnet (bid/ask)", "Volume de seance", "Dernier prix"}
    expliques = [] if ouvert else [n for n in bloquants if n in attendus]
    bloquants = [n for n in bloquants if n not in expliques]

    print()
    print("=== VERDICT ===")
    if expliques:
        print(f"  Explique par le marche ferme : {', '.join(expliques)}")
    if not bloquants:
        print("  Aucune capacite manquante imputable au compte.")
        if ouvert:
            print("  scan_market.py et sync_ibkr.py sont utilisables.")
        else:
            print("  Verdict PARTIEL : relancer en seance pour trancher le carnet.")
    else:
        print(f"  {len(bloquants)} capacite(s) manquante(s) : {', '.join(bloquants)}")
        print("  Cause la plus frequente : droits de donnees de marche absents")
        print("  (Account Management > Settings > Market Data Subscriptions).")
        print("  Le moteur reste utilisable en mode degrade — il refusera plus.")
    return len(bloquants)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight TWS/Gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=19)
    parser.add_argument("--symbol", default="AAPL",
                        help="titre de test, liquide de preference")
    args = parser.parse_args()

    ouvert, quand = marche_ouvert()
    print(f"=== Marche : {quand} ===")
    if not ouvert:
        print("  Hors seance, l'absence de carnet et de transaction est NORMALE.")
        print("  Pour conclure sur les droits de donnees, relancer en seance.")
    print()

    print("=== Ports ===")
    probes = scan_ports(args.host)
    for port, label, listening in probes:
        print(f"  {OK if listening else KO} {port:<6} {label}")

    ouverts = [p for p, _, up in probes if up]
    port = args.port or (ouverts[0] if ouverts else None)
    if port is None:
        print("\n  Aucun port IB n'ecoute. TWS ou IB Gateway n'est pas lance,")
        print("  ou l'API n'est pas activee :")
        print("    TWS      Edit > Global Configuration > API > Settings")
        print("    Gateway  Configure > Settings > API > Settings")
        print("  Cocher « Enable ActiveX and Socket Clients », laisser")
        print("  « Read-Only API » COCHE : ce depot ne passe aucun ordre.")
        return 1

    connector = IbkrConnector(
        IbkrSettings(host=args.host, port=port, client_id=args.client_id)
    )
    try:
        return 0 if diagnose(connector, args.symbol.strip().upper()) == 0 else 2
    except IbkrUnavailable as error:
        print(f"\n  {error}")
        print("  Port ouvert mais connexion refusee : verifier que l'adresse")
        print("  127.0.0.1 figure dans « Trusted IPs » et que l'ID client")
        print(f"  {args.client_id} n'est pas deja pris par une autre session.")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Importe une liste Trading212 en reproduisant la logique d'ajout UI.

Comportement aligné sur MainWindow.add_symbol:
- validation ticker via yfinance (info.regularMarketPrice ou info.symbol)
- ajout uniquement si absent de la liste cible
- tri alphabétique
- sauvegarde via qsi.save_symbols_to_txt
- synchronisation SQLite via symbol_manager.sync_txt_to_sqlite
- si cible = mes_symbols.txt, ajout automatique aussi dans popular_symbols.txt
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import yfinance as yf

from qsi import load_symbols_from_txt, save_symbols_to_txt
from config import DB_PATH, CAP_RANGE_THRESHOLDS

try:
    from symbol_manager import sync_txt_to_sqlite
    SYMBOL_MANAGER_AVAILABLE = True
except Exception:
    SYMBOL_MANAGER_AVAILABLE = False


def validate_ticker(symbol: str) -> bool:
    """Validation robuste inspiree de l'UI sans blocage long sur ticker.info."""
    try:
        # 1) Tentative rapide
        ticker = yf.Ticker(symbol)
        fi = getattr(ticker, "fast_info", None)
        if fi:
            last_price = fi.get("last_price") or fi.get("lastPrice")
            if last_price is not None:
                return True

        # 2) Fallback: petite fenetre OHLC avec timeout court
        hist = yf.download(
            symbol,
            period="5d",
            interval="1d",
            progress=False,
            threads=False,
            timeout=6,
        )
        return hist is not None and not hist.empty
    except Exception:
        return False


def map_list_type(filename: str) -> str:
    lower = filename.lower()
    if "mes_symbol" in lower:
        return "personal"
    if "optimisation" in lower or "optimization" in lower:
        return "optimization"
    return "popular"


def classify_cap_range(market_cap_b: float | None) -> str:
    if market_cap_b is None or market_cap_b <= 0:
        return "Unknown"
    for label, (min_val, max_val) in CAP_RANGE_THRESHOLDS.items():
        if min_val <= market_cap_b < max_val:
            return label
    return "Unknown"


def fetch_market_cap_and_sector(symbol: str) -> tuple[float | None, str | None]:
    """Recupere market cap (en milliards) et secteur via yfinance."""
    ticker = yf.Ticker(symbol)

    market_cap = None
    sector = None

    try:
        fi = getattr(ticker, "fast_info", None)
        if fi:
            market_cap = fi.get("market_cap") or fi.get("marketCap")
    except Exception:
        pass

    if not market_cap or not sector:
        try:
            info = ticker.info or {}
            market_cap = market_cap or info.get("marketCap")
            sector = info.get("sector")
        except Exception:
            pass

    market_cap_b = (float(market_cap) / 1e9) if market_cap else None
    return market_cap_b, sector


def enrich_symbols_cap_metadata(symbols: Iterable[str], progress_every: int = 25) -> tuple[int, int]:
    """Met a jour market_cap_value + market_cap_range pour les symboles fournis."""
    symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    if not symbols:
        return 0, 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    updated = 0
    unknown_left = 0
    total = len(symbols)

    for idx, symbol in enumerate(symbols, start=1):
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
            print(f"[enrich {idx}/{total}] {symbol}")
        try:
            market_cap_b, sector = fetch_market_cap_and_sector(symbol)
            cap_range = classify_cap_range(market_cap_b)

            updates = ["last_checked = datetime('now')"]
            params: List[object] = []

            if market_cap_b and market_cap_b > 0:
                updates.append("market_cap_value = ?")
                params.append(market_cap_b)
            if cap_range != "Unknown":
                updates.append("market_cap_range = ?")
                params.append(cap_range)
            if sector and str(sector).strip() and str(sector).strip().lower() != "unknown":
                updates.append("sector = ?")
                params.append(sector)

            params.append(symbol)
            cur.execute(f"UPDATE symbols SET {', '.join(updates)} WHERE symbol = ?", params)

            if cap_range != "Unknown":
                updated += 1
            else:
                unknown_left += 1
        except KeyboardInterrupt:
            raise
        except Exception:
            unknown_left += 1

    conn.commit()
    conn.close()
    return updated, unknown_left


def get_symbols_with_unknown_cap(symbols: Iterable[str]) -> List[str]:
    """Filtre les symboles dont market_cap_range est Unknown/incomplet en DB."""
    symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    if not symbols:
        return []

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    placeholders = ",".join(["?"] * len(symbols))
    cur.execute(
        f"""
        SELECT symbol
        FROM symbols
        WHERE symbol IN ({placeholders})
          AND (
                COALESCE(market_cap_range, 'Unknown') = 'Unknown'
                OR market_cap_value IS NULL
                OR market_cap_value <= 0
          )
        """,
        symbols,
    )
    out = [r[0] for r in cur.fetchall() if r and r[0]]
    conn.close()
    return out


def parse_instruments_from_csv(csv_path: Path) -> List[str]:
    instruments: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "Instrument" not in reader.fieldnames:
            raise ValueError("Colonne 'Instrument' introuvable dans le CSV")

        for row in reader:
            symbol = str(row.get("Instrument", "")).strip().upper()
            if symbol:
                instruments.append(symbol)

    # Unique en conservant l'ordre d'apparition
    seen: Set[str] = set()
    unique: List[str] = []
    for s in instruments:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def add_symbols_like_ui(
    symbols_to_add: Iterable[str],
    target_filename: str,
    validate: bool = True,
    progress_every: int = 25,
    enrich_added_caps: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Retourne (added, invalid, already_exists)
    """
    existing = load_symbols_from_txt(target_filename, use_sqlite=True)
    existing_set = {s.strip().upper() for s in existing if s and s.strip()}

    added: List[str] = []
    invalid: List[str] = []
    already_exists: List[str] = []

    symbols_to_add = list(symbols_to_add)
    total = len(symbols_to_add)

    for idx, symbol in enumerate(symbols_to_add, start=1):
        s = symbol.strip().upper()
        if not s:
            continue

        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
            print(f"[{idx}/{total}] Traitement: {s}")

        if s in existing_set:
            already_exists.append(s)
            continue

        is_valid = validate_ticker(s) if validate else True
        if not is_valid:
            invalid.append(s)
            continue

        existing.append(s)
        existing_set.add(s)
        added.append(s)

    # Tri alphabétique comme dans l'UI
    existing_sorted = sorted({s.strip().upper() for s in existing if s and s.strip()})
    save_symbols_to_txt(existing_sorted, target_filename)

    # Sync SQLite comme dans l'UI
    if SYMBOL_MANAGER_AVAILABLE:
        sync_txt_to_sqlite(target_filename, list_type=map_list_type(target_filename))

    # Enrichissement explicite pour eviter les cap_range Unknown persistants.
    # On traite aussi les symboles deja presents dans la cible mais incomplets en DB.
    if enrich_added_caps:
        candidates_for_enrich = list(dict.fromkeys(added + already_exists))
        to_enrich = get_symbols_with_unknown_cap(candidates_for_enrich)
        if to_enrich:
            print(f"Enrichissement market cap/cap_range sur {len(to_enrich)} symboles incomplets...")
            updated_caps, unknown_caps = enrich_symbols_cap_metadata(to_enrich, progress_every=progress_every)
            print(f"Enrichissement termine: cap_range renseignes={updated_caps}, restant Unknown={unknown_caps}")
        else:
            print("Enrichissement non necessaire: aucun symbole incomplet detecte.")

    # Comportement UI: mes_symbols -> aussi popular
    if "mes_symbol" in target_filename.lower() and added:
        popular_file = "popular_symbols.txt"
        popular_existing = load_symbols_from_txt(popular_file, use_sqlite=True)
        pop_set = {s.strip().upper() for s in popular_existing if s and s.strip()}
        for s in added:
            if s not in pop_set:
                popular_existing.append(s)
                pop_set.add(s)

        popular_sorted = sorted(pop_set)
        save_symbols_to_txt(popular_sorted, popular_file)
        if SYMBOL_MANAGER_AVAILABLE:
            sync_txt_to_sqlite(popular_file, list_type="popular")

    return added, invalid, already_exists


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Trading212 via logique UI add_symbol"
    )
    parser.add_argument(
        "--csv",
        default="/home/berkam/Downloads/trading212-stocks-list.csv",
        help="Chemin du CSV Trading212",
    )
    parser.add_argument(
        "--target",
        default="popular_symbols.txt",
        choices=["popular_symbols.txt", "mes_symbols.txt", "optimisation_symbols.txt"],
        help="Liste cible, identique a l'UI",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Desactive la validation yfinance (non UI-like)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Afficher la progression tous les N symboles",
    )
    parser.add_argument(
        "--no-enrich-caps",
        action="store_true",
        help="Desactive l'enrichissement market cap/cap_range apres ajout",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    instruments = parse_instruments_from_csv(csv_path)
    try:
        added, invalid, already_exists = add_symbols_like_ui(
            symbols_to_add=instruments,
            target_filename=args.target,
            validate=not args.no_validate,
            progress_every=max(0, args.progress_every),
            enrich_added_caps=not args.no_enrich_caps,
        )
    except KeyboardInterrupt:
        print("\nInterruption utilisateur detectee (Ctrl+C). Fin propre du script.")
        return 130

    print(f"CSV lu: {csv_path}")
    print(f"Cible: {args.target}")
    print(f"Total instruments uniques: {len(instruments)}")
    print(f"Ajoutes: {len(added)}")
    print(f"Invalides: {len(invalid)}")
    print(f"Deja presents: {len(already_exists)}")

    if invalid:
        print("Invalides (extrait):", ", ".join(invalid[:30]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

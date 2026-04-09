#!/usr/bin/env python3
"""
Rattrapage des market caps/cap ranges pour les symboles en "Unknown".

Usage:
  /home/berkam/Projets/Gestion_trade/.venv_new/bin/python \
    /home/berkam/Projets/Gestion_trade/stock-analysis-ui/src/rattrapage_cap_ranges.py
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import List

from config import DB_PATH
from import_trading212_like_ui import enrich_symbols_cap_metadata


def fetch_unknown_symbols(limit: int = 0) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
        SELECT symbol
        FROM symbols
        WHERE is_active = 1
          AND (
                COALESCE(market_cap_range, 'Unknown') = 'Unknown'
                OR market_cap_value IS NULL
                OR market_cap_value <= 0
          )
        ORDER BY id DESC
    """

    if limit and limit > 0:
        query += f" LIMIT {int(limit)}"

    cur.execute(query)
    symbols = [r[0] for r in cur.fetchall() if r and r[0]]
    conn.close()
    return symbols


def count_unknown() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM symbols
        WHERE is_active = 1
          AND (
                COALESCE(market_cap_range, 'Unknown') = 'Unknown'
                OR market_cap_value IS NULL
                OR market_cap_value <= 0
          )
        """
    )
    val = int(cur.fetchone()[0])
    conn.close()
    return val


def main() -> int:
    parser = argparse.ArgumentParser(description="Rattrapage cap ranges/maket cap en base")
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre de symboles traites (0 = tous)")
    parser.add_argument("--progress-every", type=int, default=25, help="Afficher progression tous les N symboles")
    args = parser.parse_args()

    before = count_unknown()
    symbols = fetch_unknown_symbols(limit=args.limit)

    print(f"Unknown avant: {before}")
    print(f"Symboles a traiter: {len(symbols)}")

    if not symbols:
        print("Aucun rattrapage necessaire.")
        return 0

    updated, unknown_left_from_batch = enrich_symbols_cap_metadata(
        symbols,
        progress_every=max(0, args.progress_every),
    )
    after = count_unknown()

    print(f"Rattrapage termine: mis a jour={updated}, unknown_dans_batch={unknown_left_from_batch}")
    print(f"Unknown apres: {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

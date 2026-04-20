#!/usr/bin/env python3
"""
Migration : SQLite stock_analysis.db → store Parquet / market_store.py

Usage
-----
  python migrate_sqlite_to_parquet.py                   # migre tout
  python migrate_sqlite_to_parquet.py --dry-run         # log sans écrire
  python migrate_sqlite_to_parquet.py --symbols AAPL MSFT  # subset
  python migrate_sqlite_to_parquet.py --verify          # compare nb lignes après migration

Ce script est idempotent : il peut être relancé plusieurs fois sans doublon
(les upserts Parquet gèrent les clés en dedupliquant par symbol+date).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Chemin vers le code source (même dossier que ce script)
SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

from config import DB_PATH
import market_store as ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sqlite_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _count(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    q = f"SELECT COUNT(*) FROM {table}"
    if where:
        q += f" WHERE {where}"
    return conn.execute(q).fetchone()[0]


# ---------------------------------------------------------------------------
# Migration par table
# ---------------------------------------------------------------------------

def migrate_instruments(conn: sqlite3.Connection, dry_run: bool, symbols: list[str] | None) -> int:
    if not _table_exists(conn, "market_instruments"):
        print("  [SKIP] market_instruments: table absente dans SQLite")
        return 0

    where = ""
    params: list = []
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        where = f"WHERE symbol IN ({placeholders})"
        params = symbols

    df = pd.read_sql_query(f"SELECT * FROM market_instruments {where}", conn, params=params or None)
    if df.empty:
        print("  [SKIP] market_instruments: aucune ligne")
        return 0

    if dry_run:
        print(f"  [DRY-RUN] instruments: {len(df)} lignes seraient migrées")
        return len(df)

    ms.ensure_market_data_schema()
    path = ms._instruments_path()

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[~existing["symbol"].isin(df["symbol"])]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
    print(f"  instruments: {len(df)} lignes migrées → {path}")
    return len(df)


def migrate_prices(conn: sqlite3.Connection, dry_run: bool, symbols: list[str] | None) -> int:
    if not _table_exists(conn, "market_price_history"):
        print("  [SKIP] market_price_history: table absente dans SQLite")
        return 0

    syms_in_db: list[str]
    if symbols:
        syms_in_db = symbols
    else:
        rows = conn.execute("SELECT DISTINCT symbol FROM market_price_history").fetchall()
        syms_in_db = [r[0] for r in rows]

    total = 0
    for sym in syms_in_db:
        df = pd.read_sql_query(
            "SELECT * FROM market_price_history WHERE symbol=?", conn, params=(sym,)
        )
        if df.empty:
            continue

        if dry_run:
            print(f"  [DRY-RUN] prix {sym}: {len(df)} lignes")
            total += len(df)
            continue

        path = ms._parquet_path("prices", sym)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            existing = existing[~existing["trade_date"].isin(df["trade_date"])]
            combined = pd.concat([existing, df], ignore_index=True).sort_values("trade_date")
        else:
            combined = df.sort_values("trade_date")

        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
        print(f"  prix {sym}: {len(df)} lignes → {path}")
        total += len(df)

    return total


def migrate_fundamentals(conn: sqlite3.Connection, dry_run: bool, symbols: list[str] | None) -> int:
    if not _table_exists(conn, "market_fundamental_snapshots"):
        print("  [SKIP] market_fundamental_snapshots: table absente dans SQLite")
        return 0

    syms_in_db: list[str]
    if symbols:
        syms_in_db = symbols
    else:
        rows = conn.execute("SELECT DISTINCT symbol FROM market_fundamental_snapshots").fetchall()
        syms_in_db = [r[0] for r in rows]

    total = 0
    for sym in syms_in_db:
        df = pd.read_sql_query(
            "SELECT * FROM market_fundamental_snapshots WHERE symbol=?", conn, params=(sym,)
        )
        if df.empty:
            continue

        if dry_run:
            print(f"  [DRY-RUN] fundamentals {sym}: {len(df)} lignes")
            total += len(df)
            continue

        path = ms._parquet_path("fundamentals", sym)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            existing = existing[~existing["as_of_date"].isin(df["as_of_date"])]
            combined = pd.concat([existing, df], ignore_index=True).sort_values("as_of_date")
        else:
            combined = df.sort_values("as_of_date")

        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
        print(f"  fundamentals {sym}: {len(df)} lignes → {path}")
        total += len(df)

    return total


def migrate_features(conn: sqlite3.Connection, dry_run: bool, symbols: list[str] | None) -> int:
    if not _table_exists(conn, "market_feature_series"):
        print("  [SKIP] market_feature_series: table absente dans SQLite")
        return 0

    syms_in_db: list[str]
    if symbols:
        syms_in_db = symbols
    else:
        rows = conn.execute("SELECT DISTINCT symbol FROM market_feature_series").fetchall()
        syms_in_db = [r[0] for r in rows]

    total = 0
    for sym in syms_in_db:
        df = pd.read_sql_query(
            "SELECT * FROM market_feature_series WHERE symbol=?", conn, params=(sym,)
        )
        if df.empty:
            continue

        if dry_run:
            print(f"  [DRY-RUN] features {sym}: {len(df)} lignes")
            total += len(df)
            continue

        path = ms._parquet_path("features", sym)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            existing = existing[~existing["feature_date"].isin(df["feature_date"])]
            combined = pd.concat([existing, df], ignore_index=True).sort_values("feature_date")
        else:
            combined = df.sort_values("feature_date")

        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), str(path))
        print(f"  features {sym}: {len(df)} lignes → {path}")
        total += len(df)

    return total


# ---------------------------------------------------------------------------
# Vérification post-migration
# ---------------------------------------------------------------------------

def verify_migration(conn: sqlite3.Connection, symbols: list[str] | None) -> bool:
    print("\n--- VÉRIFICATION ---")
    ok = True

    tables = {
        "market_instruments": ("instruments/instruments.parquet", "symbol"),
        "market_price_history": ("prices/{sym}/part0.parquet", "trade_date"),
        "market_feature_series": ("features/{sym}/part0.parquet", "feature_date"),
    }

    for sql_table, (pq_pattern, _key) in tables.items():
        if not _table_exists(conn, sql_table):
            continue

        if "{sym}" in pq_pattern:
            # par symbole
            if symbols:
                syms = symbols
            else:
                syms = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM {sql_table}").fetchall()]

            for sym in syms:
                sql_count = conn.execute(
                    f"SELECT COUNT(*) FROM {sql_table} WHERE symbol=?", (sym,)
                ).fetchone()[0]
                pq_path = ms.PARQUET_DIR / pq_pattern.format(sym=f"symbol={sym.replace('/', '_')}")
                if not pq_path.exists():
                    print(f"  MANQUANT  {sym} → {pq_path}")
                    ok = False
                    continue
                pq_count = len(pd.read_parquet(pq_path))
                status = "OK" if pq_count >= sql_count else "DIFF"
                if status != "OK":
                    ok = False
                print(f"  {status}  {sym} {sql_table}: SQLite={sql_count} Parquet={pq_count}")
        else:
            sql_count = conn.execute(f"SELECT COUNT(*) FROM {sql_table}").fetchone()[0]
            pq_path = ms.PARQUET_DIR / pq_pattern
            if not pq_path.exists():
                print(f"  MANQUANT {sql_table} → {pq_path}")
                ok = False
                continue
            pq_count = len(pd.read_parquet(pq_path))
            status = "OK" if pq_count >= sql_count else "DIFF"
            if status != "OK":
                ok = False
            print(f"  {status}  {sql_table}: SQLite={sql_count} Parquet={pq_count}")

    return ok


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Migre stock_analysis.db vers le store Parquet")
    parser.add_argument("--dry-run", action="store_true", help="Log sans rien écrire")
    parser.add_argument("--symbols", nargs="+", help="Restreindre la migration à ces symboles")
    parser.add_argument("--verify", action="store_true", help="Vérifier les comptes après migration")
    args = parser.parse_args()

    print(f"SQLite source : {DB_PATH}")
    print(f"Parquet cible : {ms.PARQUET_DIR}")
    if args.dry_run:
        print("[MODE DRY-RUN — aucune écriture]")

    conn = _sqlite_connect()

    print("\n1. Instruments")
    n_instruments = migrate_instruments(conn, args.dry_run, args.symbols)

    print("\n2. Prix OHLCV")
    n_prices = migrate_prices(conn, args.dry_run, args.symbols)

    print("\n3. Snapshots fondamentaux")
    n_fundamentals = migrate_fundamentals(conn, args.dry_run, args.symbols)

    print("\n4. Feature series")
    n_features = migrate_features(conn, args.dry_run, args.symbols)

    conn_verify = _sqlite_connect()
    success = True
    if args.verify and not args.dry_run:
        success = verify_migration(conn_verify, args.symbols)
    conn_verify.close()

    conn.close()

    print(f"\n=== RÉSUMÉ ===")
    print(f"  instruments : {n_instruments}")
    print(f"  prix        : {n_prices}")
    print(f"  fundamentals: {n_fundamentals}")
    print(f"  features    : {n_features}")
    if args.verify and not args.dry_run:
        print(f"  vérification: {'SUCCÈS' if success else 'ÉCHEC (voir ci-dessus)'}")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

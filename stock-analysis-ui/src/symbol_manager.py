"""
Module pour gérer les symboles boursiers dans SQLite - Version sans emojis pour Windows.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import pandas as pd

DB_PATH = "stock_analysis.db"

def init_symbols_table():
    """Crée la table symbols si elle n'existe pas."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            sector TEXT,
            market_cap_range TEXT,
            market_cap_value REAL,
            list_type TEXT,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_checked TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON symbols(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_list_type ON symbols(list_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sector ON symbols(sector)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_cap_range ON symbols(market_cap_range)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_active ON symbols(is_active)')
    
    conn.commit()
    conn.close()

def sync_txt_to_sqlite(txt_file: str, list_type: str = 'popular', force_refresh: bool = False) -> int:
    """Synchronise les symboles d'un fichier txt vers SQLite.
    
    Args:
        txt_file: Chemin vers le fichier texte contenant les symboles
        list_type: Type de liste ('popular', 'mes_symbols', etc.)
        force_refresh: Si True, force la récupération des métadonnées même si elles existent
    """
    path = Path(txt_file)
    if not path.exists():
        return 0
    
    init_symbols_table()
    symbols = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    added = 0
    updated = 0
    needs_fetch = []
    
    for symbol in symbols:
        try:
            # Vérifier si le symbole existe déjà avec des métadonnées valides
            if not force_refresh:
                cursor.execute('''
                    SELECT sector, market_cap_range, market_cap_value 
                    FROM symbols 
                    WHERE symbol = ? AND sector IS NOT NULL AND sector != 'Unknown'
                ''', (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    # Le symbole existe avec des métadonnées valides, juste mettre à jour list_type
                    cursor.execute('''
                        UPDATE symbols 
                        SET list_type = ?, is_active = 1
                        WHERE symbol = ?
                    ''', (list_type, symbol))
                    updated += 1
                    continue
            
            # Marquer pour récupération API
            needs_fetch.append(symbol)
            
        except Exception:
            pass
    
    # Récupérer les métadonnées pour les symboles qui en ont besoin
    if needs_fetch:
        print(f"   📡 Récupération métadonnées pour {len(needs_fetch)} nouveaux symboles...")
        for idx, symbol in enumerate(needs_fetch, 1):
            if idx % 10 == 0:
                print(f"      Progression: {idx}/{len(needs_fetch)}...")
            
            try:
                sector = _get_sector_safe(symbol)
                cap_range, market_cap = _get_cap_range_safe(symbol)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO symbols 
                    (symbol, sector, market_cap_range, market_cap_value, list_type, last_checked, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (symbol, sector, cap_range, market_cap, list_type, datetime.now().isoformat()))
                added += 1
            except Exception:
                pass
    
    conn.commit()
    conn.close()
    
    if updated > 0:
        print(f"   ♻️  {updated} symboles déjà en cache (récupération instantanée)")
    if added > 0:
        print(f"   ✅ {added} nouveaux symboles ajoutés")
    
    return added + updated

def get_symbols_by_list_type(list_type: str = 'popular', active_only: bool = True) -> List[str]:
    """Récupère les symboles pour un type de liste donné."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT symbol FROM symbols WHERE list_type = ?'
    params = [list_type]
    
    if active_only:
        query += ' AND is_active = 1'
    
    query += ' ORDER BY added_date DESC'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_symbols_by_sector(sector: str, list_type: str = None, active_only: bool = True) -> List[str]:
    """Récupère les symboles d'un secteur donné."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT symbol FROM symbols WHERE sector = ?'
    params = [sector]
    
    if list_type:
        query += ' AND list_type = ?'
        params.append(list_type)
    
    if active_only:
        query += ' AND is_active = 1'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_symbols_by_cap_range(cap_range: str, list_type: str = None, active_only: bool = True) -> List[str]:
    """Récupère les symboles d'une gamme de capitalisation donnée."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT symbol FROM symbols WHERE market_cap_range = ?'
    params = [cap_range]
    
    if list_type:
        query += ' AND list_type = ?'
        params.append(list_type)
    
    if active_only:
        query += ' AND is_active = 1'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_symbols_by_sector_and_cap(sector: str, cap_range: str, list_type: str = None, active_only: bool = True) -> List[str]:
    """Récupère les symboles d'un secteur ET d'une gamme de capitalisation."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT symbol FROM symbols WHERE sector = ? AND market_cap_range = ?'
    params = [sector, cap_range]
    
    if list_type:
        query += ' AND list_type = ?'
        params.append(list_type)
    
    if active_only:
        query += ' AND is_active = 1'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_all_sectors(list_type: Optional[str] = None) -> List[str]:
    """Récupère la liste unique des secteurs disponibles (optionnellement filtrés par type)."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = 'SELECT DISTINCT sector FROM symbols WHERE is_active = 1'
    params: List[str] = []

    if list_type:
        query += ' AND list_type = ?'
        params.append(list_type)

    query += ' ORDER BY sector'

    cursor.execute(query, params)
    sectors = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return sectors

def get_all_cap_ranges(list_type: Optional[str] = None) -> List[str]:
    """Récupère la liste unique des gammes de capitalisation (optionnellement filtrées par type)."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = '''
        SELECT DISTINCT market_cap_range FROM symbols 
        WHERE is_active = 1 AND market_cap_range IS NOT NULL
    '''
    params: List[str] = []

    if list_type:
        query += ' AND list_type = ?'
        params.append(list_type)

    query += '''
        ORDER BY 
            CASE market_cap_range
                WHEN 'Small' THEN 1
                WHEN 'Mid' THEN 2
                WHEN 'Large' THEN 3
                WHEN 'Mega' THEN 4
                WHEN 'Giant' THEN 4 -- compat anciens enregistrements
                ELSE 5
            END
    '''

    cursor.execute(query, params)
    cap_ranges = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return cap_ranges

def get_symbol_count(list_type: str = None, active_only: bool = True) -> int:
    """Compte le nombre de symboles."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT COUNT(*) FROM symbols'
    params = []
    
    if list_type:
        query += ' WHERE list_type = ?'
        params.append(list_type)
        if active_only:
            query += ' AND is_active = 1'
    elif active_only:
        query += ' WHERE is_active = 1'
    
    cursor.execute(query, params)
    count = cursor.fetchone()[0]
    conn.close()
    
    return count

def _get_sector_safe(symbol: str) -> str:
    """Récupère le secteur d'une action avec gestion d'erreur."""
    try:
        ticker = yf.Ticker(symbol)
        sector = ticker.info.get('sector', 'Unknown')
        return sector if sector else 'Unknown'
    except Exception:
        return 'Unknown'

def _get_cap_range_safe(symbol: str) -> Tuple[str, float]:
    """Récupère la gamme de capitalisation et la valeur en milliards."""
    try:
        ticker = yf.Ticker(symbol)
        market_cap = ticker.info.get('marketCap')
        if market_cap is None:
            return 'Unknown', None
        
        market_cap_b = market_cap / 1e9
        
        if market_cap_b < 2:
            return 'Small', market_cap_b
        elif market_cap_b < 10:
            return 'Mid', market_cap_b
        elif market_cap_b < 200:
            return 'Large', market_cap_b
        else:
            return 'Mega', market_cap_b
    except Exception:
        return 'Unknown', None

# ----------------------------
# Enrichissement depuis indices
# ----------------------------

def _normalize_symbol(symbol: str) -> str:
    """Normalise un ticker (ex: BRK.B -> BRK-B pour Yahoo)."""
    if not symbol:
        return symbol
    # Yahoo finance utilise '-' au lieu de '.' pour les classes d'actions US
    return symbol.strip().upper().replace('.', '-')

def get_sp500_constituents() -> List[str]:
    """Récupère la liste des constituants du S&P 500 depuis Wikipedia.

    Retourne une liste de tickers normalisés pour Yahoo Finance.
    """
    # 1) Fallback local: sp500_symbols.txt si présent
    local_file = Path("sp500_symbols.txt")
    if local_file.exists():
        try:
            syms = [
                _normalize_symbol(line)
                for line in local_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if syms:
                return list(dict.fromkeys(syms))
        except Exception:
            pass

    # 2) Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        for df in tables:
            # normaliser colonnes
            cols = [str(c).strip().lower() for c in df.columns]
            if any("symbol" == c or "ticker" in c for c in cols):
                cand_col = None
                for name in df.columns:
                    low = str(name).strip().lower()
                    if low == "symbol" or low.startswith("ticker"):
                        cand_col = name
                        break
                if cand_col is not None:
                    symbols = [
                        _normalize_symbol(s)
                        for s in df[cand_col].astype(str).tolist()
                    ]
                    return list(dict.fromkeys([s for s in symbols if s]))
    except Exception:
        pass
    return []

def add_sp500_to_popular(list_type: str = 'popular') -> Dict[str, int]:
    """Ajoute les constituants du S&P 500 à la liste 'popular' (ou autre list_type).

    Retourne un dict avec les compteurs {added, updated, skipped}.
    """
    init_symbols_table()
    sp500 = get_sp500_constituents()
    if not sp500:
        return {"added": 0, "updated": 0, "skipped": 0}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    added = 0
    updated = 0
    skipped = 0

    for sym in sp500:
        try:
            # Vérifier s'il existe déjà
            cursor.execute("SELECT sector, market_cap_range, list_type FROM symbols WHERE symbol = ?", (sym,))
            row = cursor.fetchone()

            sector = _get_sector_safe(sym)
            cap_range, market_cap = _get_cap_range_safe(sym)

            if row is None:
                cursor.execute(
                    '''INSERT INTO symbols (symbol, sector, market_cap_range, market_cap_value, list_type, last_checked, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, 1)''',
                    (sym, sector, cap_range, market_cap, list_type, datetime.now().isoformat())
                )
                added += 1
            else:
                # Mettre à jour le list_type à 'popular' si différent, et rafraîchir métadonnées
                cursor.execute(
                    '''UPDATE symbols SET sector=?, market_cap_range=?, market_cap_value=?, list_type=?, last_checked=?, is_active=1
                       WHERE symbol=?''',
                    (sector, cap_range, market_cap, list_type, datetime.now().isoformat(), sym)
                )
                updated += 1
        except Exception:
            skipped += 1

    conn.commit()
    conn.close()
    return {"added": added, "updated": updated, "skipped": skipped}

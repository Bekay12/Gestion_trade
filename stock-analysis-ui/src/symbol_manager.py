"""
Module pour gérer les symboles boursiers dans SQLite - Version sans emojis pour Windows.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json
from config import DB_PATH

_FX_RATE_MEM = {}

_CCY_SUBUNIT_TO_MAJOR = {
    'GBX': ('GBP', 0.01),
    'GBPX': ('GBP', 0.01),
    'GBP.P': ('GBP', 0.01),
    'GBP': ('GBP', 1.0),
    'GBp': ('GBP', 0.01),
    'ZAc': ('ZAR', 0.01),
    'ZAR': ('ZAR', 1.0),
}

def init_symbols_table():
    """Crée les tables symbols et symbol_lists si elles n'existent pas."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table principale des symboles (métadonnées uniques)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            sector TEXT,
            market_cap_range TEXT,
            market_cap_value REAL,
            currency TEXT DEFAULT 'USD',
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_checked TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Table de jonction many-to-many : un symbole peut appartenir à plusieurs listes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbol_lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            list_type TEXT NOT NULL,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, list_type),
            FOREIGN KEY (symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON symbols(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sector ON symbols(sector)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_cap_range ON symbols(market_cap_range)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_active ON symbols(is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_lists_symbol ON symbol_lists(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_lists_type ON symbol_lists(list_type)')
    
    # Migration: Migrer les anciennes données list_type vers symbol_lists si nécessaire
    # ✅ CORRIGÉ: Après la migration, on supprime la colonne legacy list_type de la table
    # symbols pour ne plus re-migrer à chaque init (ce qui annulait les suppressions de l'utilisateur).
    cursor.execute("PRAGMA table_info(symbols)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'list_type' in columns:
        # Migrer les données existantes une dernière fois
        cursor.execute('''
            INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
            SELECT symbol, list_type FROM symbols WHERE list_type IS NOT NULL
        ''')
        # Supprimer la colonne legacy en recréant la table sans elle
        # (SQLite < 3.35 ne supporte pas ALTER TABLE DROP COLUMN)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbols_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                sector TEXT,
                market_cap_range TEXT,
                market_cap_value REAL,
                currency TEXT DEFAULT 'USD',
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO symbols_new (id, symbol, sector, market_cap_range, market_cap_value, currency, added_date, last_checked, is_active)
            SELECT id, symbol, sector, market_cap_range, market_cap_value, 'USD', added_date, last_checked, is_active
            FROM symbols
        ''')
        cursor.execute('DROP TABLE symbols')
        cursor.execute('ALTER TABLE symbols_new RENAME TO symbols')
        # Re-créer les index sur la nouvelle table
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON symbols(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sector ON symbols(sector)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_cap_range ON symbols(market_cap_range)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_active ON symbols(is_active)')
        print("✅ Migration list_type terminée: colonne legacy supprimée de la table symbols")

    # Migration légère: ajout de la colonne devise si absente
    cursor.execute("PRAGMA table_info(symbols)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'currency' not in columns:
        cursor.execute("ALTER TABLE symbols ADD COLUMN currency TEXT DEFAULT 'USD'")
    
    # Table pour cacher les groupes nettoyés (complétion + limitation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cleaned_groups_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector TEXT NOT NULL,
            cap_range TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sector, cap_range)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_sector_cap ON cleaned_groups_cache(sector, cap_range)')
    
    conn.commit()
    conn.close()

def sync_txt_to_sqlite(txt_file: str, list_type: str = 'popular', force_refresh: bool = False) -> int:
    """Synchronise les symboles d'un fichier txt vers SQLite.
    
    Args:
        txt_file: Chemin vers le fichier texte contenant les symboles
        list_type: Type de liste ('popular', 'mes_symbols', etc.)
        force_refresh: Si True, force la récupération des métadonnées même si elles existent
    """
    # ✅ CORRIGÉ: Résoudre le chemin relatif vers le répertoire de ce module (src/)
    # pour rester cohérent avec save_symbols_to_txt et load_symbols_from_txt
    path = Path(txt_file)
    if not path.is_absolute():
        module_dir = Path(__file__).parent
        candidate = module_dir / txt_file
        if candidate.exists():
            path = candidate
        # sinon on garde le chemin tel quel (CWD fallback)
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
            # Vérifier si le symbole existe déjà dans la DB (avec OU sans métadonnées)
            if not force_refresh:
                cursor.execute('''
                    SELECT sector, market_cap_range, market_cap_value, currency, last_checked
                    FROM symbols 
                    WHERE symbol = ?
                ''', (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    sector_db, cap_range_db, cap_value_db, currency_db, last_checked_db = existing

                    # Si les metadonnees cap sont incomplètes, re-fetch periodique pour eviter
                    # de garder "Unknown" indefiniment apres import massif.
                    cap_range_unknown = (
                        cap_range_db is None
                        or str(cap_range_db).strip() == ''
                        or str(cap_range_db).strip().lower() == 'unknown'
                    )
                    cap_value_missing = (cap_value_db is None or float(cap_value_db) <= 0)
                    cap_incomplete = (
                        cap_range_unknown or cap_value_missing
                    )
                    stale_or_never_checked = (
                        last_checked_db is None
                    )
                    if not stale_or_never_checked and cap_incomplete:
                        try:
                            # last_checked est souvent ISO8601; comparaison tolérante coté SQLite
                            cursor.execute("""
                                SELECT CASE
                                    WHEN datetime(?) IS NULL THEN 1
                                    WHEN datetime(?) <= datetime('now', '-14 days') THEN 1
                                    ELSE 0
                                END
                            """, (last_checked_db, last_checked_db))
                            stale_or_never_checked = bool(cursor.fetchone()[0])
                        except Exception:
                            stale_or_never_checked = True

                    if cap_incomplete and stale_or_never_checked:
                        needs_fetch.append(symbol)
                    elif not currency_db:
                        needs_fetch.append(symbol)

                    # Garder le symbole actif et rattache a la liste
                    cursor.execute('UPDATE symbols SET is_active = 1 WHERE symbol = ?', (symbol,))
                    cursor.execute('''
                        INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                        VALUES (?, ?)
                    ''', (symbol, list_type))
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
                currency = _get_currency_safe(symbol)

                # Backfill: si seule la market cap est connue, dériver la tranche.
                if (not cap_range or str(cap_range).strip().lower() == 'unknown') and market_cap and float(market_cap) > 0:
                    cap_range = classify_cap_range(float(market_cap))
                
                # Insérer ou mettre à jour les métadonnées du symbole
                cursor.execute('''
                    INSERT OR REPLACE INTO symbols 
                    (symbol, sector, market_cap_range, market_cap_value, currency, last_checked, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (symbol, sector, cap_range, market_cap, currency, datetime.now().isoformat()))
                
                # Ajouter à la liste (relation many-to-many)
                cursor.execute('''
                    INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                    VALUES (?, ?)
                ''', (symbol, list_type))
                added += 1
            except Exception:
                # Même si l'enrichissement distant échoue, garder une ligne cohérente en DB.
                cursor.execute('''
                    INSERT OR IGNORE INTO symbols 
                    (symbol, sector, market_cap_range, market_cap_value, currency, last_checked, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                ''', (symbol, 'Unknown', 'Unknown', 0.0, 'USD', datetime.now().isoformat()))
                cursor.execute('''
                    INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                    VALUES (?, ?)
                ''', (symbol, list_type))
    
    # ✅ CORRIGÉ: Supprimer de la liste SQLite les symboles qui ne sont plus dans le fichier txt
    # (sinon les symboles supprimés par l'utilisateur reviennent au redémarrage)
    if symbols:
        try:
            placeholders = ','.join('?' * len(symbols))
            cursor.execute(f'''
                DELETE FROM symbol_lists 
                WHERE list_type = ? AND symbol NOT IN ({placeholders})
            ''', [list_type] + symbols)
            removed_count = cursor.rowcount
            if removed_count > 0:
                print(f"   🗑️  {removed_count} symboles retirés de la liste '{list_type}' dans SQLite")
        except Exception as e:
            print(f"   ⚠️ Erreur suppression symboles obsolètes: {e}")
    
    conn.commit()
    conn.close()
    
    if updated > 0:
        print(f"   ♻️  {updated} symboles déjà en cache (récupération instantanée)")
    if added > 0:
        print(f"   ✅ {added} nouveaux symboles ajoutés")
    
    # 🔄 Auto-sync vers popular : tous les symboles de personal/optimization vont aussi dans popular
    if list_type in ['personal', 'optimization'] and (added > 0 or updated > 0):
        synced = auto_add_to_popular(symbols)
        if synced > 0:
            print(f"   🔄 {synced} symboles auto-synchronisés vers popular")
    
    return added + updated

def get_symbols_by_list_type(list_type: str = 'popular', active_only: bool = True) -> List[str]:
    """Récupère les symboles pour un type de liste donné."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Utiliser la table de jonction symbol_lists
    query = '''
        SELECT DISTINCT s.symbol 
        FROM symbols s
        INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
        WHERE sl.list_type = ?
    '''
    params = [list_type]
    
    if active_only:
        query += ' AND s.is_active = 1'
    
    query += ' ORDER BY sl.added_date DESC'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_recent_symbols(limit: int = 30, active_only: bool = True) -> List[str]:
    """Récupère les N derniers symboles ajoutés à la base de données, triés par date d'ajout (plus récents en premier)."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Récupérer les symboles en triant par added_date DESC (plus récents en premier)
    query = '''
        SELECT DISTINCT s.symbol
        FROM symbols s
        WHERE 1=1
    '''

    if active_only:
        query += ' AND s.is_active = 1'

    query += ' ORDER BY s.added_date DESC LIMIT ?'

    cursor.execute(query, (limit,))
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()

    return symbols

def auto_add_to_popular(symbols: List[str]) -> int:
    """
    Ajoute automatiquement des symboles à la liste 'popular' s'ils n'y sont pas déjà.
    Utilisé pour maintenir popular comme liste master contenant tous les symboles.
    
    Returns:
        Nombre de symboles ajoutés à popular
    """
    if not symbols:
        return 0
    
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    added = 0
    for symbol in symbols:
        try:
            # Vérifier si le symbole existe dans la table symbols
            cursor.execute('SELECT symbol FROM symbols WHERE symbol = ?', (symbol,))
            if cursor.fetchone():
                # Ajouter à popular s'il n'y est pas déjà
                cursor.execute('''
                    INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                    VALUES (?, 'popular')
                ''', (symbol,))
                if cursor.rowcount > 0:
                    added += 1
        except Exception:
            pass
    
    conn.commit()
    conn.close()
    
    return added

def sync_all_to_popular() -> Dict[str, int]:
    """
    Synchronise tous les symboles de 'personal' et 'optimization' vers 'popular'.
    Popular devient la liste master qui contient tous les symboles.
    
    Returns:
        Dict avec les compteurs {personal_synced, optimization_synced}
    """
    init_symbols_table()
    
    # Récupérer tous les symboles de personal et optimization
    personal_symbols = get_symbols_by_list_type('personal', active_only=True)
    optimization_symbols = get_symbols_by_list_type('optimization', active_only=True)
    
    # Ajouter à popular
    personal_added = auto_add_to_popular(personal_symbols)
    optimization_added = auto_add_to_popular(optimization_symbols)
    
    return {
        'personal_synced': personal_added,
        'optimization_synced': optimization_added,
        'total': personal_added + optimization_added
    }

def get_symbols_by_sector(sector: str, list_type: str = None, active_only: bool = True) -> List[str]:
    """Récupère les symboles d'un secteur donné."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if list_type:
        query = '''
            SELECT DISTINCT s.symbol 
            FROM symbols s
            INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
            WHERE s.sector = ? AND sl.list_type = ?
        '''
        params = [sector, list_type]
    else:
        query = 'SELECT symbol FROM symbols WHERE sector = ?'
        params = [sector]
    
    if active_only:
        query += ' AND s.is_active = 1' if list_type else ' AND is_active = 1'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_symbols_by_cap_range(cap_range: str, list_type: str = None, active_only: bool = True) -> List[str]:
    """Récupère les symboles d'une gamme de capitalisation donnée."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if list_type:
        query = '''
            SELECT DISTINCT s.symbol 
            FROM symbols s
            INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
            WHERE s.market_cap_range = ? AND sl.list_type = ?
        '''
        params = [cap_range, list_type]
    else:
        query = 'SELECT symbol FROM symbols WHERE market_cap_range = ?'
        params = [cap_range]
    
    if active_only:
        query += ' AND s.is_active = 1' if list_type else ' AND is_active = 1'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_all_popular_symbols(max_count: Optional[int] = None, exclude_symbols: Optional[set] = None) -> List[str]:
    """Retourne tous les symboles populaires en respectant une éventuelle exclusion."""
    exclude = set(exclude_symbols or [])
    symbols = [s for s in get_symbols_by_list_type('popular', active_only=True) if s not in exclude]
    return symbols[:max_count] if max_count is not None else symbols

def get_symbols_by_sector_and_cap(sector: str, cap_range: str, list_type: str = None, active_only: bool = True) -> List[str]:
    """Récupère les symboles d'un secteur ET d'une gamme de capitalisation."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if list_type:
        query = '''
            SELECT DISTINCT s.symbol 
            FROM symbols s
            INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
            WHERE s.sector = ? AND s.market_cap_range = ? AND sl.list_type = ?
        '''
        params = [sector, cap_range, list_type]
    else:
        query = 'SELECT symbol FROM symbols WHERE sector = ? AND market_cap_range = ?'
        params = [sector, cap_range]
    
    if active_only:
        query += ' AND s.is_active = 1' if list_type else ' AND is_active = 1'
    
    cursor.execute(query, params)
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return symbols

def get_all_sectors(list_type: Optional[str] = None) -> List[str]:
    """Récupère la liste unique des secteurs disponibles (optionnellement filtrés par type)."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if list_type:
        query = '''
            SELECT DISTINCT s.sector 
            FROM symbols s
            INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
            WHERE s.is_active = 1 AND sl.list_type = ?
            ORDER BY s.sector
        '''
        params = [list_type]
    else:
        query = 'SELECT DISTINCT sector FROM symbols WHERE is_active = 1 ORDER BY sector'
        params = []

    cursor.execute(query, params)
    sectors = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return sectors

def get_all_cap_ranges(list_type: Optional[str] = None) -> List[str]:
    """Récupère la liste unique des gammes de capitalisation (optionnellement filtrées par type)."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if list_type:
        query = '''
            SELECT DISTINCT s.market_cap_range 
            FROM symbols s
            INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
            WHERE s.is_active = 1 AND s.market_cap_range IS NOT NULL AND sl.list_type = ?
        '''
        params = [list_type]
    else:
        query = '''
            SELECT DISTINCT market_cap_range FROM symbols 
            WHERE is_active = 1 AND market_cap_range IS NOT NULL
        '''
        params = []

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
    
    if list_type:
        query = 'SELECT COUNT(DISTINCT sl.symbol) FROM symbol_lists sl JOIN symbols s ON sl.symbol = s.symbol WHERE sl.list_type = ?'
        params = [list_type]
        if active_only:
            query += ' AND s.is_active = 1'
    else:
        query = 'SELECT COUNT(*) FROM symbols'
        params = []
        if active_only:
            query += ' WHERE is_active = 1'
    
    cursor.execute(query, params)
    count = cursor.fetchone()[0]
    conn.close()
    
    return count

def get_symbol_info_from_db(symbol: str) -> dict:
    """Récupère les infos d'un symbole depuis la DB cache.
    
    Returns:
        dict avec keys: sector, market_cap_range, market_cap_value (ou vide si not found)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT sector, market_cap_range, market_cap_value, currency FROM symbols WHERE symbol = ?',
            (symbol.upper(),)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            sector, cap_range, cap_value, currency = row
            inferred_cap_range = cap_range if cap_range else 'Unknown'
            try:
                if (not inferred_cap_range or str(inferred_cap_range).lower() == 'unknown') and cap_value and float(cap_value) > 0:
                    inferred_cap_range = classify_cap_range(float(cap_value))
            except Exception:
                inferred_cap_range = cap_range if cap_range else 'Unknown'
            return {
                'sector': sector if sector and sector != 'Inconnu' else 'Inconnu',
                'market_cap_range': inferred_cap_range,
                'market_cap_value': cap_value,
                'currency': currency if currency else 'USD'
            }
        return {}
    except Exception:
        return {}

def _get_sector_safe(symbol: str) -> str:
    """Récupère le secteur depuis le cache DB, avec fallback yfinance.
    
    Priorité:
    1. Database cache (local DB)
    2. yfinance API (online)
    """
    # Essayer la DB d'abord (zéro latence)
    info = get_symbol_info_from_db(symbol)
    if info.get('sector') and info['sector'] != 'Inconnu':
        return info['sector']
    
    # Fallback: yfinance
    try:
        ticker = yf.Ticker(symbol)
        sector = ticker.info.get('sector', 'Unknown')
        return sector if sector else 'Unknown'
    except Exception:
        return 'Unknown'

def _get_cap_range_safe(symbol: str) -> Tuple[str, float]:
    """Récupère la gamme de capitalisation depuis le cache DB, avec fallback yfinance.
    
    Priorité:
    1. Database cache (local DB)
    2. yfinance API (online)
    """
    # Essayer la DB d'abord (zéro latence)
    info = get_symbol_info_from_db(symbol)
    if info.get('market_cap_range') and info['market_cap_range'] != 'Unknown':
        return info['market_cap_range'], info.get('market_cap_value')
    
    # Fallback: yfinance
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        market_cap = info.get('marketCap')
        # Pour certains fonds/ETF, marketCap est absent mais totalAssets est disponible.
        if market_cap is None:
            market_cap = info.get('totalAssets')
        if market_cap is None:
            return 'Unknown', None

        currency = str(info.get('currency') or 'USD').strip().upper()
        rate_to_usd = _get_rate_to_usd_simple(currency)
        market_cap_b = (float(market_cap) * rate_to_usd) / 1e9
        
        if market_cap_b < 2:
            return 'Small', market_cap_b
        elif market_cap_b < 10:
            return 'Mid', market_cap_b
        elif market_cap_b < 100:
            return 'Large', market_cap_b
        else:
            return 'Mega', market_cap_b
    except Exception:
        return 'Unknown', None

def _get_rate_to_usd_simple(currency: str) -> float:
    raw = str(currency or 'USD').strip()
    if raw in _CCY_SUBUNIT_TO_MAJOR:
        cur, unit_factor = _CCY_SUBUNIT_TO_MAJOR[raw]
    else:
        cur_up = raw.upper()
        cur, unit_factor = _CCY_SUBUNIT_TO_MAJOR.get(cur_up, (cur_up, 1.0))
    if not cur or cur == 'USD':
        return float(unit_factor)
    cache_key = f"{cur}@{unit_factor}"
    if cache_key in _FX_RATE_MEM:
        return _FX_RATE_MEM[cache_key]
    try:
        fx_pair = f"{cur}USD=X"
        fx = yf.Ticker(fx_pair)
        info = fx.info or {}
        rate = info.get('regularMarketPrice')
        if rate is None:
            fi = getattr(fx, 'fast_info', {}) or {}
            rate = fi.get('last_price') or fi.get('lastPrice')
        rate = float(rate) if rate else 1.0
        if rate <= 0:
            rate = 1.0
    except Exception:
        rate = 1.0
    rate_to_usd = float(rate) * float(unit_factor)
    _FX_RATE_MEM[cache_key] = rate_to_usd
    return rate_to_usd

def _get_currency_safe(symbol: str) -> str:
    """Récupère la devise native depuis le cache DB, avec fallback yfinance."""
    info = get_symbol_info_from_db(symbol)
    cached_currency = str(info.get('currency') or '').strip().upper()
    if cached_currency and not (cached_currency == 'USD' and '.' in str(symbol)):
        return cached_currency

    try:
        ticker = yf.Ticker(symbol)
        currency = ticker.info.get('currency', 'USD')
        return str(currency or 'USD').strip().upper()
    except Exception:
        return 'USD'

# ----------------------------
# Enrichissement depuis indices
# ----------------------------

if __name__ == '__main__':
    display_popular_symbols_distribution()
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
from config import DB_PATH, CAP_RANGE_THRESHOLDS

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
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO symbols_new (id, symbol, sector, market_cap_range, market_cap_value, added_date, last_checked, is_active)
            SELECT id, symbol, sector, market_cap_range, market_cap_value, added_date, last_checked, is_active
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
                    SELECT sector, market_cap_range, market_cap_value 
                    FROM symbols 
                    WHERE symbol = ?
                ''', (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    # ✅ Le symbole existe déjà en DB — ne PAS re-fetcher même si sector='Unknown'
                    # (ETFs, futures, cryptos n'ont pas de secteur — c'est normal)
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
                
                # Insérer ou mettre à jour les métadonnées du symbole
                cursor.execute('''
                    INSERT OR REPLACE INTO symbols 
                    (symbol, sector, market_cap_range, market_cap_value, last_checked, is_active)
                    VALUES (?, ?, ?, ?, ?, 1)
                ''', (symbol, sector, cap_range, market_cap, datetime.now().isoformat()))
                
                # Ajouter à la liste (relation many-to-many)
                cursor.execute('''
                    INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                    VALUES (?, ?)
                ''', (symbol, list_type))
                added += 1
            except Exception:
                pass
    
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

def get_popular_symbols_by_sector(sector: str, max_count: Optional[int] = None, exclude_symbols: Optional[set] = None) -> List[str]:
    """Retourne les symboles populaires d'un secteur, en excluant éventuellement certains tickers."""
    exclude = set(exclude_symbols or [])
    symbols = [s for s in get_symbols_by_sector(sector, list_type='popular', active_only=True) if s not in exclude]
    return symbols[:max_count] if max_count is not None else symbols

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
            cursor.execute("SELECT sector, market_cap_range FROM symbols WHERE symbol = ?", (sym,))
            row = cursor.fetchone()

            sector = _get_sector_safe(sym)
            cap_range, market_cap = _get_cap_range_safe(sym)

            if row is None:
                # Nouveau symbole : insérer dans symbols et symbol_lists
                cursor.execute(
                    '''INSERT INTO symbols (symbol, sector, market_cap_range, market_cap_value, last_checked, is_active)
                       VALUES (?, ?, ?, ?, ?, 1)''',
                    (sym, sector, cap_range, market_cap, datetime.now().isoformat())
                )
                cursor.execute(
                    '''INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                       VALUES (?, ?)''',
                    (sym, list_type)
                )
                added += 1
            else:
                # Symbole existe : mettre à jour métadonnées et ajouter à la liste si absent
                cursor.execute(
                    '''UPDATE symbols SET sector=?, market_cap_range=?, market_cap_value=?, last_checked=?, is_active=1
                       WHERE symbol=?''',
                    (sector, cap_range, market_cap, datetime.now().isoformat(), sym)
                )
                cursor.execute(
                    '''INSERT OR IGNORE INTO symbol_lists (symbol, list_type)
                       VALUES (?, ?)''',
                    (sym, list_type)
                )
                updated += 1
        except Exception:
            skipped += 1

    conn.commit()
    conn.close()
    return {"added": added, "updated": updated, "skipped": skipped}

# ===================================================================
# CACHE INTELLIGENT DES SECTEURS (DISQUE + MEMOIRE + TTL)
# ===================================================================

import json

SECTOR_CACHE_FILE = Path("cache_data/sector_cache.json")
SECTOR_CACHE_FILE.parent.mkdir(exist_ok=True)
SECTOR_TTL_DAYS = 30
SECTOR_TTL_UNKNOWN_DAYS = 7

def _load_sector_cache() -> Dict:
    """Charge le cache des secteurs depuis le disque."""
    try:
        if SECTOR_CACHE_FILE.exists():
            with open(SECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Impossible de charger cache secteurs: {e}")
    return {}

def _save_sector_cache(cache: Dict) -> None:
    """Sauvegarde le cache des secteurs sur disque."""
    try:
        with open(SECTOR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Impossible d'ecrire cache secteurs: {e}")

def _is_sector_expired(entry: Dict) -> bool:
    """Verifie si une entree cache est expiree."""
    try:
        ts = entry.get("ts")
        if not ts:
            return True
        dt = datetime.fromisoformat(ts)
        ttl_days = SECTOR_TTL_UNKNOWN_DAYS if entry.get("sector") == "Unknown" else SECTOR_TTL_DAYS
        return (datetime.utcnow() - dt).days >= ttl_days
    except Exception:
        return True

_sector_cache = _load_sector_cache()

def get_sector_cached(symbol: str, use_cache: bool = True) -> str:
    """Recupere le secteur d'une action avec cache intelligent (mémoire + disque + TTL).
    
    Args:
        symbol: Le ticker de l'action
        use_cache: Si True, utilise le cache; sinon recompute
    
    Returns:
        Le secteur de l'action (ou 'Unknown' si indetermine)
    """
    global _sector_cache
    
    # Check cache memoire
    if use_cache:
        entry = _sector_cache.get(symbol)
        if entry and not _is_sector_expired(entry):
            return entry.get("sector", "Unknown")
    
    # Try SQLite database first
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT sector FROM symbols WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0] and row[0] != "Unknown":
            # Cache hit from DB
            _sector_cache[symbol] = {"sector": row[0], "ts": datetime.utcnow().isoformat()}
            _save_sector_cache(_sector_cache)
            return row[0]
    except Exception:
        pass
    
    # Fallback: fetch from yfinance
    try:
        ticker = yf.Ticker(symbol)
        sector = ticker.info.get('sector', 'Unknown')
        _sector_cache[symbol] = {"sector": sector, "ts": datetime.utcnow().isoformat()}
        _save_sector_cache(_sector_cache)
        return sector
    except Exception:
        # Cache the failure too
        _sector_cache[symbol] = {"sector": "Unknown", "ts": datetime.utcnow().isoformat()}
        _save_sector_cache(_sector_cache)
        return "Unknown"


# ===================================================================
# CLASSIFICATION DE CAPITALISATION
# ===================================================================

def classify_cap_range(market_cap_b: Optional[float]) -> str:
    """Classe la capitalisation en categories (Small/Mid/Large/Mega) ou Unknown.
    
    Args:
        market_cap_b: Market cap en milliards de dollars (ou None)
    
    Returns:
        Une des categories: 'Small', 'Mid', 'Large', 'Mega', 'Unknown'
    """
    try:
        if market_cap_b is None or market_cap_b <= 0:
            return 'Unknown'
        
        for label, (min_val, max_val) in CAP_RANGE_THRESHOLDS.items():
            if min_val <= market_cap_b < max_val:
                return label
        
        return 'Unknown'
    except Exception:
        return 'Unknown'


def classify_cap_range_for_symbol(symbol: str) -> str:
    """Classe la capitalisation d'un symbole en fetching depuis yfinance.
    
    Args:
        symbol: Le ticker de l'action
    
    Returns:
        La categorie de capitalisation ('Small', 'Mid', 'Large', 'Mega', 'Unknown')
    """
    try:
        ticker = yf.Ticker(symbol)
        market_cap = ticker.info.get('marketCap')
        if market_cap is None:
            return 'Unknown'
        
        market_cap_b = market_cap / 1e9
        return classify_cap_range(market_cap_b)
    except Exception:
        return 'Unknown'

def get_cleaned_group_cache(sector: str, cap_range: str, ttl_days: int = 20) -> Optional[List[str]]:
    """Récupère un groupe nettoyé depuis le cache (si <TTL jours).
    
    Args:
        sector: Secteur
        cap_range: Gamme de cap
        ttl_days: TTL en jours (défaut 20)
    
    Returns:
        Liste des symboles si trouvé et valide, None sinon
    """
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT symbols_json, cached_at 
        FROM cleaned_groups_cache 
        WHERE sector = ? AND cap_range = ?
    ''', (sector, cap_range))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    symbols_json, cached_at_str = result
    cached_at = datetime.fromisoformat(cached_at_str)
    
    # Vérifier si le cache a expiré
    if datetime.now() - cached_at > timedelta(days=ttl_days):
        return None
    
    return json.loads(symbols_json)


def save_cleaned_group_cache(sector: str, cap_range: str, symbols: List[str]) -> None:
    """Sauvegarde un groupe nettoyé dans le cache.
    
    Args:
        sector: Secteur
        cap_range: Gamme de cap
        symbols: Liste des symboles nettoyés
    """
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO cleaned_groups_cache 
        (sector, cap_range, symbols_json, cached_at)
        VALUES (?, ?, ?, ?)
    ''', (sector, cap_range, json.dumps(symbols), datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def clear_cleaned_groups_cache() -> None:
    """Efface complètement le cache des groupes nettoyés (force recalcul)."""
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM cleaned_groups_cache')
    conn.commit()
    conn.close()


def display_popular_symbols_distribution():
    """
    Affiche la répartition des symboles 'popular' par secteur et capital range.
    Utile pour analyser la composition du portefeuille de trading.
    """
    init_symbols_table()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Récupérer tous les symboles populaires avec leurs métadonnées
    cursor.execute('''
        SELECT s.sector, s.market_cap_range, COUNT(*) as count
        FROM symbols s
        INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
        WHERE sl.list_type = 'popular' AND s.is_active = 1
        GROUP BY s.sector, s.market_cap_range
        ORDER BY s.sector, s.market_cap_range
    ''')
    
    results = cursor.fetchall()
    
    # Également récupérer les totaux par secteur et par capital range
    cursor.execute('''
        SELECT s.sector, COUNT(*) as count
        FROM symbols s
        INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
        WHERE sl.list_type = 'popular' AND s.is_active = 1
        GROUP BY s.sector
        ORDER BY s.sector
    ''')
    
    sector_totals = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute('''
        SELECT s.market_cap_range, COUNT(*) as count
        FROM symbols s
        INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
        WHERE sl.list_type = 'popular' AND s.is_active = 1
        GROUP BY s.market_cap_range
        ORDER BY s.market_cap_range
    ''')
    
    cap_totals = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute('''
        SELECT COUNT(DISTINCT s.symbol) FROM symbols s
        INNER JOIN symbol_lists sl ON s.symbol = sl.symbol
        WHERE sl.list_type = 'popular' AND s.is_active = 1
    ''')
    
    total_symbols = cursor.fetchone()[0]
    conn.close()
    
    # Affichage
    print("\n" + "="*100)
    print("RÉPARTITION DES SYMBOLES POPULAIRES PAR SECTEUR × CAPITAL RANGE")
    print("="*100)
    
    # Créer un tableau
    sectors = sorted(set(row[0] for row in results))
    cap_ranges = sorted(set(row[1] for row in results))
    
    # En-tête
    header = "Secteur".ljust(25) + "  |  ".join([f"{cap:^12}" for cap in cap_ranges]) + "  |  Total"
    print("\n" + header)
    print("-" * len(header))
    
    # Créer une matrice pour les données
    data_matrix = {}
    for sector, cap_range, count in results:
        if sector not in data_matrix:
            data_matrix[sector] = {}
        data_matrix[sector][cap_range] = count
    
    # Afficher les lignes
    for sector in sectors:
        row_data = sector.ljust(25) + "  |  "
        row_counts = []
        for cap_range in cap_ranges:
            count = data_matrix.get(sector, {}).get(cap_range, 0)
            row_counts.append(count)
            row_data += f"{count:^12}"
        
        row_data += f"  |  {sector_totals.get(sector, 0):>3}"
        print(row_data)
    
    # Ligne des totaux
    print("-" * len(header))
    footer = "TOTAL".ljust(25) + "  |  "
    for cap_range in cap_ranges:
        footer += f"{cap_totals.get(cap_range, 0):^12}"
    footer += f"  |  {total_symbols:>3}"
    print(footer)
    
    print("\n" + "="*100)
    print(f"TOTAL GÉNÉRAL: {total_symbols} symboles populaires actifs")
    print("="*100 + "\n")
    
    # Statistiques supplémentaires
    print("\nSTATISTIQUES PAR SECTEUR:")
    print("-" * 50)
    for sector in sectors:
        count = sector_totals.get(sector, 0)
        pct = (count / total_symbols * 100) if total_symbols > 0 else 0
        print(f"  {sector:30s}: {count:3d} symboles ({pct:5.1f}%)")
    
    print("\nSTATISTIQUES PAR CAPITAL RANGE:")
    print("-" * 50)
    for cap_range in cap_ranges:
        count = cap_totals.get(cap_range, 0)
        pct = (count / total_symbols * 100) if total_symbols > 0 else 0
        print(f"  {cap_range:30s}: {count:3d} symboles ({pct:5.1f}%)")
    
    print("\n")
    
    return {
        'total': total_symbols,
        'by_sector': sector_totals,
        'by_cap_range': cap_totals,
        'matrix': data_matrix
    }


# Exemple d'utilisation
if __name__ == '__main__':
    display_popular_symbols_distribution()
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
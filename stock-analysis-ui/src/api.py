#!/usr/bin/env python3
# ============================================================================
# API.PY - STOCK ANALYSIS REST API
# Flask API pour servir les signaux de trading en ligne
# ============================================================================

import os
import re
import sys
import json
import gc
import logging
from datetime import datetime, timedelta
from hmac import compare_digest
from pathlib import Path
from dotenv import load_dotenv
from functools import wraps, lru_cache
from threading import Lock

logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Get template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')

# Import core modules
try:
    from qsi import (
        analyse_signaux_populaires,
        backtest_signals,
        load_symbols_from_txt,
        get_trading_signal,
        get_cap_range_for_symbol,
        extract_best_parameters
    )
    from yfinance_helper import download_stock_data
    from config import SIGNALS_DIR, DATA_CACHE_DIR
    import yfinance as yf
except ImportError as e:
    # Volontairement un print : on est avant toute configuration du logging et
    # juste avant un sys.exit(1). Un logger sans handler avalerait le message,
    # et le processus mourrait sans rien dire.
    print(f"[API] Import error: {e}", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# FRONTIERE DE SECURITE — CORS.
# L'interface web est servie par cette meme application (`render_template` sur '/'),
# donc le cas nominal est same-origin et ne requiert aucun en-tete CORS.
# CORS_ORIGINS n'est a renseigner que si un front est heberge sur un autre domaine.
# Ne jamais revenir a `CORS(app)` sans origine : cela autorise n'importe quel site
# visite par l'utilisateur a appeler /api/analyze et a consommer le budget yfinance.
_CORS_ORIGINS = [o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()]
if _CORS_ORIGINS:
    CORS(app, origins=_CORS_ORIGINS, methods=['GET', 'POST'])

# Flask >= 2.3 : JSON_SORT_KEYS / JSONIFY_PRETTYPRINT_REGULAR ont ete retires de
# app.config et sont sans effet. La configuration passe desormais par app.json.
app.json.sort_keys = False
app.json.compact = False

# Limite de requêtes simultanées pour éviter surcharge mémoire
analysis_lock = Lock()
MAX_CONCURRENT_ANALYSES = 2
current_analyses = 0

# Cache pour info yfinance (secteur, etc.) - TTL 1 heure
@lru_cache(maxsize=100)
def get_ticker_info_cached(symbol: str):
    """Cache des infos yfinance pour éviter requêtes répétées"""
    try:
        return yf.Ticker(symbol).info
    except Exception as e:
        logger.warning(f"[API] Erreur récupération info {symbol}: {e}")
        return {}

# Security headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# ============================================================================
# DECORATORS
# ============================================================================

_API_KEY = os.getenv('API_KEY')
# Echappatoire explicite pour le developpement local. Doit rester une action
# deliberee : sans elle, une API_KEY absente bloque le service au lieu de
# l'ouvrir silencieusement.
_AUTH_DISABLED = os.getenv('API_AUTH_DISABLED') == '1'

def require_api_key(f):
    """
    --------------------------------------------------------------------------
    Objectif:
        Proteger une route par cle d'API. FRONTIERE DE SECURITE : le defaut est
        fermant. Sans API_KEY dans l'environnement, la route repond 503 au lieu
        de laisser passer — une variable oubliee ne doit jamais exposer l'API.
        Pour le developpement local, poser explicitement API_AUTH_DISABLED=1.

    Entrees:
        f (Callable): la vue Flask a proteger

    Sorties:
        decorated_function (Callable): la vue enveloppee du controle de cle
    --------------------------------------------------------------------------
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if _AUTH_DISABLED:
            return f(*args, **kwargs)
        if not _API_KEY:
            logger.error("[API] API_KEY absente : route refusee (poser API_AUTH_DISABLED=1 en local)")
            return jsonify({'error': 'Service unavailable'}), 503
        provided = request.headers.get('X-API-Key', '')
        if not compare_digest(provided, _API_KEY):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    """Decorator pour gestion d'erreurs uniforme"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            # ValueError provient de nos propres validations : le message est
            # redige pour l'appelant et ne divulgue pas d'interne.
            return jsonify({'error': f'Invalid input: {str(e)}'}), 400
        except Exception:
            # Le detail (trace, chemins, internes) reste dans les logs serveur ;
            # le client ne recoit qu'un message generique.
            logger.exception("[API] Erreur non geree sur %s", request.path)
            return jsonify({'error': 'Server error'}), 500
    return decorated_function

# ============================================================================
# VALIDATION DES ENTREES
# ============================================================================
# FRONTIERE DE VALIDATION. Tout ce qui vient du client passe par ces trois
# fonctions avant d'atteindre yfinance, le store DuckDB ou le disque. Elles
# levent ValueError ; le decorateur handle_errors le traduit en 400 avec un
# message qui dit quoi corriger.

# Un symbole boursier : lettres, chiffres, point (BRK.B), tiret (BRK-B) et
# egal (SI=F pour les futures). Volontairement strict : ce meme texte finit
# interpole dans des requetes DuckDB construites par f-string.
_MOTIF_SYMBOLE = re.compile(r'^[A-Z0-9][A-Z0-9.\-=]{0,14}$')

# La forme d'une periode, pas son vocabulaire. Le code utilise des valeurs
# non standard pour yfinance (12mo, 15mo, 4y) : figer une liste blanche
# casserait des appels existants. On rejette la forme invalide, pas la valeur.
_MOTIF_PERIODE = re.compile(r'^(?:\d{1,3}(?:d|mo|y)|ytd|max)$')

MAX_SYMBOLES_PAR_LOT = 50


def valider_symbole(brut) -> str:
    """
    --------------------------------------------------------------------------
    Objectif:
        Normaliser et valider un symbole boursier recu du client.

    Inputs:
        brut (Any): valeur fournie par l'appelant

    Outputs:
        symbole (str): symbole en majuscules, garanti conforme au motif
    --------------------------------------------------------------------------
    """
    symbole = str(brut or '').strip().upper()
    if not symbole:
        raise ValueError("le champ 'symbol' est requis")
    if not _MOTIF_SYMBOLE.match(symbole):
        raise ValueError(
            f"symbole invalide: {symbole!r}. Attendu 1 a 15 caracteres parmi "
            "A-Z, 0-9, point, tiret ou egal (exemples: AAPL, BRK.B, ENR.DE, SI=F)"
        )
    return symbole


def valider_periode(brut, defaut: str = '1mo') -> str:
    """
    --------------------------------------------------------------------------
    Objectif:
        Valider la forme d'une periode d'historique avant de la transmettre a
        la couche de telechargement.

    Inputs:
        brut (Any): valeur fournie par l'appelant
        defaut (str): valeur employee si le champ est absent

    Outputs:
        periode (str): periode conforme au motif
    --------------------------------------------------------------------------
    """
    periode = str(brut or defaut).strip().lower()
    if not _MOTIF_PERIODE.match(periode):
        raise ValueError(
            f"periode invalide: {periode!r}. Attendu <nombre>d, <nombre>mo, "
            "<nombre>y, 'ytd' ou 'max' (exemples: 5d, 3mo, 12mo, 2y)"
        )
    return periode


def valider_entier(brut, nom: str, defaut: int, mini: int, maxi: int) -> int:
    """
    --------------------------------------------------------------------------
    Objectif:
        Valider un entier borne recu en parametre de requete.

    Inputs:
        brut (Any): valeur fournie par l'appelant, ou None
        nom (str): nom du champ, pour le message d'erreur
        defaut (int): valeur si le champ est absent
        mini (int), maxi (int): bornes incluses

    Outputs:
        valeur (int): entier garanti dans [mini, maxi]
    --------------------------------------------------------------------------
    """
    if brut is None:
        return defaut
    try:
        valeur = int(brut)
    except (TypeError, ValueError):
        raise ValueError(f"{nom} doit etre un entier, recu {brut!r}")
    if not (mini <= valeur <= maxi):
        raise ValueError(f"{nom} doit etre compris entre {mini} et {maxi}, recu {valeur}")
    return valeur


def valider_liste_symboles(brut, maximum: int = MAX_SYMBOLES_PAR_LOT) -> list:
    """
    --------------------------------------------------------------------------
    Objectif:
        Valider une liste de symboles et en borner la taille, pour qu'un seul
        appel ne puisse pas consommer tout le budget de requetes yfinance.

    Inputs:
        brut (Any): liste fournie par l'appelant
        maximum (int): nombre maximal de symboles acceptes

    Outputs:
        symboles (list[str]): symboles valides, dedupliques, ordre conserve
    --------------------------------------------------------------------------
    """
    if not isinstance(brut, (list, tuple)):
        raise ValueError("'symbols' doit etre une liste")
    if not brut:
        raise ValueError("'symbols' ne peut pas etre vide")
    if len(brut) > maximum:
        raise ValueError(f"'symbols' est limite a {maximum} entrees, recu {len(brut)}")

    symboles, vus = [], set()
    for element in brut:
        symbole = valider_symbole(element)
        if symbole not in vus:
            vus.add(symbole)
            symboles.append(symbole)
    return symboles


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil avec interface web"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint pour monitoring avec info mémoire"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'memory': {
                'rss_mb': round(mem_info.rss / 1024 / 1024, 2),
                'vms_mb': round(mem_info.vms / 1024 / 1024, 2)
            },
            'concurrent_analyses': current_analyses,
            'max_concurrent': MAX_CONCURRENT_ANALYSES
        }), 200
    except ImportError:
        # psutil non installé, retourner version simple
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'environment': os.getenv('FLASK_ENV', 'development')
        }), 200

@app.route('/status', methods=['GET'])
def status():
    """Status détaillé du système"""
    try:
        # Vérifier les répertoires
        signals_exist = (SIGNALS_DIR / "signaux_trading.csv").exists()
        cache_exists = DATA_CACHE_DIR.exists()
        
        return jsonify({
            'api_status': 'running',
            'timestamp': datetime.utcnow().isoformat(),
            'directories': {
                'signals': str(SIGNALS_DIR),
                'cache': str(DATA_CACHE_DIR),
                'signals_file_exists': signals_exist,
                'cache_dir_exists': cache_exists
            },
            'environment': {
                'FLASK_ENV': os.getenv('FLASK_ENV'),
                'DEBUG': os.getenv('DEBUG', 'False') == 'True'
            }
        }), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/signals', methods=['GET'])
@handle_errors
def get_signals():
    """
    Récupère les signaux de trading récents
    
    Query params:
    - limit: nombre de signaux max (default: 50)
    - symbol: filtrer par symbole (optionnel)
    - min_reliability: score minimum (default: 30)
    """
    # Validation avant le try : un ValueError doit atteindre handle_errors
    # (400) et non l'except interne, qui repondrait 500.
    limit = valider_entier(request.args.get('limit'), 'limit', 50, 1, 500)
    symbol_brut = request.args.get('symbol', None, type=str)
    symbol = valider_symbole(symbol_brut) if symbol_brut else None
    min_reliability = valider_entier(
        request.args.get('min_reliability'), 'min_reliability', 30, 0, 100
    )
    try:
        # Charger les signaux depuis CSV
        signals_file = SIGNALS_DIR / "signaux_trading.csv"
        if not signals_file.exists():
            return jsonify({'signals': [], 'message': 'No signals yet'}), 200
        
        df = pd.read_csv(signals_file)
        
        # Filtres
        if symbol:
            df = df[df['Symbol'].str.upper() == symbol.upper()]
        
        if 'Reliability' in df.columns:
            df = df[df['Reliability'] >= min_reliability]
        
        # Trier par date récente
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date', ascending=False)
        
        # Limiter
        df = df.head(limit)
        
        # Convertir en JSON
        signals = df.to_dict('records')
        
        return jsonify({
            'signals': signals,
            'count': len(signals),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/signals/<symbol>', methods=['GET'])
@handle_errors
def get_symbol_signals(symbol):
    """
    Récupère les signaux pour un symbole spécifique
    
    Params:
    - symbol: Code du ticker (ex: AAPL)
    """
    try:
        symbol = symbol.upper()
        
        # Charger signaux
        signals_file = SIGNALS_DIR / "signaux_trading.csv"
        if not signals_file.exists():
            return jsonify({'signals': [], 'message': f'No signals for {symbol}'}), 200
        
        df = pd.read_csv(signals_file)
        df = df[df['Symbol'].str.upper() == symbol]
        
        if df.empty:
            return jsonify({'signals': [], 'message': f'No signals for {symbol}'}), 200
        
        # Trier par date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date', ascending=False)
        
        signals = df.to_dict('records')
        
        return jsonify({
            'symbol': symbol,
            'signals': signals,
            'count': len(signals),
            'latest': signals[0] if signals else None,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/analyze', methods=['POST'])
@require_api_key
@handle_errors
def analyze_symbol():
    """
    Analyse un symbole - VERSION SIMPLIFIÉE comme le desktop UI
    Ne filtre PAS par fiabilité, retourne TOUS les signaux
    """
    global current_analyses
    
    # Initialiser variables pour nettoyage
    stock_data_dict = None
    prices = None
    volumes = None
    
    data = request.get_json()
    symbol = valider_symbole(data.get('symbol'))
    period = valider_periode(data.get('period'), '1mo')
    try:
        # Optional flags to control fallbacks
        use_domain_fallback = bool(data.get('use_domain_fallback', True))
        use_cap_fallback = bool(data.get('use_cap_fallback', True))
        
        if not symbol:
            return jsonify({'error': 'symbol required'}), 400
        
        # Limiter les analyses simultanées pour éviter surcharge RAM
        with analysis_lock:
            if current_analyses >= MAX_CONCURRENT_ANALYSES:
                return jsonify({
                    'error': 'Too many concurrent analyses. Please retry in a moment.',
                    'symbol': symbol,
                    'status': 'rate_limited'
                }), 429
            current_analyses += 1
        
        try:
            logger.info(f"[API] Analyse simple de {symbol} (période: {period})...")
            
            # Télécharger les données (comme le desktop UI)
            stock_data_dict = download_stock_data([symbol], period)
            
            if not stock_data_dict or symbol not in stock_data_dict:
                return jsonify({
                    'error': f'Impossible de télécharger les données pour {symbol}',
                    'symbol': symbol,
                    'status': 'error'
                }), 404
            
            stock_data = stock_data_dict[symbol]
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            
            # Récupérer le secteur (comme le desktop UI) - AVEC CACHE + NORMALISATION
            try:
                info = get_ticker_info_cached(symbol)
                domaine = info.get("sector", "Inconnu")
                
                # ✅ NEW: Normaliser le secteur
                from sector_normalizer import normalize_sector
                domaine_raw = domaine
                domaine = normalize_sector(domaine)
                if domaine_raw != domaine:
                    logger.info(f"[API] {symbol}: Secteur normalisé: '{domaine_raw}' -> '{domaine}'")
            except Exception as e:
                domaine = "Inconnu"
                logger.warning(f"[API] Erreur normalisation secteur {symbol}: {e}")
            
            # Cap range
            cap_range = get_cap_range_for_symbol(symbol)
            if use_cap_fallback and (cap_range == "Unknown" or not cap_range):
                best_params_all = extract_best_parameters()
                # ✅ AMÉLIORÉ: Chercher dans la DB d'abord
                try:
                    import sqlite3
                    db_path = 'symbols.db'
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT DISTINCT cap_range FROM symbols 
                            WHERE sector = ? AND cap_range IS NOT NULL AND cap_range != 'Unknown'
                            LIMIT 10
                        """, (domaine,))
                        db_caps = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        
                        cap_priority = ['Small', 'Mid', 'Large', 'Mega']
                        for cap in cap_priority:
                            if cap in db_caps:
                                test_key = f"{domaine}_{cap}"
                                if test_key in best_params_all:
                                    cap_range = cap
                                    logger.info(f"[API] {symbol}: Cap_range trouvé en DB: {cap}")
                                    break
                except Exception as e:
                    logger.warning(f"[API] {symbol}: Erreur recherche DB cap_range: {e}")
                
                # Fallback standard
                if cap_range == "Unknown" or not cap_range:
                    for fallback_cap in ["Large", "Mid", "Small", "Mega"]:
                        test_key = f"{domaine}_{fallback_cap}"
                        if test_key in best_params_all:
                            cap_range = fallback_cap
                            break
            
            # Fallback domaine
            original_domaine = domaine
            if use_domain_fallback and domaine == "Inconnu":
                best_params_all = extract_best_parameters()
                for fallback_sector in ["Technology", "Healthcare", "Financial Services"]:
                    if fallback_sector in best_params_all:
                        domaine = fallback_sector
                        break
                if domaine == "Inconnu" and best_params_all:
                    first_key = list(best_params_all.keys())[0]
                    domaine = first_key.split('_')[0] if '_' in first_key else first_key
            
            # Extraire seuils optimisés
            seuil_achat_opt = None
            seuil_vente_opt = None
            best_params_all = extract_best_parameters()
            param_key = None
            if cap_range and cap_range != "Unknown":
                test_key = f"{domaine}_{cap_range}"
                if test_key in best_params_all:
                    param_key = test_key
            if not param_key and domaine in best_params_all:
                param_key = domaine
            
            if param_key and param_key in best_params_all:
                params = best_params_all[param_key]
                if len(params) > 2 and params[2]:
                    globals_th = params[2]
                    if isinstance(globals_th, (tuple, list)) and len(globals_th) >= 2:
                        seuil_achat_opt = float(globals_th[0])
                        seuil_vente_opt = float(globals_th[1])
            
            # Appeler get_trading_signal DIRECTEMENT (comme le desktop UI)
            sig, last_price, trend, last_rsi, volume_mean, score, derivatives = get_trading_signal(
                prices, volumes, domaine=domaine, return_derivatives=True, symbol=symbol, 
                cap_range=cap_range, seuil_achat=seuil_achat_opt, seuil_vente=seuil_vente_opt
            )
            
            # Construire la réponse
            signal_data = {
                'symbol': symbol,
                'signal': sig,
                'prix': float(last_price) if last_price is not None else 0.0,
                'rsi': float(last_rsi) if last_rsi is not None else 0.0,
                'tendance': 'Haussière' if trend else 'Baissière',
                'volume_moyen': float(volume_mean) if volume_mean is not None else 0.0,
                'domaine': original_domaine,
                'domaine_used': domaine,
                'cap_range': cap_range,
                'fiabilite': float(score) if score is not None else 0.0,
                'score': float(score) if score is not None else 0.0
            }
            
            # Ajouter les dérivées si disponibles
            if derivatives:
                signal_data.update({
                    'prix_sma20': derivatives.get('prix_sma20'),
                    'prix_sma50': derivatives.get('prix_sma50'),
                    'prix_sma200': derivatives.get('prix_sma200'),
                    'dPrice': derivatives.get('dPrice'),
                    'dMACD': derivatives.get('dMACD'),
                    'dRSI': derivatives.get('dRSI'),
                    'dVolRel': derivatives.get('dVolRel')
                })
            
            response = {
                'symbol': symbol,
                'period': period,
                'signals': [signal_data],
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"[API] Analyse terminée: {sig} (score: {score})")
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"[API] Erreur analyse: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Analysis failed: {str(e)}',
                'symbol': symbol,
                'status': 'error'
            }), 500
        finally:
            # Nettoyage mémoire après chaque analyse
            with analysis_lock:
                current_analyses -= 1
            # Libérer mémoire des DataFrames (seulement si créés)
            try:
                if stock_data_dict is not None:
                    del stock_data_dict
                if prices is not None:
                    del prices
                if volumes is not None:
                    del volumes
            except:
                pass
            gc.collect()
        
    except Exception as e:
        with analysis_lock:
            current_analyses -= 1
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/backtest', methods=['POST'])
@require_api_key
@handle_errors
def run_backtest():
    """
    Lance un backtest pour un symbole
    
    Body JSON:
    {
        "symbol": "AAPL",
        "period": "12mo",
        "fast_ma": 12,
        "slow_ma": 26
    }
    """
    data = request.get_json()
    symbol = valider_symbole(data.get('symbol'))
    period = valider_periode(data.get('period'), '12mo')
    try:
        
        if not symbol:
            return jsonify({'error': 'symbol required'}), 400
        
        logger.info(f"[API] 🔬 Backtesting {symbol}...")
        
        # Télécharger données
        df = download_stock_data([symbol], period=period)
        if df is None or df.empty:
            return jsonify({'error': f'Could not download data for {symbol}'}), 400
        
        # Extraire paramètres optionnels
        params = {k: v for k, v in data.items() 
                  if k in ['fast_ma', 'slow_ma', 'signal_ma', 'rsi_period']}
        
        # Lancer backtest
        results = backtest_signals(df, symbol, **params)
        
        return jsonify({
            'symbol': symbol,
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"[API] Error in /backtest: {e}")
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/analyze-batch', methods=['POST'])
@require_api_key
@handle_errors
def analyze_batch():
    """Analyse plusieurs symboles (max 20 à la fois)"""
    data = request.get_json()
    symbols = valider_liste_symboles(data.get('symbols'))
    period = valider_periode(data.get('period'), '12mo')
    try:
        
        if not symbols:
            return jsonify({'error': 'symbols list required'}), 400
        
        if len(symbols) > 20:
            return jsonify({'error': 'Max 20 symbols per request'}), 400
        
        symbols = [s.upper() for s in symbols]
        
        logger.info(f"[API] Analysing {len(symbols)} symbols...")
        results = analyse_signaux_populaires(
            popular_symbols=symbols,
            mes_symbols=[],
            period=period,
            afficher_graphiques=False,
            verbose=False,
            save_csv=False
        )
        
        return jsonify({
            'symbols': symbols,
            'count': len(symbols),
            'signals': results.get('signaux_fiables', []),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"[API] Error in /analyze-batch: {e}")
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/analyze-popular', methods=['POST'])
@require_api_key
@handle_errors
def analyze_popular_signals():
    """
    Analyser les mouvements fiables (signaux populaires)
    Équivalent à 'Analyser mouvements fiables (populaires)' du UI desktop
    
    Body JSON:
    {
        "popular_symbols": ["AAPL", "MSFT"],
        "mes_symbols": ["0005.HK"],
        "period": "12mo"
    }
    """
    # Validation avant le try : un ValueError doit atteindre handle_errors
    # (400) et non l'except interne, qui repondrait 500.
    data = request.get_json()
    popular_brut = data.get('popular_symbols', [])
    mes_brut = data.get('mes_symbols', [])
    period = valider_periode(data.get('period'), '12mo')

    if not popular_brut and not mes_brut:
        return jsonify({'error': 'At least one symbol list required'}), 400

    # Les deux listes sont analysees dans le meme appel : la borne porte
    # sur leur total, pas sur chacune, sinon on double le budget yfinance.
    popular_symbols = valider_liste_symboles(popular_brut) if popular_brut else []
    mes_symbols = valider_liste_symboles(mes_brut) if mes_brut else []
    if len(popular_symbols) + len(mes_symbols) > MAX_SYMBOLES_PAR_LOT:
        raise ValueError(
            f"total des symboles limite a {MAX_SYMBOLES_PAR_LOT}, recu "
            f"{len(popular_symbols) + len(mes_symbols)}"
        )

    try:
        logger.info(f"[API] Analyzing popular signals... ({len(popular_symbols)} popular, {len(mes_symbols)} personal)")

        # Utiliser la même fonction que le UI
        results = analyse_signaux_populaires(
            popular_symbols=popular_symbols,
            mes_symbols=mes_symbols,
            period=period,
            afficher_graphiques=False,
            verbose=True,
            save_csv=False,
            plot_all=False
        )
        
        return jsonify({
            'popular_symbols': popular_symbols,
            'mes_symbols': mes_symbols,
            'signals': results.get('signaux_fiables', []),
            'backtest_results': results.get('backtest_results', []),
            'count': len(results.get('signaux_fiables', [])),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"[API] Error in /analyze-popular: {e}")
        import traceback
        traceback.print_exc()
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/lists', methods=['GET'])
@handle_errors
def get_lists():
    """
    Récupère les listes de symboles actuelles (populaires, personnels, optimisation)
    """
    try:
        from config import PROJECT_ROOT
        
        lists_data = {
            'popular': [],
            'personal': [],
            'optimization': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Charger depuis les fichiers txt
        txt_files = {
            'popular': PROJECT_ROOT / 'popular_symbols.txt',
            'personal': PROJECT_ROOT / 'mes_symbols.txt',
            'optimization': PROJECT_ROOT / 'optimisation_symbols.txt'
        }
        
        for list_type, filepath in txt_files.items():
            try:
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        symbols = [s.strip().upper() for s in f.readlines() if s.strip()]
                        lists_data[list_type] = sorted(symbols)
            except Exception as e:
                logger.warning(f"[API] Error loading {list_type}: {e}")
        
        return jsonify(lists_data), 200
        
    except Exception as e:
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/lists/<list_type>', methods=['POST'])
@require_api_key
@handle_errors
def update_list(list_type):
    """
    Ajoute/Supprime des symboles d'une liste
    
    Body JSON:
    {
        "action": "add" ou "remove",
        "symbols": ["AAPL", "MSFT"]
    }
    """
    # Validation avant le try : un ValueError doit atteindre handle_errors
    # (400) et non l'except interne, qui repondrait 500.
    if list_type not in ['popular', 'personal', 'optimization']:
        return jsonify({'error': 'Invalid list type'}), 400

    data = request.get_json()
    action = str(data.get('action', '')).strip().lower()
    symbols = valider_liste_symboles(data.get('symbols'))

    try:
        from config import PROJECT_ROOT

        if action not in ['add', 'remove']:
            return jsonify({'error': 'Invalid action (add or remove)'}), 400
        
        # Mapper list_type à filename
        list_map = {
            'popular': 'popular_symbols.txt',
            'personal': 'mes_symbols.txt',
            'optimization': 'optimisation_symbols.txt'
        }
        
        filepath = PROJECT_ROOT / list_map[list_type]
        
        # Lire la liste actuelle
        current_symbols = set()
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                current_symbols = {s.strip().upper() for s in f.readlines() if s.strip()}
        
        # Ajouter ou supprimer
        symbols_upper = {s.upper() for s in symbols}
        if action == 'add':
            current_symbols.update(symbols_upper)
            message = f"Added {len(symbols)} symbols"
        else:  # remove
            current_symbols -= symbols_upper
            message = f"Removed {len(symbols)} symbols"
        
        # Écrire la nouvelle liste (triée)
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in sorted(current_symbols):
                f.write(f"{s}\n")
        
        return jsonify({
            'list': list_type,
            'action': action,
            'message': message,
            'count': len(current_symbols),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"[API] Error in /lists/{list_type}: {e}")
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

# ============================================================================
# STATS & REPORT ENDPOINTS
# ============================================================================

@app.route('/api/stats', methods=['GET'])
@handle_errors
def get_stats():
    """Statistiques globales du système"""
    try:
        stats = {
            'total_signals': 0,
            'signals_today': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Charger les signaux
        signals_file = SIGNALS_DIR / "signaux_trading.csv"
        if signals_file.exists():
            df = pd.read_csv(signals_file)
            stats['total_signals'] = len(df)
            
            # Signaux aujourd'hui
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                today = datetime.utcnow().date()
                stats['signals_today'] = len(df[df['Date'].dt.date == today])
            
            # Métriques si présentes
            if 'Return' in df.columns:
                returns = df['Return'].dropna()
                if len(returns) > 0:
                    stats['avg_return'] = float(returns.mean())
                    stats['win_rate'] = float((returns > 0).sum() / len(returns))
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.exception("[API] Erreur sur %s", request.path)
        return jsonify({'error': 'Server error'}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# API DOCUMENTATION ENDPOINT
# ============================================================================

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """Documentation de l'API"""
    docs = {
        'title': 'Stock Analysis API',
        'version': '1.0.0',
        'description': 'API REST pour l\'analyse technique et les signaux de trading',
        'base_url': request.host_url,
        'endpoints': {
            'Health & Status': {
                '/health': 'GET - Vérifier la santé de l\'API',
                '/status': 'GET - Status détaillé du système'
            },
            'Signals': {
                '/api/signals': 'GET - Récupérer les signaux (params: limit, symbol, min_reliability)',
                '/api/signals/<symbol>': 'GET - Signaux pour un symbole spécifique'
            },
            'Analysis': {
                '/api/analyze': 'POST - Analyser un symbole (body: {symbol, period})',
                '/api/analyze-batch': 'POST - Analyser plusieurs symboles',
                '/api/backtest': 'POST - Lancer un backtest'
            },
            'Stats': {
                '/api/stats': 'GET - Statistiques globales'
            },
            'Documentation': {
                '/api/docs': 'GET - Cette documentation'
            }
        }
    }
    return jsonify(docs), 200

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Configuration
    debug = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('BIND_PORT', 5000))
    # Defaut sur la boucle locale : ce bloc ne sert qu'au developpement.
    # L'exposition sur toutes les interfaces doit etre demandee explicitement
    # (render.yaml pose BIND_ADDRESS=0.0.0.0, requis par la plateforme).
    host = os.getenv('BIND_ADDRESS', '127.0.0.1')
    
    print(f"""
    ╔═══════════════════════════════════════╗
    ║  STOCK ANALYSIS API                   ║
    ║  v1.0.0                                ║
    ╚═══════════════════════════════════════╝
    
    🚀 Starting API server...
    📍 Host: {host}:{port}
    🔧 Debug: {debug}
    📚 Docs: http://localhost:{port}/api/docs
    
    """)
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug
    )

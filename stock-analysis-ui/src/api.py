#!/usr/bin/env python3
# ============================================================================
# API.PY - STOCK ANALYSIS REST API
# Flask API pour servir les signaux de trading en ligne
# ============================================================================

import os
import sys
import json
import gc
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from functools import wraps, lru_cache
from threading import Lock

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
    print(f"⚠️ Import error: {e}")
    sys.exit(1)

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)  # Enable CORS for all routes

app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

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
        print(f"⚠️ Erreur récupération info {symbol}: {e}")
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

def require_api_key(f):
    """Decorator to check API key (optionnel)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        # En production, vérifier contre une BD ou variable d'env
        # Pour maintenant, optionnel
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    """Decorator pour gestion d'erreurs uniforme"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'error': f'Invalid input: {str(e)}'}), 400
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    return decorated_function

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
    try:
        limit = request.args.get('limit', 50, type=int)
        symbol = request.args.get('symbol', None, type=str)
        min_reliability = request.args.get('min_reliability', 30, type=int)
        
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
        return jsonify({'error': str(e)}), 500

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
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
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
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '1mo')
        
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
            print(f"📊 Analyse simple de {symbol} (période: {period})...")
            
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
            
            # Récupérer le secteur (comme le desktop UI) - AVEC CACHE
            try:
                info = get_ticker_info_cached(symbol)
                domaine = info.get("sector", "Inconnu")
            except Exception:
                domaine = "Inconnu"
            
            # Cap range
            cap_range = get_cap_range_for_symbol(symbol)
            if cap_range == "Unknown" or not cap_range:
                best_params_all = extract_best_parameters()
                for fallback_cap in ["Large", "Mid", "Mega"]:
                    test_key = f"{domaine}_{fallback_cap}"
                    if test_key in best_params_all:
                        cap_range = fallback_cap
                        break
            
            # Fallback domaine
            original_domaine = domaine
            if domaine == "Inconnu":
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
            
            print(f"✅ Analyse terminée: {sig} (score: {score})")
            return jsonify(response), 200
            
        except Exception as e:
            print(f"❌ Erreur analyse: {e}")
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
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
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
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '12mo')
        
        if not symbol:
            return jsonify({'error': 'symbol required'}), 400
        
        print(f"🔬 Backtesting {symbol}...")
        
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
        print(f"❌ Error in /backtest: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-batch', methods=['POST'])
@handle_errors
def analyze_batch():
    """Analyse plusieurs symboles (max 20 à la fois)"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        period = data.get('period', '12mo')
        
        if not symbols:
            return jsonify({'error': 'symbols list required'}), 400
        
        if len(symbols) > 20:
            return jsonify({'error': 'Max 20 symbols per request'}), 400
        
        symbols = [s.upper() for s in symbols]
        
        print(f"📊 Analysing {len(symbols)} symbols...")
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
        print(f"❌ Error in /analyze-batch: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-popular', methods=['POST'])
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
    try:
        data = request.get_json()
        popular_symbols = data.get('popular_symbols', [])
        mes_symbols = data.get('mes_symbols', [])
        period = data.get('period', '12mo')
        
        if not popular_symbols and not mes_symbols:
            return jsonify({'error': 'At least one symbol list required'}), 400
        
        print(f"📊 Analyzing popular signals... ({len(popular_symbols)} popular, {len(mes_symbols)} personal)")
        
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
        print(f"❌ Error in /analyze-popular: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
                print(f"⚠️ Error loading {list_type}: {e}")
        
        return jsonify(lists_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lists/<list_type>', methods=['POST'])
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
    try:
        from config import PROJECT_ROOT
        
        if list_type not in ['popular', 'personal', 'optimization']:
            return jsonify({'error': 'Invalid list type'}), 400
        
        data = request.get_json()
        action = data.get('action', '').lower()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'symbols required'}), 400
        
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
        print(f"❌ Error in /lists/{list_type}: {e}")
        return jsonify({'error': str(e)}), 500

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
        return jsonify({'error': str(e)}), 500

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
    host = os.getenv('BIND_ADDRESS', '0.0.0.0')
    
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

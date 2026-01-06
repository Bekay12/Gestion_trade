#!/usr/bin/env python3
# ============================================================================
# API.PY - STOCK ANALYSIS REST API
# Flask API pour servir les signaux de trading en ligne
# ============================================================================

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from functools import wraps

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
        download_stock_data,
        backtest_signals
    )
    from config import SIGNALS_DIR, DATA_CACHE_DIR
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
    """Health check endpoint pour monitoring"""
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
    Analyse un symbole et génère un signal de trading
    Utilise EXACTEMENT le même backend que le UI desktop (main_window.py)
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '12mo')
        include_backtest = data.get('include_backtest', False)
        
        if not symbol:
            return jsonify({'error': 'symbol required'}), 400
        
        print(f"📊 Analysing {symbol}...")
        
        try:
            # Utiliser analyse_signaux_populaires EXACTEMENT comme le UI desktop
            results = analyse_signaux_populaires(
                popular_symbols=[symbol],
                mes_symbols=[],
                period=period,
                afficher_graphiques=False,
                verbose=True,  # Verbose pour debug
                save_csv=False,
                plot_all=False
            )
        except Exception as e:
            print(f"❌ analyse_signaux_populaires failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Analysis failed: {str(e)}',
                'symbol': symbol,
                'status': 'error'
            }), 500
        
        signals_list = results.get('signaux_fiables', [])
        backtest_list = results.get('backtest_results', [])
        
        # Structure de réponse unifiée
        response = {
            'symbol': symbol,
            'period': period,
            'signals': [],
            'backtest_results': [],
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'success' if signals_list else 'no_signals'
        }
        
        # Traiter chaque signal (normalement 1 seul pour un symbole unique)
        for sig in signals_list:
            signal_data = {
                'symbol': symbol,
                'signal': sig.get('signal', 'HOLD'),
                'score': sig.get('score', 0.0),
                'prix': sig.get('prix'),
                'rsi': sig.get('rsi'),
                'tendance': sig.get('tendance'),
                'volume_moyen': sig.get('volume_moyen'),
                'domaine': sig.get('domaine'),
                'cap_range': sig.get('cap_range'),
                'fiabilite': sig.get('fiabilite'),
                'prix_sma20': sig.get('prix_sma20'),
                'prix_sma50': sig.get('prix_sma50'),
                'prix_sma200': sig.get('prix_sma200'),
                'variations': {
                    'last_30d': sig.get('variation_30j'),
                    'last_180d': sig.get('variation_180j')
                },
                'derivatives': {
                    'price_slope': sig.get('dPrice'),
                    'macd_slope': sig.get('dMACD'),
                    'rsi_slope': sig.get('dRSI'),
                    'volume_slope': sig.get('dVolRel')
                },
                'fundamentals': {
                    'rev_growth': sig.get('Rev. Growth (%)'),
                    'ebitda_yield': sig.get('EBITDA Yield (%)'),
                    'fcf_yield': sig.get('FCF Yield (%)'),
                    'd_e_ratio': sig.get('D/E Ratio'),
                    'market_cap': sig.get('Market Cap (B$)')
                }
            }
            response['signals'].append(signal_data)
        
        # Ajouter les résultats de backtest si disponibles
        if include_backtest and backtest_list:
            for bt in backtest_list:
                backtest_data = {
                    'symbol': bt.get('Symbole'),
                    'gain_total': bt.get('gain_total'),
                    'gain_moyen': bt.get('gain_moyen'),
                    'taux_reussite': bt.get('taux_reussite'),
                    'trades': bt.get('trades'),
                    'gagnants': bt.get('gagnants'),
                    'perdants': bt.get('perdants'),
                    'params': {
                        'fast_ma': bt.get('fast_ma'),
                        'slow_ma': bt.get('slow_ma'),
                        'signal_period': bt.get('signal_period')
                    }
                }
                response['backtest_results'].append(backtest_data)
        
        # Si pas de signaux, retourner quand même la structure avec un message
        if not signals_list:
            response['message'] = f'No reliable signals found for {symbol} in period {period}'
            return jsonify(response), 200
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error in /analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'symbol': data.get('symbol') if 'data' in locals() else 'unknown',
            'status': 'error'
        }), 500

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

# ============================================================================
# BATCH ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/analyze-batch', methods=['POST'])
@handle_errors
def analyze_batch():
    """
    Analyse plusieurs symboles (max 20 à la fois)
    
    Body JSON:
    {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "period": "12mo"
    }
    """
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
        
        # Analyser tous les symboles
        results = analyse_signaux_populaires(
            symbols=symbols,
            mes_symbols=[],
            period=period,
            afficher_graphiques=False,
            verbose=False
        )
        
        return jsonify({
            'symbols': symbols,
            'count': len(symbols),
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error in /analyze-batch: {e}")
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

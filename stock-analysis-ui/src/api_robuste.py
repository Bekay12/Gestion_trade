"""
API Flask robuste: orchestration seulement, pas de calcul long
Les tâches longues (download, analyse) sont enqueue dans Redis/Celery
"""
import os
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import redis

from config import SIGNALS_DIR, DATA_CACHE_DIR
from src.tasks import analyze_symbol_task, download_stock_data_task

# Redis connection
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
try:
    redis_client = redis.from_url(redis_url)
    redis_client.ping()
    print("[API] Redis connecté ✓")
except Exception as e:
    print(f"[API] ⚠️  Redis non disponible: {e}")
    redis_client = None

# Flask app
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


def handle_errors(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'error': f'Invalid input: {str(e)}', 'status': 'error'}), 400
        except Exception as e:
            print(f"[API] ❌ {type(e).__name__}: {e}")
            return jsonify({'error': str(e), 'status': 'error'}), 500
    return decorated


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check avec état des dépendances"""
    try:
        redis_status = "ok" if redis_client and redis_client.ping() else "unavailable"
    except:
        redis_status = "unavailable"
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0-robust',
        'dependencies': {
            'redis': redis_status,
            'celery': 'configured'
        }
    }), 200


@app.route('/')
def index():
    """Web UI"""
    return render_template('index.html')


# ============================================================================
# API ROBUSTE - Enqueue tâches dans Redis
# ============================================================================

@app.route('/api/analyze', methods=['POST'])
@handle_errors
def analyze_symbol():
    """
    Analyse asynchrone d'un symbole
    Retourne immédiatement task_id, résultat disponible après quelques secondes
    
    POST /api/analyze
    {
        "symbol": "AAPL",
        "period": "1mo"
    }
    """
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    period = data.get('period', '1mo')
    
    if not symbol:
        return jsonify({'error': 'symbol required'}), 400
    
    if not redis_client:
        return jsonify({
            'error': 'Task queue unavailable (Redis offline)',
            'status': 'error'
        }), 503
    
    try:
        # Enqueue la tâche
        task = analyze_symbol_task.apply_async(args=[symbol, period], queue='analysis')
        
        return jsonify({
            'status': 'queued',
            'symbol': symbol,
            'period': period,
            'task_id': task.id,
            'timestamp': datetime.utcnow().isoformat(),
            'poll_url': f'/api/task/{task.id}'
        }), 202  # 202 Accepted
        
    except Exception as e:
        print(f"[API] Erreur enqueue: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/task/<task_id>', methods=['GET'])
@handle_errors
def get_task_status(task_id):
    """
    Récupère le statut d'une tâche
    GET /api/task/<task_id>
    """
    from src.tasks import app as celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {'status': 'pending', 'task_id': task_id}
    elif task.state == 'SUCCESS':
        response = {'status': 'success', 'task_id': task_id, 'result': task.result}
    elif task.state == 'FAILURE':
        response = {'status': 'failed', 'task_id': task_id, 'error': str(task.info)}
    elif task.state == 'RETRY':
        response = {'status': 'retrying', 'task_id': task_id, 'message': 'Task retrying...'}
    else:
        response = {'status': task.state.lower(), 'task_id': task_id}
    
    response['timestamp'] = datetime.utcnow().isoformat()
    return jsonify(response), 200


@app.route('/api/download', methods=['POST'])
@handle_errors
def download_symbol():
    """
    Télécharge uniquement les données (sans analyse)
    """
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    period = data.get('period', '1mo')
    
    if not symbol:
        return jsonify({'error': 'symbol required'}), 400
    
    if not redis_client:
        return jsonify({'error': 'Task queue unavailable'}, 503)
    
    try:
        task = download_stock_data_task.apply_async(args=[symbol, period], queue='downloads')
        
        return jsonify({
            'status': 'queued',
            'symbol': symbol,
            'task_id': task.id,
            'poll_url': f'/api/task/{task.id}'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API LEGACY (fast path - si result en cache Redis)
# ============================================================================

@app.route('/api/analyze-cached', methods=['POST'])
@handle_errors
def analyze_cached():
    """
    Retourne le résultat en cache si disponible (très rapide)
    Sinon enqueue et retourne task_id
    """
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({'error': 'symbol required'}), 400
    
    # Vérifier le cache
    cache_key = f"analysis:{symbol}:1mo"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                import json
                result = json.loads(cached)
                return jsonify({
                    'status': 'cached',
                    'symbol': symbol,
                    'result': result,
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
        except Exception as e:
            print(f"[API] Cache miss: {e}")
    
    # Sinon, enqueue
    try:
        task = analyze_symbol_task.apply_async(
            args=[symbol, '1mo'],
            queue='analysis',
            expires=300  # Expire après 5 min
        )
        return jsonify({
            'status': 'queued',
            'task_id': task.id,
            'poll_url': f'/api/task/{task.id}'
        }), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.route('/api/status', methods=['GET'])
@handle_errors
def api_status():
    """Statut complet du système"""
    return jsonify({
        'api': 'running',
        'worker': 'see /flower for details',
        'redis': 'ok' if redis_client and redis_client.ping() else 'offline',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'status': 'error'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# ============================================================================
# SECURITY HEADERS
# ============================================================================

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_ENV') == 'development', host='0.0.0.0', port=5000)

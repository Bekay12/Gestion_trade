# VPS DEPLOYMENT GUIDE - Architecture Robuste

## 🎯 Vue d'ensemble

Cette branche `robuste/vps-docker` implémente une architecture production-ready :

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │ HTTP
       │
┌──────▼──────────┐
│     Nginx       │ ← Reverse proxy, HTTPS, static files
│   (reverse)     │
└──────┬──────────┘
       │
┌──────▼──────────┐
│   Flask API     │ ← Orchestration légère, pas de calcul
│  (4 workers)    │
└──────┬──────────┘
       │ Enqueue
       │
┌──────▼──────────┐
│     Redis       │ ← Cache + message queue
│  (persistent)   │   (Volume: /data)
└──────┬──────────┘
       │ Pop tâches
       │
┌──────▼──────────┐
│  Celery Worker  │ ← Download, analyse, backtest
│  (2 workers)    │   (CPU-bound)
└──────┬──────────┘
       │ Persist results
       │
┌──────▼──────────┐
│   PostgreSQL    │ ← Résultats persistants
│  (stock_analysis)│  (Volume: postgres_data)
└─────────────────┘
```

## 📋 Fichiers clés

| Fichier | Rôle |
|---------|------|
| `docker-compose.yml` | Orchestration des 6 services |
| `Dockerfile.api` | Image Flask (léger) |
| `Dockerfile.worker` | Image Celery (avec dépendances calcul) |
| `src/api_robuste.py` | Flask API - enqueue only |
| `src/tasks.py` | Celery tasks (tâches longues) |
| `src/yfinance_helper.py` | Download avec retry + cache disque |
| `requirements-worker.txt` | Dépendances Celery + Redis |
| `.env.example` | Variables d'environnement |

## 🚀 Déploiement sur VPS (Ubuntu 22.04)

### Étape 1: Préparer le serveur

```bash
# SSH sur le VPS
ssh user@your-vps-ip

# Installer Docker + Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo apt install docker-compose-plugin

# Créer les répertoires persistants
mkdir -p /data/cache /data/signals
chmod 755 /data

# Vérifier
docker --version
docker compose version
```

### Étape 2: Cloner et configurer

```bash
cd /home/user
git clone https://github.com/Bekay12/Gestion_trade.git
cd Gestion_trade/stock-analysis-ui

# Checkout la branche robuste
git checkout robuste/vps-docker

# Configurer .env
cp .env.example .env
nano .env

# ⚠️  IMPORTANT: Changer les mots de passe!
# - DB_PASSWORD: min 20 caractères
# - REDIS_PASSWORD: min 20 caractères
# - SECRET_KEY: générer avec: python -c "import secrets; print(secrets.token_hex(32))"
```

### Étape 3: Lancer les services

```bash
# Build images (peut prendre 5-10 min)
docker compose build

# Démarrer tous les services
docker compose up -d

# Vérifier le statut
docker compose ps

# Logs
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f scheduler
```

### Étape 4: Vérifier la santé

```bash
# Health check API
curl -s http://localhost:5000/health | jq

# Redis
docker exec stock-cache redis-cli -a $REDIS_PASSWORD ping
# Output: PONG

# PostgreSQL
docker exec stock-db psql -U stock_user -d stock_analysis -c "SELECT 1"
# Output: 1

# Worker logs
docker compose logs worker | grep "ready to accept"
```

## 📊 Utilisation de l'API

### Analyse asynchrone (recommandé)

```bash
# 1. Enqueue une analyse
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "period": "1mo"}'

# Retour:
# {
#   "status": "queued",
#   "task_id": "abc123...",
#   "poll_url": "/api/task/abc123..."
# }

# 2. Poll le résultat (toutes les 2-3 sec)
curl http://localhost:5000/api/task/abc123...

# Retour (en cours):
# {"status": "pending", "task_id": "abc123..."}

# Retour (fini):
# {"status": "success", "result": {...}}
```

### Depuis le front (index.html)

```javascript
// Analyse async avec polling
async function analyzeAsync(symbol) {
  // 1. Enqueue
  const res1 = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, period: '1mo' })
  });
  const { task_id } = await res1.json();
  
  // 2. Poll jusqu'à succès
  let result;
  while (true) {
    const res2 = await fetch(`/api/task/${task_id}`);
    const data = await res2.json();
    if (data.status === 'success') {
      result = data.result;
      break;
    }
    await new Promise(r => setTimeout(r, 2000)); // Wait 2s
  }
  
  return result;
}
```

## 🔄 Pre-calcul automatique (Celery Beat)

Le scheduler lance automatiquement:

- **Toutes les heures**: `refresh_popular_symbols()` 
  - Analyse: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, BRK.B, JNJ, V
  - Résultats en cache pour requêtes ultra-rapides

- **Chaque matin 9h**: `cache_sp500_symbols()`
  - Pré-télécharge top 20 du S&P 500
  - Cache disque pour 7 jours

Voir `src/tasks.py` pour modifier les horaires.

## 📈 Monitoring

### Flower (Celery web UI)

```bash
# Accessible sur http://localhost:5555
docker compose up -d flower
```

Features:
- En-cours/terminées/échouées tâches
- CPU/RAM par worker
- Retry history
- Real-time monitoring

### Logs

```bash
# Tous les services
docker compose logs -f

# Service spécifique
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f scheduler
```

### Santé

```bash
curl http://localhost:5000/api/status
# {
#   "api": "running",
#   "redis": "ok",
#   "worker": "see /flower for details"
# }
```

## 🔒 Sécurité

### Nginx HTTPS (recommandé)

```bash
# Installer Let's Encrypt + Certbot
sudo apt install certbot python3-certbot-nginx

# Générer cert
sudo certbot certonly --standalone -d your-domain.com

# Monter le cert dans docker-compose
# Modifier nginx.conf pour les chemins /etc/letsencrypt/...
# Redémarrer Nginx
docker compose restart nginx
```

### Firewall

```bash
# Autoriser seulement HTTP/HTTPS
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw default deny incoming
sudo ufw enable
```

### .env sécurisé

```bash
# Jamais commiter .env en production!
echo ".env" >> .gitignore
chmod 600 .env
```

## 🔧 Maintenance

### Backup PostgreSQL

```bash
# Dump la DB
docker exec stock-db pg_dump -U stock_user stock_analysis > backup.sql

# Restaurer
docker exec -i stock-db psql -U stock_user stock_analysis < backup.sql
```

### Clear Redis cache

```bash
docker exec stock-cache redis-cli -a $REDIS_PASSWORD FLUSHDB
```

### Scale worker (CPU-intensive)

```bash
# Modifier docker-compose: ajouter worker_2, worker_3, etc.
# Ou augmenter concurrency dans Dockerfile.worker:
# CMD ["celery", "-A", "src.tasks", "worker", "--concurrency=4"]
```

### Update code

```bash
git pull origin robuste/vps-docker
docker compose build
docker compose restart api worker scheduler
```

## 💰 Coût estimé (VPS)

| Provider | Specs | Prix |
|----------|-------|------|
| Vultr | 2 vCPU, 4 GB RAM, 80 GB SSD | ~$12/mois |
| DigitalOcean | 2 vCPU, 2 GB RAM, 50 GB SSD | $12/mois |
| Linode | 2 vCPU, 4 GB RAM, 80 GB SSD | $12/mois |
| AWS Lightsail | 2 vCPU, 2 GB RAM | $10/mois |

**Vs Render** (free = unstable, pro = $7/mois + $1 par GB RAM supplémentaire)

## 🆘 Troubleshooting

### API lent
```
→ Check Redis: docker exec stock-cache redis-cli info
→ Check Worker: docker compose logs worker
→ Augmenter workers Celery (concurrency) dans Dockerfile.worker
```

### Worker crash
```
→ Logs: docker compose logs worker
→ Memory leak: ajouter --max-tasks-per-child=1000 dans Celery
→ Timeout: augmenter task_soft_time_limit dans src/tasks.py
```

### Erreur yfinance DNS
```
→ Vérifier: docker exec stock-api curl -I https://query1.finance.yahoo.com
→ Si bloqué: utiliser un DNS public (8.8.8.8 dans /etc/resolv.conf du container)
→ Fallback: pré-télécharger via Celery Beat (automatique toutes les heures)
```

### Espace disque plein
```
→ Nettoyer cache: docker exec stock-cache redis-cli FLUSHDB
→ Nettoyer logs: docker compose logs --rm
→ Augmenter volume: réinstancier VPS avec plus d'espace
```

## 📚 Ressources

- [Celery docs](https://docs.celeryproject.org/)
- [Flask + Redis](https://flask.palletsprojects.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [PostgreSQL sur Docker](https://hub.docker.com/_/postgres)

---

**Branche créée**: `robuste/vps-docker`  
**Commit basé sur**: 49589a9 (avant les problèmes Render)  
**Réutilisé de master**: `yfinance_helper.py`, `get_trading_signal()`, analyse core

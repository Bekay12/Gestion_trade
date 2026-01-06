# 🚀 GUIDE COMPLET DE DÉPLOIEMENT EN LIGNE - PHASE 1

## 📋 Table des Matières
1. [Vue d'ensemble](#overview)
2. [Fichiers créés](#fichiers-créés)
3. [Déploiement Local (test)](#déploiement-local)
4. [Déploiement Docker](#déploiement-docker)
5. [Déploiement Heroku](#déploiement-heroku)
6. [Déploiement AWS/Cloud](#déploiement-aws)
7. [Monitoring & Logs](#monitoring--logs)

---

## <a id="overview"></a>🎯 Vue d'Ensemble

Votre système est maintenant prêt pour le déploiement en ligne avec:
- ✅ API REST complète (Flask)
- ✅ Docker containerization
- ✅ Support Heroku, AWS, Azure
- ✅ Background workers pour tâches planifiées
- ✅ Nginx reverse proxy
- ✅ Configuration centralisée

**Architecture**:
```
Internet
   ↓
[Nginx Reverse Proxy] (port 80/443)
   ↓
[Flask API] (port 5000)
   ↓
[Python Core - QSI]
   ↓
[yfinance] + [Cache SQLite]
```

---

## <a id="fichiers-créés"></a>📁 Fichiers Créés - Phase 1

| Fichier | Description | Priorité |
|---------|-------------|----------|
| `requirements.txt` | Dépendances Python | 🔴 CRITIQUE |
| `Dockerfile` | Image Docker | 🔴 CRITIQUE |
| `docker-compose.yml` | Orchestration services | 🟡 IMPORTANT |
| `.env.example` | Variables d'environnement | 🔴 CRITIQUE |
| `src/api.py` | API REST Flask | 🔴 CRITIQUE |
| `Procfile` | Config Heroku | 🟡 IMPORTANT |
| `runtime.txt` | Version Python Heroku | 🟡 IMPORTANT |
| `heroku.yml` | Config Heroku avancée | 🟢 OPTIONNEL |
| `nginx.conf` | Config reverse proxy | 🟡 IMPORTANT |
| `src/background_worker.py` | Worker tasks | 🟢 OPTIONNEL |
| `src/scheduler.py` | Task scheduler | 🟢 OPTIONNEL |
| `.gitignore` | Fichiers à exclure | 🔴 CRITIQUE |

---

## <a id="déploiement-local"></a>💻 Déploiement Local (Test)

### Prérequis
```bash
# Python 3.11+
python --version

# pip à jour
pip install --upgrade pip
```

### Installation

1. **Cloner/préparer le projet**
```bash
cd stock-analysis-ui
```

2. **Créer l'environnement virtuel**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer l'environnement**
```bash
# Copier le fichier de config exemple
cp .env.example .env

# Éditer .env avec vos paramètres
# (Optionnel pour test local)
```

5. **Lancer l'API**
```bash
# Mode développement
python src/api.py

# Output:
# ╔═══════════════════════════════════════╗
# ║  STOCK ANALYSIS API                   ║
# ║  v1.0.0                                ║
# ╚═══════════════════════════════════════╝
# 
# 🚀 Starting API server...
# 📍 Host: 0.0.0.0:5000
# 🔧 Debug: True
# 📚 Docs: http://localhost:5000/api/docs
```

6. **Tester l'API**
```bash
# Dans un autre terminal:

# Health check
curl http://localhost:5000/health

# Documentation
curl http://localhost:5000/api/docs

# Récupérer les signaux
curl http://localhost:5000/api/signals?limit=10

# Analyser un symbole
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "period": "12mo"}'
```

### Commandes Utiles

```bash
# Logs en temps réel
tail -f logs/stock_analysis.log

# Tester un module
python -m pytest tests/

# Vérifier les imports
python -c "import api; print('✅ API imports OK')"

# Arrêter le serveur
Ctrl + C
```

---

## <a id="déploiement-docker"></a>🐳 Déploiement Docker

### Prérequis
```bash
# Installer Docker
# https://www.docker.com/products/docker-desktop

# Vérifier installation
docker --version
docker-compose --version
```

### Build & Run

1. **Build l'image Docker**
```bash
docker build -t stock-analysis:latest .

# Avec output détaillé
docker build -t stock-analysis:latest --progress=plain .
```

2. **Lancer le conteneur**

**Option A: Mode développement (plus simple)**
```bash
docker run -p 5000:5000 \
  -v $(pwd)/data_cache:/app/data_cache \
  -v $(pwd)/signaux:/app/signaux \
  --env-file .env \
  stock-analysis:latest
```

**Option B: Mode production (gunicorn)**
```bash
docker run -p 5000:5000 \
  -v $(pwd)/data_cache:/app/data_cache \
  -v $(pwd)/signaux:/app/signaux \
  -e FLASK_ENV=production \
  --env-file .env \
  stock-analysis:latest
```

3. **Avec Docker Compose (recommandé)**

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier les logs
docker-compose logs -f api

# Arrêter les services
docker-compose down

# Avec services optionnels (Nginx, PostgreSQL)
docker-compose --profile with-nginx up -d
```

### Commandes Docker Utiles

```bash
# Lister les images
docker images

# Lister les conteneurs actifs
docker ps

# Consulter les logs
docker logs -f stock-analysis-api

# Exécuter une commande dans le conteneur
docker exec -it stock-analysis-api bash

# Arrêter/Redémarrer
docker stop stock-analysis-api
docker restart stock-analysis-api

# Nettoyer
docker prune -a
```

---

## <a id="déploiement-heroku"></a>☁️ Déploiement Heroku (GRATUIT & FACILE)

### Prérequis
```bash
# Installer Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Vérifier installation
heroku --version

# Se connecter
heroku login
```

### Déploiement

1. **Préparer le git repo**
```bash
git init
git add .
git commit -m "Initial commit - ready for Heroku deployment"
```

2. **Créer l'app Heroku**
```bash
# Créer une nouvelle app
heroku create stock-analysis-app

# Ou lier une app existante
heroku apps:create stock-analysis-trading
```

3. **Configurer les variables d'environnement**
```bash
# Définir la clé secrète
heroku config:set FLASK_SECRET_KEY=your-random-secret-key-here

# Autres variables (optionnelles)
heroku config:set FLASK_ENV=production
heroku config:set DEBUG=False

# Vérifier la config
heroku config
```

4. **Déployer le code**
```bash
git push heroku main

# Ou si la branche est 'master'
git push heroku master
```

5. **Ouvrir l'app**
```bash
heroku open

# Ou accéder manuellement
https://stock-analysis-app.herokuapp.com/
```

### Monitoring Heroku

```bash
# Voir les logs en temps réel
heroku logs --tail

# Voir les logs d'une dyno spécifique
heroku logs --dyno=worker --tail

# Vérifier les processes
heroku ps

# Scaler les workers (payant)
heroku ps:scale worker=1

# Vérifier l'utilisation des ressources
heroku resources
```

### Limitations Heroku Gratuit

- ⚠️ Application mise en sleep après 30 min d'inactivité
- ⚠️ Pas de persistance de données (fichiers perdus au redéploiement)
- ⚠️ 550 heures/mois gratuites

**Solution pour persistance**: Migrer vers PostgreSQL Heroku Postgres Add-on (payant)

---

## <a id="déploiement-aws"></a>☁️ Déploiement AWS/Azure/GCP

### AWS Elastic Beanstalk (Recommandé)

1. **Installer AWS CLI**
```bash
pip install awsebcli
```

2. **Initialiser l'app**
```bash
eb init -p docker stock-analysis-api
```

3. **Créer l'environnement**
```bash
eb create stock-analysis-prod
```

4. **Déployer**
```bash
git push && eb deploy
```

### Google Cloud Run (Serverless)

```bash
# Installer Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authentifier
gcloud auth login

# Déployer
gcloud run deploy stock-analysis \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# URL de l'app affichée automatiquement
```

### Azure Container Instances

```bash
# Construire et pousser vers Container Registry
az acr build --registry <name> -t stock-analysis:latest .

# Déployer
az container create \
  --resource-group <group> \
  --name stock-analysis \
  --image <registry>/stock-analysis:latest \
  --ports 5000
```

---

## <a id="monitoring--logs"></a>📊 Monitoring & Logs

### Logging Application

```bash
# Voir les logs (fichier local)
tail -f logs/stock_analysis.log

# Filtrer par niveau
grep "ERROR" logs/stock_analysis.log
grep "WARNING" logs/stock_analysis.log

# Archiver les logs
gzip logs/stock_analysis.log.2026-01-*
```

### Monitoring Endpoints

```bash
# Health check (200 = OK)
curl http://localhost:5000/health

# Status détaillé
curl http://localhost:5000/status

# Statistiques
curl http://localhost:5000/api/stats
```

### Monitoring en Production

**Option 1: Sentry (Error tracking)**
```bash
pip install sentry-sdk
```

```python
import sentry_sdk
sentry_sdk.init("your-sentry-dsn")
```

**Option 2: DataDog (APM)**
```bash
pip install datadog
```

**Option 3: New Relic**
```bash
pip install newrelic
newrelic-admin run-program gunicorn ...
```

---

## ✅ Checklist Phase 1

- ✅ `requirements.txt` créé
- ✅ `Dockerfile` créé
- ✅ `docker-compose.yml` créé
- ✅ `.env.example` créé
- ✅ `src/api.py` créé (API Flask complète)
- ✅ `Procfile` créé (pour Heroku)
- ✅ `runtime.txt` créé (Python 3.11)
- ✅ `nginx.conf` créé (reverse proxy)
- ✅ `src/background_worker.py` créé
- ✅ `src/scheduler.py` créé
- ✅ `.gitignore` créé
- ⏭️ Phase 2: Créer `scheduler_setup.py` pour Heroku
- ⏭️ Phase 2: Mettre en place PostgreSQL
- ⏭️ Phase 3: Créer l'UI web (React/Vue)

---

## 🚀 Prochaines Étapes (Phase 2)

1. **Tests de l'API** - Vérifier tous les endpoints
2. **Intégration CI/CD** - GitHub Actions / GitLab CI
3. **Persistance de données** - PostgreSQL
4. **UI Web** - Dashboard React/Vue
5. **SSL/HTTPS** - Certificat Let's Encrypt
6. **Monitoring** - Sentry, DataDog ou similaire
7. **Base de données** - Migrer SQLite → PostgreSQL

---

## 📞 Besoin d'Aide?

Voir aussi:
- [Flask Deployment](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [Heroku Python Buildpack](https://devcenter.heroku.com/articles/python-support)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Nginx Docs](https://nginx.org/en/docs/)

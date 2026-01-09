# 🚀 DÉMARRAGE RAPIDE - API STOCK ANALYSIS

## ⚡ Test en 3 Minutes

### 1. Installer les dépendances
```bash
pip install flask flask-cors python-dotenv
```

### 2. Démarrer l'API
```bash
# Dans un terminal
cd "c:\Users\berti\Desktop\Mes documents\Gestion_trade\stock-analysis-ui"
python src/api.py
```

Vous devriez voir :
```
╔═══════════════════════════════════════╗
║  STOCK ANALYSIS API                   ║
║  v1.0.0                                ║
╚═══════════════════════════════════════╝

🚀 Starting API server...
📍 Host: 0.0.0.0:5000
🔧 Debug: True
📚 Docs: http://localhost:5000/api/docs
```

### 3. Tester l'API

**Option A: Via navigateur**
- Ouvrir http://localhost:5000/health
- Ouvrir http://localhost:5000/api/docs

**Option B: Via PowerShell**
```powershell
# Health check
Invoke-WebRequest -Uri "http://localhost:5000/health" | Select-Object -Expand Content

# Documentation
Invoke-WebRequest -Uri "http://localhost:5000/api/docs" | Select-Object -Expand Content

# Récupérer les signaux
Invoke-WebRequest -Uri "http://localhost:5000/api/signals?limit=10" | Select-Object -Expand Content
```

**Option C: Via Python**
```bash
# Dans un autre terminal
python test_api.py
```

---

## 🌐 Endpoints Disponibles

### Health & Status
- `GET /health` - Vérifier la santé de l'API
- `GET /status` - Status détaillé du système
- `GET /api/docs` - Documentation complète

### Signals
- `GET /api/signals` - Récupérer les signaux récents
  - Query params: `limit`, `symbol`, `min_reliability`
- `GET /api/signals/{symbol}` - Signaux pour un symbole spécifique

### Analysis
- `POST /api/analyze` - Analyser un symbole
  ```json
  {
    "symbol": "AAPL",
    "period": "12mo"
  }
  ```

- `POST /api/analyze-batch` - Analyser plusieurs symboles
  ```json
  {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "period": "12mo"
  }
  ```

- `POST /api/backtest` - Lancer un backtest
  ```json
  {
    "symbol": "AAPL",
    "period": "12mo"
  }
  ```

### Stats
- `GET /api/stats` - Statistiques globales

---

## 📋 Prochaines Étapes

### Test Local Complet
1. ✅ API démarrée
2. ⏭️ Exécuter `python test_api.py`
3. ⏭️ Tester avec Postman/Insomnia
4. ⏭️ Vérifier les logs

### Déploiement Docker
```bash
# Build
docker build -t stock-analysis:latest .

# Run
docker-compose up -d

# Vérifier
curl http://localhost:5000/health
```

### Déploiement Heroku
```bash
# Se connecter
heroku login

# Créer l'app
heroku create stock-analysis-app

# Déployer
git push heroku main

# Ouvrir
heroku open
```

---

## 🔧 Configuration

### Variables d'Environnement
Copier `.env.example` vers `.env` et configurer:
```bash
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key
DEBUG=True
```

### Fichiers Importants
- `src/api.py` - API Flask principale
- `requirements.txt` - Dépendances Python
- `Dockerfile` - Configuration Docker
- `docker-compose.yml` - Orchestration services
- `Procfile` - Configuration Heroku

---

## 📚 Documentation Complète

Voir [DEPLOYMENT_PHASE_1.md](DEPLOYMENT_PHASE_1.md) pour:
- Guide de déploiement complet
- Instructions Docker détaillées
- Configuration Heroku/AWS/Azure
- Monitoring & Logs
- Troubleshooting

---

## ⚠️ Troubleshooting

### L'API ne démarre pas
```bash
# Vérifier les imports
python -c "import sys; sys.path.insert(0, 'src'); from api import app; print('OK')"

# Vérifier les dépendances
pip install -r requirements.txt
```

### Port 5000 déjà utilisé
```bash
# Changer le port dans .env
BIND_PORT=5001

# Ou en variable d'environnement
set BIND_PORT=5001
python src/api.py
```

### Erreur d'import de modules
```bash
# Ajouter src au PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;c:\Users\berti\Desktop\Mes documents\Gestion_trade\stock-analysis-ui\src
```

---

## 💡 Commandes Utiles

```bash
# Démarrer l'API
python src/api.py

# Tester l'API
python test_api.py

# Installer dépendances
pip install -r requirements.txt

# Build Docker
docker build -t stock-analysis .

# Lancer Docker
docker-compose up -d

# Voir les logs Docker
docker-compose logs -f api

# Arrêter Docker
docker-compose down

# Health check
curl http://localhost:5000/health

# Documentation API
curl http://localhost:5000/api/docs
```

---

## ✅ Checklist Phase 1

- [x] API Flask créée
- [x] Endpoints implémentés
- [x] Docker configuré
- [x] Heroku configuré
- [x] Documentation créée
- [ ] Tests exécutés
- [ ] API déployée en production
- [ ] Monitoring configuré

---

**Besoin d'aide ?** Voir [DEPLOYMENT_PHASE_1.md](DEPLOYMENT_PHASE_1.md)

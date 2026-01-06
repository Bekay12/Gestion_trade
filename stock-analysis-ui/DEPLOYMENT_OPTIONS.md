# 🚀 OPTIONS DE DÉPLOIEMENT - GUIDE PRATIQUE

## ✅ Votre API est fonctionnelle localement !

Bravo ! L'API tourne correctement sur `http://localhost:5000`

---

## 📋 3 Options de Déploiement

### Option 1 : Heroku (Cloud Gratuit) ☁️

**Installer Heroku CLI sur Windows** :

**Méthode A : Via Scoop (Recommandé)**
```powershell
# Installer Scoop si pas déjà installé
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Installer Heroku CLI
scoop install heroku-cli

# Vérifier
heroku --version
```

**Méthode B : Via Chocolatey**
```powershell
# Installer Chocolatey si pas déjà installé (admin)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Installer Heroku CLI
choco install heroku-cli

# Vérifier
heroku --version
```

**Méthode C : Installer manuellement**
1. Télécharger : https://devcenter.heroku.com/articles/heroku-cli#install-the-heroku-cli
2. Exécuter l'installeur
3. Redémarrer PowerShell
4. Vérifier : `heroku --version`

**Après installation** :
```bash
# Login Heroku
heroku login

# Créer l'app
heroku create stock-analysis-trading

# Ajouter remote
heroku git:remote -a stock-analysis-trading

# Déployer
git push heroku master

# Ouvrir l'app
heroku open
```

---

### Option 2 : Docker Local (Recommandé pour Test) 🐳

**Si Docker Desktop est installé** :

```powershell
# Build l'image
docker build -t stock-analysis:latest .

# Lancer avec docker-compose
docker-compose up -d

# Vérifier
Invoke-RestMethod -Uri "http://localhost:5000/health"

# Voir les logs
docker-compose logs -f api

# Arrêter
docker-compose down
```

**Avantages** :
- ✅ Test local exact de la production
- ✅ Pas besoin de compte cloud
- ✅ Isolation complète
- ✅ Facile à debugger

---

### Option 3 : Render.com (Alternative Heroku - GRATUIT) 🌐

**Plus simple que Heroku, sans CLI** :

1. **Créer un compte sur render.com**
   - https://render.com/

2. **Créer un nouveau Web Service**
   - "New" → "Web Service"
   - Connecter votre repo GitHub/GitLab

3. **Configuration** :
   ```
   Name: stock-analysis
   Environment: Docker
   Branch: master ou main
   ```

4. **Variables d'environnement** :
   ```
   FLASK_ENV=production
   FLASK_SECRET_KEY=your-secret-key-here
   ```

5. **Déployer** :
   - Render détecte automatiquement le Dockerfile
   - Déploiement automatique à chaque push

**Avantages** :
- ✅ Gratuit (750h/mois)
- ✅ SSL automatique
- ✅ Déploiement auto
- ✅ Pas de CLI nécessaire

---

### Option 4 : Railway.app (Alternative Simple) 🚂

1. Aller sur https://railway.app/
2. "Start a New Project"
3. Connecter GitHub repo
4. Railway détecte automatiquement
5. Déploiement en 1 clic

---

### Option 5 : Google Cloud Run (Serverless) ☁️

**Si vous avez gcloud installé** :

```powershell
# Login Google Cloud
gcloud auth login

# Déployer
gcloud run deploy stock-analysis `
  --source . `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated

# URL affichée automatiquement
```

---

## 🎯 Recommandation

### Pour Test Rapide (5 min)
→ **Option 2 : Docker Local**
```powershell
docker-compose up -d
```

### Pour Production Gratuite (15 min)
→ **Option 3 : Render.com** (plus simple, pas de CLI)

### Pour Intégration Pro (30 min)
→ **Option 1 : Heroku** (standard industrie)

---

## 📊 Comparaison

| Plateforme | Gratuit | CLI Requis | SSL | Auto Deploy | Difficulté |
|-----------|---------|------------|-----|-------------|------------|
| **Heroku** | ✅ 750h/mois | ✅ | ✅ | ✅ | ⭐⭐ |
| **Render** | ✅ 750h/mois | ❌ | ✅ | ✅ | ⭐ |
| **Railway** | ✅ $5/mois | ❌ | ✅ | ✅ | ⭐ |
| **Docker Local** | ✅ Illimité | ❌ | ❌ | ❌ | ⭐⭐ |
| **Google Cloud Run** | ✅ 2M req/mois | ✅ | ✅ | ✅ | ⭐⭐⭐ |

---

## 🛠️ Test Local Complet (Maintenant)

**Votre API tourne déjà ! Testons-la** :

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:5000/health"

# Documentation
Invoke-RestMethod -Uri "http://localhost:5000/api/docs"

# Statistiques
Invoke-RestMethod -Uri "http://localhost:5000/api/stats"

# Signaux (si fichier existe)
Invoke-RestMethod -Uri "http://localhost:5000/api/signals?limit=5"
```

**Ouvrir dans le navigateur** :
- http://localhost:5000/health
- http://localhost:5000/api/docs
- http://localhost:5000/status

---

## ✅ État Actuel

- ✅ API créée et fonctionnelle
- ✅ Tests locaux réussis
- ✅ Git commit effectué
- ⏭️ Choisir plateforme de déploiement

---

## 🎯 Action Recommandée

**Pour déployer SANS installer Heroku CLI** :

### 1. Créer un compte Render.com
https://render.com/

### 2. Pousser sur GitHub (si pas déjà fait)
```powershell
# Créer repo sur github.com
# Puis :
git remote add origin https://github.com/votre-username/stock-analysis-ui.git
git push -u origin master
```

### 3. Connecter Render à GitHub
- New Web Service
- Sélectionner votre repo
- Render détecte le Dockerfile
- Déployer !

**Temps total : 10 minutes** ⏱️

---

## 💡 Commandes Utiles Maintenant

```powershell
# Arrêter l'API locale
# Ctrl + C dans le terminal Python

# Tester avec le script de test
python test_api.py

# Voir les fichiers créés
Get-ChildItem -Recurse -Include "*.py","Dockerfile","*.yml","*.txt" | Select-Object Name, Length

# Vérifier Docker (si installé)
docker --version
docker-compose --version
```

---

**Question : Quelle option préférez-vous ?**
1. Installer Heroku CLI (20 min)
2. Utiliser Render.com (10 min, plus simple)
3. Docker local (5 min, test uniquement)
4. Railway.app (10 min)

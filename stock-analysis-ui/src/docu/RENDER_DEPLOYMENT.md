# 🚀 DÉPLOIEMENT SUR RENDER.COM - GUIDE COMPLET

## ✅ Pourquoi Render.com ?

- ✅ **Plus simple que Heroku** - Pas de CLI à installer
- ✅ **Gratuit** - 750 heures/mois
- ✅ **SSL automatique** - HTTPS inclus
- ✅ **Déploiement auto** - Push Git = déploiement
- ✅ **Dashboard intuitif** - Interface graphique simple

---

## 📋 Prérequis

- [x] Code sur GitHub/GitLab/Bitbucket
- [x] Fichiers créés : `render.yaml`, `Dockerfile`, `requirements.txt`
- [x] Compte Render.com (gratuit)

---

## 🚀 Déploiement en 5 Minutes

### Étape 1 : Pousser sur GitHub

Si pas encore fait :

```powershell
# Créer un repo sur github.com
# Puis dans votre terminal :

git remote add origin https://github.com/VOTRE-USERNAME/stock-analysis-ui.git
git branch -M main
git push -u origin main
```

### Étape 2 : Créer un Compte Render

1. Aller sur https://render.com
2. Cliquer "Get Started for Free"
3. Se connecter avec GitHub (recommandé)

### Étape 3 : Créer un Web Service

1. **Dans le Dashboard Render** :
   - Cliquer "New +" en haut à droite
   - Sélectionner "Web Service"

2. **Connecter le Repository** :
   - Autoriser Render à accéder à votre GitHub
   - Sélectionner le repo `stock-analysis-ui`
   - Cliquer "Connect"

3. **Configuration Automatique** :
   Render détecte automatiquement :
   - ✅ `render.yaml` → Configuration automatique
   - ✅ `Dockerfile` → Build Docker
   - ✅ Branch `main` → Auto-deploy

4. **Vérifier la Configuration** :
   ```
   Name: stock-analysis-api
   Runtime: Docker
   Branch: main
   Plan: Free
   ```

5. **Variables d'environnement** (déjà dans render.yaml) :
   - `FLASK_ENV=production`
   - `FLASK_SECRET_KEY=auto-généré`
   - `BIND_PORT=10000`

6. **Cliquer "Create Web Service"** ✅

### Étape 4 : Attendre le Déploiement

```
⏳ Building... (2-3 minutes)
   ├── Cloning repository
   ├── Building Docker image
   ├── Pushing to registry
   └── Starting service

✅ Live! (URL fournie)
```

### Étape 5 : Tester Votre API

Render vous fournit une URL comme :
```
https://stock-analysis-api.onrender.com
```

**Tester** :
```powershell
# Health check
Invoke-RestMethod -Uri "https://stock-analysis-api.onrender.com/health"

# Documentation
Invoke-RestMethod -Uri "https://stock-analysis-api.onrender.com/api/docs"

# Stats
Invoke-RestMethod -Uri "https://stock-analysis-api.onrender.com/api/stats"
```

---

## 📊 Après le Déploiement

### Accéder aux Logs

1. Dashboard Render → Votre service
2. Onglet "Logs"
3. Logs en temps réel disponibles

### Monitoring

1. **Health Check** :
   - Render ping automatiquement `/health`
   - Si échec → redémarre le service

2. **Métriques** :
   - CPU, RAM, Requêtes
   - Disponibles dans "Metrics"

### Déploiement Automatique

Chaque `git push` sur `main` → redéploiement automatique :

```powershell
git add .
git commit -m "Update API"
git push origin main

# Render déploie automatiquement
```

### Variables d'Environnement

Ajouter/modifier dans le dashboard :
1. Service → "Environment"
2. Ajouter nouvelle variable
3. Sauvegarder → Redéploiement auto

---

## ⚙️ Configuration Avancée

### Custom Domain

1. Dashboard → Votre service
2. "Settings" → "Custom Domains"
3. Ajouter votre domaine
4. Configurer DNS (CNAME)

### Base de Données PostgreSQL

Ajouter dans `render.yaml` :
```yaml
databases:
  - name: stock-analysis-db
    databaseName: stock_analysis
    plan: free
```

### Worker Background

Pour tâches planifiées (plan payant) :
```yaml
services:
  - type: worker
    name: stock-analysis-worker
    runtime: docker
    startCommand: python src/background_worker.py
```

---

## 🔧 Troubleshooting

### Service ne démarre pas

**Vérifier les logs** :
1. Dashboard → Service → Logs
2. Chercher les erreurs

**Erreurs communes** :
```
Port binding error → Vérifier BIND_PORT=10000
Module not found → Vérifier requirements.txt
Docker build failed → Vérifier Dockerfile
```

### Service en "Sleep"

Le plan gratuit met le service en sleep après 15 min d'inactivité.

**Solution** :
- Premier accès : 10-30s pour réveiller
- Ou upgrader vers plan payant ($7/mois)

### Déploiement Lent

Premier déploiement : 3-5 minutes (normal)  
Déploiements suivants : 1-2 minutes

---

## 💰 Plans Render

### Free Plan
- ✅ 750 heures/mois
- ✅ SSL gratuit
- ✅ 512 MB RAM
- ⚠️ Service sleep après 15 min
- ⚠️ Limitation : 1 service web gratuit

### Starter Plan ($7/mois)
- ✅ Always-on (pas de sleep)
- ✅ 512 MB RAM
- ✅ Plusieurs services
- ✅ Support email

### Pro Plan ($25/mois)
- ✅ 4 GB RAM
- ✅ Support prioritaire
- ✅ Scaling horizontal

---

## 📈 Comparaison Heroku vs Render

| Feature | Render (Free) | Heroku (Free) |
|---------|--------------|---------------|
| **Sleep après** | 15 min | 30 min |
| **Heures/mois** | 750h | 550h |
| **RAM** | 512 MB | 512 MB |
| **SSL** | ✅ Auto | ✅ Auto |
| **CLI Required** | ❌ | ✅ |
| **Interface** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Auto Deploy** | ✅ | ✅ |
| **Custom Domain** | ✅ | ✅ |

**Verdict** : Render est plus simple et moderne ! 🏆

---

## ✅ Checklist Déploiement

- [x] `render.yaml` créé
- [x] Code sur GitHub
- [x] Compte Render créé
- [ ] Service créé sur Render
- [ ] Déploiement réussi
- [ ] API testée en ligne
- [ ] URL partagée

---

## 🎯 Commandes Git Utiles

```powershell
# Commit et push
git add render.yaml Dockerfile
git commit -m "Add Render.com configuration"
git push origin main

# Vérifier le statut
git status

# Voir l'historique
git log --oneline -5

# Créer une nouvelle branche (test)
git checkout -b test-deployment
git push -u origin test-deployment
```

---

## 📚 Ressources

- **Render Documentation** : https://render.com/docs
- **Dashboard** : https://dashboard.render.com
- **Status Page** : https://status.render.com
- **Community** : https://community.render.com

---

## 🚀 Prochaines Étapes

1. ✅ Déployer sur Render
2. ⏭️ Tester l'API en production
3. ⏭️ Configurer un domaine personnalisé
4. ⏭️ Ajouter monitoring (Sentry, etc.)
5. ⏭️ Créer une UI web (React/Vue)

---

**Votre API sera accessible 24/7 à l'URL fournie par Render !** 🎉

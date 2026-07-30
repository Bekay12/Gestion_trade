# 📋 Changelog - Stock Analysis Web Dashboard

## Version 1.1.0 - Correction de biais de look-ahead et durcissement de l'API (2026-07-30)

### 🐛 Corrections

- **Biais de look-ahead dans le calcul des fondamentaux point-in-time**
  - Fichier : `src/fundamentals_cache.py`, fonction `compute_pit_fundamentals`.
  - La branche de repli terminale retournait les fondamentaux du trimestre le plus récent du cache lorsque aucun trimestre n'était encore publié à la date simulée. Elle retourne maintenant `None`.
  - Cette branche était atteinte à chaque barre par la boucle de backtest (`src/trading_c_acceleration/qsi_optimized.py`), injectant des données futures dans les barres anciennes.
  - Les performances de backtest en étaient surévaluées, et l'optimiseur sélectionnait ses paramètres sur cette base.
  - Les appelants traitent déjà `None` : la barre est évaluée sans composante fondamentale.
  - Verrouillé par `src/tests/test_pit_fundamentals.py` (7 tests). Vérifié : 3 de ces tests échouent sur le code d'avant correctif et passent après.

### 🎯 Sécurité et durcissement

- **Durcissement de l'API Flask**
  - Mise à jour des dépendances : Flask 2.2.5 → 3.1.3, Flask-Cors 4.0.0 → 6.0.0, gunicorn 21.2.0 → 23.0.0, requests 2.31.0 → 2.33.0, python-dotenv 1.0.1 → 1.2.2. `pip-audit` passe de 14 vulnérabilités connues à 0.
  - `src/api.py` :
    - Décorateur `require_api_key` fermant. Sans `API_KEY` en environnement, les routes protégées répondent 503 au lieu de laisser passer toutes les requêtes. Échappatoire explicite pour le développement : `API_AUTH_DISABLED=1`.
    - Comparaison de clé par `hmac.compare_digest`.
    - `CORS(app)` sans restriction remplacé par liste d'origines lue dans `CORS_ORIGINS`. Vide par défaut, aucun en-tête CORS émis (same-origin uniquement).
    - Routes `POST /api/backtest` et `POST /api/lists/<type>` désormais protégées par authentification (auparavant aucune).
    - Gestionnaire d'erreurs n'expose plus `str(e)` au client. Détail dans les logs via `logger.exception`, réponse générique au client.
    - Adaptation à Flask 3 : `app.json.sort_keys` et `app.json.compact` remplacent les clés dépréciées.
    - Adresse d'écoute par défaut du bloc de développement : `0.0.0.0` → `127.0.0.1`.
  - `render.yaml` :
    - Ajout de `API_KEY` (`generateValue: true`) et de `CORS_ORIGINS`.
    - `autoDeploy` passe à `false` (un push sur master ne met plus l'API en ligne sans relecture).
  - `.env.example` :
    - Documentation des trois nouvelles variables.
    - Correction de la note sur `API_KEY` qui indiquait à tort qu'une valeur vide désactive l'authentification.

---

## Version 1.0.0 - Interface Web Complète (Janvier 2025)

### ✨ Nouvelles Fonctionnalités

#### 🎨 Interface Utilisateur Complètement Restructurée
- **Tabbed Dashboard** avec 4 onglets principaux
  - 🔍 **Analyser** - Analyse de symboles individuels
  - 📋 **Listes** - Gestion des symboles populaires, personnels et d'optimisation
  - 📊 **Batch** - Analyse multiple (jusqu'à 20 symboles)
  - 🔬 **Backtest** - Test de stratégies historiques

#### 🖥️ Dashboard Amélioré
- Statistiques en temps réel (Signaux Total, BUY, SELL, Fiabilité Moyenne)
- Affichage des 20 derniers signaux dans un tableau interactif
- Codes couleur intelligents (🟢 BUY, 🔴 SELL, 🟡 HOLD)
- Design moderne avec dégradés et animations CSS

#### 📡 Nouveaux Endpoints API
- `POST /api/analyze-popular` - Analyser les listes populaires et personnelles
- `POST /api/analyze-batch` - Analyser plusieurs symboles en une seule requête
- `GET /api/lists` - Récupérer les 3 listes de symboles
- `POST /api/lists/<type>` - Ajouter/retirer symboles de listes
- `POST /api/backtest` - Exécuter un backtest avec paramètres

#### 🔧 Fonctionnalités JavaScript Ajoutées
- **Tab Switching** - Navigation fluide entre les onglets
- **Form Validation** - Validation des entrées utilisateur
- **API Integration** - Communication seamless avec le backend
- **Result Rendering** - Affichage dynamique des résultats
- **Error Handling** - Gestion gracieuse des erreurs

#### 📊 Fonctionnalités par Onglet

**Onglet Analyser:**
- Analyse d'un symbole unique
- Affichage du signal (BUY/SELL/HOLD)
- Détails: Prix, RSI, Tendance, Domaine, Volume, Fiabilité
- Chargement animé pendant l'analyse

**Onglet Listes:**
- Affichage des 3 listes (Populaires, Personnels, Optimisation)
- Formulaires pour ajouter/retirer des symboles
- Support de multiples symboles par ajout
- Gestion instantanée sans rechargement de page

**Onglet Batch:**
- Champ pour entrer jusqu'à 20 symboles
- Sélection de la période
- Tableau de résultats avec tous les détails
- Limitation et validation automatique

**Onglet Backtest:**
- Champ symbole unique
- Sélection de la période historique
- Paramètres de moyennes mobiles (défaut: 9/21)
- Résultats formatés: Gain Total, Win Rate, Nb Trades, Gagnants

### 🐛 Corrections et Améliorations

- **JavaScript Optimisé** - Évite les mutations du DOM inutiles
- **CSS Responsive** - Interface adaptée à tous les écrans
- **Animation Fluides** - Transitions CSS pour meilleure UX
- **Gestion d'Erreurs** - Messages clairs pour chaque type d'erreur
- **Performance** - Chargement initial rapide avec cache

### 🎯 Améliorations de Stabilité

- Tous les endpoints testés et validés
- Intégration avec les mêmes fonctions Python que le desktop
- Utilisation cohérente du format de réponse JSON
- Support des périodes complètes (1M, 3M, 6M, 1A, 2A, 5A)

### 📝 Documentation Ajoutée

- **INTERFACE_GUIDE.md** - Guide complet d'utilisation
- **Commentaires en code** - JavaScript bien documenté
- **Exemples d'usage** - Cas d'usage dans la documentation

### 🚀 Déploiement

- Rendu automatique sur Render.com
- URL: https://stock-analysis-api-8dz1.onrender.com/
- Redéploiement automatique à chaque push Git
- Health check endpoint disponible

---

## Version 0.9.0 - API Endpoints Implémentés (Décembre 2024)

### ✨ Nouvelles Fonctionnalités
- Endpoints `/api/analyze`, `/api/lists`, `/api/backtest`
- Flask app avec `render_template()` pour servir HTML
- Template HTML basique de l'interface

### 🐛 Corrections
- Configuration des chemins absolus avec `Path(__file__).parent.resolve()`
- Fix des imports `Archives.qsi` → `qsi`
- Python 3.11 compatible dependencies

---

## Version 0.8.0 - Docker & Render Setup (Décembre 2024)

### ✨ Nouvelles Fonctionnalités
- Dockerfile avec `PYTHONPATH=/app/src`
- render.yaml Blueprint configuration
- Requirements.txt optimisé pour Python 3.11

### 🐛 Corrections
- Numpy 1.26.4 pour Python 3.11
- Ta-lib 0.11.0 (0.10.2 n'existe pas)
- Flask-Cors 4.0.0 ajouté

---

## Utilisation Conseillée

### Pour les Utilisateurs
1. Visitez https://stock-analysis-api-8dz1.onrender.com/
2. Explorez les 4 onglets
3. Utilisez le guide INTERFACE_GUIDE.md

### Pour les Développeurs
1. Clonez le repo
2. Installez les dépendances: `pip install -r requirements.txt`
3. Lancez l'API: `python api.py`
4. Consultez le README.md pour plus de détails

---

## Feuille de Route Future

- [ ] Authentification utilisateur
- [ ] Sauvegarde des listes en base de données
- [ ] Graphiques et charts (Chart.js)
- [ ] Notifications en temps réel (WebSocket)
- [ ] Export des résultats (PDF, Excel)
- [ ] Historique des analyses
- [ ] Alertes personnalisées
- [ ] Version mobile native

---

**Contributeurs:** Bekay12  
**Licence:** MIT  
**Support:** Voir INTERFACE_GUIDE.md pour le dépannage

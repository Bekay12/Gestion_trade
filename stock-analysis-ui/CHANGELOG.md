# 📋 Changelog - Stock Analysis Web Dashboard

## Version 1.3.0 - Requête du store réparée, validation des entrées, journalisation (2026-07-30)

### 🐛 Corrections

- **`query_features()` retournait toujours un résultat vide**
  - `src/market_store.py` : la requête employait `hive_partitioning=false` sans `union_by_name=true` et échouait dès qu'un fichier Parquet n'avait pas les mêmes colonnes que le premier lu — ce qui est le cas courant, les schémas variant selon la date d'écriture de chaque symbole. Le `try/except` qui enveloppait l'appel masquait la panne en retournant un DataFrame vide. Mesuré après correction : 3522 lignes pour AAPL et MSFT, contre zéro auparavant.
  - `src/api.py` : neuf routes interceptaient leurs propres exceptions, retournant `str(e)` au client avec un code 500. Les détails sont désormais journalisés, le client reçoit un message générique.

### 🎯 Sécurité et validation

- **Protection contre les injections SQL et validation centralisée**
  - `src/market_store.py` : la clause `WHERE` de `query_features()` utilise des paramètres liés au lieu d'interpoler les valeurs, empêchant les erreurs dues à des apostrophes dans les symboles.
  - `src/api.py` :
    - Validation centralisée via trois fonctions (`valider_symbole`, `valider_periode`, `valider_liste_symboles`) : symboles validés sur 1-15 caractères (A-Z, chiffres, `.` `-` `=`), périodes sur leur format (`12mo`, `15mo`, `4y`), `limit` bornée à 1-500, `min_reliability` à 0-100.
    - Limite de 50 symboles par lot dans `/api/analyze-popular` pour éviter l'épuisement du budget yfinance.

### 📝 Journalisation

- **Amélioration de la traçabilité des erreurs**
  - `src/market_store.py` : les onze blocs `try/except/pass` journalisent désormais la cause des erreurs, permettant de détecter les problèmes comme la panne de `query_features()`.
  - `src/api.py` : dix-sept appels à `print()` remplacés par le logger du module avec préfixe `[API]`. L'erreur d'import initiale reste un `print` (précédant la configuration du logging).

---

## Version 1.2.0 - Mises à jour de fonctionnalités (2026-07-30)

### 🐛 Corrections

- **Ajout de fonctions et correction des appels**
  - `src/symbol_manager.py` : fonction `classify_cap_range(market_cap_b)` ajoutée, appelée à deux endroits sans jamais avoir été définie. L'argument est une capitalisation en milliards de dollars US ; seuils 2 / 10 / 100 pour Small / Mid / Large / Mega, `Unknown` si absente/nulle/négative/non-numérique.
  - `src/qsi.py` : chaîne d'import rétablie. Le module importait `classify_cap_range` depuis `symbol_manager` dans un bloc `try/except ImportError`. Ce nom manquant faisait échouer l'import entier, si bien que `qsi` perdait aussi `init_symbols_table`, `sync_txt_to_sqlite`, `get_symbols_by_list_type` et `get_symbols_by_sector_and_cap`, et basculait sur un repli « méthode txt ».
  - `src/symbol_manager.py` : `_get_cap_range_safe` réutilise désormais `classify_cap_range` au lieu de redupliquer les seuils.
  - `src/symbol_manager.py` : `get_all_sectors()` exclut les secteurs nuls ou vides. Ils étaient retournés tels quels et faisaient lever `TypeError` à tout appelant qui triait la liste.
  - `src/symbol_manager.py` : le bloc `if __name__ == '__main__'` appelait `display_popular_symbols_distribution()`, fonction inexistante ; il affiche maintenant un résumé bâti sur les fonctions réellement définies.
  - `src/symbol_manager.py` : le bloc `except` d'enrichissement journalise désormais la cause au lieu de l'avaler en silence.

### 🎯 Nettoyage

- **Réduction de code et suppression d'artefacts**
  - `src/cache_db.py` réduit de 1308 à 108 lignes. Le module ne contient plus que la ré-exportation de l'API publique de `market_store`. Les ~1200 lignes d'implémentation SQLite qu'il conservait étaient masquées deux fois (par les imports de tête puis par un bloc de reliaison final) et n'étaient donc jamais exécutées, sans que rien ne le signale : modifier une fonction au milieu du fichier n'avait aucun effet.
  - `src/fundamentals_cache.py` : l'appel `_ensure_fundamentals_table()` au niveau module est supprimé. Chaque point d'entrée public l'appelle désormais lui-même, avec une mémorisation par chemin de base. Un simple `import cache_db` ouvrait jusqu'ici une connexion SQLite sur `stock_analysis.db`, par la cascade `cache_db` → `market_store` → `fundamentals_cache`. Vérifié : zéro connexion ouverte à l'import.
  - `src/tests/conftest.py` : les tests s'exécutent sur une copie temporaire de la base. Sans cette isolation, `sync_txt_to_sqlite` appelée avec les fixtures de test purgeait les vraies listes de symboles de l'utilisateur.
  - 57 fichiers suivis par git malgré le `.gitignore` du projet ont été détachés du suivi (sorties de signaux, artefacts de build Windows, fichiers d'IDE, journal). Ils restent sur le disque.
  - `.gitignore` : ajout de `*.db-wal` et `*.db-shm`.

### ✅ Tests

- Nouveaux fichiers `src/tests/test_cap_range.py` (16 cas) et `src/tests/test_pit_fundamentals.py` (7 cas).
- Le sous-ensemble exécuté en intégration continue passe de 4 à 27 tests.

---

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

# 📋 Changelog - Stock Analysis Web Dashboard

## Version 1.6.0 - Tickers Finviz corrigés, colonnes « Nom » et « Pays », nombres arrondis (2026-08-03)

### 🐛 Corrections

- **Le tri des colonnes chiffrées du tableau de résultats était lexicographique**
  - `ui/main_window.py` : `_set_item()` posait le texte `str(value)` puis une valeur numérique sur `Qt.EditRole`, que `QTableWidgetItem` ramène au rôle d'affichage. Mesuré sur la colonne Score : l'ordre croissant donnait **10.2, 100, puis 9.5**. La nouvelle `CelluleNumerique` garde la valeur réelle dans un attribut et surcharge la comparaison ; vérifié croissant et décroissant, valeurs négatives comprises (`-3.75, 9.5, 10.2, 100`). Le tableau comparatif hérite du correctif, ses cellules étant recopiées.
  - Corollaire : `compute_domain_stats()` faisait `int(item.data(Qt.EditRole))`, qui lève sur un « 3.0 » — la ligne était alors abandonnée en silence par le `except: continue`, faussant les totaux par domaine. La lecture passe par `valeur_cellule()`, qui prend la valeur portée par la cellule et non son texte arrondi.

- **Tous les tickers venant de Finviz partaient avec leur première lettre doublée**
  - `src/core/finviz_screeners.py` : Finviz place un avatar-lettre dans la cellule Ticker (`<a class="company-ticker"><img …/><span>I</span></a>`, la lettre servant de repli le temps que le logo charge), devant le lien du symbole. `finvizfinance` 1.3.0 remplit chaque cellule avec `td.text`, qui concatène **tout** le texte de la cellule : `IESC` devenait `IIESC`, `AMZN` → `AAMZN`, `TW` → `TTW`. Chaque symbole injecté dans le champ d'analyse échouait ensuite côté yfinance (« No history available for FFUTU »). Mesuré sur une session : 14 symboles sur 14 en échec, 0 synchronisé.
  - Le ticker est désormais lu sur l'attribut `data-boxover-ticker` du `<td>`, qui fait autorité ; à défaut, l'avatar est retiré du DOM avant que la librairie ne lise le texte. Aucune réparation par heuristique : `AAPL`, `MMM` ou `TTWO` sont des symboles légitimes à première lettre doublée, indiscernables d'un symbole corrompu. Si les deux mécanismes échouent (3ᵉ refonte de la page Finviz), `run_screen()` lève une `RuntimeError` explicite plutôt que de renvoyer une liste corrompue.
  - `run_screen()` devient le point d'entrée unique vers finvizfinance : il porte à la fois la session `curl_cffi` et l'assainissement du ticker. Le screener Gapper, qui construisait son propre `Overview` dans `ui/mixins/screeners.py`, passe par lui — il avait exactement le même bug.
  - Verrouillé par `src/tests/test_finviz_screeners.py` (5 tests, HTML figé, aucun accès réseau).

- **Effet de bord : 18 profils fantômes écrits dans le store**
  - `ensure_instrument_profiles()` acceptait le `info` renvoyé par yfinance pour un symbole inexistant. yfinance ne lève pas : il renvoie un dict non vide mais sans identité, parfois un pseudo-fonds de l'échange « YHD » au nom numérique (`164` pour `AABT`, `24564` pour `TTYL`). 18 profils avaient été écrits, `name` valant le ticker corrompu ; ils comptaient ensuite comme profils **frais**, donc n'étaient jamais corrigés, et polluaient les colonnes Nom et Pays.
  - `_info_sans_identite()` refuse désormais un profil sans `shortName` ni `longName`, ainsi qu'un nom purement numérique. Les 18 profils existants ont été déplacés vers `market_parquet/instruments_fantomes/` (déplacement, pas suppression : le dossier est hors du glob de lecture). 186 profils valides conservés.

### ✨ Nouveautés

- **Colonne « Nom » sur tous les tableaux de l'interface**
  - Tableau de résultats (`merged_table`), tableau comparatif multicritère, tableau de comparaison historique, et tous les screeners (Finviz market-wide, Finviz Gapper, Yahoo Screener, top movers, vues store Combined et Golden Cross, Événements 48 h).
  - Coût réseau nul. Finviz et le screener Yahoo renvoient déjà le nom dans leur réponse ; ailleurs, `market_store.get_name_map()` le lit dans les profils d'instruments (1 requête DuckDB), avec la colonne `name` des features en secours. Un symbole sans profil affiche N/A et se remplit à la passe suivante, comme la colonne Pays.
  - Nom tronqué à l'affichage et repris en entier en infobulle : sur 28 colonnes, un nom complet poussait les colonnes chiffrées hors de l'écran.
  - La complétion des profils en arrière-plan se déclenche désormais sur la présence de « Nom » **ou** de « Pays » : les deux colonnes viennent du même profil, et le screener Événements 48 h (sans colonne Pays) ne l'aurait jamais déclenchée.

- **Colonne « Pays » dans le tableau de résultats**
  - Même source que le nom (`get_country_map()`, 0 requête réseau), N/A tant que le profil n'est pas récupéré. Le tableau comparatif la recopie.

- **Nombres limités à 6 décimales à l'affichage**
  - Les valeurs calculées arrivaient en double précision et s'affichaient sur 15 chiffres : `Score/Seuil` à `1.312463256881238`, `dRSI` à `2.00000000001`, `dPrice` en notation scientifique `1.23456789e-05`. `formater_nombre()` affiche au plus `DECIMALES_MAX` (6) décimales, sans zéro inutile ni notation scientifique (`0.000012`), et les valeurs pleine précision restent disponibles pour le tri et les statistiques.

### ♻️ Interne

- **Disposition du tableau de résultats déclarée une seule fois**
  - `ui/main_window.py` : `MERGED_COLUMNS` (clé logique → en-tête) est la source unique, avec `MERGED_COL['clé']` en remplacement des index littéraux. Une vingtaine d'accès désignaient les colonnes par leur numéro (coloration, statistiques par domaine, recopie vers le tableau comparatif) ; insérer « Nom » les aurait tous décalés en silence. Le tableau comparatif dérive maintenant ses colonnes de la même source.
  - Verrouillé par `src/tests/test_results_table_columns.py` et `src/tests/test_screener_dialog.py` (8 tests, PyQt offscreen).

## Version 1.5.0 - Profils d'instruments : stockage partitionné et complétion progressive (2026-08-02)

### 🐛 Corrections

- **Les screeners store-only renvoyaient une sélection différente à chaque appel**
  - `src/core/store_screeners.py` : les vues Combined et Golden Cross trient puis tronquent à `MAX_ROWS`, mais leur clé de tri laissait de nombreux ex aequo (même profil, mêmes scores, ou même écart). Ces égalités étaient départagées par l'ordre de sortie de DuckDB, qui varie d'une exécution à l'autre : **5 symboles sur 80 changeaient entre deux affichages consécutifs**. Le symbole clôt désormais la clé de tri. Vérifié : sélection identique sur trois exécutions successives.
  - Conséquence visible : la colonne « Pays » ne se complétait jamais, puisque chaque affichage introduisait des symboles jamais rencontrés. La complétion progressive converge maintenant — 80 lignes sur 80 renseignées après deux passes.

- **Les profils d'instruments se perdaient entre processus**
  - `src/market_store.py` : `upsert_instrument()` écrit désormais dans un fichier par symbole (`instruments/symbol=XXX/part0.parquet`), comme les features. Il réécrivait auparavant un fichier unique partagé en entier à chaque appel — lire tout, remplacer une ligne, réécrire tout — sous la seule protection d'un `threading.Lock` local au processus. Les scripts `*_scan.py` et le worker en sous-processus écrivant en parallèle, le dernier écrivain gagnait : **124 profils subsistaient pour 1853 symboles** présents dans le store de features. La colonne « Pays » des screeners affichait N/A pour 94 % des lignes, et `sector`, `industry`, `exchange` et `market_cap` manquaient de la même façon.
  - Migration incluse via `migrate_instruments_to_partitioned()`, idempotente. L'ancien fichier n'est pas supprimé. Vérifié : `get_country_map()` et `_get_instrument_profile()` renvoient exactement les mêmes valeurs qu'avant sur les 124 symboles, sans aucun écart de champ.

### ✨ Nouveautés

- **Complétion progressive des profils**
  - `ensure_instrument_profiles(symbols, max_fetch, max_age_days)` complète les profils manquants ou périmés **par petits lots**. Appelée après l'affichage d'un screener, dans un thread détaché : la liste s'affiche immédiatement, et se complète pour la fois suivante.
  - Plafond configurable dans `src/config.py` : `INSTRUMENT_PROFILE_FETCH_LIMIT` (25) et `INSTRUMENT_PROFILE_MAX_AGE_DAYS` (90). Le budget de requêtes yfinance étant une contrainte dure, une liste de 500 résultats ne peut jamais déclencher 500 appels.
  - Interrupteur dédié `QSI_DISABLE_PROFILE_FETCH=1`. Volontairement distinct de `QSI_CONSENSUS_OFFLINE`, dont le sens est étroit (lookups de consensus) et que `ui/main_window.py` pose systématiquement au démarrage : s'appuyer dessus empêchait toute complétion dans l'application. Le conftest des tests pose l'interrupteur par défaut, pour qu'aucun test ne puisse consommer de requêtes.
  - Un symbole en échec n'interrompt pas les suivants et ne remonte jamais jusqu'à l'interface.
  - `missing_instrument_profiles()` liste les symboles à compléter sans aucune requête réseau.
  - `read_instruments()` lit tous les profils en une requête DuckDB avec `union_by_name=true`, ce qui tolère les fichiers écrits avant l'ajout des nouvelles colonnes : aucune migration de schéma n'est nécessaire.

- **Champs ajoutés au profil**
  - `financial_currency` : devise de publication des comptes, distincte de la devise de cotation. Un titre coté en HKD mais publiant en USD voyait ses fondamentaux convertis une fois de trop, `fx_rate_to_usd` étant dérivé de la cotation.
  - `beta` : lu par le screener Sichere Unternehmen (critère S3), qui devait le redemander à yfinance faute d'être stocké.
  - `float_shares` : flottant réel. `shares_outstanding` surestime la quantité négociable des titres à actionnariat concentré, ce qui fausse les critères de liquidité.
  - `first_trade_date` : début de l'historique disponible, pour savoir si une fenêtre de backtest est couverte.
  - `exchange_timezone`, et `isin` — ce dernier reste vide, yfinance ne le fournissant pas dans `.info` sans une requête supplémentaire par symbole.

### ✅ Tests

- Nouveau fichier `src/tests/test_instrument_profiles.py` (6 cas, hors ligne) : mode hors ligne, respect du plafond, non-rechargement d'un profil frais, rechargement d'un profil périmé, isolation des échecs, liste vide. yfinance y est simulé et `PARQUET_DIR` redirigé vers un répertoire temporaire — vérifié : le store réel reste bit à bit identique après exécution.
- Le sous-ensemble exécuté en intégration continue passe de 27 à 33 tests.

---

## Version 1.4.0 - Mises à jour techniques (2026-07-31)

### 🔧 Outillage et CI

- **Ajout de règles de linting ciblées**
  - `.github/workflows/tests.yml` : étape de lint ajoutée avant les tests. Trois règles `ruff` activées : `F821` (nom indéfini), `F811` (redéfinition masquant la précédente), `E9` (erreur de syntaxe). Ces règles sont volontairement limitées à des problèmes critiques, sans règles de style.

- **Corrections de redéfinitions F811**
  - `src/trading_c_acceleration/qsi_optimized.py` : deux redéfinitions corrigées (`Dict` et `Union` importés deux fois), qui bloquaient l'exécution de la nouvelle étape de linting.

### 🧹 Nettoyage

- **Corrections automatiques à grande échelle**
  - 102 corrections appliquées : suppression d'imports inutilisés, variables assignées sans utilisation, f-strings sans champ de substitution.

- **Mise à jour des annotations d'imports**
  - Les imports réellement ré-exportés sont désormais marqués `# noqa: F401` avec la raison. Trois emplacements de `src/qsi.py` étaient concernés : `_BoundedCache`, `backtest_signals`, et le bloc importé de `symbol_manager`. D'autres modules en dépendent, un nettoyage automatique les aurait supprimés.

- **Amélioration du test d'import**
  - `src/tests/test_cap_range.py` : le test vérifie désormais cinq noms importés au lieu de deux, assurant une couverture plus précise.

### 📌 Dépendances

- **Épinglage des versions critiques**
  - `requirements.txt` : quatre dernières dépendances épinglées à des versions spécifiques — `yfinance==1.2.2`, `curl_cffi==0.15.0`, `duckdb==1.5.2`, `lxml==6.1.1`. Cela empêche les mises à jour majeures non planifiées.

- **Fichier de dépendances verrouillé**
  - Nouveau fichier `requirements.locked.txt` : 54 paquets, 1198 hachages SHA-256 générés avec `uv pip compile --generate-hashes`. Contrôle d'intégrité réussi sans anomalie.

### 📝 Documentation du code

- **Clarification de la sécurité de `pickle.load()`**
  - `src/_subprocess_worker.py` : ajout d'une note expliquant pourquoi `pickle.load()` est sûr dans ce contexte (fichier temporaire `tempfile.mkstemp()` en 0600, écrit par le processus parent, lu par l'enfant). La note précise les cas où cette sécurité ne serait plus garantie.

- **Correction d'une référence inexacte**
  - `src/trading_c_acceleration/qsi_optimized.py` : un commentaire pointant vers des numéros de ligne obsolètes a été modifié pour référencer directement la variable concernée.

---

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

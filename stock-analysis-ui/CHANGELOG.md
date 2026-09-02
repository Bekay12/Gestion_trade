# 📋 Changelog - Stock Analysis Web Dashboard

## Version 1.8.1 - Optimisateur hybride : ce que le lot 2 avait laissé (2026-08-07)

Le lot 2 avait mesuré son gain en agrégat (7,18 s à ~0,58 s) sans profiler ce qui
restait. Un profil de l'évaluation à cache chaud montre que l'essentiel du temps
restant était encore du travail redondant. Mesure sur huit évaluations, deux
séries de 1210 barres, **gain, trades et gagnants identiques au bit près avant et
après** : moyenne à chaud **0,895 s → 0,057 s, soit 15,7x**.

### 🚀 Performances

- **`DERIV_CACHE` était resté dimensionné à 500, exactement le défaut corrigé sur son jumeau**
  - Il porte la même clé que `TA_CACHE` (`symbol`, dernier prix, dernier volume, `prices_len`) et la même volumétrie : 1160 entrées pour un seul symbole sur un backtest de 5 ans. À 500, il évinçait les premières barres avant réutilisation et son taux de réussite était nul, si bien que le repli reconstruisait le RSI complet (`ta.momentum.RSIIndicator` sur toute la tranche) **à chaque barre** dès que les features de prix étaient actives : 1160 reconstructions, 52 % du temps de l'évaluation au profil. Porté au même plafond de 100 000, et vidé aux deux mêmes frontières naturelles que `TA_CACHE`, dont il partage le cycle de vie. Effet seul sur un vecteur avec extras de prix : 0,858 s → 0,055 s.

- **Trois `pct_change` sur la tranche complète, une fois par barre, dont deux totalement morts**
  - `momentum_10` et `sharpe` (avec son intermédiaire `returns`, dont `.std()` était évalué deux fois) étaient calculés à chaque barre et **lus nulle part**. Supprimés, sans effet possible sur un résultat. Le troisième, `volatility`, est bien consommé (il fixe `m4` et multiplie le score) mais ne dépend que des prix : il rejoint l'instantané `TA_CACHE`, comme les variations et les moyennes de volume. Ces trois appels formaient le O(n²) restant, 3480 appels pour 57 % du temps d'une évaluation à cache chaud. Le NaN est préservé tel quel, `NaN > 0.05` étant faux.

- **Les deux EMA de repli étaient recalculées à chaque cache hit**
  - `snap.get(cle, defaut)` évalue `defaut` **avant** d'appeler `get()`. Écrites en argument par défaut, `prices.ewm(span=20).mean()` et `span=50` étaient donc recalculées sur toute la tranche à chaque barre, y compris quand le cache répondait : 2320 calculs inutiles par backtest. Le repli reste possible mais n'est plus payé que s'il sert.

### ✅ Tests

- Trois tests verrouillent `DERIV_CACHE` sur le modèle de ceux du lot 2 : plafond, rétention d'un backtest complet, et identité du résultat entre cache neutralisé et cache chaud avec les features de prix actives. Suite complète : 137 tests, hors réseau et hors base réelle.

## Version 1.8.0 - Optimisateur hybride : lot 2, coût de l'objectif (2026-08-06)

### 🚀 Performances

- **Le cache d'instantanés techniques était dimensionné à 500 entrées pour 1160 barres**
  - `TA_CACHE` mémoïse les indicateurs par barre et sa clé se répète bien d'une évaluation à l'autre, la série de prix ne changeant pas. Mais il retenait 500 instantanés quand un backtest de 5 ans en produit 1160 pour un seul symbole : il évinçait les premières barres avant d'avoir pu les réutiliser, et son taux de réussite était nul. Porté à 100 000 entrées, un plafond LRU qui pèse environ 196 Mo une fois plein (mesure au conteneur réel, environ 2,06 Ko par instantané, pas une allocation immédiate), il couvre 1160 barres pour une cinquantaine de symboles à environ 2,4 Mo par symbole réellement en cache. Effet de ce seul dimensionnement, mesuré sur la même série et à résultats identiques au bit près : la deuxième évaluation passe de 7,18 s à 2,36 s. Ces 2,36 s ont été mesurées à un plafond de 5 000, la seule autre valeur du banc ; tout plafond supérieur à 1160 donne le même taux de réussite pour un symbole, puisque le cache retient alors l'intégralité de ses barres, et 100 000 est retenu pour couvrir un groupe entier. Le chiffre à retenir pour l'utilisateur est celui des deux étages réunis, plus bas : environ 0,58 s.

- **`extract_best_parameters` interrogeait SQLite une fois par barre**
  - Soit 1160 requêtes par backtest, pour 14 % du temps, alors que la réponse ne change pas pendant un run. Elle est désormais mémoïsée avec une clé portant la date de modification et la taille du fichier de base, ce qui rend l'invalidation automatique : une écriture en base, même par un autre processus, invalide le cache d'elle-même. C'est ce qui lève l'objection du lot 1, qui avait écarté cette mémoïsation parce qu'elle aurait changé le comportement en cours de run.

- **Étage 3 (sortir les indicateurs de la boucle) remesuré et refermé sans objet**
  - La spec conditionnait l'ouverture de ce troisième étage à une remesure après les deux corrections ci-dessus. Remesuré sur le même script de référence (1210 barres, quatre évaluations), les deux étages ci-dessus étant cette fois actifs ensemble : la deuxième évaluation, celle du régime stable d'une optimisation qui enchaîne les jeux de coefficients sur la même série, passe de 7,18 s à environ 0,58 s, un gain d'environ 12,6x, très au-delà du seuil de 3x fixé pour justifier l'étage 3. Il n'est pas ouvert et ne fait l'objet d'aucun plan de suite ; voir la section « Étage 3 » de `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md`.

### 🐛 Corrections

- **`_BoundedCache` n'était pas sûr en multithread, et ses caches décident maintenant d'un résultat**
  - `get` et `__setitem__` testaient l'appartenance de la clé avant d'appeler `move_to_end`. L'optimisateur évalue les symboles sur un `ThreadPoolExecutor` imbriqué dans un second pour les populations : si un thread évince exactement cette clé entre les deux, `move_to_end` lève une `KeyError`. Elle ne remontait nulle part, car le `except Exception` par barre de `qsi_optimized.py` l'absorbait en ajoutant un signal 'NEUTRE' ; la panne se serait donc traduite par un résultat silencieusement différent, ce que ce lot s'interdit. Les deux fenêtres sont fermées par un garde local plutôt que par un verrou : le GIL rend déjà chaque opération d'`OrderedDict` atomique, le verrou n'aurait fermé que ces deux mêmes fenêtres, et mesuré sur 8 threads et 200 000 lectures chacun il coûte 1,618 s contre 0,272 s, soit 5,9 fois plus cher sur la structure la plus chaude du run. Deux tests posent l'entrelacement exact au lieu de l'espérer d'un test de charge.

- **Le cache d'instantanés n'avait aucun point de libération**
  - Porté à 100 000 entrées, il pesait jusqu'à environ 196 Mo immobilisés pour toute la vie du process, et le plafond est atteignable depuis l'interface graphique : le bouton « Analyser + Backtester » lance un backtest par symbole sur une période allant jusqu'à 10y, soit une quarantaine de symboles pour saturer, quand `popular_symbols.txt` en compte 3243. Sa clé portant le nom du symbole, les entrées d'un symbole déjà traité ne resservent jamais. Le cache est désormais vidé à deux frontières où l'ensemble de travail change entièrement : la fin de l'optimisation d'un groupe secteur × cap_range et la fin de la boucle de backtest de l'interface. Les groupes ne partagent pas de symboles, donc aucune entrée réutilisable n'est jetée.

- **La suite de tests par défaut lisait la base d'optimisation réelle**
  - `get_trading_signal` appelle `extract_best_parameters()` sans argument à chaque barre : tout backtest lancé depuis les tests ouvrait `signaux/optimization_hist.db`. La lecture seule protégeait la base, mais rendait la suite dépendante de l'environnement. `OPTIMIZATION_DB_PATH` rejoint `DB_PATH` et `MARKET_DATA_DB_PATH` dans la fixture d'isolation, redirigé vers une copie temporaire selon le même principe.

- **Le test d'identité ne couvrait que le premier étage**
  - Il neutralisait `TA_CACHE` mais laissait `_BEST_PARAMS_CACHE` chaud des deux côtés, si bien que la référence « froide » n'en était pas une pour l'étage 2. Les deux caches sont désormais neutralisés, et l'assertion « aucun résultat ne change » porte enfin sur les deux étages.

### ⚠️ Connu, non traité dans ce lot

- Le module C reste une implémentation divergente de la stratégie, hors du chemin d'optimisation. À seuils alignés, les deux moteurs rendent des résultats de signe opposé sur trois séries sur trois, sept divergences structurelles l'expliquant, dont Ichimoku absent du C et un ADX calculé une seule fois pour toute la série. Voir `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md`.
- `a3` et `th_score` sont deux dimensions inertes de l'espace de recherche, prouvées telles par lecture du code et par mesure. Les retirer ferait passer le vecteur de 14 à 12 dimensions.

## Version 1.7.0 - Optimisateur hybride : lot 1, démarrage et cohérence (2026-08-05)

### 🐛 Corrections

- **Le CLI de l'optimisateur mourait à l'import**
  - `trading_c.cpython-310-x86_64-linux-gnu.so` avait été compilé le 2026-04-18 avec AddressSanitizer (`QSI_DEBUG_C_MODE=1 QSI_USE_ASAN=1`). Son chargement ne lève pas, il **avorte le processus**, donc le `try/except` de `_diagnose_import` était impuissant et `python optimisateur_hybride.py` mourait sans message. Un garde-fou inspecte désormais le binaire avant tout `dlopen` et se replie sur le chemin Python, et le module a été recompilé avec les drapeaux de production.

- **Les 4 seuils optimisés n'avaient aucun effet mais partaient en production**
  - `evaluate_config` calculait les seuils RSI, Volume, ADX et Score puis appelait `backtest_signals_c_extended`, qui n'a aucun paramètre de seuils : `py_backtest_symbol` ne prend que `(prices, volumes, coeffs, montant, cost)`. Quatre des quatorze dimensions étaient donc du bruit, et ces valeurs jamais évaluées étaient sauvegardées en `th1`, `th4`, `th5`, `th8` puis appliquées aux signaux réels. L'objectif passe par `backtest_signals_with_events` avec `domain_thresholds`, pour un surcoût mesuré de 1 %.

- **Les `trades` et le `success_rate` sauvegardés venaient d'une autre configuration**
  - La sauvegarde lisait `optimizer.meilleur_trades`, qui suit la meilleure configuration jamais vue, pas le vecteur écrit sur la même ligne. Sous `workers=-1` ces compteurs restaient dans les sous-processus, si bien que `strategy='differential'` sur un groupe sans historique n'écrivait **jamais** rien. Les métriques voyagent désormais avec leur vecteur.

- **Trois plages concurrentes par paramètre, quatre divergences**
  - `core/optim_params.py` devient la seule description du vecteur. Exemple : `a_price_slope` était cherché sur (-1.5, 3.0), bridé à (-0.5, 3.0) à l'évaluation et à (0.0, 3.0) à la sauvegarde. `th_score`, non relevé par l'audit, était cherché sur (2.0, 6.0) et bridé à (1.0, 6.0).

- **Le budget d'évaluations était ignoré d'un facteur 19**
  - `popsize` de SciPy est un multiplicateur : la population vaut `popsize * dimension`. Passer 200 avec 36 dimensions donnait 7 200 individus par génération, soit environ 1,45 million d'évaluations par groupe au lieu des 75 000 visés. Le budget est désormais calculé, réparti entre stratégies en mode `hybrid`, et le menu affiche la valeur réellement utilisée, contre 3 500 annoncés pour 30 000 utilisés.

- **Le croisement génétique produisait des individus hors bornes**
  - BLX-α étend l'intervalle parental sans borner ; seule la mutation bornait, et avec 10 % de probabilité par gène. Le meilleur individu retourné pouvait donc sortir du domaine et être sauvegardé tel quel.

- **Coût de transaction incohérent d'un facteur 50**
  - C'est un montant absolu par trade et non un pourcentage, contrairement au docstring. Les deux défauts contradictoires, `1.0` dans la fonction et `0.02` passé par le CLI, sont remplacés par une constante unique à `1.0`, et une colonne `transaction_cost` dit désormais dans quel monde chaque ligne a été mesurée.

- **La relecture des lignes historiques d'avant la colonne `use_price_extras` perdait ou faussait leurs features de prix**
  - La base réelle (`signaux/optimization_hist.db`, 545 lignes, 43 colonnes) ne porte pas la colonne `use_price_extras` : `depuis_colonnes()` retombait sur le milieu des bornes pour un drapeau absent, ce qui l'arrondit à 1 (activé). Les 545 lignes auraient donc été rejouées avec les features de prix activées et leurs 3 extras les plus récents (`a16..a18`, `th16..th18`) au milieu de leurs bornes, soit des configurations qu'aucun run n'a jamais évaluées. Un drapeau de feature absent d'une ligne historique vaut désormais 0 (désactivé), jamais le milieu des bornes.
  - Séparément, 185 de ces 545 lignes portent un ancien drapeau par feature (`use_price_slope`, `use_price_acc`) que le vecteur courant ne modélise plus. Après arbitrage de l'utilisateur, leur rejeu restaure `use_price_extras=1` et reporte ces anciens poids sur `a_price_slope`/`a_price_acc`, plutôt que de les faire retomber sur les 14 paramètres de base comme l'aurait fait une relecture stricte du seul contrat courant.

- **Le baseline historique était mesuré par un autre moteur que le score auquel il était comparé**
  - Une fois l'objectif basculé sur `backtest_signals_with_events`, `hist_avg_gain`, `hist_total_trades` et `hist_success_rate` venaient encore d'une boucle `backtest_signals_c_extended`, qui n'a aucun paramètre de seuils et ignore donc les 8 seuils, les 2 seuils globaux et les deux dictionnaires d'extras. La ligne de fin de groupe, `✅ {domaine}: gain 22.94 vs 5.10`, opposait ainsi un backtest entièrement paramétré à un backtest de coefficients seuls, et le résumé repris dans le rapport final portait le même défaut. La décision de sauvegarde, elle, n'était pas touchée : elle compare `hist_objective_score`, déjà mesuré par `evaluate_config`. Le baseline passe désormais par `optimizer.mesure_de()`, et `backtest_signals_c_extended` n'est plus importé par le module.

- **Une quatrième description du vecteur avait survécu, la plus longue**
  - Le bloc qui reconstruit le vecteur historique posait à la main 8 coefficients, 4 seuils, 2 globaux, 11 extras de prix puis 11 extras fondamentaux, dans un ordre littéral, alors que `core/optim_params.py` prévient que cet ordre EST celui du vecteur. Il dérive maintenant de `params.indices()` et lève si un emplacement reste sans valeur, au lieu de laisser passer un zéro silencieux. Un test vérifie que chaque valeur atterrit à l'index annoncé par le contrat.

### 🔁 Conséquence sur l'historique déjà en base

- **76 des 545 lignes de `signaux/optimization_hist.db` rejouent désormais différemment**
  - C'est le comportement voulu, rendu visible pour la première fois parce que les 4 seuils mordent enfin. Deux causes, mesurées en lecture seule sur la base réelle. D'abord, 48 lignes portent `NULL` dans `th1`, `th4`, `th5` et `th8` ; de ces quatre seuils, seul `th_vol` voit son défaut de rejeu bouger, de 1.0 vers le milieu de ses bornes, 1.5. Ensuite, `contraindre()` bride ce qui sortait du domaine déclaré : 48 lignes ont un `seuil_vente` hors des bornes (-6.0, -1.0), 3 un `seuil_achat` montant jusqu'à 46.4, et 2 un `th_score` descendu à 0.5.
  - Ces ensembles se recoupent. Le décompte distinct : 48 lignes au titre du défaut de `th_vol`, 26 lignes de plus au titre du `seuil_vente`, 2 de plus au titre du `seuil_achat`, 0 de plus au titre du `th_score`, soit 76 lignes sur 545.
  - Les colonnes `transaction_cost` et `seed`, nouvelles, distinguent une ligne écrite avant le lot d'une ligne écrite après : une ligne qui ne les porte pas a été mesurée dans l'ancien monde.

### ✨ Nouveautés

- **Runs rejouables** : la graine devient explicite, amorce `numpy` et `random`, et est stockée dans une colonne `seed`.
- **Contrat de paramètres testable** : `core/optim_params.py` et `core/optim_budget.py`, verrouillés par `test_optim_params.py` et `test_optim_budget.py`, soit 30 fonctions de test pour 47 cas collectés, hors réseau. Le module n'en avait aucun. En y ajoutant `test_optim_sauvegarde.py` et `test_c_module_guard.py`, le lot apporte 47 fonctions de test pour 64 cas collectés, et la suite par défaut (`pytest -m "not integration"`) est à 123 passed, 15 deselected.

### ♻️ Interne

- Suppression du `warnings.filterwarnings("ignore")` de niveau module. Hors pytest, il rendait muet tout avertissement Python émis par le processus après l'import du module, GUI comprise, effet mesuré et confirmé (avant/après, avec un avertissement de contrôle). Sous pytest, l'effet était nul : pytest encadre chaque test de son propre `warnings.catch_warnings()` et réinitialise ses filtres, si bien que le filtre global posé à l'import n'a jamais influencé le résumé d'avertissements de la suite (39 avertissements tiers mesurés à l'identique avant et après, sur la suite ciblée comme sur la suite complète).
- Le cache secteur prend `config.CACHE_DIR`, absolu, et crée son dossier à l'écriture. Trois dossiers `cache_data/` parasites existaient, dont un créé par une simple exécution de pytest.
- Le pool de threads est créé une fois par optimiseur au lieu d'être reconstruit à chaque appel d'objectif, et `workers=-1` passe à `workers=1`, ce qui supprime la sur-souscription et la sérialisation des séries à chaque génération.
- Suppression de `get_best_gain_csv`, morte, et des imports dupliqués.
- `save_optimization_results` tient enfin la promesse de son docstring de ne rien propager à l'appelant : `params.vers_colonnes()` et la composition des colonnes sont passées à l'intérieur du `try`, alors qu'un vecteur de mauvaise taille y aurait fait remonter une `ValueError`.

### ⚠️ Connu, non traité dans ce lot

- **Régression de performance assumée en ligne de commande.** Rendre les 4 seuils effectifs a imposé d'épingler l'objectif sur `backtest_signals_with_events`, le moteur C n'ayant aucun paramètre de seuils. Mesuré le 2026-08-06 sur 1210 barres : 7,17 s pour le chemin Python contre 0,0002 s pour `backtest_signals_c_extended` avec le module C, soit un facteur 37 000. L'interface graphique n'est pas concernée, elle pose `QSI_DISABLE_C_ACCELERATION=1` et empruntait déjà le chemin Python. Le compromis est assumé : garder le moteur C laissait 4 des 14 dimensions inertes à l'optimisation tout en écrivant ces valeurs en base, où elles pilotaient les signaux réels.
- Un run complet reste donc hors de portée. Le profil situe précisément le coût : 96 % du temps part dans `get_trading_signal`, appelé une fois par barre, dont l'ADX recalculé intégralement à chaque barre (47 %), `extract_best_parameters` et ses 1160 requêtes SQLite (14 %) et le RSI (7 %). C'est un O(n²) : sortir ces calculs de la boucle ne change aucun résultat, seulement le temps. Objet du lot 2.
- L'évolution différentielle est passée de `workers=-1` à `workers=1`, pour que les mesures enregistrées par `evaluate_config` restent accessibles au processus parent. Restaurer le parallélisme demanderait de rendre l'optimiseur sérialisable, ce que le `ThreadPoolExecutor` porté par l'instance empêche aujourd'hui. Sans intérêt tant que l'objectif coûte 37 000 fois trop cher.
- **Sortie prévue, mesurée.** Sur les 4 seuils rendus effectifs, un seul pèse : `th_rsi` fait varier le gain de 73 points sur une même série (de +43,6 à −29,6), `th_vol` de 3 à 4, `th_adx` de moins de 1,4, et `th_score` de **exactement 0**, étant écrasé par `seuil_achat` (`qsi.py:879-890`). Le lot 2 descendra donc `th_rsi` et `th_vol` dans le module C, dont `backtest.c:48-50` code trois constantes RSI en dur, puis rebasculera l'objectif sur `backtest_signals_c_extended` : justesse et vitesse à la fois. `th_score` sortira de l'espace de recherche, 14 dimensions à 13. Constats détaillés dans `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-constats.md`.
- `get_sector` et `classify_cap_range` consomment toujours une requête yfinance par symbole. Lot 3.

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

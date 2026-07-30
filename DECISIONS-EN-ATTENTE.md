# Décisions en attente

Points relevés pendant l'audit et les correctifs qui **ne relèvent pas d'un choix technique
évident** : ils dépendent de ton intention, de ton usage réel de l'outil, ou engagent des
conséquences que je ne peux pas arbitrer à ta place.

Rien de ce qui suit n'a été fait. Chaque entrée décrit le constat, les options, et ce que je
recommanderais — mais la décision reste ouverte.

Dernière mise à jour : 2026-07-30.

---

## 0. Travail restant — ne demande pas ton avis, demande du temps

**M-04, extraction de la couche réseau yfinance.** Pas fait. Ce n'est pas un arbitrage :
c'est un chantier que je n'ai pas mené au bout dans cette session.

Les trois scripts `Big_Growth_scan.py`, `Sichere_Unternehmen_scan.py` et `Combined_scan.py`
partagent 18 noms de fonctions identiques, dont toute la couche réseau : `_throttle`,
`_fetch_history_with_retry`, `_fetch_info_with_retry`, `_is_rate_limit_error`,
`_is_valid_history`, `_is_valid_info`, `_get_fast_info_snapshot`, `analyze_safe`,
`load_symbols`, `run_scan`, `print_summary`.

C'est précisément le code qui gère ton budget de requêtes yfinance — en trois exemplaires aux
constantes divergentes. Un ajustement du backoff après un blocage doit être répété trois fois,
et en oublier un suffit à faire bannir l'IP.

**Plan proposé :** extraire dans `core/yf_fetch.py` le throttle, le backoff, la détection de
rate-limit et la validation de réponse ; les trois scripts ne gardent que leurs critères
propres. Une seule constante de backoff, un seul endroit à ajuster.

**Pourquoi je m'arrête là :** ces trois scripts n'ont aucun test. Refactoriser du code réseau
non testé en fin de session, sans pouvoir vérifier chaque script bout à bout, c'est
exactement le genre de changement qui casse en silence — et le premier symptôme serait un
bannissement yfinance. À reprendre au début d'une session, avec des tests d'abord.

---

## 1. Deux tests en échec dont je ne connais pas l'intention

### 1.1 `test_best_param_extras_loading`

**Constat.** Le test exige que le dictionnaire retourné contienne la clé `use_price_slope` :

```
AssertionError: Missing price param: use_price_slope
assert 'use_price_slope' in {'use_price_extras': 0, 'a_price_slope': 0.0,
                             'a_price_acc': 0.0, 'th_price_slope': 0.0, ...}
```

Le code produit `use_price_extras`, `a_price_slope`, `th_price_slope` — mais pas
`use_price_slope`. Dans `qsi_optimized.py`, `use_price_slope` n'apparaît que comme *nom de
colonne lue en base*, pas comme clé de sortie.

**Deux lectures possibles, opposées :**

- Le **test décrit un contrat abandonné** : les drapeaux par indicateur ont été fusionnés en
  un seul `use_price_extras`. Le test n'a pas suivi → il faut le mettre à jour.
- Le **code a régressé** : chaque indicateur devait garder son propre drapeau, et la fusion
  est une perte de granularité → il faut restaurer les clés.

Je ne peux pas trancher : les deux sont cohérentes avec le code tel qu'il est. La réponse
dépend de si tu voulais pouvoir activer la pente de prix **indépendamment** de l'accélération
et du RSI, ou si un drapeau global suffisait.

**Question :** les 5 sous-indicateurs de prix doivent-ils être activables séparément ?

### 1.2 `test_optimizer_bounds` et `test_price_features`

**Constat.** `ModuleNotFoundError: No module named 'optimisateur_hybride'`. Ce module
n'existe nulle part dans le dépôt. Il fait échouer `test_optimizer_bounds` et met
`test_price_features` en `SKIPPED` permanent depuis le début.

**Options :**

- Le module existe ailleurs (autre branche, machine, dossier non versionné) → le rapatrier.
- Le module a été abandonné → supprimer les deux tests, qui mesurent alors une couverture
  fictive.

**Recommandation :** si tu ne le retrouves pas rapidement, supprimer. Un test qui se saute en
permanence ment sur la couverture, et c'est précisément ce qui a permis à la CI de ne
valider que 4 tests sans que cela se voie.

---

## 2. Purge de l'historique git

**Constat mesuré.** `.git` pèse **421 Mo** (272 Mo empaquetés) pour ~13 000 lignes de Python.
L'historique contient 99 blobs `.db`, dont **62 versions de `stock_analysis.db` à ~12 Mo**
chacune, et un PDF de 33 Mo. SQLite étant binaire, deux versions ne se compressent pas entre
elles : chaque commit a ajouté ~12 Mo définitifs.

Les 57 fichiers suivis à tort ont été détachés (fait), ce qui **arrête l'hémorragie** mais ne
récupère rien : le poids est dans l'historique, pas dans l'arbre de travail.

**Option A — ne rien faire.** Le dépôt reste à 421 Mo. Chaque clone frais télécharge 273 Mo.
Aucun risque.

**Option B — purger.**

```bash
git filter-repo --path-glob '*.db' --path-glob '*.log' --invert-paths
```

Gain estimé : ~250 Mo.

**Conséquences à peser :** tous les SHA sont réécrits. Tes 6 branches distantes
(`Version-amelioree`, `Prioritee-taux-de-reussite`, `Version_amelioree_de_priorite_taux_de_reussite`,
`copilot/add-streamlit-interface`, `robuste/vps-docker`, `master`) doivent être force-push.
Toute copie locale ailleurs devient incompatible. À faire sur un clone de secours d'abord.

**Recommandation :** ne le faire que si le poids te gêne réellement (clone lent, quota).
Sinon l'option A est raisonnable — le mal est fait, il ne s'aggrave plus.

---

## 3. Les artefacts de `Results/`

**Constat.** `stock-analysis-ui/src/Results/` contient des sorties générées **suivies
délibérément** (elles ne correspondent à aucune règle du `.gitignore`) : un PDF de 33 Mo, un
de 2,2 Mo, et une trentaine de PNG `temp_graph_*.png` de ~850 Ko chacun.

Le préfixe `temp_` suggère des fichiers temporaires ; ils sont pourtant versionnés.

**Question :** ces sorties ont-elles une valeur d'archive (comparer une analyse de février à
une d'aujourd'hui), ou sont-elles régénérables à volonté ?

- **Régénérables** → les détacher du suivi et ajouter `Results/` au `.gitignore`.
- **Archive** → les garder, mais peut-être déplacer les `temp_graph_*.png` hors de `Results/`,
  puisque ceux-là sont clairement jetables.

Je ne les ai pas touchés.

---

## 4. Numérotation de version incohérente

**Constat.** Le `CHANGELOG.md` va maintenant jusqu'à **1.2.0**, mais `src/api.py` code encore
`'version': '1.0.0'` en dur à trois endroits (routes `/health`, `/status`, `/api/docs`).

Le premier numéro (1.1.0) a été proposé par l'agent de rédaction, pas par toi.

**Options :**

- Aligner `api.py` sur `1.2.0` — et accepter que la version de l'API suive celle du CHANGELOG.
- Découpler : l'API a sa propre version, le CHANGELOG décrit l'application desktop.
- Lire la version depuis une source unique (`config.py` ou `importlib.metadata`).

**Recommandation :** la troisième. Trois littéraux en dur finiront par diverger.

---

## 5. Fichiers d'IDE détachés du suivi

**Constat.** Parmi les 57 fichiers détachés se trouvaient `stock-analysis-ui/.vscode/launch.json`,
`.vscode/tasks.json` et `src/.vscode/tasks.json`. Ils correspondaient à la règle `.vscode/` du
`.gitignore`, donc l'intention affichée du projet était de les ignorer.

Certaines équipes versionnent pourtant `launch.json` volontairement, pour partager les
configurations de lancement.

**Si tu les veux de nouveau suivis :**

```bash
git add -f stock-analysis-ui/.vscode/launch.json stock-analysis-ui/.vscode/tasks.json
```

Les fichiers sont toujours sur le disque, rien n'est perdu.

---

## 6. Découpage de `get_trading_signal` (M-03)

**Constat mesuré.** Complexité cyclomatique **204** pour `get_trading_signal`
(`qsi.py`), 161 pour `analyse_signaux_populaires`, 99 pour
`market_store._build_daily_feature_frame`. Indice de maintenabilité `radon` à **0,00** (rang C,
plancher de l'échelle) pour `qsi.py`, `market_store.py` et `ui/main_window.py`.

Une CC de 204 signifie plus de 200 chemins d'exécution indépendants dans la fonction qui
décide d'acheter ou de vendre — et **aucun n'est couvert par un test**.

**Pourquoi je ne l'ai pas fait.** Refactoriser cette fonction sans test de caractérisation
préalable, c'est parier. Le test de caractérisation lui-même est un chantier : il faut figer
le comportement actuel sur un jeu de séries de prix représentatif, ce qui suppose de décider
**quel comportement est le bon** — or une partie des 204 chemins encode probablement des
correctifs empiriques dont toi seul connais la raison.

**Ce que je propose, si tu veux l'attaquer :**

1. Enregistrer les entrées/sorties réelles de `get_trading_signal` sur une centaine de
   symboles (un fichier de fixtures).
2. Écrire un test qui rejoue ces fixtures et vérifie que la sortie ne bouge pas.
3. Extraire alors par blocs vers `core/signals.py` : d'abord le calcul de score (pur), puis
   les seuils, puis les fondamentaux.

C'est plusieurs sessions de travail, et l'étape 1 exige ton arbitrage sur le choix des
symboles témoins. À décider avant de commencer, pas en cours de route.

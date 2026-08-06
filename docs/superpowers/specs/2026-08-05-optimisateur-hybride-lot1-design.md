# Optimisateur hybride, lot 1 : démarrer et ne plus mentir

Date : 2026-08-05
Fichier concerné : `stock-analysis-ui/src/optimisateur_hybride.py` (2 128 lignes)
Statut : design validé, prêt pour le plan d'implémentation

## 1. Constat

L'audit du 2026-08-05 a relevé quatorze défauts. Deux autres, plus graves, sont
apparus pendant le brainstorming, par la mesure.

### 1.1 Le module ne démarre pas

`trading_c.cpython-310-x86_64-linux-gnu.so` a été compilé le 2026-04-18 en mode
debug avec AddressSanitizer (`nm -D` y trouve douze symboles `asan`). Son
chargement provoque un abort du processus :

```
ASan runtime does not come first in initial library list
```

Comme `optimisateur_hybride.py` importe `trading_c_acceleration.qsi_optimized` à
la ligne 16, `python optimisateur_hybride.py`, son unique point d'entrée, meurt à
l'import sans le moindre message Python. Seule la variable
`QSI_DISABLE_C_ACCELERATION=1` évite l'abort, en sautant l'import.

L'ASan est opt-in dans le build (`QSI_DEBUG_C_MODE=1` et `QSI_USE_ASAN=1`,
`setup.py` lignes 8 à 21). C'est un artefact de debug oublié, pas un choix.

### 1.2 L'objectif coûte 7,6 secondes

Profil d'un backtest unique sur 1 260 barres, cinq ans, chemin Python :

| Poste | Coût |
|---|---|
| `backtest_signals_with_events` total | 16,0 s sous profileur, 7,6 s sans |
| dont `get_trading_signal`, appelé **1 210 fois**, une fois par barre | 15,5 s |
| dont `ta.ADXIndicator` reconstruit 1 210 fois | 7,6 s |
| dont 2,3 millions d'accès `pandas.Series.__getitem__` | 5,6 s |
| dont `extract_best_parameters()`, **1 210 requêtes SQLite** | 2,2 s |

L'objectif est en O(n²) avec une requête base par barre. À dix symboles par
groupe et 75 000 évaluations, un groupe demande environ dix-huit jours. Avec le
défaut B4 décrit plus bas, environ un an.

Conséquence de méthode : corriger la justesse sans corriger ce coût produirait un
optimisateur juste et inutilisable. Le travail est donc découpé en trois lots.

### 1.3 Mesure qui simplifie le reste

Avec le C indisponible, `backtest_signals_c_extended` retombe déjà sur
`backtest_signals_with_events` (`qsi_optimized.py` ligne 347). Mesuré sur la même
série, faire honorer les quatre seuils coûte **1 %** :

| Appel | Coût |
|---|---|
| `c_extended`, repli Python, seuils ignorés | 7 618 ms |
| `with_events` avec `domain_thresholds` | 7 550 ms |

Rendre les seuils effectifs est donc gratuit, et n'attend pas le lot 2.

## 2. Découpage

| Lot | Objet | Contenu |
|---|---|---|
| **1, cette spec** | L'optimisateur démarre et ce qu'il écrit est ce qu'il a mesuré | N1, B1, B2, B3, B4, S1, S5, reproductibilité, hygiène, tests |
| 2 | Évaluateur vectorisé | N2 : features précalculées une fois par symbole, évaluation en tableaux |
| 3 | Hygiène de données | S3 budget yfinance vers le store, reste de S4, dette résiduelle |

## 3. Périmètre du lot 1

### 3.1 Traitement de chaque défaut

| Réf | Défaut | Traitement |
|---|---|---|
| N1 | `.so` ASan, abort à l'import | Garde-fou avant `dlopen` + recompilation propre |
| B1 | Les 4 seuils optimisés n'ont aucun effet | Objectif épinglé sur `with_events` avec `domain_thresholds` |
| B2 | `trades` et `success_rate` d'un autre vecteur | Mesure indexée par vecteur, réévaluation si absente |
| B3 | Trois plages par paramètre, quatre instances | `core/optim_params.py`, source unique |
| B4 | `popsize` mal interprété, budget ignoré | Effectif calculé, `maxiter` plafonné, budget réparti |
| S1 | Croisement du GA hors bornes | `contraindre()` en sortie de croisement |
| S5 | Coût de transaction incohérent | Constante unique + colonne `transaction_cost` en base |
| M1 | `filterwarnings("ignore")` global | Supprimé, réduction locale si besoin |
| M2 | Runs non reproductibles | Graine explicite, journalisée, stockée |
| M3 | Budget affiché ≠ budget utilisé | `budget_effectif()` partagée menu et lancement |
| M4 | Assertions de test périmées | Remplacées par les valeurs réelles |
| M5 | Dette diverse | Voir section 7 |
| S4, moitié | `mkdir` à l'import, chemin relatif | `config.CACHE_DIR`, création à l'écriture |

### 3.2 Hors périmètre, explicitement

- Une optimisation complète qui finit en temps raisonnable. À 7,6 s par backtest
  un groupe reste hors de portée ; c'est l'objet du lot 2.
- La levée de `QSI_DISABLE_C_ACCELERATION=1` dans l'interface. Elle mérite sa
  propre vérification, `-march=native` et `-ffast-math` rendant le binaire non
  portable et pouvant décaler les flottants par rapport au chemin Python.
- Les requêtes yfinance par symbole de `get_sector` et `classify_cap_range`,
  reportées au lot 3.
- L'extension du moteur C aux huit seuils. Le chemin Python suffit, et le lot 2
  rendra la question sans objet.

## 4. Architecture

### 4.1 `core/optim_params.py`, le contrat unique

Nouveau module, seule description du vecteur de paramètres. Motif : B3 a quatre
instances parce que chaque plage est écrite trois fois, dans les bornes de
recherche, dans le bridage à l'évaluation, dans le bridage à la sauvegarde, sans
rien qui les tienne synchronisées.

```python
@dataclass(frozen=True)
class SpecParam:
    cle: str                    # 'a1', 'th_rsi', 'seuil_achat'…
    bornes: tuple | None        # (min, max), None si gelé
    colonne_db: str             # 'a1', 'th1', 'seuil_achat'…
    gele: float | None = None   # valeur figée, absente du vecteur de recherche
```

Contenu déclaré :

| Groupe | Paramètres | Bornes | Colonnes |
|---|---|---|---|
| Coefficients | `a1` à `a8` | (-1.5, 3.0) | `a1` à `a8` |
| Seuils optimisés | `th_rsi` | (30.0, 70.0) | `th1` |
| | `th_vol` | (0.5, 2.5) | `th4` |
| | `th_adx` | (15.0, 35.0) | `th5` |
| | `th_score` | (2.0, 6.0) | `th8` |
| Seuils gelés | `th_macd`, `th_ema`, `th_ichimoku` | gelés à 0.0 | `th2`, `th3`, `th6` |
| | `th_boll` | gelé à 0.5 | `th7` |
| Globaux | `seuil_achat` | (1.0, 6.0) | `seuil_achat` |
| | `seuil_vente` | (-6.0, -1.0) | `seuil_vente` |
| Extras prix, 11 | drapeau, 5 poids, 5 seuils | celles des lignes 542 à 554 | `use_price_extras`, `a9`, `a10`, `a16` à `a18`, `th9`, `th10`, `th16` à `th18` |
| Extras fondamentaux, 11 | drapeau, 5 poids, 5 seuils | celles des lignes 559 à 571 | `use_fundamentals`, `a11` à `a15`, `th11` à `th15` |

Les bornes retenues sont celles de la recherche, `self.bounds` lignes 527 à 572,
qui sont les plus larges et les seules que les optimiseurs explorent réellement.
Les valeurs de bridage plus étroites présentes aujourd'hui aux lignes 644 à 701
et 1564 à 1617 sont abandonnées : elles créaient des plateaux et faisaient
diverger le vecteur sauvegardé du vecteur évalué.

Interface publique :

```python
def bornes(prix: bool, fond: bool) -> list[tuple]
def contraindre(v, prix: bool, fond: bool) -> np.ndarray   # le seul clip du projet
def indices(prix: bool, fond: bool) -> dict[str, int]      # remplace les offsets 14 et 25
def vers_colonnes(v, prix: bool, fond: bool) -> dict[str, float]   # gelés inclus
def depuis_colonnes(row, prix: bool, fond: bool) -> np.ndarray
```

`contraindre()` est appelée à trois endroits et à ces trois endroits seulement :
à l'entrée de `evaluate_config`, en sortie du croisement du GA, ce qui règle S1,
et avant la sauvegarde. Les trois donnent le même résultat par construction.

`indices()` supprime `fundamentals_index_offset = 25 if use_price_features else 14`.
`vers_colonnes()` et `depuis_colonnes()` remplacent la correspondance SQL écrite à
la main deux fois, dans `save_optimization_results` et à l'envers dans
`replay_all_historical`.

### 4.2 Quatrième instance de B3, trouvée pendant le design

`th_score` a pour bornes `(2.0, 6.0)` ligne 532 mais est bridé à `(1.0, 6.0)`
ligne 617. Le segment `[1.0, 2.0]` est atteignable à la sauvegarde et jamais
exploré. L'audit ne l'avait pas relevé.

## 5. Exécution correcte

### 5.1 B2, les métriques suivent leur vecteur

`evaluate_config` continue de renvoyer un score, l'interface des optimiseurs ne
change pas, mais enregistre au passage une mesure indexée par le vecteur
contraint :

```python
Mesure = namedtuple('Mesure', 'score gain_moyen trades gagnants')
```

À la sauvegarde, on lit la mesure du vecteur gagnant. Si elle est absente, cas du
DE en sous-processus, on réévalue une fois en processus courant. Les `trades` et
le `success_rate` écrits en base appartiennent donc toujours aux coefficients de
la même ligne.

`meilleur_score`, `meilleur_trades` et `meilleur_success` ne servent plus qu'à
l'affichage tqdm. Ils ne décident plus rien et ne sont plus sauvegardés. Cela
supprime aussi le cas où `strategy='differential'` sur un groupe sans historique
n'écrivait jamais rien, parce que `meilleur_trades` restait à zéro.

Coût : au plus une évaluation supplémentaire par groupe.

### 5.2 Parallélisme

`workers=-1` devient `workers=1` dans l'appel à `differential_evolution`, et le
`ThreadPoolExecutor` par symbole est hissé au niveau de l'instance, créé une fois
et fermé à la fin, au lieu d'être reconstruit à chaque appel d'objectif.

Deux raisons. L'état de l'optimiseur redevient fiable, ce dont 5.1 dépend, et la
sur-souscription de N processus par 12 threads disparaît, ainsi que la
sérialisation de toutes les séries de prix à chaque génération. La vitesse ne
viendra pas de là mais du lot 2 ; avec un objectif à 7,6 s un pool de processus
ne sauve rien.

### 5.3 B4, le budget devient une contrainte

`popsize` est un multiplicateur chez SciPy : la population vaut
`popsize * dimension`. Avec `popsize=200` et 36 dimensions, cela fait 7 200
individus par génération, donc environ 1,45 million d'évaluations par groupe au
lieu des 75 000 visés.

Correction :

- l'effectif est calculé depuis le budget et la dimension ;
- `maxiter` est plafonné pour que `population * (maxiter + 1) <= budget` ;
- chaque stratégie déclare son nombre d'évaluations prévu ;
- en mode `hybrid` le budget est réparti entre DE, PSO et LHS au lieu que chacune
  prenne le budget entier ;
- une seule fonction `budget_effectif()` sert au menu et au lancement, ce qui
  supprime l'écart entre les 3 500 affichés ligne 1965 et les 30 000 réels
  ligne 2050 ;
- le nombre prévu est journalisé avant de démarrer.

### 5.4 S5, coût de transaction

Le moteur soustrait un montant absolu par trade,
`profit = (close - entry) / entry * montant - transaction_cost`
(`qsi_optimized.py` ligne 726), alors que son docstring annonce un pourcentage.
Le docstring est corrigé.

Les deux défauts contradictoires, `1.0` dans `optimize_sector_coefficients_hybrid`
ligne 1135 et `0.02` passé par le CLI ligne 2079, sont remplacés par une constante
nommée unique. Une colonne `transaction_cost` est ajoutée à `optimization_runs`,
pour que chaque ligne dise dans quel monde elle a été mesurée ; les 545 lignes
existantes gardent la valeur nulle, qui signifie inconnue.

Décision retenue, à confirmer en relecture : valeur par défaut `1.0`, soit un
dollar par trade sur une position de cinquante dollars. `0.02`, deux centimes,
flatte mécaniquement les configurations qui multiplient les trades, ce que la
pénalité d'efficacité de 0,02 par trade ne compense pas. La colonne rend le choix
réversible sans perdre la comparabilité de l'historique.

### 5.5 Reproductibilité

`seed=np.random.randint(0, 10000)` ligne 904 devient un paramètre explicite,
journalisé et stocké dans une colonne `seed`. La graine amorce aussi le module
`random`, utilisé par le GA, le PSO et `clean_sector_cap_groups`. Un run contesté
peut être rejoué à l'identique.

## 6. Build

### 6.1 Garde-fou

Avant tout `dlopen`, le chargeur inspecte le fichier `.so` et refuse un binaire
instrumenté, en cherchant `__asan_` dans ses octets sans le charger. Message
explicite, repli sur le chemin Python. Un artefact de debug oublié ne pourra plus
abattre le processus.

### 6.2 Recompilation

`python setup.py` depuis la racine, drapeaux par défaut
`-O3 -march=native -ffast-math -funroll-loops`, sans `QSI_DEBUG_C_MODE` ni
`QSI_USE_ASAN`.

### 6.3 Épinglage de l'objectif, et B1

Une fois le C actif, `backtest_signals_c_extended` repasse par le moteur C, dont
`py_backtest_symbol` n'accepte que `(prices, volumes, coeffs, montant, cost)`
(`python_interface.c` ligne 237). Le moteur rapide est donc structurellement
incapable d'honorer les seuils.

L'objectif de l'optimisateur est donc épinglé sur `backtest_signals_with_events`
avec `domain_thresholds`, au lieu de `c_extended` qui les met à `None`
(`qsi_optimized.py` ligne 351). Coût mesuré : 1 %. C'est ce qui fait atterrir B1
dans ce lot : les quatre seuils deviennent effectifs immédiatement, et le lot 2
ne fera que les rendre rapides.

Le C recompilé profite donc à l'application, pas à l'optimisateur.

## 7. Hygiène

- `warnings.filterwarnings("ignore")` ligne 24, effet de bord sur tout le
  processus y compris l'application Qt, est supprimé. Si un avertissement précis
  inonde la sortie, il est réduit localement par `catch_warnings`.
- `best_cache` devient un `_BoundedCache` importé de `core/cache.py`, qui existe
  déjà avec éviction LRU.
- Les connexions SQLite de `save_optimization_results` et
  `replay_all_historical` passent en `try/finally`.
- `get_best_gain_csv`, définie ligne 211 et jamais appelée, est supprimée.
- Les deux `from pathlib import Path` en trop, lignes 11 et 17, sont supprimés.
- Le commentaire « Extra 13 params » ligne 558 en décrit onze, il est corrigé.
- Le cache secteur prend `config.CACHE_DIR`, déjà absolu, et son dossier est créé
  à l'écriture et non à l'import. Trois dossiers `cache_data/` existent
  aujourd'hui, dont un créé le 2026-08-05 par une simple exécution de pytest.

## 8. Tests

Tous hors réseau et hors base réelle, donc dans le sous-ensemble par défaut.

| Cible | Vérifications |
|---|---|
| `core/optim_params.py` | `contraindre` idempotente et toujours dans les bornes ; aller-retour `vers_colonnes` puis `depuis_colonnes` sans perte ; dimensions 14, 25, 25, 36 selon les drapeaux ; aucune colonne SQL en double ; cohérence de `indices()` |
| Budget | évaluations prévues inférieures ou égales au budget pour chaque stratégie et pour `hybrid` ; valeur du menu identique à celle du lancement |
| B2 | micro-optimisation avec backtest simulé sur base temporaire, puis vérification que les `trades` de la ligne écrite correspondent à un recalcul sur ses propres coefficients |
| Garde-fou ASan | un faux `.so` contenant `__asan_init` est refusé, et le repli Python est pris |
| Effets de bord | importer `optimisateur_hybride` ne crée aucun dossier |
| `test_fundamentals_integration.py` | les assertions 18, 24, 28 et 34 deviennent 14, 25, 25 et 36 |

### 8.1 Critères d'acceptation

1. `python optimisateur_hybride.py` atteint le menu sans variable
   d'environnement.
2. Un run de fumée, un groupe, deux symboles, stratégie `lhs`, budget de 40
   évaluations, backtest simulé, écrit exactement une ligne dont les métriques
   correspondent à ses propres coefficients.
3. `pytest -m "not integration"` reste vert, nouveaux modules compris.
4. Le nombre d'évaluations prévu égale le nombre observé, pour chaque stratégie.
5. Importer le module ne crée aucun dossier `cache_data/` parasite.

## 9. Risques

| Risque | Portée | Traitement |
|---|---|---|
| L'élargissement des plages de bridage change les résultats par rapport aux 545 runs existants | Réel et voulu : les valeurs sauvegardées cessent de différer des valeurs évaluées | La colonne `transaction_cost` et la colonne `seed` rendent l'ancien et le nouveau distinguables en base |
| La recompilation active le C et change le comportement des backtests de l'application | Hors optimisateur, qui reste épinglé sur le chemin Python | La levée du drapeau par défaut reste hors périmètre, à vérifier séparément |
| `-ffast-math` peut décaler les flottants entre chemin C et chemin Python | Application uniquement | Constaté, non traité dans ce lot |
| Le lot 1 ne rend pas les runs praticables | Assumé | Annoncé en 3.2, objet du lot 2 |

## 10. Suite

Lot 2, évaluateur vectorisé : précalculer les features une fois par symbole, les
indicateurs ne dépendant ni des coefficients ni des seuils, puis évaluer chaque
configuration en arithmétique de tableaux. Le contrat de la section 4 y est
réutilisé tel quel. Critère de validation : conformité au chemin de référence sur
un jeu de configurations tirées au hasard, à tolérance près.

Lot 3, hygiène de données : `get_sector` et `classify_cap_range` lisent le store
au lieu d'interroger yfinance par symbole, écriture atomique et verrouillée du
cache secteur, reste de la dette.

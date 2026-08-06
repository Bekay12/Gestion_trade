# Optimisateur hybride, lot 2 : coût de l'objectif. Conception

> Spec validée le 2026-08-06. Constats chiffrés qui la fondent :
> [2026-08-06-optimisateur-hybride-lot2-constats.md](2026-08-06-optimisateur-hybride-lot2-constats.md).

## Objectif

Réduire le coût d'une évaluation de l'objectif d'optimisation, **sans changer
d'un iota le résultat produit**. Toute modification de cette spec qui ferait
diverger un gain ou un nombre de trades est hors périmètre par construction.

## Contexte mesuré

Environnement : `.venv_new`, 16 cœurs, séries de 1210 barres, module C compilé.

Une évaluation coûte **7,17 s** par symbole. Le profil situe la dépense :
`get_trading_signal` est appelé **une fois par barre**, soit 1160 appels, et
pèse 96 % du temps, dont l'ADX reconstruit à chaque barre (47 %),
`extract_best_parameters` et ses 1160 requêtes SQLite (14 %) et le RSI (7 %).

La cause n'est pas le langage. `qsi.py` possède déjà un cache d'instantanés
techniques, `TA_CACHE`, alimenté en `qsi.py:402` et lu en `qsi.py:306`, dont la
clé `(symbol, last_close, last_volume, prices_len)` se répète bien d'une
évaluation à l'autre puisque la série de prix ne change pas. La boucle de
backtest lui passe bien `symbol=symbol_name`.

Il est simplement **dimensionné à 500 entrées pour 1160 barres**
(`core/cache.py:39`). Il évince donc les premières barres avant de pouvoir les
réutiliser, et son taux de réussite est nul. Mesure directe :

| `maxsize` | éval 1 | éval 2 | éval 3 | entrées retenues |
|---|---|---|---|---|
| 500, actuel | 7,23 s | 7,18 s | 7,13 s | 500, saturé |
| 5000 | 7,09 s | **2,36 s** | **2,35 s** | 1160, complet |

Résultats identiques dans les deux cas, `(-31.114652404972077, 46)`, au bit
près. Une optimisation effectuant des dizaines de milliers d'évaluations sur les
mêmes symboles, seule la première paye le prix fort.

## Ce que cette spec ne fait pas, et pourquoi

**Elle ne touche pas au module C.** Le lot 1 avait épinglé l'objectif sur le
chemin Python pour honorer les seuils, ce qui coûte un facteur 37 000 face au C.
La tentation était de rebasculer sur le C. La vérification d'équivalence l'a
interdit : à seuils alignés, les deux moteurs rendent des résultats de **signe
opposé** sur trois séries sur trois.

| Série | Python gain | trades | C gain | trades |
|---|---|---|---|---|
| 7 | −59,83 | 30 | +13,49 | 38 |
| 21 | +41,12 | 13 | −62,72 | 24 |
| 42 | −11,26 | 12 | +4,88 | 26 |

Sept divergences structurelles l'expliquent, dont deux lourdes : le Python fonde
`strong_uptrend` et `strong_downtrend` sur **Ichimoku**, que le C n'implémente
pas du tout, et le C calcule **un seul ADX pour toute la série** là où le Python
en a un par barre. Le C est une autre stratégie, pas une version rapide de la
même.

Le rendre équivalent supposerait de réécrire tout son chemin de signal et d'y
ajouter un indicateur. Surtout, cela maintiendrait **deux implémentations de la
même stratégie**, ce qui est précisément la cause de la situation actuelle :
elles ont divergé sans que personne ne le voie. Le module C reste donc en place,
documenté comme divergent, et hors du chemin d'optimisation.

## Architecture

Trois étages, dans cet ordre, du gratuit vers le coûteux. L'étage 3 n'est ouvert
qu'après remesure et peut se révéler sans objet.

### Étage 1 : dimensionner `TA_CACHE`

**Décision : une constante nommée dans `core/cache.py`, pas un dimensionnement
dynamique.** Dimensionner dynamiquement supposerait que l'optimisateur mute un
cache global au moment où il connaît sa charge, ce qui crée un couplage entre
l'optimisateur et le cache et une course entre groupes traités successivement.

**La constante reste dans `core/cache.py` et non dans `config.py`**, bien qu'il
n'y ait aucun cycle d'import (`config.py` n'importe que `pathlib`, `pandas` et
`datetime`). Raison : `config.py` crée des dossiers à l'import (`CACHE_DIR.mkdir`
et `DATA_CACHE_DIR.mkdir`), effet de bord que la tâche 9 du lot 1 a justement
combattu. Faire dépendre un module feuille, dont le docstring annonce qu'il est
« process-local, sans lien avec Parquet/SQLite », d'un module à effet de bord
serait un recul. Le but est que la valeur soit nommée, documentée et justifiée
par sa mesure, pas qu'elle vive dans un fichier particulier.

Valeur retenue : **100 000 entrées**. Justification : 1160 barres × 50 symboles
d'un groupe font 58 000 entrées, et un instantané pèse une vingtaine de flottants
plus la clé, soit de l'ordre de 200 octets, donc environ **20 Mo** au plafond.
La constante porte en commentaire la mesure qui la justifie, pour qu'elle ne
redevienne pas un nombre arbitraire.

**Critère d'acceptation.** Sur un backtest de 1210 barres, le cache retient les
1160 instantanés attendus, et la deuxième évaluation du même symbole coûte au
plus la moitié de la première.

### Étage 2 : mémoïser `extract_best_parameters`

14 % du temps, soit 1160 requêtes SQLite par backtest pour une réponse qui ne
change pas pendant un run.

Le plan du lot 1 avait explicitement écarté ce point, au motif que mémoïser
« change le comportement de la production en cours de run ». **La conception qui
lève l'objection : une clé de cache dérivée de l'état du fichier de base**, sa
date de modification et sa taille. Une écriture en base invalide alors le cache
d'elle-même, sans portée explicite à gérer ni objet de contexte à faire
circuler, y compris si un autre processus écrit pendant un run.

**Critère d'acceptation.** Sur un backtest de 1210 barres,
`extract_best_parameters` touche la base **une fois**, pas 1160. Après une
écriture en base, l'appel suivant relit et rend la nouvelle valeur.

### Étage 3 : sortir les indicateurs de la boucle, verdict

**Sans objet. L'étage 1 a absorbé le coût.** Remesuré le 2026-08-06 sur le
script de référence (1210 barres, quatre évaluations consécutives, `TA_CACHE`
partagé d'une évaluation à l'autre comme en production) :

| Évaluation | Avant le lot 2 | Après étages 1 et 2 |
|---|---|---|
| 1 (cache froid) | 7,17 s | 5,75 à 5,77 s |
| 2 (cache chaud) | 7,18 s | 0,57 à 0,59 s |
| 3 | (non mesurée) | 0,57 s |
| 4 | (non mesurée) | 0,58 s |

Deux exécutions indépendantes du script donnent les mêmes chiffres à 0,02 s
près, et le même gain et le même nombre de trades sur les quatre évaluations
(déterminisme confirmé). La deuxième évaluation, celle qui représente le
régime stable d'une optimisation qui enchaîne les jeux de coefficients sur la
même série, passe de 7,18 s à environ 0,58 s, soit un gain d'environ 12,6x,
très au-delà du seuil de 3x fixé pour ouvrir l'étage 3 (le brief demandait de
profiler seulement si l'évaluation 2 restait au-dessus de 2,4 s). Le profilage
de l'étape 2 n'a donc pas été nécessaire.

L'argument de conception reste correct a posteriori, il explique le résultat :
la boucle passe `prices.iloc[:i+1]`, et **toutes** les grandeurs dérivées sont
des fenêtres glissantes fixes lues au dernier point (`rolling(window=30)`,
EMA, RSI Wilder, ADX), donc causales et indépendantes de la longueur de la
tranche. Une fois que `TA_CACHE` retient effectivement les 1160 instantanés
d'une série (étage 1) et que la lecture des paramètres ne repasse plus par
SQLite à chaque barre (étage 2), il ne reste plus, à l'intérieur de la boucle,
de calcul redondant assez coûteux pour justifier de sortir le calcul des
indicateurs en un passage vectorisé préalable. L'étage 3 n'est pas ouvert et
ne fait l'objet d'aucun plan de suite.

## Tests

Le test central est un test **d'identité, pas de performance** : mêmes séries,
mêmes coefficients, mêmes seuils, on compare gain et trades avant et après. La
mesure a montré que l'égalité est exacte, donc l'assertion est une **égalité
stricte**, sans tolérance.

Les deux autres portent sur des **compteurs, pas sur des durées** : nombre
d'entrées retenues par le cache, et nombre d'accès à la base. Une assertion sur
le temps serait instable en intégration continue et ne dirait rien de la cause.

Tous les tests restent hors réseau et hors base réelle, dans le sous-ensemble
`pytest -m "not integration"`. Le test de l'étage 2 écrit dans une base SQLite
temporaire via `tmp_path`.

## Risques nommés

- **Mémoire.** Le plafond de 100 000 entrées vaut environ 196 Mo au plafond
  (mesuré directement au conteneur réel, tâche 1, environ 2,06 Ko par
  instantané, pas les ~20 Mo initialement estimés sur les seuls flottants
  bruts). C'est un plafond LRU, pas une allocation : le cache ne monte qu'à
  l'ensemble réellement utilisé, soit environ 2,4 Mo par symbole pour un
  backtest de 1160 barres.
- **Clé du cache.** Elle arrondit le prix à deux décimales et le volume à la
  centaine. Sans danger aujourd'hui, `prices_len` et `symbol` en faisant partie,
  mais une collision deviendrait silencieuse. À consigner à côté de la clé.
- **Invalidation par date de modification.** Deux écritures dans la même seconde
  sur un système de fichiers à granularité d'une seconde pourraient ne pas
  invalider. La taille du fichier entre dans la clé pour réduire ce cas ; le
  risque résiduel est à consigner.
- **Étage 3.** Remesuré et tranché sans objet (tâche 3) : voir la section
  « Étage 3 » ci-dessus.

## Traçabilité

| Constat | Origine | Étage |
|---|---|---|
| `TA_CACHE` saturé à 500 pour 1160 barres | mesure du 2026-08-06 | 1 |
| 1160 requêtes SQLite par backtest | profil du 2026-08-06 | 2 |
| ADX recalculé à chaque barre, 47 % | profil du 2026-08-06 | 3, sans objet après remesure (tâche 3) |
| C et Python non équivalents | mesure du 2026-08-06 | hors périmètre, documenté |

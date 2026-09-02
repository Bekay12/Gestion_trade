# Optimisateur hybride, lot 2 : constats mesurés et priorités

> Écrit le 2026-08-06, à la clôture du lot 1. Ce document porte les mesures qui
> justifient l'ordre des travaux du lot 2. Il évite de les refaire.

## Contexte

Le lot 1 a rendu l'optimisateur démarrable et cohérent avec ce qu'il mesure.
Il a aussi introduit une régression de performance assumée, en ligne de
commande uniquement, dont ce document établit le coût et la sortie.

Environnement des mesures : `.venv_new`, 16 cœurs, séries de 1210 barres,
module C compilé en production (`C_ACCELERATION = True`).

## Mesure 1 : le coût des deux moteurs de backtest

| Moteur | Temps par backtest |
|---|---|
| `backtest_signals_with_events` (chemin Python, objectif actuel) | **7,17 s** |
| `backtest_signals_c_extended` (module C, ancien objectif) | **0,0002 s** |

Rapport : **37 000**. Le commentaire du plan du lot 1 annonçait « +1 % » ; ce
chiffre n'était vrai que sans module C, donc dans l'interface graphique, qui
pose `QSI_DISABLE_C_ACCELERATION=1`. En ligne de commande, le surcoût est
entier.

## Mesure 2 : où passent les 7,17 s

Profil `cProfile` du chemin Python. `get_trading_signal` est appelé **une fois
par barre**, soit 1160 appels, et pèse 96 % du temps.

| Poste | Part |
|---|---|
| `ta.trend.ADXIndicator`, reconstruit et recalculé à chaque barre | 47 % |
| `extract_best_parameters`, soit 1160 requêtes SQLite par backtest | 14 % |
| `ta.momentum` (RSI), même schéma | 7 % |
| `pandas.Series.__getitem__`, 2,1 millions d'accès scalaires | conséquence des trois lignes ci-dessus |

C'est un O(n²) : les indicateurs sont recalculés sur tout l'historique à chaque
barre. Les sortir de la boucle ne change **aucun résultat**, seulement le temps.

## Mesure 3 : l'impact réel des 4 seuils optimisés

Coefficients tirés d'une ligne réelle de `optimization_runs` (Technology,
149 trades, `gain_moy` 27,3). Chaque seuil balayé sur son étendue réelle en
base, les autres tenus à leur moyenne. Amplitude du gain sur deux séries.

| Seuil | Amplitude série 1 | Amplitude série 2 | Verdict |
|---|---|---|---|
| `th_rsi` | 20,85 | **73,17** | **Décisif** |
| `th_vol` | 4,11 | 3,01 | Marginal |
| `th_adx` | 0,25 | 1,36 | Quasi nul |
| `th_score` | **0,00** | **0,00** | **Inerte** |

Sur la deuxième série, `th_rsi = 30` donne **+43,59** en 15 trades et
`th_rsi = 70` donne **−29,58** en 46 trades : le même jeu de coefficients passe
de gagnant à perdant selon ce seul seuil. La correction B1 était donc
nécessaire, la production lisant ces seuils depuis la base (`qsi.py:2945`) pour
piloter les signaux réels. Mais son impact est concentré sur **un** seuil.

`th_score` est mort par construction, pas seulement par mesure :
`buy_threshold = seuil_achat` écrase `score_threshold` dès que `seuil_achat`
est fourni (`qsi.py:879-890`), ce que l'optimiseur comme la production font
toujours.

Limite : séries synthétiques, un seul jeu de coefficients réels. Cela établit
le mécanisme et les ordres de grandeur, pas un chiffre par secteur. Le rang des
quatre seuils est robuste, celui de `th_score` découlant du code.

## Priorités du lot 2

### 1. Descendre `th_rsi` et `th_vol` dans le module C, puis rebasculer l'objectif

C'est le gain le plus important du projet : il rend la lenteur caduque sans
rien sacrifier de la justesse acquise au lot 1.

Le moteur C code trois constantes RSI en dur :

```c
// backtest.c:48-50
int rsi_cross_up   = (prev_rsi < 30.0) && (last_rsi >= 30.0);
int rsi_cross_mid  = (prev_rsi < 50.0) && (last_rsi >= 50.0);
int rsi_cross_down = (prev_rsi > 65.0) && (last_rsi <= 65.0);
```

`python_interface.c` transporte déjà des seuils vers la structure de
coefficients (`th_price_rsi_slope`, `th_price_vol_slope`), donc le motif de
plomberie existe et il suffit de le suivre. Une fois les deux seuils passés,
`evaluate_config` peut revenir à `backtest_signals_c_extended`.

Vérification attendue : à seuils égaux, le C et le chemin Python doivent rendre
le même nombre de trades et le même gain, à tolérance près. C'est le test qui
autorise la bascule.

### 2. Retirer `th_score` de l'espace de recherche

14 dimensions vers 13. Gratuit, et cela supprime une dimension qui ne fait que
brouiller l'optimiseur. Retirer `th_score` de `VECTEUR_BASE` dans
`core/optim_params.py` et le déplacer vers `SEUILS_GELES`, la colonne `th8`
continuant d'être écrite pour que les lignes restent relisibles.

Attention : les tests du lot 1 verrouillent les dimensions à 14, 25, 25 et 36
(`test_optim_params.py`). Ils devront suivre, en même temps et pas après.

### 3. Le O(n²) du chemin Python

Reste nécessaire pour l'interface graphique, qui n'a pas le module C, mais
cesse d'être le chemin de l'optimisation une fois le point 1 fait. Sortir le
calcul des indicateurs de la boucle par barre et mémoïser
`extract_best_parameters`.

## Reste connu, non prioritaire

- L'évolution différentielle est passée à `workers=1`. Restaurer `workers=-1`
  demanderait de rendre l'optimiseur sérialisable, ce que le
  `ThreadPoolExecutor` porté par l'instance empêche
  (`cannot pickle '_queue.SimpleQueue'`). Sans intérêt tant que l'objectif
  coûte 37 000 fois trop cher, et probablement inutile après le point 1.
- La parallélisation par symbole n'accélère rien, ni avant ni après le lot 1 :
  cinq symboles coûtent cinq fois un symbole, le GIL neutralisant ce
  parallélisme sur du pandas pur.
- `get_sector` et `classify_cap_range` consomment une requête yfinance par
  symbole. Lot 3.

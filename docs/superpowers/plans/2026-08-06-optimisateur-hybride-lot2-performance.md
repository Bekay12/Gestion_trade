# Optimisateur hybride, lot 2 : coût de l'objectif. Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réduire le coût d'une évaluation de l'objectif d'optimisation sans changer d'un iota le résultat produit.

**Architecture:** Deux corrections indépendantes, chacune sur un cache déjà présent mais mal réglé. `TA_CACHE` est dimensionné à 500 entrées pour 1160 barres, donc il évince les premières barres avant de pouvoir les réutiliser : on le dimensionne pour la charge réelle. `extract_best_parameters` interroge SQLite une fois par barre pour une réponse qui ne change pas : on la mémoïse avec une clé dérivée de l'état du fichier, ce qui rend l'invalidation automatique. Une troisième tâche remesure et tranche si un troisième étage reste utile.

**Tech Stack:** Python 3.10, pandas, SQLite, pytest.

## Global Constraints

- Environnement : `.venv_new` à la racine du dépôt. Toute commande Python utilise `../.venv_new/bin/python` depuis `stock-analysis-ui/`.
- Les tests tournent depuis `stock-analysis-ui/`. Le sous-ensemble par défaut est `pytest -m "not integration"` et doit rester **hors réseau et hors base réelle**. Aucun nouveau test de ce lot ne porte le marqueur `integration`.
- **Interdiction absolue d'écrire dans la vraie base** `stock-analysis-ui/src/signaux/optimization_hist.db`. Toute inspection est en lecture seule (`file:...?mode=ro`), toute écriture de test passe par `tmp_path`.
- **Aucun résultat ne doit changer.** Gain et nombre de trades sont comparés par **égalité stricte**, sans tolérance. La mesure du 2026-08-06 a montré que l'égalité est exacte.
- Conventions : docstrings et messages en français, `snake_case`, types dans les signatures, `logging.getLogger(__name__)` avec préfixe `[TAG]` si journalisation.
- Interdiction du tiret d'incise en prose et en commentaires (`--`, `—`, `–`). Utiliser virgule, deux-points ou parenthèses. Les lignes de séparation `-----` des docstrings du projet sont du code, donc autorisées.
- Ne jamais ajouter de boucle yfinance par symbole. Ce lot n'ajoute aucun appel réseau.
- **Ne pas toucher au module C.** Il est une implémentation divergente, documentée comme telle, hors du chemin d'optimisation. Voir la spec.
- Spec de référence : `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md`.

---

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `stock-analysis-ui/src/core/cache.py` (modifier) | Constante de dimensionnement de `TA_CACHE`, avec sa justification mesurée |
| `stock-analysis-ui/src/qsi.py` (modifier) | Mémoïsation de `extract_best_parameters`, invalidée par l'état du fichier |
| `stock-analysis-ui/src/tests/test_optim_performance.py` (créer) | Rétention du cache, identité des résultats, compteur d'accès base |
| `stock-analysis-ui/CHANGELOG.md` (modifier) | Entrée de version |

---

### Task 1: Dimensionner `TA_CACHE` pour la charge réelle

**Files:**
- Modify: `stock-analysis-ui/src/core/cache.py:39`
- Test: `stock-analysis-ui/src/tests/test_optim_performance.py` (créer)

**Interfaces:**
- Consumes: rien.
- Produces: `core.cache.TA_CACHE_MAXSIZE: int`, constante de module.

- [ ] **Step 1: Écrire les tests, qui échouent**

Créer `stock-analysis-ui/src/tests/test_optim_performance.py` :

```python
"""
Verrouillage du cout de l'objectif d'optimisation.

TA_CACHE memoise les instantanes techniques par barre. Il etait dimensionne a
500 entrees alors qu'un backtest de 5 ans en produit environ 1160 pour un seul
symbole : il evincait donc les premieres barres avant de pouvoir les reutiliser
et son taux de reussite etait nul. Mesure du 2026-08-06, meme serie, resultats
identiques au bit pres :
    maxsize=500    eval1 7,23 s   eval2 7,18 s   eval3 7,13 s
    maxsize=5000   eval1 7,09 s   eval2 2,36 s   eval3 2,35 s

Aucun acces reseau, aucune base reelle.
"""
import numpy as np
import pandas as pd
import pytest

from core import cache as cache_module
from trading_c_acceleration.qsi_optimized import backtest_signals_with_events

# Une ligne reelle de optimization_runs (Technology, 149 trades), pour que le
# backtest produise de vrais signaux plutot qu'un score constamment nul.
COEFFS = (1.28, 3.0, 0.94, 2.19, 1.57, 0.5, 0.77, 3.0)
SEUILS = (49.9, 0.0, 0.0, 1.48, 24.8, 0.0, 0.5, 3.84)
ACHAT, VENTE = 2.0, -2.62
NB_BARRES = 1210
PREMIERE_BARRE = 50   # backtest_signals_with_events demarre a l'indice 50


def _serie(graine: int = 7) -> tuple[pd.Series, pd.Series]:
    """Serie de prix et de volumes deterministe, sans reseau."""
    rng = np.random.default_rng(graine)
    index = pd.date_range("2021-01-01", periods=NB_BARRES, freq="D")
    prix = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.018, NB_BARRES)), index=index)
    volumes = pd.Series(rng.lognormal(13.5, 0.6, NB_BARRES), index=index)
    return prix, volumes


def _backtest(prix, volumes, symbole: str) -> tuple[float, int]:
    resultat, _evenements = backtest_signals_with_events(
        prix, volumes, "default", 50, 1.0,
        domain_coeffs={"default": COEFFS},
        domain_thresholds={"default": SEUILS},
        seuil_achat=ACHAT, seuil_vente=VENTE,
        symbol_name=symbole)
    return resultat["gain_total"], resultat["trades"]


def test_ta_cache_dimensionne_pour_un_groupe_entier() -> None:
    """Le plafond doit couvrir plusieurs symboles, pas une fraction d'un seul.

    Un backtest produit environ 1160 instantanes pour UN symbole, et un groupe
    en compte des dizaines.
    """
    assert cache_module.TA_CACHE_MAXSIZE >= 50_000
    assert cache_module.TA_CACHE._maxsize == cache_module.TA_CACHE_MAXSIZE


def test_ta_cache_retient_un_backtest_complet() -> None:
    """Verrouille la regression : a 500, le cache saturait et n'aidait jamais."""
    cache_module.TA_CACHE.clear()
    prix, volumes = _serie()

    _backtest(prix, volumes, "TEST_RETENTION")

    attendu = NB_BARRES - PREMIERE_BARRE
    assert len(cache_module.TA_CACHE) >= attendu, (
        f"{len(cache_module.TA_CACHE)} instantanes retenus, {attendu} attendus")


def test_le_cache_ne_change_aucun_resultat() -> None:
    """Deuxieme evaluation servie par le cache : resultat identique au bit pres."""
    cache_module.TA_CACHE.clear()
    prix, volumes = _serie()

    froid = _backtest(prix, volumes, "TEST_IDENTITE")
    chaud = _backtest(prix, volumes, "TEST_IDENTITE")

    assert froid == chaud
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_performance.py -q`
Expected: FAIL. `test_ta_cache_dimensionne_pour_un_groupe_entier` lève `AttributeError: module 'core.cache' has no attribute 'TA_CACHE_MAXSIZE'`, et `test_ta_cache_retient_un_backtest_complet` échoue avec 500 instantanés retenus pour 1160 attendus.

- [ ] **Step 3: Dimensionner le cache**

Dans `stock-analysis-ui/src/core/cache.py`, remplacer la ligne 39 :

```python
TA_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=500)
```

par :

```python
# Un backtest de 5 ans parcourt environ 1160 barres et produit donc 1160
# instantanes pour UN seul symbole. A 500, le cache evincait les premieres
# barres avant de pouvoir les reutiliser : son taux de reussite etait nul et
# chaque evaluation repayait le calcul complet des indicateurs.
# Mesure du 2026-08-06, meme serie, resultats identiques au bit pres :
#     maxsize=500    eval1 7,23 s   eval2 7,18 s   eval3 7,13 s
#     maxsize=5000   eval1 7,09 s   eval2 2,36 s   eval3 2,35 s
# 100 000 entrees couvrent 1160 barres pour une cinquantaine de symboles, soit
# environ 20 Mo : un instantane porte une vingtaine de flottants plus sa cle.
TA_CACHE_MAXSIZE = 100_000

TA_CACHE: Dict[tuple, Dict[str, float]] = _BoundedCache(maxsize=TA_CACHE_MAXSIZE)
```

Ne pas toucher au défaut `maxsize: int = 500` de `_BoundedCache` ni à `DERIV_CACHE` : d'autres appelants en dépendent et leur charge n'a pas été mesurée.

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_performance.py -q`
Expected: PASS, 3 tests.

- [ ] **Step 5: Lancer la suite complète**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q`
Expected: PASS, aucun test cassé.

- [ ] **Step 6: Commit**

```bash
git add stock-analysis-ui/src/core/cache.py stock-analysis-ui/src/tests/test_optim_performance.py
git commit -m "perf(qsi): dimensionner TA_CACHE pour la charge reelle

Il retenait 500 instantanes alors qu'un backtest en produit 1160 pour un seul
symbole, donc il evincait les premieres barres avant de pouvoir les reutiliser
et son taux de reussite etait nul. Mesure : la deuxieme evaluation passe de
7,18 s a 2,36 s, resultats identiques au bit pres."
```

---

### Task 2: Mémoïser `extract_best_parameters`

**Files:**
- Modify: `stock-analysis-ui/src/qsi.py:82-...` (la fonction entière)
- Test: `stock-analysis-ui/src/tests/test_optim_performance.py` (ajout)

**Interfaces:**
- Consumes: `core.cache._BoundedCache` (déjà importé dans `qsi.py:20`).
- Produces, dans `qsi` :
  - `_BEST_PARAMS_CACHE: _BoundedCache`, cache de module
  - `_extract_best_parameters_sans_cache(db_path: str) -> Dict`, l'ancien corps
  - `extract_best_parameters(db_path: str = None) -> Dict`, signature inchangée

**Contexte que l'implémenteur ne peut pas deviner :** `get_trading_signal` appelle `extract_best_parameters()` sans argument (`qsi.py:480`), une fois par barre, soit 1160 requêtes SQLite par backtest, pour 14 % du temps total. Tous les appelants n'utilisent le résultat qu'en **lecture** (recherches par clé et dépaquetage de tuples), vérifié sur les huit sites d'appel : la valeur mémoïsée peut donc être partagée sans copie. Le plan du lot 1 avait écarté cette mémoïsation parce qu'elle « change le comportement de la production en cours de run » ; la clé dérivée de l'état du fichier lève cette objection, une écriture en base invalidant le cache d'elle-même.

- [ ] **Step 1: Écrire les tests, qui échouent**

Ajouter à `stock-analysis-ui/src/tests/test_optim_performance.py` :

```python
def _base_avec_une_ligne(chemin, secteur: str = "Technology", a1: float = 1.28):
    """
    Cree une base optimization_runs portant une ligne.

    ATTENTION, piege verifie le 2026-08-06 : la requete de
    _extract_best_parameters_sans_cache exige SANS REPLI les colonnes
    a9, a10, th9, th10, use_price_slope, use_price_acc, a11 a a15,
    th11 a th15 et use_fundamentals (qsi.py:126-138). Seules a16 a a18,
    th16 a th18 et use_price_extras recoivent un `NULL AS`. Une table plus
    courte ferait echouer la requete et rendre {} silencieusement, donnant un
    echec de test sans rapport avec le sujet.
    """
    import sqlite3

    conn = sqlite3.connect(chemin)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            sector TEXT NOT NULL,
            market_cap_range TEXT,
            gain_moy REAL, success_rate REAL, trades INTEGER,
            seuil_achat REAL, seuil_vente REAL,
            a1 REAL, a2 REAL, a3 REAL, a4 REAL, a5 REAL, a6 REAL, a7 REAL, a8 REAL,
            th1 REAL, th2 REAL, th3 REAL, th4 REAL, th5 REAL, th6 REAL, th7 REAL, th8 REAL,
            a9 REAL, a10 REAL, th9 REAL, th10 REAL,
            use_price_slope INTEGER DEFAULT 0, use_price_acc INTEGER DEFAULT 0,
            a11 REAL, a12 REAL, a13 REAL, a14 REAL, a15 REAL,
            th11 REAL, th12 REAL, th13 REAL, th14 REAL, th15 REAL,
            use_fundamentals INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        "INSERT INTO optimization_runs ("
        " timestamp, sector, market_cap_range, gain_moy, trades,"
        " seuil_achat, seuil_vente,"
        " a1, a2, a3, a4, a5, a6, a7, a8,"
        " th1, th2, th3, th4, th5, th6, th7, th8,"
        " a9, a10, th9, th10, use_price_slope, use_price_acc,"
        " a11, a12, a13, a14, a15, th11, th12, th13, th14, th15, use_fundamentals"
        ") VALUES ("
        " '2026-01-01 00:00:00', ?, 'Large', 10.0, 5,"
        " 4.2, -2.0,"
        " ?, 1, 1, 1, 1, 1, 1, 1,"
        " 50, 0, 0, 1.5, 25, 0, 0.5, 4,"
        " 0, 0, 0, 0, 0, 0,"
        " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
        (secteur, a1))
    conn.commit()
    conn.close()


def test_extract_best_parameters_ne_lit_la_base_qu_une_fois(monkeypatch, tmp_path) -> None:
    """Verrouille : 1160 requetes SQLite par backtest devenaient une seule."""
    import qsi

    chemin = str(tmp_path / "optimization_hist.db")
    _base_avec_une_ligne(chemin)

    appels = []
    vrai = qsi._extract_best_parameters_sans_cache

    def compte(db_path):
        appels.append(db_path)
        return vrai(db_path)

    monkeypatch.setattr(qsi, "_extract_best_parameters_sans_cache", compte)
    qsi._BEST_PARAMS_CACHE.clear()

    for _ in range(50):
        qsi.extract_best_parameters(chemin)

    assert len(appels) == 1, f"{len(appels)} lectures de base, 1 attendue"


def test_une_ecriture_en_base_invalide_le_cache(tmp_path) -> None:
    """La cle derivee de l'etat du fichier doit rendre l'invalidation automatique."""
    import time

    import qsi

    chemin = str(tmp_path / "optimization_hist.db")
    _base_avec_une_ligne(chemin, a1=1.0)
    qsi._BEST_PARAMS_CACHE.clear()

    premier = qsi.extract_best_parameters(chemin)
    assert premier, "la base de test devrait produire au moins un secteur"

    # La granularite de mtime peut valoir une seconde sur certains systemes de
    # fichiers ; la taille du fichier entre aussi dans la cle, et une ligne
    # supplementaire la change.
    time.sleep(1.1)
    _base_avec_une_ligne(chemin, secteur="Healthcare", a1=2.0)

    second = qsi.extract_best_parameters(chemin)

    assert set(second) != set(premier), "le cache n'a pas ete invalide"


def test_une_base_absente_ne_leve_pas(tmp_path) -> None:
    """Un chemin invalide doit rendre un dict vide, pas une exception."""
    import qsi

    qsi._BEST_PARAMS_CACHE.clear()
    assert qsi.extract_best_parameters(str(tmp_path / "absente.db")) == {}
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_performance.py -q`
Expected: FAIL, `AttributeError: module 'qsi' has no attribute '_extract_best_parameters_sans_cache'`.

- [ ] **Step 3: Extraire le corps existant**

Dans `stock-analysis-ui/src/qsi.py`, renommer la fonction `extract_best_parameters` en `_extract_best_parameters_sans_cache` et lui donner cette signature, le corps restant **inchangé** à partir du `try:` :

```python
def _extract_best_parameters_sans_cache(db_path: str) -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, float]]]:
    """
    --------------------------------------------------------------------------
    Objectif:
        Lire reellement la base. Ne jamais appeler directement : passer par
        extract_best_parameters(), qui memoise.

    Inputs:
        db_path (str): chemin de la base, deja resolu

    Outputs:
        parametres (Dict): {secteur: (coeffs_8, seuils_8, globaux_2, gain, extras)}
    --------------------------------------------------------------------------
    """
```

Supprimer de ce corps le bloc de résolution du chemin, qui remonte dans l'enveloppe :

```python
    if db_path is None:
        from config import OPTIMIZATION_DB_PATH
        db_path = OPTIMIZATION_DB_PATH
```

- [ ] **Step 4: Ajouter l'enveloppe mémoïsante**

Juste avant `_extract_best_parameters_sans_cache`, ajouter :

```python
# Memoisation de la lecture des meilleurs parametres. get_trading_signal
# l'appelle une fois PAR BARRE, soit 1160 requetes SQLite par backtest pour
# 14 % du temps, alors que la reponse ne change pas pendant un run.
# La cle porte l'etat du fichier, donc une ecriture en base invalide le cache
# d'elle-meme : aucune portee explicite a gerer, et le comportement reste
# correct si un autre processus ecrit pendant un run.
_BEST_PARAMS_CACHE = _BoundedCache(maxsize=8)


def extract_best_parameters(db_path: str = None) -> Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, float]]]:
    """
    --------------------------------------------------------------------------
    Objectif:
        Rendre les meilleurs coefficients et seuils par secteur, en memoisant
        la lecture tant que le fichier de base n'a pas change.

    Inputs:
        db_path (str | None): chemin de la base, defaut config.OPTIMIZATION_DB_PATH

    Outputs:
        parametres (Dict): {secteur: (coeffs_8, seuils_8, globaux_2, gain, extras)}
        Le dictionnaire est PARTAGE entre appelants : le lire, ne pas le muter.
    --------------------------------------------------------------------------
    """
    if db_path is None:
        from config import OPTIMIZATION_DB_PATH
        db_path = OPTIMIZATION_DB_PATH

    cle = None
    try:
        etat = os.stat(db_path)
        cle = (str(db_path), etat.st_mtime_ns, etat.st_size)
    except OSError:
        # Base absente ou illisible : on ne memoise pas, la fonction interne
        # journalise et rend un dict vide.
        cle = None

    if cle is not None:
        connu = _BEST_PARAMS_CACHE.get(cle)
        if connu is not None:
            return connu

    resultat = _extract_best_parameters_sans_cache(db_path)

    if cle is not None:
        _BEST_PARAMS_CACHE[cle] = resultat
    return resultat
```

Vérifier que `os` est importé en tête de `qsi.py`. Si non, l'ajouter. `_BoundedCache` est déjà importé ligne 20.

- [ ] **Step 5: Lancer les tests pour vérifier qu'ils passent**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest src/tests/test_optim_performance.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 6: Vérifier que rien d'autre n'appelait la fonction interne**

Run: `cd stock-analysis-ui/src && grep -rn "extract_best_parameters" --include=*.py . | grep -v Archiv`
Expected: tous les appelants passent par `extract_best_parameters`, aucun n'utilise le nom interne. Les huit sites connus sont `api.py` (trois), `optimisateur_hybride.py`, `qsi.py` (trois) et `ui/main_window.py`.

- [ ] **Step 7: Lancer la suite complète**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add stock-analysis-ui/src/qsi.py stock-analysis-ui/src/tests/test_optim_performance.py
git commit -m "perf(qsi): memoiser extract_best_parameters

Elle etait appelee une fois par barre, soit 1160 requetes SQLite par backtest
pour 14 % du temps, alors que la reponse ne change pas pendant un run. La cle
porte la date de modification et la taille du fichier, donc une ecriture en
base invalide le cache d'elle-meme, ce qui leve l'objection du lot 1."
```

---

### Task 3: Remesurer et trancher sur le troisième étage

**Files:**
- Modify: `stock-analysis-ui/CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md` (section « Étage 3 »)

**Interfaces:**
- Consumes: les tâches 1 et 2.
- Produces: aucune interface nouvelle.

**Pourquoi cette tâche existe :** la spec ouvre un troisième étage, sortir le calcul des indicateurs de la boucle, mais **conditionne son ouverture à une remesure**. L'étage 1 rendant ce calcul quasi gratuit dès la deuxième évaluation, l'étage 3 peut n'avoir plus d'objet. Trancher sans mesurer serait exactement l'erreur que ce lot corrige.

- [ ] **Step 1: Mesurer le coût après les deux étages**

Créer un script jetable hors du dépôt, par exemple dans `/tmp`, qui reproduit la mesure de référence :

```python
"""Cout d'une evaluation apres les etages 1 et 2."""
import os, time
os.environ.setdefault("QSI_CONSENSUS_OFFLINE", "1")
os.environ.setdefault("QSI_DISABLE_PROFILE_FETCH", "1")
import numpy as np, pandas as pd
from trading_c_acceleration.qsi_optimized import backtest_signals_with_events

COEFFS = (1.28, 3.0, 0.94, 2.19, 1.57, 0.5, 0.77, 3.0)
SEUILS = (49.9, 0.0, 0.0, 1.48, 24.8, 0.0, 0.5, 3.84)
n = 1210
rng = np.random.default_rng(7)
index = pd.date_range("2021-01-01", periods=n, freq="D")
prix = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.018, n)), index=index)
vol = pd.Series(rng.lognormal(13.5, 0.6, n), index=index)

for k in range(4):
    t0 = time.perf_counter()
    r, _ = backtest_signals_with_events(
        prix, vol, "default", 50, 1.0,
        domain_coeffs={"default": (COEFFS[0] + 0.01 * k,) + COEFFS[1:]},
        domain_thresholds={"default": SEUILS},
        seuil_achat=2.0, seuil_vente=-2.62, symbol_name="MESURE")
    print(f"eval {k + 1}: {time.perf_counter() - t0:6.2f} s   "
          f"gain {r['gain_total']:9.2f}  trades {r['trades']}")
```

Run: `cd stock-analysis-ui/src && PYTHONPATH=. ../../.venv_new/bin/python /tmp/mesure_lot2.py`

Référence d'avant le lot, à comparer : 7,17 s pour chaque évaluation.

- [ ] **Step 2: Profiler ce qui reste, si le gain est inférieur à 3x**

Si l'évaluation 2 reste au-dessus de 2,4 s, profiler pour savoir où part le temps restant :

```bash
cd stock-analysis-ui/src && PYTHONPATH=. ../../.venv_new/bin/python -c "
import cProfile, pstats, io, runpy
pr = cProfile.Profile(); pr.enable()
runpy.run_path('/tmp/mesure_lot2.py', run_name='__main__')
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
print(s.getvalue())
"
```

- [ ] **Step 3: Trancher, dans la spec**

Remplacer la section « Étage 3 » de `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md` par le verdict, en y portant les chiffres mesurés à l'étape 1 : soit « sans objet, l'étage 1 a absorbé le coût », soit « ouvert, il reste X s par évaluation dont Y % dans Z », avec le profil à l'appui.

Ne pas implémenter l'étage 3 dans cette tâche. S'il est ouvert, il fera l'objet de son propre plan.

- [ ] **Step 4: Journaliser dans le CHANGELOG**

Ajouter en tête de `stock-analysis-ui/CHANGELOG.md`, avant la section `## Version 1.7.0` :

```markdown
## Version 1.8.0 - Optimisateur hybride : lot 2, coût de l'objectif (2026-08-06)

### 🚀 Performances

- **Le cache d'instantanés techniques était dimensionné à 500 entrées pour 1160 barres**
  - `TA_CACHE` mémoïse les indicateurs par barre et sa clé se répète bien d'une évaluation à l'autre, la série de prix ne changeant pas. Mais il retenait 500 instantanés quand un backtest de 5 ans en produit 1160 pour un seul symbole : il évinçait les premières barres avant d'avoir pu les réutiliser, et son taux de réussite était nul. Porté à 100 000 entrées, soit environ 20 Mo, il couvre 1160 barres pour une cinquantaine de symboles. Mesuré sur la même série, résultats identiques au bit près : la deuxième évaluation passe de 7,18 s à 2,36 s.

- **`extract_best_parameters` interrogeait SQLite une fois par barre**
  - Soit 1160 requêtes par backtest, pour 14 % du temps, alors que la réponse ne change pas pendant un run. Elle est désormais mémoïsée avec une clé portant la date de modification et la taille du fichier de base, ce qui rend l'invalidation automatique : une écriture en base, même par un autre processus, invalide le cache d'elle-même. C'est ce qui lève l'objection du lot 1, qui avait écarté cette mémoïsation parce qu'elle aurait changé le comportement en cours de run.

### ⚠️ Connu, non traité dans ce lot

- Le module C reste une implémentation divergente de la stratégie, hors du chemin d'optimisation. À seuils alignés, les deux moteurs rendent des résultats de signe opposé sur trois séries sur trois, sept divergences structurelles l'expliquant, dont Ichimoku absent du C et un ADX calculé une seule fois pour toute la série. Voir `docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md`.
- `a3` et `th_score` sont deux dimensions inertes de l'espace de recherche, prouvées telles par lecture du code et par mesure. Les retirer ferait passer le vecteur de 14 à 12 dimensions.
```

- [ ] **Step 5: Lancer la suite complète**

Run: `cd stock-analysis-ui && ../.venv_new/bin/python -m pytest -m "not integration" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add stock-analysis-ui/CHANGELOG.md \
        docs/superpowers/specs/2026-08-06-optimisateur-hybride-lot2-performance-design.md
git commit -m "docs(optim): verdict du troisieme etage et journal du lot 2

Remesure apres dimensionnement du cache et memoisation de la lecture des
parametres, puis decision documentee sur l'opportunite de sortir le calcul des
indicateurs de la boucle."
```

---

## Traçabilité vers la spec

| Réf spec | Constat | Tâche |
|---|---|---|
| Étage 1 | `TA_CACHE` saturé à 500 pour 1160 barres | 1 |
| Étage 2 | 1160 requêtes SQLite par backtest | 2 |
| Étage 3 | ADX recalculé à chaque barre, conditionnel | 3, décision seulement |
| Hors périmètre | C et Python non équivalents | 3, étape 4, consigné au CHANGELOG |

## Notes pour l'implémenteur

**Ce que ce lot ne corrige volontairement pas.** Le module C n'est pas touché. La tentation de rebasculer l'objectif dessus a été mesurée et écartée : les deux moteurs ne calculent pas la même stratégie. Y revenir demanderait de réécrire tout le chemin de signal du C et d'y ajouter Ichimoku, et surtout de maintenir deux implémentations de la même stratégie, ce qui est la cause de la situation actuelle.

**Piège de test.** `TA_CACHE` est un global de module partagé entre tests. Chaque test qui compte ses entrées doit l'appeler `.clear()` en première ligne, sans quoi l'ordre d'exécution des tests le fait échouer par intermittence.

**Piège de cache.** La clé de `TA_CACHE` vaut `(symbol, last_close arrondi, last_volume arrondi, prices_len)`. Elle n'est calculée que si `symbol` est fourni. Un test qui omet `symbol_name` ne remplira jamais le cache et échouera sans rapport avec le code testé.

**Piège de schéma, vérifié le 2026-08-06.** La requête de `_extract_best_parameters_sans_cache` exige **sans repli** les colonnes `a9`, `a10`, `th9`, `th10`, `use_price_slope`, `use_price_acc`, `a11` à `a15`, `th11` à `th15` et `use_fundamentals` (`qsi.py:126-138`). Seules `a16` à `a18`, `th16` à `th18` et `use_price_extras` reçoivent un `NULL AS`. Une table de test plus courte fait échouer la requête, que le `try/except` englobant transforme en dictionnaire vide : le test échoue alors pour une raison étrangère au sujet. La fabrique `_base_avec_une_ligne` du plan porte déjà toutes ces colonnes.

**Piège d'invalidation.** La granularité de `st_mtime` vaut une seconde sur certains systèmes de fichiers. La taille du fichier entre dans la clé pour réduire ce cas, mais deux écritures rapides de même taille pourraient ne pas invalider. C'est pourquoi le test d'invalidation attend une seconde et ajoute une ligne, ce qui change les deux composantes.

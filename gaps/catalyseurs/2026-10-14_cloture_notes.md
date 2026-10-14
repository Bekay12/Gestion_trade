# Notes de cloture - 2026-10-14

## Echec des scans mecaniques (cassure_scan.py et gap_scan.py --mode close)

Les deux scripts ont echoue avec la meme erreur :

```
ModuleNotFoundError: No module named 'core'
```

Cause factuelle verifiee : les deux scripts (`scripts/cassure_scan.py` ligne 35 et
`scripts/gap_scan.py` ligne 41) definissent une constante codee en dur :

```python
APP_SRC = "/home/berkam/Projets/Gestion_trade/stock-analysis-ui/src"
```

Ce chemin n'existe pas dans cet environnement d'execution (le depot est clone sous
`/home/user/Gestion_trade`, et le module `core/finviz_screeners.py` se trouve reellement sous
`/home/user/Gestion_trade/stock-analysis-ui/src/core/finviz_screeners.py`). `sys.path.insert(0, APP_SRC)`
insere donc un chemin invalide, et `from core.finviz_screeners import run_screen` leve
`ModuleNotFoundError`.

- cassure_scan.py (--min-hausse 5 --limit 50) : echec, exit code 1, meme cause.
- gap_scan.py (--mode close --min-gap 5) : echec, exit code 1, meme cause.

Aucun fichier JSON de detections n'a pu etre genere pour 2026-10-14 (run 2012Z).

Cette note se limite au constat factuel de l'echec technique ; aucune evaluation de
catalyseur ni jugement n'est formule ici, conformement a la consigne de la routine.

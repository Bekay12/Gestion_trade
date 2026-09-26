# skills/ — copie d'archive

Le skill vit dans `~/.claude/skills/`, hors du dépôt : c'est là que Claude Code
le charge. Cette copie existe pour qu'un `git pull` sur une autre machine
récupère le code, les références et les tests.

> **La version qui fait foi est celle de `~/.claude/skills/`.** Celle-ci en est
> une photographie. En cas de divergence, c'est le chemin de chargement qui
> tranche.

| Skill | Version faisant foi | Pour quoi |
|---|---|---|
| `gap-trading-germain` | `~/.claude/skills/gap-trading-germain/` | Trading de gaps intraday / intraweek, méthode Germain. Détections dans [`../gaps/`](../gaps/) |

## Installer sur une autre machine

```bash
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  Trading_Agent/skills/gap-trading-germain/ \
  ~/.claude/skills/gap-trading-germain/
```

## Rafraîchir la copie après modification

```bash
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.gitkeep' \
  ~/.claude/skills/gap-trading-germain/ \
  Trading_Agent/skills/gap-trading-germain/
```

À lancer à chaque changement de la version faisant foi, sinon la copie est une
photographie avec l'apparence de l'actualité. État de cette copie :
**26.09.2026**.

## Vérifier que la copie est saine

Cinq suites, toutes hors ligne, bibliothèque standard seule, **162 tests** au
total. Elles doivent passer depuis la copie comme depuis la version faisant foi.

```bash
cd Trading_Agent/skills/gap-trading-germain/scripts
python3 Test/test_gap_qualifier.py        # 68  classement des gaps
python3 Test/test_cassure_qualifier.py    # 33  classement des cassures
python3 Test/test_edgar_depots.py         # 22  dépôts SEC, dilution et frontière
python3 Test/test_germain_revues.py       # 22  ingestion des revues, gardes de sécurité
python3 Test/test_lecture_finviz.py       # 17  lecture des colonnes Finviz
```

## Ce que la copie contient

| Fichier | Rôle |
|---|---|
| `scripts/gap_scan.py` | Screener des gaps, Finviz plus yfinance groupé plus EDGAR |
| `scripts/gap_qualifier.py` | Classement, direction et stop (pur, sans réseau) |
| `scripts/cassure_scan.py` | Screener des cassures sans gap |
| `scripts/cassure_qualifier.py` | Classement des cassures (pur) |
| `scripts/edgar_depots.py` | Dépôts SEC récents, remplit `formulaire_sec` |
| `scripts/germain_revues.py` | Ingestion des revues de séance d'Academy Germain |
| `scripts/backtest_gaps.py` | Rejeu du classificateur sur l'historique |
| `docs/methode-gaps-et-cassures.md` | Le chapitre : pourquoi chaque seuil est ce qu'il est |

L'orchestrateur du cycle vit **côté projet**, dans
[`../gaps/cycle_quotidien.py`](../gaps/cycle_quotidien.py), parce qu'il ordonne
des phases plutôt qu'il ne calcule une règle.

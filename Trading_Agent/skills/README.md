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
**22.09.2026**.

## Vérifier que la copie est saine

```bash
python3 Trading_Agent/skills/gap-trading-germain/scripts/Test/test_gap_qualifier.py
```

27 tests, hors ligne, stdlib seule.

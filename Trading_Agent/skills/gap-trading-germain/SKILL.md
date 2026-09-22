---
name: gap-trading-germain
description: Use for short-term trading of opening gaps (intraday and intraweek) following the Germain method - screens gappers on Finviz, enriches them with fundamental and technical stock data, classifies each gap as continuation / fade / pump-risk, and returns the signals to validate before the order plus a temporal take-profit horizon. Trigger on gap scan, gappers, pre-market watchlist, gap and go, intraday setup.
---

# Gap Trading — Méthode Germain

Transforme une liste de gappers en décisions traçables. Le screener trouve les
candidats ; ce skill fait le travail qui commence après : vérifier qu'un catalyseur
réel existe, distinguer un short squeeze d'un pump and dump, et dire combien de
temps la position est censée vivre.

**La règle qui donne sa valeur au skill : un gap sans catalyseur vérifié n'est pas
un trade, c'est une loterie.** La méthode Germain l'énonce ainsi : « RVOL élevé
sans catalyseur fondamental = suspect → possible manipulation »
(Academy Germain, Partie 7 Ch.01).

## Quand l'utiliser

- Préparer la watchlist du jour avant l'ouverture US (mode `premarket`)
- Décider si un gap qui a tenu la séance se joue sur plusieurs jours (mode `close`)
- Qualifier un gap précis sur un titre donné (mode `ticker`)

Pas pour : l'investissement long terme (c'est `startup-investment-analyzer`), le
screening de qualité fondamentale (`Combined_scan.py`), ou la détection de creux
pluriannuels (`Valley_scan.py`).

## Les deux fenêtres, deux usages différents

| Mode | Heure ET | Heure France | Ce qu'il produit |
|---|---|---|---|
| `premarket` | 7h00–9h30 | 13h00–15h30 | Watchlist du jour, 5 à 15 noms, classés par volume |
| `close` | 16h00–17h00 | 22h00–23h00 | Gaps ayant **tenu** la séance → candidats intraweek |

Le mode `premarket` est la fenêtre principale pour l'intraday. Le mode `close` ne
sert pas à trader le jour même : il mesure si le gap a survécu, ce qui est le seul
critère qui fait basculer un gap de « fade probable » à « continuation probable ».
Détail et sources : `references/timing-et-sessions.md`.

## Comment l'exécuter

### 1. Scanner

```bash
python3 scripts/gap_scan.py --mode premarket --min-gap 5 --limit 50
python3 scripts/gap_scan.py --mode close --min-gap 5
python3 scripts/gap_scan.py --mode ticker --tickers AAPL,TSLA
```

Le script réutilise `core/finviz_screeners.run_screen` de l'application (session
curl_cffi et lecture correcte de la colonne Ticker déjà réglées) puis enrichit
chaque candidat via **un seul appel yfinance groupé**. Le budget de requêtes
yfinance est une contrainte dure du dépôt : jamais de boucle par symbole.

### 2. Qualifier

`scripts/gap_qualifier.py` classe chaque gap sans appel réseau (fonctions pures,
donc testables) en quatre verdicts :

| Verdict | Signification | Horizon |
|---|---|---|
| `CONTINUATION` | Gap + catalyseur + RVOL ≥ 3 | Intraday, éventuellement intraweek |
| `SQUEEZE` | Short interest élevé **avant** le mouvement, volume progressif | Plusieurs jours |
| `FADE` | Petit gap, RVOL faible, pas de catalyseur | Comblement attendu le jour même |
| `PUMP_RISK` | Signaux de manipulation présents | Ne pas jouer |

Le seuil qui sépare `CONTINUATION` de `FADE` n'est pas arbitraire : la taille du
gap est normalisée par l'ATR, parce que la probabilité de comblement en dépend
directement (gap < 0,3 ATR : ~78 % comblé ; gap > 1,2 ATR : ~8 %). Tableau complet
et sources dans `references/methode-germain.md`.

### 3. Rendre le rapport

Format fixe, trois blocs, décrit dans `references/rapport-template.md` :

1. **Analyse du gap et du contexte** — ce qui s'est passé et sur quel titre
2. **Signaux à valider avant l'ordre** — la check-list qui passe *avant* le clic
3. **Horizon de réalisation** — jour / semaine / mois, avec le motif qui le fixe

## Non négociables

1. **Aucun trade sans catalyseur identifié et daté.** Le catalyseur se vérifie sur
   EDGAR (8-K, S-3, 424B) ou une source de news, pas sur l'intuition. Un gap dont
   le catalyseur reste introuvable est déclaré comme tel, jamais supposé.
2. **Le dépôt 424B et le S-3 sont des signaux baissiers**, pas des catalyseurs
   haussiers : ils annoncent une dilution. Un gap haussier sur un S-3 récent est
   une alerte, pas une confirmation.
3. **Distinguer squeeze et pump avant de conclure.** Les deux montent vite. Le
   tableau de discrimination est dans `references/methode-germain.md` et il est
   obligatoire : short interest avant le mouvement, progressivité du volume,
   réalité de la news, comportement après le pic.
4. **VWAP comme arbitre de direction.** Cours sous le VWAP = pression vendeuse
   dominante ; un « gap up » sous son VWAP n'est pas un gap haussier en cours,
   c'est une distribution.
5. **Le float et la capitalisation gouvernent le risque, pas le potentiel.** Nano
   cap + low float = mouvements violents dans les deux sens. Taille de position
   réduite, jamais l'inverse.
6. **L'horizon est annoncé avant l'entrée**, avec le motif qui le justifie. Un
   trade sans horizon devient un investissement par accident.
7. **Les données pré-marché de Finviz gratuit n'existent pas.** Le mode
   `premarket` lit le gap tel que Finviz le calcule à l'ouverture ; avant 9h30 ET,
   la couverture est partielle. La limite est déclarée dans le rapport, jamais
   masquée.

## Fichiers

| Fichier | À lire quand |
|---|---|
| `references/methode-germain.md` | Avant toute qualification — filtres, RVOL, squeeze vs pump |
| `references/timing-et-sessions.md` | Pour choisir la fenêtre et comprendre pourquoi |
| `references/rapport-template.md` | Avant d'écrire la sortie |
| `scripts/gap_scan.py` | Le scan (Finviz + enrichissement yfinance groupé) |
| `scripts/gap_qualifier.py` | La classification (pur, sans réseau) |
| `scripts/Test/test_gap_qualifier.py` | Tests hors ligne, stdlib seule |

## Environnement

Python 3.10, environnement `.venv_new` du dépôt. Dépendances déjà présentes :
`finvizfinance`, `curl_cffi`, `lxml`, `yfinance`, `pandas`.

```bash
source /home/berkam/Projets/Gestion_trade/.venv_new/bin/activate
python3 scripts/Test/test_gap_qualifier.py     # hors ligne, aucune clé requise
```

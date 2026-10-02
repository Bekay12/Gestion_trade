# Backtests des scanners

Deux backtests point-in-time, rejouables hors réseau (sauf un téléchargement groupé des indices).
Ils lisent le store local (`market_parquet/`, `stock_analysis.db`) et réutilisent les fonctions
mêmes des scanners, pour que la mesure porte sur le code réellement utilisé.

| Script | Question | Sortie |
|---|---|---|
| [Combined_backtest.py](../Combined_backtest.py) | Que sont devenus les profils du Combined (Dual Champion*, Dual Champion, Pure Safe, Pure Growth, Balanced) à 6 et 12 mois ? | `combined/` |
| [Valley_backtest.py](../Valley_backtest.py) | Les signaux du détecteur de creux (INFLEXION, DIVERGENCE, PIÈGE, À SUIVRE) battent-ils l'indice ? | `valley/` |

## Rejouer

Depuis `stock-analysis-ui/src/`, avec `.venv_new` activé :

```bash
python Combined_backtest.py                                  # dates trimestrielles par défaut
python Combined_backtest.py --dates 2025-04-01 2025-07-01    # dates choisies
python Combined_backtest.py --candidats                      # note le store aujourd'hui, 0 requête
python Valley_backtest.py                                    # mensuel, 2021-07 à 2026-03
python Valley_backtest.py --debut 2022-01-01 --fin 2026-03-01
```

`--candidats` écrit une liste restreinte (`candidats_AAAA-MM-JJ*.csv` et `.symbols.txt`) à
confirmer en direct, puisque le store est ancien :

```bash
python Combined_scan.py --symbols-file backtests/combined/candidats_2026-10-02.symbols.txt --profile dual
```

## Ce qui garantit l'absence de look-ahead

- Comptes annuels publiés 90 jours après la clôture de l'exercice, trimestriels 45 jours après.
- Rendement total : dividendes réinvestis, lus dans l'écart cours ajusté / cours brut.
- Écart à l'indice local (table par suffixe Yahoo) et à la médiane de l'univers à la même date.
- Rendements et écarts écrêtés aux 1er et 99e centiles par date ; saut de cours x4 ou /4 en une
  séance écarté (regroupements d'actions non ajustés, par exemple +916 567 % sur PPCB).
- Une échéance non atteinte à la date des cours n'est pas comptée.
- `p_hasard` (Valley, sélections du Combined) : part des tirages au hasard de même taille, date
  par date, qui font au moins aussi bien. Une valeur proche de 1 veut dire « pas mieux que le hasard ».

## Limites à citer avec tout résultat

- **Univers survivant** : seuls les titres encore dans le store aujourd'hui.
- **PEG rétrospectif** : à la date T, G3 utilise PER / croissance passée du BPA, pas les
  prévisions des analystes (introuvables à T). Le PEG Yahoo ou Finviz en direct n'est pas le même.
- **Nombre d'actions actuel** pour S1, **bêta hebdomadaire 52 semaines** pour S3, comptes retraités
  par Yahoo.
- **Seules les dates T >= 2025-04-01 sont fiables** pour les 12 critères : avant, S6 et S7
  (quatre exercices annuels) sont presque incalculables.
- **Peu de dates indépendantes** : 2 dates à 12 mois, 4 à 6 mois, un seul marché haussier, et un
  poids fort des mines d'or dans les Dual Champion*. Un écart important sur 35 titres n'est pas
  une preuve.

## Résultats de référence (mesure du 02.10.2026)

Détail et tableaux complets dans [combined/backtest_combined.md](combined/backtest_combined.md)
et [valley/valley_backtest.md](valley/valley_backtest.md). Ces chiffres vieillissent : relancer
les scripts avant de les citer.

| Sélection | Horizon | Titres | Perf. moyenne | Écart médian à l'univers | Bat l'indice |
|---|---|---|---|---|---|
| Dual Champion* | 12 mois | 35 (2 dates) | +55,7 % | +38,3 pts | 77 % |
| Dual Champion | 12 mois | 85 (2 dates) | +32,7 % | +15,5 pts | 64 % |
| Dual Champion* | 6 mois | 64 (4 dates) | +20,8 % | +13,7 pts | 58 % |
| Dual Champion | 6 mois | 169 (4 dates) | +14,3 % | +7,4 pts | 52 % |
| Pure Safe | 6 mois | 1380 (4 dates) | +7,4 % | +1,6 pts | 44 % |

- **Dual Champion\*** = Dual Champion qui remplit aussi G3 (sous-valorisation), G4 (momentum 3 mois)
  et S4 (dividende). La définition est dans `core/scan_fondamentaux.est_etoile`.
- **Finviz seul (règle `dual_star` émulée)** : écart à l'indice de +4,6 pts à 6 mois, sur 4 dates
  sur 4 ; confirmé Dual par le Combined, +7,7 pts à 6 mois et +37,1 pts à 12 mois. C'est la raison
  d'être de la forme « Finviz + Combined » de l'application (`core/combined_finviz.py`).
- **Valley** : INFLEXION est négatif (écart à l'indice médian de -5,7 pts à 3 mois, -38,8 pts à 12
  mois sur 38 signaux seulement). PIÈGE fonctionne comme filtre d'exclusion. DIVERGENCE n'est utile
  que sur les grandes capitalisations. Aucun des signaux ne bat clairement le hasard (`p_hasard`).

## Fichiers

| Fichier | Contenu |
|---|---|
| `combined/backtest_combined.md` | Rapport : profils, scores, Dual par date, sélections par critère, seuils par date |
| `combined/backtest_detail.csv` | Une ligne par titre et par date, avec les 12 critères |
| `combined/backtest_par_profil.csv`, `backtest_par_date.csv` | Synthèses reprises dans le rapport |
| `combined/backtest_finviz_9_absents.csv` | Titres détectés par Finviz mais absents du store, notés a posteriori |
| `combined/candidats_*.csv`, `.symbols.txt` | Sortie de `--candidats` (liste restreinte et univers noté) |
| `combined/indices.parquet`, `valley/cours.parquet` | Caches de cours (ignorés par git, recréés au besoin) |
| `valley/valley_backtest.md`, `valley_backtest_detail.csv` | Rapport et détail des signaux Valley |

Tests des briques : `src/tests/test_combined_backtest.py`, `src/tests/test_valley_backtest.py`.

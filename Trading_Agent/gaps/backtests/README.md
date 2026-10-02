# backtests/ : rejeu du classificateur sur l'historique

Produit par `backtest_gaps.py`. Les deux fichiers JSON du 26.09.2026 sont la
première mesure du classificateur sur un échantillon utilisable.

## Ce que le backtest mesure, et ce qu'il ne mesure pas

Il mesure **le classificateur, pas la découverte**. Le screener Finviz n'a pas
d'historique : impossible de reconstruire la liste qu'il aurait rendue le 15 août.
Le module part d'un univers de tickers, repère dans leur historique les séances
remplissant les critères, et juge le verdict. Un taux de justesse issu d'ici ne
dit rien de la couverture du screener.

La notation réutilise `evaluer_gaps.noter()`, donc le backtest juge avec les mêmes
promesses que la notation du soir.

## Résultats du 26.09.2026

37 tickers, deux ans, gap minimal 5 %, fenêtre de notation 3 séances.
**226 verdicts**, contre 20 auparavant.

| | Avec champs statiques | Sans (sensibilité) |
|---|---|---|
| Justesse globale | 170/226 (75 %) | 172/226 (76 %) |
| `FADE` | 61/64 (**95 %**) | 67/70 (96 %) |
| `PUMP_RISK` | 109/162 (67 %) | 105/156 (67 %) |

Le 75 % du backtest **coïncide avec le 15/20 mesuré en direct** sur deux journées.

`FADE` est la classe la plus fiable du dispositif, à 95 %, et sa notation n'est pas
circulaire : elle vérifie si le plus bas de la séance est revenu sous l'ouverture,
critère indépendant des alertes.

## Le résultat qui renverse une conclusion

La vente à découvert des `PUMP_RISK`, mesurée sur deux journées en septembre,
donnait +4,13 %. Sur 161 observations elle donne **−9,57 %**.

| | Valeur |
|---|---|
| Moyenne | **−9,57 %** |
| Médiane | +9,09 % |
| Gagnants | 109 sur 161 (68 %) |
| Gain moyen des gagnants | +21,6 % |
| Perte moyenne des perdants | **−74,8 %** |
| Pire cas | IPDN, 10.09.2026, **−1750 %** |

Médiane positive, espérance nettement négative : la distribution est écrasée par
quelques short squeezes. L'échantillon de deux journées n'en contenait aucun.

**Avec un stop, déclenché sur le plus haut de séance donc au pire moment du jour :**

| Stop | Espérance | Médiane | Stoppés |
|---|---|---|---|
| 10 % | +9,09 % | −10,00 % | 104/161 |
| 15 % | +10,35 % | −15,00 % | 90/161 |
| **20 %** | **+11,91 %** | **+2,86 %** | 73/161 |
| 30 % | +11,12 % | +5,81 % | 61/161 |
| 50 % | +10,82 % | +9,02 % | 39/161 |

Le stop à 20 % maximise l'espérance et rend la médiane positive. La zone 20 à 50 %
est plate (11,91 / 11,12 / 10,82), donc le réglage n'est pas ajusté au bruit.

**Conséquence de méthode : le signal n'a de valeur qu'avec un stop.** Sans lui la
stratégie est ruineuse malgré 68 % de réussite. Ce n'est pas le signal qu'il faut
améliorer, c'est la gestion du risque qu'il faut poser.

## Une justesse qui ne mesure rien

L'alerte « gap effacé » ressort à **47/47**. Elle se déclenche quand le cours passe
sous la clôture de la veille ; `PUMP_RISK` est noté juste quand le cours finit sous
l'ouverture du jour du gap, laquelle est par construction au-dessus de cette même
clôture. Les deux mesures sont liées par construction. **Ce 100 % n'établit aucun
pouvoir prédictif** et ne doit pas être cité comme tel.

## Ce que le backtest ne peut pas dire

- **Le borrow.** Tout le volet vendeur suppose qu'un emprunt existait et était
  abordable. Aucune source publique ne le donne, et sur des nano caps à faible
  flottant c'est l'hypothèse la plus douteuse du tableau.
- **Le spread et le slippage.** Non modélisés.
- **Le VWAP**, absent au-delà de trente jours d'historique. Le backtest reproduit
  donc le mode pré-marché, pas le mode séance.
- **Le catalyseur**, jamais disponible après coup. Fidèle à la production, qui ne
  le renseigne pas non plus, mais cela signifie qu'aucune classe exigeant un
  catalyseur (`CONTINUATION`, `SQUEEZE`, `A_CONFIRMER`) n'apparaît dans les 226.

Cette dernière limite est la plus lourde : le backtest ne juge que `FADE`,
`PUMP_RISK` et `INSUFFISANT`. Les trois classes haussières restent non mesurées.

## Exécution

```bash
SKILL=~/.claude/skills/gap-trading-germain/scripts
python3 $SKILL/backtest_gaps.py --annees 2 --min-gap 5 --jours 3 \
    --json backtests/$(date +%F)_univers-detections.json
python3 $SKILL/backtest_gaps.py --annees 2 --sans-statique   # sensibilité
```

L'univers par défaut est lu dans `../detections/`, donc il grandit à chaque
séance. Le cache `_statique.json` rend les rejeux suivants gratuits en requêtes.

## Re-mesure du 28.09.2026 : promesse FADE corrigée

`evaluer_gaps.noter()` juge désormais FADE sur la **clôture de la veille** (plus bas ≤
veille × 1,005) au lieu de « plus bas ≤ ouverture × 0,995 ». Rejeu sur le **même univers**
(les 37 titres des détections du 22 au 25.09), deux ans, gap ≥ 5 %, fenêtre 3 séances:
`2026-09-28_regle-fade-veille.json`, 234 verdicts (la fenêtre de deux ans a glissé).

| | 26.09 (ancienne règle) | 28.09 (règle corrigée) |
|---|---|---|
| `FADE` justesse | 61/64 (95 %) | **36/63 (57 %)** |
| `FADE` vente ouverture→clôture | +4,75 % | +2,27 % (médiane +3,33 %, 41/63 gagnants) |
| `PUMP_RISK` justesse | 109/162 (67 %) | 118/170 (69 %) |
| `PUMP_RISK` vente sans stop | −9,57 % | −7,88 % hors CTNT (voir ci-dessous) |
| `PUMP_RISK` vente, stop 20 % | +11,83 % | +12,21 % |

**Artefact de données : CTNT, séance du 25.09.2026, +10 488 %.** Le 28.09 le cours passe
de 0,032 à 4,17 $ pendant que le volume tombe de 507 M à 0,4 M: regroupement d'actions pas
encore ajusté par yfinance (dernier split connu: 29.04.2026). Cette ligne seule porte la
moyenne brute à −69,53 %; elle est exclue des chiffres ci-dessus. Le backtest ne détecte pas
encore ce cas; un garde-fou (saut de cours > 20× avec effondrement du volume) reste à écrire.

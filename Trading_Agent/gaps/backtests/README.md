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

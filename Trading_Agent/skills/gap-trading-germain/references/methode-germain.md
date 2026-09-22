# La méthode Germain appliquée aux gaps

Sources primaires : *Academy Germain — Formation aux Marchés Financiers*, Lionel
Germain, Parties 7 (formation des prix, manipulation), 9 (catalyseurs) et 12
(screeners et scanners). Les statistiques de comblement viennent de sources
externes, signalées comme telles.

---

## 1. Le filtre de base (Partie 12, Ch.01)

Configuration littérale de la formation pour trouver les gappers du matin :

| Filtre Finviz | Valeur | Raison donnée |
|---|---|---|
| Market Cap | Nano + Micro + Small | Mouvements amples possibles |
| Price | 0,50 $ à 20 $ | Zone des mouvements violents |
| Average Volume | > 500 K | Liquidité minimale ; sous 500 K, le trading actif est risqué |
| Relative Volume | > 2 | Activité anormale : premier filtre du matin |
| Gap | Up 5 %+ | Le mouvement lui-même |
| Float Short | > 10 % | Potentiel de short squeeze si catalyseur |

Puis, dans l'ordre : trier par `Change`, et **pour chaque action, vérifier le
catalyseur sur EDGAR ou les news**. Cette dernière étape n'est pas optionnelle
dans la méthode ; c'est elle qui sépare une opportunité d'une manipulation.

Le préréglage existe déjà dans le dépôt sous le nom `morning_gappers`
(`stock-analysis-ui/src/Code_Germain/Gap_Screen.py`) et dans l'interface via
`_show_finviz_gapper_screener`.

## 2. RVOL : l'échelle de lecture (Partie 7, Ch.01)

| RVOL | Lecture | Action prescrite |
|---|---|---|
| < 0,5 | Volume très faible | Ignorer, aucun intérêt pour le trading actif du jour |
| 0,5 – 1 | Normal | Aucun signal |
| 1 – 2 | Légèrement élevé | À surveiller |
| 2 – 5 | Significativement élevé | Chercher le catalyseur |
| > 5 | Anormal | News ? Catalyseur ? Manipulation ? |
| > 10 | Explosif | Vérifier **immédiatement** : news, 8-K, rumeur, squeeze en cours |

Définition retenue : RVOL = volume actuel / volume moyen **à la même heure**.
Comparer un volume de 10h00 à une moyenne journalière produit un chiffre faux.

## 3. VWAP : l'arbitre de direction (Partie 7, Ch.01)

Le cours sous son VWAP signifie que la pression vendeuse domine, quelle que soit
la couleur de la bougie journalière. L'exemple de la formation est net : VWAP à
1,0241 $ contre un cours à 0,82 $, soit 20 % sous le VWAP, pendant une séance qui
avait pourtant ouvert en forte hausse.

**Conséquence opérationnelle** : un gap up dont le cours repasse sous le VWAP n'est
plus un gap haussier en cours, c'est une distribution. Ce basculement est le
signal de sortie le plus simple à surveiller.

## 4. Squeeze contre pump : la discrimination obligatoire (Partie 7, Ch.03)

Les deux produisent une hausse rapide sur faible flottant. La distinction n'est pas
cosmétique : l'un peut se tenir plusieurs jours, l'autre retombe sous son niveau
d'avant-pump.

| Critère | Short squeeze (naturel) | Pump and dump (manipulation) |
|---|---|---|
| Short interest **avant** le mouvement | Élevé | Faible |
| Progression du volume | Élevé mais progressif | Explose très soudainement |
| Durée du mouvement | Continue sur plusieurs jours | Pump rapide puis dump brutal le même jour |
| Catalyseur | News réelle | Pas de news, ou news très vague |
| Origine des achats | Shorts qui couvrent (achats forcés) | Retail poussé par spam / hype |
| Après consolidation | Peut rebondir | Retombe sous le niveau d'avant-pump |

### Les sept signaux d'alarme du pump (mêmes pages)

1. Montée rapide sans catalyseur fondamental clair
2. Volume explosif **au sommet** — le pic de volume exactement au plus haut est une distribution
3. Nano cap et/ou low float — facile à manipuler
4. Chute rapide après le sommet
5. Cours retombant sous le VWAP
6. Email ou SMS de recommandation — « le signal le plus fiable du pump »
7. Absence de dépôts SEC récents

La formation insiste : le short squeeze n'est pas illégal, le pump and dump l'est,
**et la prudence s'impose dans les deux cas**.

## 5. Le catalyseur : ce qui compte et ce qui trompe (Partie 9)

Le gap doit avoir une cause nommable et datable. Hiérarchie :

- **Haussiers** : earnings au-dessus du consensus, approbation ou avancée FDA,
  contrat majeur, entrée d'un institutionnel visible en 13F (pour une microcap,
  la formation la qualifie de « catalyseur haussier potentiel fort »).
- **Baissiers déguisés en gap haussier** : dépôt S-3 ou 424B (dilution à venir),
  conversion d'actions privilégiées — « catalyseur baissier pour les actionnaires
  ordinaires ». Un gap up sur un S-3 récent est une alerte, pas une confirmation.
- **Spéculatifs** : gamma squeeze, options 0DTE (Partie 9, Ch.04) — mouvement réel
  mais mécanique, qui se dégonfle avec l'échéance.

Règle de composition donnée par la formation :
**Short Interest élevé + Low Float + Catalyseur = configuration de squeeze.**

## 6. Taille du gap et probabilité de comblement (source externe)

La méthode Germain ne chiffre pas les probabilités de comblement. Ces valeurs
viennent de statistiques publiées et sont utilisées ici **normalisées par l'ATR**,
parce qu'un gap de 5 % n'a pas le même sens sur un titre qui bouge de 2 % par jour
que sur un titre qui en bouge de 15 %.

| Taille du gap | Probabilité de comblement le jour même |
|---|---|
| < 0,3 × ATR | ~78 % |
| 0,3 – 0,7 × ATR | ~42 % |
| 0,7 – 1,2 × ATR | ~25 % |
| > 1,2 × ATR | ~8 % |

Deux règles complémentaires, même origine :

- **La règle des 30 minutes** : un gap qui survit aux 30 premières minutes sans
  être comblé bascule statistiquement du côté de la continuation. L'essentiel de
  l'action de comblement est très précoce (49 % des combles ES, 39 % NQ
  interviennent après la première demi-heure seulement).
- **Volume et catalyseur priment sur le pourcentage** : un gap avec RVOL ≥ 3 et un
  catalyseur clair tend vers la continuation ; un gap à faible volume sans news
  identifiable tend vers le comblement dans la première heure.
- Les gaps d'earnings se comblent nettement moins souvent que les gaps communs.

Sources : [TradingStats — When Do Gaps Fill](https://tradingstats.net/when-do-gaps-fill/),
[QuantifiedStrategies — Gap Fill Trading Strategies](https://www.quantifiedstrategies.com/gap-fill-trading-strategies/),
[TradeZella — Gap and Go](https://www.tradezella.com/blog/gap-and-go-strategy),
consultées le 22.09.2026.

## 7. Ce que la méthode ne dit pas

À déclarer dans le rapport plutôt qu'à combler par hypothèse :

- **Aucune taille de position chiffrée.** La formation donne le principe (low float
  = risque, donc position réduite) mais pas de formule. Le skill ne prétend pas en
  inventer une.
- **Pas de stop-loss chiffré.** Le VWAP sert de signal de sortie qualitatif ; le
  niveau exact relève du trader.
- **Données pré-marché absentes du Finviz gratuit.** La formation le note
  explicitement : les données pre-market sont réservées à Finviz Elite. Avant 9h30
  ET, la couverture du scan est partielle, et cette limite figure dans le rapport.

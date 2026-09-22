# gaps/ — détections de gaps et leur notation

Détections datées du skill `gap-trading-germain`, et le script qui les note
après coup. Complémentaire de [`agent/scan_market.py`](../agent/scan_market.py) :
celui-ci balaye via TWS et exige Gateway ouvert, celui-là passe par Finviz et ne
demande aucun broker.

## Cycle

```
13h00–15h30 (France)   détection      → detections/AAAA-MM-JJ_premarket.json
22h00–23h00 (France)   notation       → detections/AAAA-MM-JJ_premarket_evalue_J1.json
```

```bash
# Détection (fenêtre pré-marché : 7h00–9h30 ET, soit 13h00–15h30 en France)
python3 ~/.claude/skills/gap-trading-germain/scripts/gap_scan.py \
    --mode premarket --min-gap 5 --limit 40 \
    --json detections/$(TZ=America/New_York date +%F)_premarket.json

# Notation, le soir après la clôture US
python3 gaps/evaluer_gaps.py --jours 1        # la détection du jour
python3 gaps/evaluer_gaps.py --jours 3        # rouvrir 3 séances plus tard
```

## Ce que la notation mesure

Chaque verdict est confronté à **ce qu'il promettait**, pas à une hausse en
général :

| Verdict | Promesse | Juste si |
|---|---|---|
| `FADE` | le gap se comble | le bas de séance revient sous l'ouverture |
| `CONTINUATION` | le gap tient | clôture ≥ ouverture |
| `SQUEEZE` | le mouvement dure | encore ≥ ouverture après plusieurs séances |
| `PUMP_RISK` | le titre retombe | clôture < ouverture sur la fenêtre |
| `A_CONFIRMER` | rien : le verdict attend la règle des 30 min | noté `tenu` ou `comblé`, jamais juste/faux |
| `INSUFFISANT` | aucune | non jugé (écarté à la détection) |

Un verdict sans cours disponible ressort **incomplet** et n'est compté ni juste
ni faux. La règle du dépôt vaut ici aussi : une donnée absente ne vaut pas feu
vert, et elle ne vaut pas non plus succès.

## Deux pièges déjà rencontrés

**Le RVOL se calcule sur la médiane, pas la moyenne.** Mesuré le 22.09.2026 sur
VEEA : deux séances à 14 M de volume juste avant la détection tiraient la
moyenne à ~9 M et écrasaient le RVOL à 0,1 pour une séance ordinaire. La médiane
(163 K) montre le volume réellement typique du titre — et le fait basculer en
`INSUFFISANT`, ce qui est le bon verdict pour une nano cap qui ne traite
normalement pas 500 K titres par jour.

**Finviz gratuit n'a pas de données pré-marché.** Avant 9h30 ET, le champ `Gap`
reflète la dernière séance close, pas le jour à venir. Lancer la détection trop
tôt (avant 13h00 en France) produit une liste de la veille. Le script le signale
mais ne le corrige pas : il n'y a rien à corriger sans Finviz Elite.

## Le catalyseur reste manuel

Le scan ne vérifie aucun catalyseur. C'est délibéré et conforme à la méthode :
« RVOL élevé sans catalyseur fondamental = suspect ». Chaque ligne retenue exige
une vérification EDGAR (8-K, S-3, 424B) ou news **avant** toute entrée, et un
`S-3` ou `424B` récent est une alerte de dilution, pas une confirmation.

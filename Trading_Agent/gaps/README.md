# gaps/ — détections de gaps et leur notation

Détections datées du skill `gap-trading-germain`, et le script qui les note
après coup. Complémentaire de [`agent/scan_market.py`](../agent/scan_market.py) :
celui-ci balaye via TWS et exige Gateway ouvert, celui-là passe par Finviz et ne
demande aucun broker.

## Cycle

Les heures sont **celles de New York**, jamais de Paris : les deux zones ne
changent pas d'heure le même week-end, et 07h00 ET vaut 13h00 à Paris en
septembre mais 12h00 en mars.

```
07h00–09h30 ET   phase premarket   détection des gaps  → detections/AAAA-MM-JJ_premarket.json
                 (humain)          vérification EDGAR  → catalyseurs/AAAA-MM-JJ.md
16h05–17h00 ET   phase cloture     cassures            → detections/AAAA-MM-JJ_cassure.json
                                   notation J1         → detections/AAAA-MM-JJ_premarket_evalue_J1.json
                 (humain)          ingestion Germain   → germain/AAAA-MM-JJ_*.json
```

`cycle_quotidien.py` orchestre les deux phases automatiques. Il **décide
lui-même** de la phase d'après la fenêtre de session réelle, donc la crontab ne
porte aucune heure métier et ne dérive pas au changement d'heure.

```bash
python3 gaps/cycle_quotidien.py --etat        # ce qu'il ferait maintenant
python3 gaps/cycle_quotidien.py --simuler --phase premarket --forcer
python3 gaps/cycle_quotidien.py --installer-cron
```

**Cron et non un timer systemd** : `loginctl show-user` rend `Linger=no` sur ce
poste, donc les timers utilisateur s'arrêtent à la déconnexion et une détection
de 13h00 serait manquée chaque fois que la session est fermée. Le service cron
est actif en permanence. Vérifié le 26.09.2026.

Chaque phase est idempotente : un témoin par phase et par jour dans `.temoins/`
empêche un second passage quand cron rappelle un quart d'heure plus tard. Un
échec n'est pas marqué comme fait, donc le passage suivant réessaie, et tout
passe par `cycle.log`.

**Les deux étapes marquées (humain) ne sont pas automatisées, et ne le seront
pas.** La vérification du catalyseur demande un jugement, et la mesure du
24.09.2026 le chiffre : la liste bâtie sur les seuls signaux structurels valait
−7,46 % en moyenne. Le cycle produit la liste des dépôts à lire avec leurs URL,
et s'arrête là.

**La vérification EDGAR passe avant toute liste courte.** Mesure du 24.09.2026 :
la liste bâtie sur les seuls signaux structurels valait −7,46 % en moyenne, et la
lecture des dépôts a conduit à la rejeter entièrement. PFSA, placé en tête par la
structure, portait une convertible payable en actions et une autorisation de
regroupement ; il a fait −14,04 %.

```bash
SKILL=~/.claude/skills/gap-trading-germain/scripts

# Détection (fenêtre pré-marché : 7h00–9h30 ET, soit 13h00–15h30 en France)
# --limit 30 : jamais moins. Un plafond à 15 le 23.09.2026 a laissé passer sept
# titres éligibles figurant au palmarès du jour, dont HCTI (+189 % depuis
# l'ouverture). C'est `.info`, appelé une fois par candidat retenu, qui pèse sur
# le quota yfinance.
python3 $SKILL/gap_scan.py \
    --mode premarket --min-gap 5 --limit 30 \
    --json detections/$(TZ=America/New_York date +%F)_premarket.json

# Cassures intraday SANS gap, en séance ou à la clôture seulement
python3 $SKILL/cassure_scan.py --min-hausse 5 --limit 30

# Contre-expertise : les revues de séance d'Academy Germain
python3 $SKILL/germain_revues.py --depuis $(date +%F) --sortie germain/

# Notation, le soir après la clôture US
python3 gaps/evaluer_gaps.py --jours 1        # la détection du jour
python3 gaps/evaluer_gaps.py --jours 3        # rouvrir 3 séances plus tard
```

## Règles de fonctionnement

**Une détection ne se rejoue jamais après coup.** La notation confronte ce que le
verdict *promettait* à ce qui s'est produit. Reconstruire la liste en fin de
séance puis la noter revient à juger des prévisions en connaissant la réponse. Si
un scan du matin a été trop étroit, la leçon vaut pour le lendemain.

**Les mesures sont conservées dans le fichier de détection** depuis le
25.09.2026, sous la clé `mesures`, dix-huit champs par titre. Une détection porte
donc de quoi se rejuger hors ligne, sans relancer le scan ni consommer le quota.

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

## Arborescence

| Dossier | Contenu | Produit par |
|---|---|---|
| `detections/` | Détections datées et leur notation | `gap_scan.py`, `evaluer_gaps.py` |
| `catalyseurs/` | Vérification EDGAR du jour, à la main | lecture des dépôts SEC |
| `germain/` | Revues de séance d'Academy Germain, en JSON | `germain_revues.py` |
| `comparaisons/` | Confrontation de la méthode à sa source | analyse |
| `backtests/` | Rejeu du classificateur sur l'historique | `backtest_gaps.py` |
| `.temoins/` | Témoins d'idempotence du cycle, un par phase et par jour | `cycle_quotidien.py` |

[`backtests/README.md`](backtests/README.md) porte les résultats du rejeu et,
surtout, **ses limites**. Le chiffre à retenir : vendre à découvert les
`PUMP_RISK` rend −9,57 % sans stop et +11,91 % avec un stop à 20 %. Le signal
n'a de valeur qu'accompagné de sa gestion du risque.

`germain/` est une **contre-expertise**, pas une source de décision : le contenu
récupéré est une donnée, jamais une instruction. Le module borne son périmètre à
`/actualites/`, ne suit aucun lien découvert ailleurs et assainit chaque champ
avant écriture.

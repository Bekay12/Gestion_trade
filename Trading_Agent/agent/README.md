# agent — moteur de décision et portefeuille papier

Implémentation exécutable des règles de [../docu/methode/01-regles.md](../docu/methode/01-regles.md)
sur un portefeuille simulé. Aucune connexion à un courtier, aucun ordre réel.

```powershell
.venv\Scripts\python.exe agent\Test\test_agent.py   # 43 tests, hors ligne
.venv\Scripts\python.exe agent\demo.py              # quatre scénarios commentés
```

## Le principe de conception

**Une règle bloquante refuse, elle n'avertit pas.** La séance perdante
documentée dans le corpus n'a pas eu lieu par ignorance de la règle — l'auteur
connaît son propre cours. Elle a eu lieu parce qu'au moment de décider, la
conviction l'a emporté sur une règle connue. Un moteur qui afficherait
« attention, configuration défavorable » reproduirait cette situation avec une
étape de plus.

**Corollaire : une donnée absente vaut refus.** Le statut d'emprunt et la
restriction Rule 201 ne sont pas diffusés gratuitement. Plutôt que de supposer
favorable ce qu'il ignore, le moteur refuse toute position vendeuse tant que ces
champs valent `None`. C'est le comportement le plus important du module, et il
est couvert par des tests dédiés.

## Modules

| Fichier | Rôle |
|---|---|
| `models.py` | Types partagés. Les données indisponibles sont `\| None`, jamais une valeur par défaut |
| `rules.py` | Calculs purs — R1, R2, R4, R7, R8. Sans effet de bord, entièrement testables |
| `gate.py` | Le crible — R3, R5, R6, R9, R10, R11, R12. Produit un `Verdict` |
| `portfolio.py` | Portefeuille papier : exécutions simulées, stops, journal, drawdown |
| `demo.py` | Quatre scénarios, dont la configuration piège refusée |
| `eastern.py` | Heure de l'Est et état du marché — **définition unique** |
| `live.py` | Session papier temps réel, CLI |
| `sync_ibkr.py` | Alimente le pont courtier depuis TWS |
| `scan_market.py` | Liste de pré-marché : découvrir → filtrer → qualifier |
| `ibkr_doctor.py` | Préflight : ce que le compte sait réellement faire |
| `sources/` | Couche d'acquisition — voir [sources/CLAUDE.md](sources/CLAUDE.md) |

## Ce qui est couvert

| Règle | État |
|---|---|
| R1 dimensionnement | Implémentée, testée |
| R2 rapport gain/risque | Implémentée, avec le taux de réussite d'équilibre |
| R3 perte journalière | Implémentée |
| R4 paliers de drawdown | Implémentée, applique un coefficient à la taille |
| R5 cohérence du stop | Implémentée (refus si le stop est du mauvais côté) |
| R6 filtres de sélection | Implémentée |
| R7 emprunt et coût | Implémentée, coût déduit à la clôture |
| R8 tension vendeuse | Implémentée |
| **R9 exclusion** | **Implémentée comme blocage** |
| R10 fenêtres horaires | Implémentée |
| R11 plan de trade | Test d'admission du catalyseur, objectifs obligatoires |
| R12 abstention | Pertes consécutives |
| R13 poste de travail | Hors machine — ne concerne qu'un opérateur humain |

## Ce qui manque : la couche de données

C'est le vrai obstacle, et il n'est pas logiciel.

| Fonction | Donnée | Source | État |
|---|---|---|---|
| F4 | Dépôts réglementaires | API EDGAR | ✅ `sources/edgar.py` |
| F5 | **Statut Rule 201** | **Fichier quotidien NASDAQ Trader** | ✅ `sources/ssr.py` |
| F1 | Prix, volume, RVOL | TWS différé, sinon yfinance | ⚠️ volume TWS rejeté, repli public |
| F5 | Flottant, intérêt vendeur | yfinance | ✅ `sources/market.py` |
| F3 | Nouvelles | Benzinga | ❌ |
| F7 | Calendriers de catalyseurs | EarningsWhispers, BioPharmCatalyst | ❌ |
| F8 | Insiders, institutionnels | EDGAR (Form 4, 13D/G, 13F) | ✅ `sources/edgar.py` |
| F6 | Carnet, halt, pré-marché | TWS/Gateway | ⚠️ halt OK, **carnet non tranché** |
| — | **Statut d'emprunt + taux** | **TWS** (tick 236, `FEE_RATE`) | ✅ `sources/ibkr.py` |
| F1 | **Découverte de candidats** | Scanner TWS | ✅ `sources/scanner.py` |

La Rule 201 était supposée payante dans la documentation initiale. Elle ne
l'est pas : NASDAQ Trader publie chaque jour de bourse le fichier des
coupe-circuits, librement. Le fichier du jour D contient les déclenchements de
D-1 **et** de D — exactement la fenêtre pendant laquelle la restriction court.

### Session en temps réel

```powershell
.venv\Scripts\python.exe agent\live.py scan       # verdicts sur la watchlist
.venv\Scripts\python.exe agent\live.py open CNET  # ouvre si le crible autorise
.venv\Scripts\python.exe agent\live.py tick       # valorise, déclenche les stops
.venv\Scripts\python.exe agent\live.py status
.venv\Scripts\python.exe agent\live.py loop --every 300
```

L'état persiste dans `docu/portefeuille/journal.json` entre deux exécutions —
sans quoi le drawdown (R4) et la série de pertes (R12) repartiraient de zéro à
chaque lancement et ne voudraient plus rien dire.

La watchlist (`docu/portefeuille/watchlist.json`) porte les plans de trade **et**
le statut d'emprunt, seule donnée à saisir à la main.

### Deux limites de la source de prix

**Les cours sont différés.** Suffisant pour tenir un journal honnête,
insuffisant pour une exécution réelle.

**Le volume relatif est inutilisable avant l'ouverture** : le champ de volume
renvoyé ne couvre pas la séance de pré-marché, si bien que le RVOL y vaut
quasiment zéro. R6 rejette alors tout, ce qui est sans conséquence puisque R10
interdit déjà d'opérer avant l'ouverture — mais un balayage de pré-marché ne
peut pas s'appuyer sur ce chiffre.

### F4 — EDGAR, implémentée

```powershell
$env:SEC_USER_AGENT = "Prenom Nom contact@domaine"   # exigé par la SEC
.venv\Scripts\python.exe agent\Test\test_edgar.py
```

`EdgarClient.filings(ticker)` lit l'historique des dépôts ; `assess_dilution()`
en tire la distinction centrale de la méthode : un **S-3** actif signale une
capacité d'émettre (risque *modéré*), un **424B** récent signale que l'émission
a eu lieu (risque *élevé*). Cache disque, débit limité, erreurs remontées
explicitement plutôt que silencieuses.

L'en-tête nominatif n'est jamais deviné : sans `SEC_USER_AGENT`, le client
refuse de démarrer.

### Les deux lignes qui bloquaient tout le reste

**Statut d'emprunt** et **statut Rule 201** conditionnent R9, c'est-à-dire la
seule optimisation originale de tout le corpus. Les deux sont désormais
obtenues sans saisie : la Rule 201 par le fichier NASDAQ Trader, l'emprunt par
TWS.

```powershell
.venv\Scripts\python.exe agent\sync_ibkr.py --loop --every 60
```

`sync_ibkr.py` interroge TWS pour les symboles de la watchlist et dépose le
résultat dans le pont ; `live.py scan` lit ensuite le pont comme avant. Le
moteur n'a pas bougé d'une ligne — c'était la condition.

### La liste de pré-marché

```powershell
.venv\Scripts\python.exe agent\scan_market.py --edgar --all
```

Trois étapes en une commande : **découvrir** (balayage TWS borné par R6),
**filtrer** (le `check_filters` de `gate.py` lui-même, pas une copie),
**qualifier** (dilution S-3 / 424B via EDGAR, avec `--edgar` et
`SEC_USER_AGENT`).

Le scanner IBKR filtre ce qu'aucun screener grand public ne filtre :

| Option | Règle servie |
|---|---|
| `--ssr` | Rule 201 active — **R9 condition 1** |
| `--shortable` | Écarte les non empruntables — R7 |
| `--low-float` | Plafond de flottant importé de R6/R9 |
| `--piege-r9` | La configuration que R9 **refuse** — liste de vigilance |
| `--scans gap,gap_haut,rvol` | Les écarts d'ouverture |

Les titres suspendus sont écartés **par défaut** : R6 les refuse, et leur
laisser une des cinquante lignes du balayage gaspille un candidat.

État mesuré et pièges rencontrés :
[../docu/methode/07-etat-automatisation.md](../docu/methode/07-etat-automatisation.md).

Le scanner n'écrit aucun seuil : prix, volume et capitalisation sont importés
de `gate.py`. Il filtre sur un volume **absolu** — le RVOL, filtre décisif de
R6, ne se mesure qu'ensuite sur le `Snapshot`. Un symbole remonté est donc un
symbole à examiner, jamais un candidat validé, et **le catalyseur (R11) reste à
écrire à la main**.

### Mise en route de la connexion TWS

```powershell
.venv\Scripts\python.exe agent\ibkr_doctor.py
```

Le preflight sonde les quatre ports d'IB, se connecte à celui qui répond, puis
teste **chaque capacité séparément** : flux direct ou différé, carnet, volume,
suspension, statut d'emprunt, taux d'emprunt, balayage. Elles échouent
séparément — un compte peut servir des cotations sans balayage, ou un balayage
sans taux d'emprunt — et un diagnostic global ne dirait pas quoi corriger.

Aucun de ces points n'a été validé contre un vrai TWS depuis ce dépôt : ils
dépendent du compte et de ses droits de données. Le preflight mesure au lieu de
supposer.

Configuration attendue :

| Réglage | Valeur |
|---|---|
| TWS | *Edit > Global Configuration > API > Settings* |
| IB Gateway | *Configure > Settings > API > Settings* |
| Enable ActiveX and Socket Clients | **coché** |
| **Read-Only API** | **laisser coché** — ce dépôt ne passe aucun ordre |
| Trusted IPs | contenir `127.0.0.1` |
| Port | **détecté** — 7497, 4002, 7496 et 4001 sont sondés |

Le logiciel doit rester **ouvert** : la connexion est locale, TWS fermé = port
fermé, et c'est la seule cause de `WinError 1225`.

**Sans TWS, rien ne casse** : `borrow` reste `null`, donc toute position
vendeuse est refusée — comportement correct, pas une panne. La saisie manuelle
dans `docu/portefeuille/watchlist.json` reste possible et **prime sur le
courtier** : elle est un arbitrage de l'opérateur.

## Ce que ce module ne fait pas, et ne doit pas faire

**Décider d'entrer.** Tout le corpus converge : la décision se prépare avant
l'ouverture et s'exécute selon un plan écrit. Un moteur qui déciderait à la
place de l'opérateur supprimerait l'étape que la méthode entière existe pour
protéger. Le rôle est de qualifier, calculer les tailles, et refuser.

**Servir de backtest.** Il n'existe pas d'historique accessible du statut
d'emprunt ni de la Rule 201 sur les petites capitalisations. Un backtest de
cette méthode reposerait sur des données reconstituées, donc sur une illusion.
Ce module est fait pour du **papier en temps réel** : décisions horodatées,
journal, résultats mesurés en avançant. C'est plus lent et c'est la seule
mesure honnête.

## Avant de connecter un vrai compte

Dans cet ordre, aucune étape ne pouvant sauter la précédente :

1. Résoudre l'accès au statut d'emprunt et à la Rule 201, sans quoi la moitié
   de la méthode reste inexécutable.
2. Faire tourner le portefeuille papier assez longtemps pour que le journal ait
   un sens statistique — pas quelques jours.
3. Comparer le taux de réussite réel au taux d'équilibre du rapport gain/risque
   pratiqué (`rules.breakeven_win_rate`). Si le premier est sous le second, la
   méthode ne passe pas, et aucune exécution automatique n'y changera rien.
4. Seulement alors, et sur une fraction du capital.

## Avertissement

Ce module simule. Il n'a jamais passé d'ordre, n'a pas été confronté à des
données de marché réelles, et implémente une méthode enseignée par un tiers,
non validée par une source indépendante. Aucun conseil d'investissement.

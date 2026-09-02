---
niveau: 2
maj: 2026-08-23
source: etat du depot, verifie par lecture du code et par mesure sur un Gateway reel
---

# État de l'automatisation

Où en est le moteur face à la méthode, fonction par fonction. Document
d'**état**, pas de conception : il dit ce qui tourne aujourd'hui, pas ce qui
devrait tourner. La conception est en
[05-specification-agent.md](05-specification-agent.md).

> **À actualiser à chaque évolution du code.** La procédure est en fin de
> document. Un état des lieux faux est pire que pas d'état des lieux : il fait
> croire qu'une fonction est couverte alors qu'elle est saisie à la main.

## Le niveau en une phrase

**Le pipeline est complet de la découverte au verdict, sauf la lecture du
catalyseur.** Le moteur trouve, filtre, dimensionne et refuse ; il ne sait
toujours pas dire si une nouvelle est haussière ou dilutive.

| | Fonctions |
|---|---|
| ✅ Automatisé | 5 |
| ⚠️ Partiel ou dégradé | 3 |
| ❌ Manuel ou absent | 3 |
| **Coût mensuel** | **0 $** — voir la section coût |

Les acquis des 22 et 23 août 2026 sont conditionnés à **TWS ou IB Gateway
ouvert**. Fermé, le dispositif retombe sur l'état antérieur : cours différés,
emprunt saisi à la main, watchlist manuelle. C'est une dégradation annoncée,
pas une panne — le balayage le dit au lieu de rendre une liste vide qui
ressemblerait à une séance sans candidat.

## Où s'arrête l'automatisation

```
  PIPELINE DE LA METHODE          ETAT DE L'AUTOMATISATION
  ─────────────────────────────   ────────────────────────
  1. Balayage du marche (F1)      ✓ scanner.py (si TWS)   mesure OK
  2. Nouvelle / catalyseur (F3)   ✗ MANUEL  <- le mur (couche 2)
  3. Qualification depots (F4)    ✓ edgar.py              gratuit
  4. Flottant / interet vendeur   ~ market.py (yfinance, differe)
  5. Rule 201 (F5)                ✓ ssr.py                gratuit
  6. Statut d'emprunt + taux      ✓ ibkr.py (si TWS)      mesure OK
  7. Carnet / halt / pre-marche   ~ ibkr.py — carnet NON tranche
  8. Plan de trade (R11)          ~ watchlist.json a la main
  9. Crible + dimensionnement     ✓ gate.py + rules.py    COMPLET
 10. Journal / drawdown           ✓ portfolio.py + live.py
 11. Execution (F6)               ✗ hors perimetre volontaire
```

## Tableau 1 — Fonction par fonction : son outil vs le nôtre

Les fonctions F1 à F8 sont définies en [03-outils.md](03-outils.md), les outils
nommés en [06-inventaire-outils.md](06-inventaire-outils.md).

| # | Fonction | Outil du cours | Ce qu'on utilise | État |
|---|---|---|---|---|
| F4 | Dépôts réglementaires | SEC EDGAR (101 mentions) | **API EDGAR directe** | ✅ complet, gratuit |
| F5 | Rule 201 | Fintel.io / Short-selling.com | **Fichier NASDAQ Trader** | ✅ complet, gratuit |
| F8 | Insiders / institutionnels | OpenInsider, WhaleWisdom, Dataroma | **EDGAR** (Form 4, 13D/G, 13F) | ✅ complet, gratuit |
| F1 | **Découverte de candidats** | Finviz screener | **`scanner.py`** (TWS) | ✅ mesuré, bornes importées de R6 |
| — | **Statut d'emprunt + taux** | Courtier (bouton *Locate*) | **`ibkr.py`** : tick 236 + `FEE_RATE` | ✅ **verrou levé**, mesuré |
| F1 | Prix, volume, RVOL | Finviz | TWS différé, sinon yfinance | ⚠️ différé, volume décodé |
| F5 | Flottant, intérêt vendeur | Floatchecker | yfinance | ⚠️ `None` si absent, jamais approximé |
| F6 | Carnet, halt, pré-marché | DAS Trader / IBKR | `ibkr.py` → `broker.py` | ⚠️ halt OK, **carnet non tranché** |
| F2 | Veille temps réel | TradeIdeas (~100-180 $/m) | rien | ❌ |
| F3 | Nouvelles rapides | Benzinga Pro | rien | ❌ |
| F7 | Calendriers catalyseurs | EarningsWhispers, BioPharmCatalyst | rien | ❌ |

**Trois alternatives sont meilleures que les outils nommés par le cours.**
[06-inventaire-outils.md](06-inventaire-outils.md) annonçait la Rule 201 comme
« la seule dépense réellement structurante » : NASDAQ Trader la publie
librement. EDGAR rend OpenInsider / WhaleWisdom / Dataroma redondants — ils
agrègent ce qu'on lit à la source. Et **le scanner IBKR filtre sur la
restriction Rule 201, sur le caractère non empruntable et sur la suspension**,
ce qu'aucun screener grand public ne sait faire : trois des quatre conditions de
R9 sont donc filtrables **à la source**.

Ce que Finviz garde en propre : un filtre d'**intérêt vendeur**, absent du
scanner IBKR. On le récupère via yfinance, après le balayage.

## Tableau 2 — Contre la spécification en trois couches

Découpage de [05-specification-agent.md](05-specification-agent.md).

| Couche | Contenu | État |
|---|---|---|
| **3 — Garde-fous** (R3, R4, R9, R10 bloquants) | La plus importante selon la spec | ✅ **complète et testée** |
| **1 — Filtrage et calcul** (R1, R2, R6, R7, R8) | Déterministe | ✅ complète — la liste d'entrée existe enfin |
| **2 — Qualification** (lire un dépôt, juger haussier ou dilutif) | « là où un modèle de langage a une valeur réelle » | ❌ **absente en code** |

Le point à ne pas confondre : `assess_dilution()` classe les **formulaires**
(S-3 = autorisation d'émettre, 424B = émission effectuée), mais rien ne **lit**
le contenu d'un 8-K pour dire si le catalyseur est haussier ou dilutif. Ce
jugement se fait à la main, en écrivant le champ `catalyst` de la watchlist.

**185 tests hors ligne**, sept suites, ni réseau ni compte — voir
[../../agent/Test/CLAUDE.md](../../agent/Test/CLAUDE.md).

## Ce qui a été mesuré sur un Gateway réel

Le 2026-08-23, compte papier `DUR062454`, IB Gateway port 4002, **marché fermé
(dimanche)**. Mesuré par [`ibkr_doctor.py`](../../agent/ibkr_doctor.py), qui
teste chaque capacité séparément parce qu'elles échouent séparément.

| Capacité | Résultat |
|---|---|
| Connexion, compte papier | ✅ |
| Qualification de contrat (`conId`) | ✅ — obligatoire, un `Stock` brut est refusé |
| **Statut d'emprunt** (tick 236 → 46) | ✅ `'easy'` sur AAPL |
| **Actions empruntables** (tick 89) | ✅ 82 104 556 |
| **Taux d'emprunt** (`FEE_RATE`) | ✅ 0,0 %/an — plausible pour AAPL |
| **Balayage** (scanner) | ✅ 10 symboles, petites capitalisations dans R6 |
| Prix | ⚠️ en différé seulement (type 3) |
| Volume | ⚠️ virgule fixe 10⁶, **décodé et validé** |
| Carnet (bid/ask) | ❓ **non tranché** — marché fermé |

**Les deux verrous annoncés levés le sont réellement.** R7 et R9 sont
calculables sans saisie, et sans abonnement payant.

### Quatre pièges découverts en confrontant le code au vrai Gateway

**Un compte sans abonnement ne reçoit RIEN en direct**, pas une version
dégradée : erreur 10089 et zéro tick. Le repli sur le différé est automatique
(`auto_delayed`), journalisé, et `realtime` reste `False` — jamais de différé
présenté comme du direct.

**Le volume différé arrive en virgule fixe, échelle 10⁶** — élucidé le
2026-08-23. TWS envoie lui-même `48 591 578 764 254` pour AAPL : le callback brut
porte déjà cette valeur, alors que le tick 89 arrive juste **par le même chemin
de code**. Ce n'est donc ni `ib_async`, ni la locale — c'est l'encodage du tick
74. Divisée par un million : 48,6 M, du même ordre que les 42,2 M de la source
publique, et corroboré sur cinq titres couvrant trois ordres de grandeur
(JUNS 52,4 M · SDOT 19,6 M · ADXN 727 069 · AAOX 97 242).

`session_volume()` décode, mais **seulement si la valeur brute est impossible ET
que la valeur décodée devient plausible**. Hors de cette fenêtre elle refuse et
le moteur retombe sur le volume public. Une mise à l'échelle appliquée au jugé
fausserait le RVOL dans le sens permissif, ce qu'une règle bloquante ne doit
jamais faire. Mesure limitée au flux **différé** : à revalider en direct.

**Un filtre refusé rend une liste VIDE, pas une erreur.** Erreur 10360,
« Scan filter `floatSharesBelow` is not allowed » — arrivée par le canal
d'erreurs, sans exception, requête rendant zéro symbole. Un marché calme y
ressemble trait pour trait. `scan()` écoute désormais `errorEvent` et lève.

**Hors séance, les balayages intrajournaliers rendent du bruit**, pas une
erreur : un ordre **alphabétique**, faute de classement à calculer (`rvol` a
rendu AADX, AAL, AAOG, AAOX, AAOZ). Avertissement posé quand le marché est
fermé — et le préflight, lui, n'impute plus à un droit manquant ce qui n'est
qu'un week-end.

## Ce que coûterait le temps réel

| Abonnement IBKR | Prix/mois | Waiver |
|---|---|---|
| US Securities Snapshot and Futures Value Bundle | **10 $** (non-professionnel) | Gratuit si ≥ 30 $ de commissions |
| US Equity and Options Add-On Streaming Bundle | 4,50 $ | Aucun |
| Snapshots à la demande | 0,01 $/requête (actions US) | 1 $/mois offert |

**Ce que ces 10 $ achèteraient pour ce projet : le carnet, et lui seul.** Prix,
flottant, intérêt vendeur, Rule 201, emprunt, taux et balayage fonctionnent
déjà gratuitement. Sans carnet, `spread_pct` vaut `None` et le garde-fou d'écart
de R6 (2 % / 5 %) est **inactif** — un contrôle qui ne s'exerce pas, ce qui vaut
mieux d'être su.

**La décision est suspendue**, parce que la mesure a eu lieu un dimanche : un
carnet vide hors séance est normal. À trancher en relançant le préflight en
séance.

### Les alternatives gratuites, et leur piège

| Source | Ce qu'elle donne | Le piège |
|---|---|---|
| **Alpaca** | Bid/ask temps réel, 200 req/min | **IEX seulement** — une place, ~2 % du volume |
| **Finnhub** | Cotations temps réel, 60 req/min | Basé IEX également |
| **Tiingo**, Twelve Data | Cotations, quotas généreux | Idem |
| **Finviz gratuit** | Screener web, 15 min de retard | **Pas d'API** — scraper violerait les CGU |
| yfinance *(en place)* | Prix, volume, flottant, intérêt vendeur | ~15 min de retard |

Un écart calculé sur IEX seul **n'est pas le NBBO** : il est systématiquement
plus large. R6 refuserait des titres négociables, pour un écart qui n'existe
pas. Ce n'est pas prudent, c'est faux dans le sens du bruit — et une règle qui
refuse pour de mauvaises raisons finit contournée, ce que ce projet existe pour
empêcher.

**Aucune source gratuite ne donne le statut d'emprunt.** Ni Alpaca, ni Finnhub,
ni Finviz, ni yfinance. C'est IBKR ou rien — et il fonctionne déjà, en différé,
gratuitement.

## Les leviers livrés

**Découverte de candidats (F1)** — 2026-08-23.
[`scanner.py`](../../agent/sources/scanner.py) balaie,
[`scan_filters.py`](../../agent/sources/scan_filters.py) porte les filtres, et
[`scan_market.py`](../../agent/scan_market.py) enchaîne la routine de
pré-marché : découvrir → filtrer par le **vrai** `check_filters` de R6 →
qualifier la dilution (F4, avec `--edgar`).

Point d'architecture : le scanner **n'écrit aucun seuil**, il les importe de
`gate.py`. Un test lit le fichier source et échoue si un chiffre de R6 y
réapparaît en dur. Défaut volontairement modeste — 10 lignes par balayage —
parce que le cours signale qu'un volume d'alertes trop grand produit l'effet
inverse de celui recherché.

Nuance à ne pas perdre : IB n'offre **aucun filtre** de volume relatif, mais un
**classement** (`rvol`). Le RVOL, filtre décisif de R6, ne se mesure qu'ensuite
sur le `Snapshot`. Le balayage retrécit, `gate.py` décide.

**Connexion TWS/IB Gateway** — 2026-08-22.
[`ibkr.py`](../../agent/sources/ibkr.py) est le producteur,
[`sync_ibkr.py`](../../agent/sync_ibkr.py) le point d'entrée. Le pont JSON n'a
**pas** été remplacé : il est devenu la couture entre un vrai producteur et un
moteur qui n'a pas bougé d'une ligne. C'est ce qui permet au journal papier de
tourner sans TWS, et aux 185 tests de rester hors ligne.

### Carte des modules ajoutés

| Module | Rôle |
|---|---|
| [`sources/ibkr.py`](../../agent/sources/ibkr.py) | Connexion, requêtes, publication dans le pont |
| [`sources/ib_ticks.py`](../../agent/sources/ib_ticks.py) | Traduction pure des ticks, seuils, garde-fous |
| [`sources/ib_ports.py`](../../agent/sources/ib_ports.py) | Où TWS écoute — sonde les quatre ports |
| [`sources/scanner.py`](../../agent/sources/scanner.py) | Balayage, borné par R6 |
| [`sources/scan_filters.py`](../../agent/sources/scan_filters.py) | Filtres Rule 201 / non empruntable / halt |
| [`eastern.py`](../../agent/eastern.py) | Heure de l'Est et état du marché, **définition unique** |
| [`ibkr_doctor.py`](../../agent/ibkr_doctor.py) | Préflight : ce que le compte sait réellement faire |
| [`scan_market.py`](../../agent/scan_market.py) | Liste de pré-marché en une commande |
| [`sync_ibkr.py`](../../agent/sync_ibkr.py) | Alimente le pont depuis TWS |

### Prérequis TWS

Le port se **détecte** : les quatre défauts (7497, 4002, 7496, 4001) sont
sondés, le premier qui répond est utilisé. Une erreur de port produisait le même
message qu'un logiciel fermé, ce qui envoyait chercher la panne au mauvais
endroit.

| Réglage | Valeur |
|---|---|
| TWS | *Edit > Global Configuration > API > Settings* |
| IB Gateway | *Configure > Settings > API > Settings* |
| Enable ActiveX and Socket Clients | coché |
| **Read-Only API** | **laisser coché** — ce dépôt ne passe aucun ordre |
| Trusted IPs | contenir `127.0.0.1` |

`ib_async` est une dépendance **optionnelle** : son absence produit un message
explicite, pas un plantage.

## Le prochain levier

**La couche 2 — la qualification du catalyseur.** Seul trou structurant
restant, et le pipeline vient de le rendre visible : `scan_market.py` produit
une liste filtrée par R6 et annotée du risque de dilution, puis s'arrête sur la
ligne « le catalyseur reste à établir à la main (R11) ».

Ce qui manque précisément : **lire un 8-K et dire s'il est haussier ou
dilutif**. `assess_dilution()` classe les *formulaires*, pas leur contenu.

Deux garde-fous à poser avant d'écrire une ligne :

1. **Le jugement d'un modèle n'est pas une donnée de marché.** Il produit un
   avertissement documenté, jamais un feu vert. La règle du dépôt — donnée
   absente vaut refus — devient : jugement incertain vaut refus.
2. **R11 reste un test d'admission humain.** Le catalyseur doit tenir en une
   phrase écrite par l'opérateur. Un catalyseur rédigé par la machine
   supprimerait l'étape que la méthode entière existe pour protéger.

Après seulement, la question d'un abonnement (F2, F3, F7) se pose. La réserve du
cours tient : payer des alertes qu'on ne sait pas qualifier reste la dépense la
moins rentable de la liste.

## Questions ouvertes

| Question | Comment trancher |
|---|---|
| Le carnet différé répond-il en séance ? | `python agent/ibkr_doctor.py` entre 10 h et 11 h 30 (Est) |
| Le décodage 10⁶ vaut-il aussi en flux **direct** ? | Mesuré sur le différé seulement — relancer le préflight avec un abonnement |
| `floatSharesBelow` marche-t-il ailleurs ? | Dépend du compte ; `--low-float` reste disponible |

## Comment actualiser ce document

Trois évènements l'invalident. Dans chaque cas, corriger **le tableau
concerné** et la date `maj:` du frontmatter.

| Évènement | Ce qui change ici |
|---|---|
| Un module de `agent/sources/` est ajouté ou change d'état | Tableau 1, la carte des modules, et les compteurs ✅/⚠️/❌ |
| Une couche de la spec avance | Tableau 2, et le diagramme du pipeline |
| Une mesure est faite sur un vrai Gateway | « Ce qui a été mesuré », et les questions ouvertes |

Deux fichiers renvoient ici pour que l'oubli se voie :
[../../agent/sources/CLAUDE.md](../../agent/sources/CLAUDE.md), dans sa
checklist « Ajouter une source » — le déclencheur le plus fréquent — et
[../../agent/CLAUDE.md](../../agent/CLAUDE.md), en tête de ses invariants.

**Ne pas dupliquer ici** ce que porte déjà
[../../agent/README.md](../../agent/README.md) : ce dernier détaille l'état
règle par règle (R1…R13). Ce document raisonne par **fonction** (F1…F8) et par
**couche**. Deux axes, une seule vérité — si les deux se contredisent, c'est le
code qui tranche.

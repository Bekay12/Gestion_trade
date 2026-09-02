---
niveau: 2
maj: 2026-08-21
source: cours Academy Germain (35 chapitres du noyau day-trading)
---

# Inventaire nommé des outils

Tous les sites, applications et plateformes effectivement cités dans le corpus.
[03-outils.md](03-outils.md) raisonne par **fonction** ; ce document donne les
**noms**.

## Avertissement de provenance

**Ces outils viennent du cours, pas de la chaîne.** Les 13 transcripts vidéo ne
citent aucun site : seul « myinvesting.com » y apparaît une fois, et l'auteur
indique explicitement ne pas utiliser le scanner de son courtier sans dire ce
qu'il utilise à la place. Ce qu'il a réellement à l'écran pendant ses séances
reste inconnu.

Aucun de ces outils n'a été testé ni vérifié. Les tarifs sont ceux mentionnés
dans le cours, à confirmer à la source — ils datent et changent.

## Le classement par fréquence dit l'essentiel

| Outil | Mentions | Ce que ça révèle |
|---|---|---|
| **SEC EDGAR** | 101 | Le pivot de toute la méthode, et il est gratuit |
| Level 2 / carnet | 28 | Outil de lecture permanent |
| **Finviz** | 14 | Le filtre par défaut |
| TradeIdeas | 13 | Le scanner de référence |
| Floatchecker | 12 | Systématique sur flottant réduit |
| Benzinga Pro | 10 | Le fil de nouvelles |

Un enseignement à lui seul : **l'outil le plus utilisé du corpus ne coûte
rien.** La dépense n'est pas le facteur limitant.

## F4 — Dépôts réglementaires

| Outil | Coût | Usage |
|---|---|---|
| **SEC EDGAR** (`sec.gov`) | Gratuit | La source. 8-K, S-1, S-3, 424B, 10-K/10-Q, Form 4, 13F |
| SECAlerts.com | — | Alertes 8-K par courriel ou SMS |
| StockAnalysis.com | Gratuit | Alertes 8-K, et movers de pré-marché |

## F1 — Filtrage statique

| Outil | Coût | Usage |
|---|---|---|
| **Finviz** | Gratuit | Le screener de référence. Paramétrage selon [R6](01-regles.md) |
| Finviz Elite | ~25 $/mois | Ajoute alertes temps réel et données de pré-marché |
| StockAnalysis.com | Gratuit | Movers du matin avec variation en % |

## F2 — Veille temps réel

| Outil | Coût | Usage |
|---|---|---|
| **TradeIdeas** | ~100-180 $/mois | Scanner professionnel, alertes instantanées. Inclut « Holly », générateur d'idées automatisé |

Réserve du cours, qui vaut d'être reprise : commencer avec **trois à cinq
alertes maximum**. Le volume par défaut est ingérable et produit l'effet
inverse de celui recherché.

## F3 — Nouvelles

| Outil | Coût | Usage |
|---|---|---|
| **Benzinga Pro** | Gratuit → ~100 $/mois | Fil temps réel. Les paliers ajoutent la suppression du délai de 15 min, puis les alertes personnalisées et le *squawk* audio |

Ce qu'on achète ici est de la **vitesse**, seule variable qui compte sur un
écart d'ouverture. Le squawk audio permet de suivre les nouvelles sans quitter
les écrans des yeux.

## F5 — Flottant et données de vente à découvert

| Outil | Usage |
|---|---|
| **Floatchecker** | Flottant réel et titres bloqués — la donnée que Finviz peut fausser |
| Finviz · Shortsqueeze.com · MarketBeat.com | Intérêt vendeur |
| **Fintel.io · Short-selling.com** | Liste des titres sous restriction Rule 201 active |

Fintel et Short-selling.com méritent l'attention : ils fournissent l'entrée
directe de la [règle d'exclusion R9](01-regles.md), qui est la principale
optimisation de tout ce corpus.

## F6 — Exécution

| Outil | Usage |
|---|---|
| **DAS Trader** | La plateforme documentée par la chaîne. Voir [le tutoriel](../academy-germain-trading/knowledge/05-plateforme-das-trader.md) |
| Interactive Brokers | Cité pour son bouton *Locate* et son affichage du statut Rule 201 |

## F7 — Calendriers de catalyseurs (fonction absente de 03-outils.md)

Cette famille n'apparaissait pas dans mon découpage initial en six fonctions.
Elle le mérite : elle permet de savoir **à l'avance** quand un catalyseur va
tomber, au lieu de le subir.

| Outil | Usage |
|---|---|
| **EarningsWhispers.com** | Calendrier des résultats, et attentes non officielles du marché — celles contre lesquelles la réaction se juge réellement |
| **BioPharmCatalyst.com** | Calendrier des échéances réglementaires FDA (dates PDUFA) |
| Investing.com/calendar | Calendrier macro : inflation, emploi, réunions de banque centrale |
| IRCalendar.com | Calendrier des communications d'entreprises |
| IPOMonitor.com | Suivi des introductions en bourse |
| Biospace.com | Actualité du secteur biotechnologique |

## F8 — Insiders et institutionnels

| Outil | Usage |
|---|---|
| OpenInsider.com | Agrège les Form 4 — opérations des dirigeants sur leurs propres titres |
| WhaleWisdom.com | Agrège les 13F — positions des gestionnaires |
| Dataroma.com | Suivi de portefeuilles d'investisseurs connus |
| Macrotrends.net | Graphiques financiers construits sur les dépôts réglementaires |
| Yahoo Finance | Détention institutionnelle (onglet *Holders*), volume moyen |

## Flux d'options

| Outil | Usage |
|---|---|
| Unusual Whales · Market Chameleon | Activité inhabituelle sur les options |

Rattaché aux catalyseurs spéculatifs — voir
[../synthese/04-catalyseurs-et-filings.md](../synthese/04-catalyseurs-et-filings.md).

## Ce qui n'est PAS un outil

Piège rencontré en dépouillant le corpus, consigné pour qui referait
l'exercice : **Twitter** apparaît 19 fois et **Reddit** 7 fois, mais dans des
passages de cas d'école — Twitter en tant qu'entreprise ayant levé des fonds
avant son introduction en bourse, Reddit dans le récit des rachats forcés de
2021. Ce ne sont pas des outils recommandés. Un simple comptage d'occurrences
les aurait promus au rang de sources de veille.

## Ce que le cours tarife réellement

Distinction importante : le cours ne donne un prix que pour six outils. Pour
tous les autres, il les cite sans rien dire de leur modèle économique.

**Gratuité affirmée par le cours :**

| Outil | Formulation de la source |
|---|---|
| **SEC EDGAR** | « base de données gratuite de la SEC », « l'outil gratuit le plus puissant du trader » |
| **Finviz** (version de base) | « le screener boursier gratuit le plus utilisé par les traders retail » |
| **StockAnalysis.com** | cité comme « alternative gratuite » |
| **Benzinga Pro** — palier Basic | gratuit, avec un délai de 15 min sur les nouvelles |
| **Floatchecker** — version de base | « dispose d'une version gratuite avec les données de base » |

Le cours ajoute que ces versions gratuites **suffisent pour débuter** : Finviz
gratuit pour préparer la liste, Benzinga gratuit pour le pré-marché.

**Payant, prix annoncé :**

| Outil | Prix cité | Ce que ça achète |
|---|---|---|
| Finviz Elite | ~25 $/mois | Alertes temps réel, données de pré-marché |
| Benzinga Pro | ~30 / ~50 / ~100 $/mois | Suppression du délai, puis alertes personnalisées et squawk audio |
| TradeIdeas | ~100-180 $/mois | Scanner temps réel, générateur d'idées « Holly » |
| Floatchecker Premium | non chiffré | Alertes et données historiques |

Note du cours sur Benzinga : négocier pendant les périodes promotionnelles, des
réductions de moitié étant fréquentes.

**Tarif non indiqué par le cours** — à vérifier avant de bâtir une routine
dessus. Le corpus les mentionne comme ressources utiles sans préciser s'ils
sont gratuits, en accès limité, ou payants :

EarningsWhispers · BioPharmCatalyst · Investing.com/calendar · IRCalendar ·
IPOMonitor · Biospace · OpenInsider · WhaleWisdom · Dataroma · Macrotrends ·
Yahoo Finance · Shortsqueeze.com · MarketBeat.com · Fintel.io ·
Short-selling.com · SECAlerts.com · Unusual Whales · Market Chameleon ·
DAS Trader · Interactive Brokers

Deux méritent une vérification prioritaire : **Fintel.io** et
**Short-selling.com**, qui fournissent la liste des titres sous Rule 201 active
— l'entrée directe de la [règle d'exclusion R9](01-regles.md). Si cette donnée
n'est accessible que par abonnement, c'est la seule dépense réellement
structurante de tout le dispositif.

## Dispositif de départ à coût nul

Constructible avec la seule gratuité affirmée par le cours :

| Fonction | Outil | Statut |
|---|---|---|
| Dépôts réglementaires (F4) | SEC EDGAR | Gratuit, public |
| Filtrage (F1) | Finviz | Version gratuite suffisante |
| Movers du matin (F1) | StockAnalysis.com | Gratuit |
| Nouvelles (F3) | Benzinga Basic | Gratuit, délai 15 min |
| Flottant (F5) | Floatchecker | Version de base gratuite |
| Exécution (F6) | Plateforme du courtier | Selon le courtier |

Non couvert : la **veille temps réel** (F2), sans équivalent gratuit —
compensée par la surveillance manuelle d'une liste restreinte, cohérent avec
[R10](01-regles.md). Et les **calendriers** (F7) et **suivi insiders** (F8),
dont la gratuité n'est pas établie par le corpus.

Le délai de 15 minutes de Benzinga Basic est la vraie limite de ce dispositif :
sur un écart d'ouverture, quinze minutes sont une éternité. C'est ce qui fait
de la vitesse des nouvelles le premier achat sensé.

## Ordre d'acquisition si budget

1. **Rien** — le dispositif ci-dessus couvre le pipeline complet.
2. **Vitesse des nouvelles** (Benzinga payant) — agit sur le facteur limitant.
3. **Veille temps réel** (TradeIdeas) — seulement en activité quotidienne, et
   seulement une fois EDGAR maîtrisé. Payer des alertes qu'on ne sait pas
   qualifier est la dépense la moins rentable de la liste.

---
niveau: 2
maj: 2026-08-21
---

# Outils nécessaires

Ce qu'il faut réellement pour exécuter la méthode, par **fonction** plutôt que
par marque : une fonction non couverte est un trou dans le pipeline, quel que
soit le nombre d'abonnements souscrits.

> Les **noms** des outils cités dans le corpus — sites, applications, tarifs —
> sont dans [06-inventaire-outils.md](06-inventaire-outils.md), qui ajoute aussi
> deux fonctions absentes d'ici : les calendriers de catalyseurs (F7) et le suivi
> des insiders et institutionnels (F8).

## Les six fonctions

| # | Fonction | Sans elle | Moment d'usage |
|---|---|---|---|
| F1 | **Filtrage statique** | Pas de liste de départ | Veille au soir, matin avant l'ouverture |
| F2 | **Veille temps réel** | On rate ce qui démarre en séance | Pendant la séance |
| F3 | **Nouvelles rapides** | Le catalyseur arrive après le mouvement | Pré-marché et séance |
| F4 | **Dépôts réglementaires** | Dilution invisible avant qu'elle frappe | Pré-marché, avant chaque trade |
| F5 | **Données de flottant** | Calculs faussés (flottant ≠ actions émises) | Avant tout trade sur flottant réduit |
| F6 | **Exécution** | — | En séance |

## Ce que couvre l'outillage documenté

**F1 — Filtrage.** Un screener gratuit suffit pour débuter ; la version payante
n'ajoute que les alertes et les données de pré-marché. Paramétrage selon R6.

**F2 — Veille temps réel.** C'est le poste le plus cher (ordre de la centaine
d'euros mensuels) et le seul réellement optionnel au départ. Le cours
recommande de démarrer avec trois à cinq alertes maximum : le volume d'alertes
par défaut est ingérable et produit l'effet inverse de celui recherché.

**F3 — Nouvelles.** Une offre gratuite couvre l'essentiel du pré-marché. La
version payante achète de la **vitesse**, qui est la variable qui compte sur un
écart d'ouverture.

**F4 — Dépôts réglementaires.** EDGAR est public et gratuit. C'est la fonction
la moins chère et la plus discriminante de la liste : c'est elle qui distingue
un catalyseur de dilution d'une histoire de croissance.

**F5 — Flottant.** Un outil spécialisé donne le flottant réel et le détail des
titres bloqués. Substituable par un screener + EDGAR, au prix d'un travail
manuel.

**F6 — Exécution.** La plateforme documentée par la chaîne est DAS Trader, dont
le tutoriel construit la disposition : graphique, carnet (appelé *montage*, et
c'est depuis lui que partent les ordres), time & sales, positions et ordres.
Détail dans
[../academy-germain-trading/knowledge/05-plateforme-das-trader.md](../academy-germain-trading/knowledge/05-plateforme-das-trader.md).

## Configuration minimale viable

Pour exécuter la méthode sans abonnement payant :

- **F1** screener gratuit, **F3** fil de nouvelles gratuit, **F4** EDGAR,
  **F5** screener + EDGAR en manuel, **F6** plateforme du courtier.
- **F2 non couverte** — compensée par une surveillance manuelle de la liste
  restreinte constituée en pré-marché.

Cette configuration est cohérente avec R10 : si l'essentiel du travail se fait
avant l'ouverture, la veille temps réel devient un confort plutôt qu'une
nécessité. C'est l'argument qui justifie de ne pas payer F2 au départ.

## Ce qui reste non documenté

**Les raccourcis clavier.** Ni le cours ni la chaîne ne les traitent. Sur une
méthode qui exige des entrées rapides dans des mouvements violents, c'est la
lacune la plus sérieuse de tout le corpus — et elle ne sera pas comblée par une
source secondaire, puisqu'elle dépend de la plateforme et du style de chacun.

**Le scanner effectivement utilisé par l'auteur.** Il indique ne pas se servir
de celui du courtier, sans dire ce qu'il utilise à la place.

**La lecture du carnet et du tape.** Leur emplacement à l'écran est documenté,
leur interprétation ne l'est pas — au-delà des principes généraux vus en
[../synthese/03-marches-formation-des-prix.md](../synthese/03-marches-formation-des-prix.md).

## Coût et séquence d'acquisition

L'ordre dans lequel investir, si budget il y a :

1. **Rien** — F1, F3, F4 gratuits couvrent le pipeline complet en mode préparé.
2. **Vitesse des nouvelles** (F3 payant) — premier gain réel, car il agit sur
   le facteur limitant : arriver avant le mouvement.
3. **Veille temps réel** (F2) — seulement si l'activité est quotidienne et si
   les trois premières fonctions sont déjà maîtrisées.

Payer F2 avant de maîtriser F4 revient à acheter des alertes qu'on ne saura pas
qualifier.

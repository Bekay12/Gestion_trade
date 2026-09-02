---
niveau: 1
maj: 2026-08-21
sources: [cours P12, chaine]
---

# Outils et routine

La partie la plus directement transposable, et celle où la chaîne était la plus
muette.

## Filtre statique contre veille temps réel

Deux catégories d'outils à ne pas confondre, parce qu'elles servent à des
moments différents :

Un **screener** interroge une base à un instant donné selon des critères. Son
résultat est figé. Il sert à **construire la liste de surveillance**, la veille
au soir ou tôt le matin.

Un **scanner** surveille le marché en continu et alerte dès qu'une condition
est remplie. Il sert **pendant la séance**, à ne pas rater ce qui démarre.

Le premier est généralement gratuit, le second généralement payant. Le cours
recommande de les combiner plutôt que de choisir : filtrer avant l'ouverture,
être alerté pendant.

## Ce qu'on cherche dans un filtre

Les critères qui reviennent définissent le terrain de jeu de la méthode :
marché américain, capitalisation petite à très petite, prix dans une fourchette
basse mais pas dérisoire, volume moyen suffisant pour entrer et sortir, volume
relatif anormalement élevé — c'est-à-dire *il se passe quelque chose
aujourd'hui* — et écart d'ouverture à la hausse.

Le volume relatif est le filtre décisif : il ne demande pas si le titre est
gros ou petit, il demande si l'activité du jour sort de l'ordinaire. Les
valeurs de seuil sont dans [../methode/01-regles.md](../methode/01-regles.md).

Un outil spécialisé dans le flottant complète l'ensemble, pour la raison
donnée en [02](02-entreprise-float-dilution.md) : le chiffre facilement
accessible n'est pas toujours le bon.

## La journée type

Le cours structure la séance en phases dont chacune a une fonction distincte —
et c'est le point qui manquait entièrement à la chaîne.

**Avant l'ouverture**, trois blocs successifs : lecture des nouvelles de la
nuit et vérification des dépôts réglementaires ; analyse approfondie des
candidats retenus ; puis rédaction du plan de trade. La règle qui structure ce
temps : le pré-marché sert à **analyser**, pas à opérer — le volume y est trop
faible et les fourchettes trop larges.

**À l'ouverture**, une phase d'observation. Les premières minutes concentrent
l'exécution des ordres accumulés pendant la nuit et donnent des signaux peu
fiables. Le cours recommande de laisser le marché trouver sa direction.

**Puis la fenêtre principale**, où volume et liquidité sont maximaux — c'est là
que les meilleures configurations se présentent. Le milieu de journée est
signalé comme une période à activité réduite et mouvements moins fiables.

**Après la clôture**, l'analyse des opérations et la mise à jour de la liste.

## Le plan de trade

L'élément que le cours désigne comme le plus souvent sauté par les débutants,
et comme la cause de leurs pertes. C'est une fiche remplie **avant**
l'ouverture, pour chaque titre retenu, qui fixe : le catalyseur en une phrase,
le sens de la position, la zone d'entrée, le niveau où le scénario est
invalidé, les objectifs échelonnés, la taille de position calculée, et
l'événement qui annulerait le trade.

Sa logique est de déplacer la décision hors du moment où l'on est exposé. Si le
marché fait ce qui était prévu, on exécute ; s'il fait autre chose, on
n'improvise pas.

Formulé autrement par le cours : une opération non planifiée n'est pas une
opération, c'est un pari.

## Ce que la chaîne montre, et ne montre pas

La chaîne documente **l'espace de travail** — le tutoriel DAS Trader construit
la disposition : graphique, carnet, exécutions, positions et ordres, chacun
dans sa zone. Elle documente aussi, par ses récapitulatifs quotidiens, la phase
d'analyse post-séance.

Elle ne documente ni le paramétrage du scanner — l'auteur indique explicitement
ne pas utiliser celui du courtier, sans dire ce qu'il utilise — ni les
raccourcis clavier, ni la routine de pré-marché. Sur ces trois points, le cours
est la seule source disponible.

Source : cours, Partie 12 (5 chapitres, p. 322-347) et tutoriel du 20 août.

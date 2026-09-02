---
niveau: 1
maj: 2026-08-21
sources: [cours P5, cours P7, chaine]
---

# Marchés, formation des prix et vente à découvert

La partie la plus technique du domaine, et celle qui explique le mieux pourquoi
la méthode réussit ou échoue.

## Le carnet d'ordres et le tape

Le **carnet** (level 2, appelé *montage* dans DAS Trader) montre les intentions :
qui propose d'acheter et de vendre, à quel prix, en quelle quantité. Le **tape**
(time & sales) montre les faits : chaque transaction réellement exécutée, avec
son prix, sa taille et sa place d'exécution.

Le cours qualifie le tape de document le plus honnête du marché, et la
distinction est utile : une intention se retire, une exécution non. La taille
des transactions y est lisible — des lots de quelques actions signalent une
activité de particuliers, des blocs de plusieurs milliers signalent un
intervenant institutionnel.

Le pré-marché est le moment où cette lecture est la plus révélatrice, parce que
le volume y est faible : chaque transaction pèse.

## Le VWAP comme niveau de référence

Le prix moyen pondéré par le volume depuis l'ouverture. Sa fonction n'est pas
de prédire mais de **situer** : un cours sous le VWAP signale que les vendeurs
dominent la séance ; un cours au-dessus, l'inverse. C'est le repère par défaut
de la journée, et un support de placement de stop courant en day trading.

## La vente à découvert, mécanisme complet

C'est ici que le cours comble la lacune la plus visible de la chaîne.

**Le locate.** Avant de vendre à découvert, il faut l'accord du courtier
confirmant que les titres sont empruntables. Sans lui, l'opération est un
*naked short*, illégal aux États-Unis. Le locate est une ressource limitée : si
de nombreux intervenants veulent vendre le même titre difficile à emprunter,
certains n'obtiendront rien. Il peut aussi être révoqué en cours de séance, ce
qui force un rachat.

Trois états à connaître : facile à emprunter, **difficile à emprunter** (peu de
titres, coût élevé), et indisponible — auquel cas le titre n'est simplement pas
jouable ce jour-là.

C'est exactement la situation décrite dans la chaîne : une configuration jugée
bonne, écartée parce que les titres n'étaient pas disponibles, avec la remarque
qu'une position fermée ne libère pas de quoi en rouvrir une.

**Le coût d'emprunt.** Un taux annualisé, prélevé chaque jour sur la position.
Sur un titre facile à emprunter il est négligeable ; sur un titre difficile il
peut dépasser 100 % par an. Le coût court même si le cours ne bouge pas, ce qui
en fait une pression au rachat rapide — et un des moteurs des mouvements de
rachat forcé.

**La restriction de vente à découvert (Rule 201).** Déclenchée quand un titre
perd 10 % ou plus par rapport à la clôture de la veille, elle reste active
jusqu'à la fin de la séance suivante. Son effet : on ne peut plus vendre à
découvert au prix acheteur, seulement à l'offre ou au-dessus. Concrètement, les
vendeurs à découvert ne peuvent plus accélérer la baisse. Elle protège
partiellement les positions acheteuses et complique les positions vendeuses.

## Les indicateurs de tension vendeuse

Trois mesures, à lire ensemble :

- **L'intérêt vendeur** en pourcentage du flottant — le pessimisme exprimé.
- **Les jours pour couvrir** — le volume moyen nécessaire aux vendeurs pour
  déboucler.
- **La rotation du flottant** — volume du jour rapporté au flottant, qui mesure
  l'intensité réelle de la séance.

Les seuils chiffrés sont dans [../methode/01-regles.md](../methode/01-regles.md).

## La configuration qui piège les vendeurs

Point capital, et le cours l'énonce sans détour : la combinaison **restriction
active + titre difficile à emprunter + flottant réduit + intérêt vendeur
élevé** est un piège pour les vendeurs à découvert. Les vendeurs existants
paient cher chaque jour, ne peuvent pas accélérer la baisse, et la moindre
poussée acheteuse les force à racheter — ce qui alimente la hausse.

Autrement dit : **la configuration qui ressemble le plus à une opportunité de
vente à découvert est aussi celle qui peut coûter le plus cher.** Cette phrase
est la clé de lecture de la séance perdante de la chaîne
([07](07-chaine-face-au-cours.md)).

## Les mécanismes de protection

Les suspensions de cotation (*halts*) interrompent les échanges sur nouvelle
importante ou volatilité excessive. La chaîne les mentionne comme un événement
subi ; le cours les traite comme un mécanisme prévisible, avec ses types et ses
règles de reprise.

Source : cours, Parties 5 et 7 (7 chapitres, p. 162-221).

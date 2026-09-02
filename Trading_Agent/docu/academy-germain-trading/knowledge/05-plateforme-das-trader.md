---
concept: plateforme-das-trader
maj: 2026-08-21
videos_sources: [B33yDi_2fr4]
confiance: moyenne
---

# DAS Trader : construction de l'espace de travail

Source unique : le tutoriel du 20 août 2026 (50 min), la plus longue vidéo du
corpus et la seule entièrement pédagogique.

⚠️ **Limite de restitution.** Cette vidéo est un partage d'écran construit sur
la désignation visuelle (« vous cliquez *ici* », « ça ressemble à *ça* »). Le
transcript conserve la **séquence des menus** mais perd les cibles à l'écran.
Cette note est donc une carte des étapes, pas un substitut au visionnage.

## L'objectif annoncé

Construire un espace de travail propre *avant* d'aborder les vidéos avancées.
Le tutoriel énonce d'emblée ce qu'il faut obtenir : un emplacement réservé au
graphique, une zone pour le level 2 et le time & sales, une zone pour les
positions et les ordres, le tout redimensionné pour que seul l'essentiel reste
visible.

## Séquence de configuration

**Repartir de zéro.** `File` → `Clear Desktop`, pour ne pas configurer par
dessus la disposition livrée par défaut.

**Détacher chaque fenêtre.** Chaque outil est créé depuis le menu puis sorti de
son cadre par un clic droit → `pop up`. C'est le point le plus important du
tutoriel sur le plan pratique : sans cette étape, la fenêtre reste prisonnière
du cadre principal et l'espace de travail ne peut pas être organisé. La
manœuvre se répète pour le graphique, le montage et le time & sales.

**Le graphique.** Couleur des chandeliers d'abord. Puis l'affichage distinct
du pré-marché et de l'after market par une couleur de fond dédiée — l'auteur
retient un gris, la couleur par défaut étant jugée trop bruyante. Puis la
vitesse de zoom de la souris, réglée au maximum parce que le défaut rend la
navigation trop lente. Le titre du graphique et les bordures sont désactivés,
pour rendre l'écran au contenu.

Deux réglages persistants méritent l'attention : la ligne de clôture de marché,
et la sauvegarde automatique des lignes tracées, qui permet de retrouver ses
niveaux d'une session à l'autre.

**Le montage (level 2).** À retenir, parce que le vocabulaire dérouterait :
dans DAS Trader, la fenêtre de level 2 s'appelle le **montage**, et c'est
depuis elle que passent les ordres — type d'ordre, envoi, annulation, rappel
d'un ordre précédent. Ce n'est pas un simple afficheur de carnet.

**Le time & sales**, le suivi des **positions** et celui des **ordres** en
cours complètent la disposition.

**Le compte.** Une étape est explicitement signalée comme délicate : la
sélection du compte auquel la fenêtre est rattachée. Point de vigilance évident
mais réel — une fenêtre d'ordres pointant sur le mauvais compte est une erreur
silencieuse.

**Sauvegarder le layout**, pour ne pas refaire ce travail.

## Un choix révélateur

L'auteur indique ne pas utiliser le scanner du courtier. Le repérage des
titres se fait donc ailleurs, avec un outil que le corpus ne nomme pas — c'est
la lacune signalée dans
[03-selection-et-filtres.md](03-selection-et-filtres.md), et elle est ici
confirmée par la négative.

## Lacunes connues

- **Les hotkeys ne sont pas couvertes.** Sur une méthode qui repose sur des
  entrées rapides dans des mouvements violents
  ([01](01-methode-short-sur-spike.md)), c'est la lacune la plus importante du
  tutoriel.
- Le tutoriel construit l'espace mais n'enseigne pas la lecture du level 2 ni
  du tape.
- Aucun réglage de gestion du risque au niveau de la plateforme (limites de
  perte, taille par défaut) n'est abordé.

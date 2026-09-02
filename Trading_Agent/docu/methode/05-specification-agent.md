---
niveau: 2
maj: 2026-08-21
---

# Spécification pour l'agent

Ce que les règles impliquent si on veut les faire exécuter par un programme
plutôt que par une personne. Document de conception, pas d'implémentation.

## La ligne de partage

Toutes les règles ne sont pas également automatisables, et confondre les deux
catégories est la façon la plus courante de construire un agent qui échoue.

**Déterministe** — calculable à partir de données, sans jugement :

| Règle | Entrée nécessaire | Sortie |
|---|---|---|
| R1 dimensionnement | Capital, prix d'entrée, stop | Nombre d'actions |
| R2 rapport gain/risque | Entrée, stop, cible | Ratio, verdict d'admission |
| R3 bornes de perte | P&L de la séance | Autorisation d'opérer |
| R4 paliers de drawdown | Historique du capital | Coefficient de taille |
| R6 filtres de sélection | Données de marché | Liste de candidats |
| R7 coût d'emprunt | Valeur, taux, durée | Coût, impact sur le R/R |
| R8 tension vendeuse | Flottant, intérêt vendeur, volume | Indicateurs, seuils franchis |
| R9 exclusion | Rule 201, emprunt, flottant, intérêt vendeur | **Booléen bloquant** |
| R10 fenêtres | Heure | Autorisation d'opérer |

**Non déterministe** — exige une interprétation :

- Lire un communiqué et décider s'il est haussier ou dilutif.
- Juger si un catalyseur tient en une phrase (R11).
- Lire le carnet et le tape.
- Évaluer son propre état (R12).

**Structurellement hors machine** : R13 — le poste de travail ne s'applique
qu'à un opérateur humain.

## Architecture qui en découle

Trois couches, la troisième étant la plus importante et la plus facile à
oublier.

**Couche 1 — Filtrage et calcul.** Purement déterministe. Implémente R6, R1,
R2, R7, R8. Produit une liste de candidats avec leurs métriques.

**Couche 2 — Qualification.** Requiert la lecture de documents (dépôts
réglementaires, communiqués). C'est là qu'un modèle de langage a une valeur
réelle : classer un dépôt, résumer un catalyseur, détecter une mention
d'émission d'actions. Sortie : catalyseur qualifié ou rejet.

**Couche 3 — Garde-fous.** Implémente R3, R4, R9, R10 comme des **blocages**,
pas comme des avis. Une couche qui produit une recommandation ignorable ne sert
à rien : la valeur d'un garde-fou est qu'il refuse.

## Le point de conception le plus important

**R9 doit bloquer, pas alerter.**

La séance perdante documentée par la chaîne n'a pas eu lieu par ignorance de la
règle — l'auteur connaît son cours. Elle a eu lieu parce qu'au moment de la
décision, la conviction l'a emporté sur la règle connue.

Un agent qui affiche « attention, configuration défavorable » reproduit
exactement cette situation, avec une étape de plus. Un agent qui refuse
d'armer l'ordre change le résultat.

Corollaire : R3 et R4 aussi. La perte journalière maximale atteinte doit
désarmer, pas notifier.

## Données nécessaires

| Donnée | Fréquence | Criticité |
|---|---|---|
| Cours, volume, VWAP | Temps réel | Indispensable |
| Flottant réel | Quotidienne | Indispensable (R6, R8, R9) |
| Intérêt vendeur, jours pour couvrir | Bimensuelle | Élevée |
| Statut d'emprunt et taux | Temps réel | **Indispensable pour toute position vendeuse** |
| Statut Rule 201 | Temps réel | Indispensable (R9) |
| Dépôts réglementaires | Temps réel | Élevée |
| Suspensions de cotation | Temps réel | Moyenne |

Le statut d'emprunt est la dépendance la plus contraignante : il vient du
courtier, pas d'une source de marché publique. Sans lui, R7 et R9 ne sont pas
calculables, et la moitié de la méthode devient inexécutable par programme.

C'est la contrainte à vérifier **en premier** avant de construire quoi que ce
soit : si le courtier n'expose pas cette donnée par interface programmable, il
faut soit changer d'approche, soit accepter que l'agent ne couvre que le côté
acheteur.

## Ce que l'agent ne doit pas faire

Décider d'entrer. Le corpus entier converge sur un point : la décision se
prépare avant et s'exécute selon un plan écrit. Un agent qui déciderait à la
place de l'opérateur supprimerait la seule étape que toute la méthode existe
pour protéger.

Le rôle utile est plus modeste et plus solide : **présenter des candidats
qualifiés, calculer les tailles, et refuser ce qui viole une règle.**

## Articulation avec le questionnaire

L'application d'apprentissage et l'agent partagent la même base de règles. Une
règle modifiée doit se répercuter dans les deux — voir l'étape 6 de
[04-grille-nouvelle-source.md](04-grille-nouvelle-source.md).

L'ordre logique est d'ailleurs celui-là : comprendre les règles (questionnaire),
puis les faire appliquer (agent). Un agent dont l'opérateur ne comprend pas les
refus finit par être contourné.

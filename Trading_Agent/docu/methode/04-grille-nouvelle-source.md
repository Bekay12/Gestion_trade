---
niveau: 2
maj: 2026-08-21
usage: intégrer une nouvelle chaîne YouTube ou un nouveau formateur
---

# Grille d'intégration d'une nouvelle source

Pour quand tu ajoutes quelqu'un d'autre à suivre. Le problème n'est pas
d'accumuler des transcripts — le pipeline
[../../scripts/youtube/](../../scripts/youtube/) le fait déjà — mais d'éviter
que la documentation devienne un empilement d'avis contradictoires sans
arbitrage.

Cette grille répond à trois questions dans l'ordre : **cette source vaut-elle
d'être suivie**, **qu'apporte-t-elle que je n'ai pas**, et **que faire quand
elle contredit ce que je sais déjà**.

## Étape 1 — Ingérer

```powershell
.venv\Scripts\python.exe scripts\youtube\ingest_channel.py "<url de la chaîne>"
```

Crée `docu/<nom-chaine>/` avec transcripts et filigrane. Aucune décision à ce
stade.

## Étape 2 — Qualifier la source

Avant d'investir du temps de synthèse, sept critères. Ils viennent en partie du
premier chapitre du cours, qui traite précisément de la reconnaissance d'un
formateur sérieux — un test que la source appliquait donc déjà à elle-même.

| # | Critère | Signal favorable | Signal défavorable |
|---|---|---|---|
| C1 | **Point de départ** | Commence par les bases et les actions | Commence par le Forex, la crypto, l'effet de levier |
| C2 | **Traitement du risque** | Parle des pertes autant que des gains | Ne montre que des gains |
| C3 | **Traçabilité** | Montre ses opérations, y compris perdantes | Résultats invérifiables, captures sélectives |
| C4 | **Explication** | Explique le raisonnement | Vend des signaux sans justification |
| C5 | **Modèle économique** | Transparent sur ses revenus | Rémunéré par le courtier qu'il recommande |
| C6 | **Chiffrage** | Donne des seuils vérifiables | Reste dans le qualitatif |
| C7 | **Avertissement** | Précise que ce ne sont pas des conseils | Promet des gains |

**Seuil** : moins de 4 signaux favorables → ingérer les transcripts mais ne pas
construire de documentation. La source devient une matière de comparaison, pas
une référence.

Pour mémoire, la chaîne de référence satisfait C2, C3, C4 et C7 de façon
documentée ; C6 est faible (peu de seuils chiffrés) et compensé par le cours.

## Étape 3 — Cartographier l'apport

Positionner la nouvelle source sur les fonctions existantes, pour voir
immédiatement si elle apporte ou si elle répète.

| Domaine | Couvert actuellement | Nouvelle source ? |
|---|---|---|
| Sélection et filtres | R6 | |
| Catalyseurs et dépôts | [../synthese/04](../synthese/04-catalyseurs-et-filings.md) | |
| Mécanique vente à découvert | R7, R8 | |
| Dimensionnement et risque | R1–R5 | |
| Routine et plan | R10, R11 | |
| Outils | [03-outils.md](03-outils.md) | |
| **Raccourcis clavier** | **non couvert** | |
| **Lecture carnet / tape** | **non couvert** | |
| **Paramétrage scanner** | **non couvert** | |

Les trois lignes en gras sont les lacunes ouvertes. Une source qui les comble a
une valeur immédiate ; une source qui ne fait que redire R1–R5 n'en a presque
aucune.

## Étape 4 — Arbitrer les contradictions

Le vrai travail, et la raison d'être de cette grille. Deux sources sérieuses
peuvent diverger. Procédure, dans l'ordre :

**1. Distinguer le désaccord du contexte.** Un seuil différent n'est pas
forcément une contradiction : il peut refléter un capital, un marché ou un
horizon différents. Vérifier d'abord que les deux parlent de la même chose.

**2. Classer la nature de l'affirmation.**

- *Fait réglementaire* (seuil SEC, délai légal) → il n'y a pas d'arbitrage à
  faire, il y a une source officielle à consulter. Trancher hors corpus.
- *Fait de marché* (mécanisme, formule) → vérifiable ; celui qui démontre gagne.
- *Choix de méthode* (seuil de risque, fenêtre horaire) → pas de vérité ;
  documenter les deux avec leur justification.
- *Opinion* → ne pas intégrer aux règles.

**3. En cas de choix de méthode divergent**, retenir le plus conservateur par
défaut et noter l'autre. Une règle de risque plus stricte coûte du gain
potentiel ; une règle plus permissive coûte du capital.

**4. Consigner l'arbitrage**, pas seulement son résultat. Une contradiction
tranchée sans trace resurgit à la source suivante.

## Étape 5 — Écrire

Règle d'or, identique à celle de la documentation de chaîne : **une nouvelle
source modifie les notes existantes, elle n'ajoute pas une couche parallèle.**

- Concept déjà couvert, source d'accord → enrichir la note, ajouter la source
  en frontmatter.
- Concept déjà couvert, source en désaccord → arbitrer (étape 4), puis
  documenter la divergence *dans la note existante*.
- Concept non couvert → nouvelle note, référencée depuis l'index.

Ce qu'il ne faut pas faire : créer `docu/<nouvelle-chaine>/knowledge/` en
miroir. Ça produit deux documentations qui ne se parlent pas — exactement
l'empilement qu'on cherche à éviter. Les transcripts restent séparés par
chaîne ; **la connaissance reste unifiée par concept.**

## Étape 6 — Répercuter

Une règle modifiée se répercute dans trois endroits :

1. [01-regles.md](01-regles.md) — la règle elle-même.
2. [02-logique-decision.md](02-logique-decision.md) — si le pipeline change.
3. Le questionnaire — si une question devient fausse ou obsolète.

Le troisième point est celui qu'on oublie : une question d'évaluation qui teste
une règle périmée enseigne l'erreur.

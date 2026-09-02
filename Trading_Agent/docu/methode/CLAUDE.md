# docu/methode/

**Niveau 2 — l'opérationnel.** Règles exécutables, pipeline de décision,
outillage, extensibilité.

| Fichier | Contenu |
|---|---|
| `01-regles.md` | R1 à R13 — le contrat |
| `02-logique-decision.md` | Le pipeline, ses quatre points de rupture |
| `03-outils.md` | Les fonctions F1 à F8 |
| `04-grille-nouvelle-source.md` | Intégrer un autre formateur |
| `05-specification-agent.md` | Ce qui est automatisable |
| `06-inventaire-outils.md` | Les outils nommés, tarifs |
| `07-etat-automatisation.md` | L'état réel du moteur — **à réactualiser** |

## Le contrat des numéros de règle

**R1…R13 sont référencés hors de ce dossier** : par le code de
[../../agent/](../../agent/) dans ses messages de refus, et par les questions du
[quiz](../quiz/). Renuméroter casse les deux silencieusement.

Modifier une règle impose trois répercussions, dans cet ordre :

1. `01-regles.md` — la règle elle-même ;
2. `../../agent/gate.py` ou `rules.py` — l'implémentation, plus son test ;
3. le questionnaire — **le point qu'on oublie**. Une question d'évaluation qui
   teste une règle périmée enseigne l'erreur.

## Les règles qui ne se négocient pas

**R9 — exclusion.** Ne vient d'aucune source directement : c'est le produit du
croisement. Elle interdit la position vendeuse quand se cumulent restriction
Rule 201, titre difficile à emprunter, flottant réduit et intérêt vendeur
élevé — la configuration qui ressemble le plus à une opportunité et qui coûte
le plus cher. C'est la seule optimisation originale de tout le corpus.

**R5 — le stop ne recule jamais.** Il ne se déplace que pour protéger un gain
acquis.

**R1 — l'ordre est imposé** : risque accepté → invalidation → taille.

## Origine des seuils

Tous viennent du cours PDF. Deux exceptions signalées comme telles dans le
texte :

- **R13** (poste de travail contre téléphone) vient de la chaîne — la seule
  règle dont la transgression soit documentée avec son coût ;
- les **seuils d'écart acheteur-vendeur** (2 % / 5 %) sont dérivés : le corpus
  signale que les écarts de pré-marché sont trois à cinq fois plus larges sans
  donner de chiffre. Ils sont marqués comme garde-fou dérivé dans `gate.py`.

Ne pas introduire d'autre seuil sans le marquer.

## Fonctions F1 à F8

`03-outils.md` raisonne par fonction, `06-inventaire-outils.md` donne les noms
et ce que la source tarife réellement. Distinction utile : le cours ne donne un
prix que pour six outils ; pour la vingtaine restante il ne dit rien, et ce
silence ne vaut pas gratuité.

## Ajouter une source d'enseignement

Suivre `04-grille-nouvelle-source.md` intégralement, en particulier l'étape
d'arbitrage des contradictions. Sans elle, une deuxième source ne fait pas
grandir la documentation — elle la contredit.

# docu/

Toute la documentation, organisée en deux niveaux plus les sources.

| Dossier | Nature |
|---|---|
| [Livres/](Livres/) | PDF sources, propriété de l'utilisateur |
| [academy-germain-trading/](academy-germain-trading/) | Corpus vidéo : transcripts + notes |
| [synthese/](synthese/) | **Niveau 1** — le savoir, par mécanisme |
| [methode/](methode/) | **Niveau 2** — l'opérationnel, règles exécutables |
| [quiz/](quiz/) | Questionnaire d'évaluation (page web) |
| [portefeuille/](portefeuille/) | État du compte papier — données vivantes |
| `_extraction/`, `_cache/` | Caches dérivés, régénérables |

## La séparation qui structure tout

**Niveau 1 explique, niveau 2 exécute.** Une notion se comprend dans
`synthese/`, s'applique dans `methode/`. Ne pas mettre de seuil chiffré
actionnable dans le niveau 1, ni d'exposé pédagogique dans le niveau 2.

Les numéros de règle (R1…R13) de `methode/01-regles.md` sont un **contrat** :
ils sont cités par le code de [../agent/](../agent/) dans ses messages de refus
et par les questions du quiz. Ne pas renuméroter.

## Provenance des affirmations

| Type | Source | Traitement |
|---|---|---|
| Valeur chiffrée | **Cours PDF uniquement** | Reprise telle quelle |
| Situation, geste, erreur vécue | Transcripts vidéo | Repris, chiffres exclus |
| Règle réglementaire | Énoncée de mémoire par l'auteur | Marquée *à vérifier hors corpus* |

Les transcripts viennent de sous-titres automatiques qui corrompent les
nombres. Cette règle n'a pas d'exception.

## Enrichir plutôt qu'empiler

Quand une source nouvelle traite un concept déjà couvert, **modifier la note
existante** — ajouter la nuance, signaler la contradiction avec les deux
sources datées. Ne pas créer une note parallèle ni une section « vu ailleurs ».

Les transcripts restent séparés par chaîne ; la **connaissance reste unifiée
par concept**. La procédure complète est dans
[methode/04-grille-nouvelle-source.md](methode/04-grille-nouvelle-source.md).

## Réserve permanente

Le cours et la chaîne ont le **même auteur**. Leur accord vérifie une cohérence
interne, pas une validité externe. Aucune affirmation n'a été confrontée à une
source indépendante. Toute note doit rester lisible avec cette réserve en tête.

## Langue

Français. Les termes techniques du domaine ne se traduisent pas : *float*,
*short interest*, *borrow rate*, *locate*, *time & sales*.

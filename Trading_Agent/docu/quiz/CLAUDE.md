# docu/quiz/

Questionnaire d'évaluation façon auto-école, publié comme page web.
`index.html` est autonome : aucune dépendance externe hormis les polices
Google Fonts.

## Banque de questions

51 questions dans le tableau `Q` du script, réparties sur huit domaines
(`DOM`). L'examen blanc en tire **40** au hasard, seuil de réussite **35** —
donc deux sessions ne sont jamais identiques, et il faut garder au moins une
dizaine de questions de marge au-dessus de 40.

Format d'une entrée :

```js
{d:"risk", q:"...", o:["...","..."], c:[1], x:"explication", s:"methode/R1"}
```

`c` est une liste d'index : plusieurs valeurs font une question à réponses
multiples. `s` est le renvoi à la source — il doit pointer sur une règle ou une
note réelle, c'est ce qui permet au lecteur d'aller vérifier.

## Validation avant publication

```bash
node -e "…"   # cf. historique : compte les questions, vérifie les index,
              # détecte les doublons d'options et les entrées incomplètes
```

Contrôler : aucun index hors bornes, pas d'option dupliquée dans une question,
`x` et `s` renseignés partout, et au moins 40 questions disponibles.

## Le lien avec les règles

**Une règle modifiée dans [../methode/01-regles.md](../methode/01-regles.md)
rend potentiellement une question fausse.** C'est le point qu'on oublie quand
on ajuste un seuil. Une question d'évaluation qui teste une règle périmée
enseigne l'erreur — c'est pire que pas de question du tout.

Vérifier en particulier les questions du domaine `risk` (seuils chiffrés) et
`short` (mécanique Rule 201, emprunt) après toute révision.

## Publication

L'artefact se met à jour en republiant **le même chemin de fichier**, ce qui
conserve l'URL. Ne pas créer un nouveau fichier pour une révision.

Titre et favicon restent stables : les lecteurs retrouvent la page par son
icône d'onglet.

## Design

Palette encre chaude / ambre, sémantiques vert et brique réservées au
juste/faux — jamais décoratives. Trois familles : Bricolage Grotesque
(titres), Newsreader (lecture, le français long se lit mieux en serif), IBM
Plex Mono (chiffres et étiquettes).

Thèmes clair et sombre définis par jetons sur `:root`, redéfinis sous
`prefers-color-scheme` **et** sous `[data-theme]`. Ne jamais déclarer une
couleur uniquement dans un bloc média : elle ne s'appliquerait pas dans l'état
« système » non marqué, et la page rendrait le texte d'un thème sur le fond de
l'autre.

`localStorage` est enveloppé dans `try/catch` — il peut lever selon le
contexte d'affichage.

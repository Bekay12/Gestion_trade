# Trading_Agent

Deux moitiés qui se répondent : une **documentation** d'une méthode de day
trading (actions US, petites capitalisations, catalyseurs de dilution) et un
**moteur** qui applique ses règles sur un portefeuille papier.

| Dossier | Rôle |
|---|---|
| [docu/](docu/) | Sources, synthèse, règles, questionnaire |
| [agent/](agent/) | Moteur de décision et portefeuille papier |
| [scripts/youtube/](scripts/youtube/) | Ingestion incrémentale des transcripts |

## Langue

**Français** pour toute la documentation et les messages utilisateur. Le code
et les identifiants restent en anglais ; les termes techniques du domaine
(*float*, *short interest*, *borrow rate*, *locate*) ne se traduisent pas.

Les fichiers Python sont écrits **sans accents** dans les commentaires et
docstrings : la sortie console Windows les mange, et un message d'erreur
illisible au mauvais moment coûte plus que l'élégance typographique.

## Environnement

Tout passe par le venv du projet. Rien n'est installé globalement.

```powershell
.venv\Scripts\python.exe <script>
```

`yt-dlp`, `pymupdf`, `yfinance`, `requests`, `pip-audit` y sont installés,
ainsi que `ib_async` pour la connexion TWS/Gateway — optionnelle : le moteur
et toutes les suites tournent sans elle.
Après tout ajout de dépendance : `.venv\Scripts\python.exe -m pip_audit`.

## Le principe qui gouverne tout le code

**Une règle bloquante refuse, elle n'avertit pas.** Et **une donnée absente
vaut refus, jamais feu vert.**

Ce n'est pas de la prudence décorative. La séance perdante qui a motivé tout ce
travail n'a pas eu lieu par ignorance de la règle — l'opérateur connaissait son
propre cours. Elle a eu lieu parce qu'au moment de décider, la conviction l'a
emporté. Un système qui se contente d'afficher un avertissement reproduit cette
situation avec une étape de plus.

Corollaire pratique : ne jamais combler un champ manquant par une valeur
plausible. Voir [agent/sources/CLAUDE.md](agent/sources/CLAUDE.md).

## Provenance des chiffres

**Toute valeur numérique vient du cours PDF, jamais des transcripts vidéo.**
Ces derniers sont issus de sous-titres automatiques qui corrompent
systématiquement les nombres — deux montants contradictoires apparaissent à
quinze secondes d'intervalle dans une même séance. Les transcripts fournissent
les situations, le cours fournit les seuils.

## Tests

Toutes les suites sont hors ligne : ni réseau, ni clé, ni compte.

```powershell
.venv\Scripts\python.exe agent\Test\test_agent.py     # 43
.venv\Scripts\python.exe agent\Test\test_edgar.py     # 22
.venv\Scripts\python.exe agent\Test\test_sources.py   # 21
.venv\Scripts\python.exe agent\Test\test_broker.py    # 17
.venv\Scripts\python.exe agent\Test\test_ibkr.py      # 40
.venv\Scripts\python.exe agent\Test\test_scanner.py   # 31
.venv\Scripts\python.exe agent\Test\test_doctor.py    # 11
```

Pas de CI. Les lancer manuellement avant toute modification de règle.

## Ce que ce projet ne fait pas

Il ne passe aucun ordre réel et ne décide jamais d'entrer à la place de
l'opérateur. Son rôle est de qualifier, calculer les tailles, et refuser ce qui
viole une règle. La méthode documentée est enseignée par un tiers ; elle n'est
validée par aucune source indépendante et ne constitue pas un conseil
d'investissement.

## Taille des fichiers

Aucun fichier source ne dépasse ~4 000 jetons. Au-delà, découper plutôt
qu'étendre.

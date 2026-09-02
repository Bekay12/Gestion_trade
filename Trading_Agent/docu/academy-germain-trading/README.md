# Academy Germain Trading — documentation de chaîne

Chaîne : [@AcademyGermainTrading](https://www.youtube.com/@AcademyGermainTrading)
(`UCdSNM4mXuWBr3EYKe0ScWHw`)
Dernière actualisation : **2026-08-21** — 13 vidéos sur 13 ingérées.

Documentation vivante construite à partir des sous-titres de la chaîne. Elle
n'est pas un empilement de résumés : les transcripts bruts sont conservés
séparément, et la connaissance est organisée **par concept**. Une nouvelle
vidéo modifie les notes existantes plutôt que d'en ajouter une de plus.

## Par où commencer

**[00 — Comment lire ce corpus](knowledge/00-lecture-du-corpus.md)** —
à lire en premier, sans exception. Les sous-titres sont automatiques et
corrompent systématiquement les nombres et le jargon. Cette note dit ce qui est
fiable, ce qui ne l'est pas, et fournit le lexique de décodage.

## Les notes

| Note | Contenu |
|---|---|
| [00 — Lecture du corpus](knowledge/00-lecture-du-corpus.md) | Fiabilité des sources, lexique ASR, règles d'usage |
| [01 — Short sur spike de dilution](knowledge/01-methode-short-sur-spike.md) | La méthode centrale et son point faible |
| [02 — Catalyseurs et dilution](knowledge/02-catalyseurs-et-dilution.md) | Placement privé, warrants, calendrier d'annonce |
| [03 — Sélection et filtres](knowledge/03-selection-et-filtres.md) | Float, volume, news, contrainte d'emprunt |
| [04 — Gestion du risque](knowledge/04-gestion-du-risque.md) | Règles nommées, mécanisme d'échec récurrent |
| [05 — Plateforme DAS Trader](knowledge/05-plateforme-das-trader.md) | Construction de l'espace de travail |
| [06 — Journal des séances](knowledge/06-journal-des-seances.md) | Index chronologique des 13 vidéos |

## Carte de couverture

Ce que le corpus couvre, et à quel point. La colonne « solidité » dit si la
note repose sur plusieurs vidéos concordantes ou sur un passage isolé.

| Sujet | Solidité | Commentaire |
|---|---|---|
| Discipline et règles de risque | **Bonne** | Énoncées à partir d'échecs datés, donc concrètes |
| Catalyseurs de dilution | Moyenne | Mécanique claire, chiffres inexploitables |
| Méthode d'entrée | Moyenne | Raisonnement lisible, aucun seuil chiffré |
| Espace de travail DAS Trader | Moyenne | Séquence conservée, cibles visuelles perdues |
| Filtres de sélection | Faible | Qualitatifs uniquement |
| Lecture level 2 / tape | **Absente** | Emplacement documenté, lecture jamais enseignée |
| Hotkeys | **Absente** | Lacune la plus coûteuse vu la méthode |
| Warrants | **Absente** | Explicitement remis à plus tard par l'auteur |
| Paramétrage du scanner | **Absente** | L'auteur n'utilise pas celui du courtier |

Ces lacunes sont le programme des prochaines actualisations. Si une future
vidéo traite les hotkeys ou le scanner, elle enrichit
[05](knowledge/05-plateforme-das-trader.md) et
[03](knowledge/03-selection-et-filtres.md) — elle ne crée pas de note nouvelle.

## Actualiser

```powershell
.venv\Scripts\python.exe scripts\youtube\ingest_channel.py --docu-dir docu\academy-germain-trading
```

L'URL de la chaîne est mémorisée : pas besoin de la redonner. Seules les vidéos
publiées depuis le dernier passage sont récupérées. Le filigrane vit dans
`.state/index.json` (vidéos vues, statut, dernière traitée, date du dernier
passage) et est réécrit après **chaque** vidéo, donc une interruption ne coûte
que la vidéo en cours.

Après ingestion, la mise à jour des notes de `knowledge/` est un travail de
lecture, pas un traitement automatique — c'est ce qui fait la différence entre
cette documentation et un tas de transcripts. Reporter les changements dans
[CHANGELOG.md](CHANGELOG.md).

Détail du pipeline : [scripts/youtube/README.md](../../scripts/youtube/README.md).

## Structure

```
academy-germain-trading/
  README.md            ce fichier — index et carte de couverture
  CHANGELOG.md         ce que chaque actualisation a changé
  knowledge/           la doc vivante, une note par concept
  transcripts/         un .md par vidéo, horodaté, jamais réécrit
  .state/index.json    le filigrane d'ingestion
```

## Avertissement

Les vidéos sources portent une clause explicite : ce ne sont pas des conseils
d'investissement et les résultats montrés ne sont pas typiques. Cette
documentation décrit une pratique observée dans un corpus de deux semaines ;
elle ne la valide pas et ne la recommande pas.

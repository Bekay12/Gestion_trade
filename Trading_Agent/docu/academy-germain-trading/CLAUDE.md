# docu/academy-germain-trading/

Corpus de la chaîne YouTube du même auteur que le cours. 13 vidéos, du
2026-06-26 au 2026-08-20.

| Dossier | Nature |
|---|---|
| `transcripts/` | Un fichier par vidéo — **immuables** |
| `knowledge/` | Notes par concept — **modifiées en place** |
| `.state/` | Filigrane d'ingestion, ne pas éditer à la main |

## La règle qui fait la valeur du dossier

`transcripts/` est la matière première : une entrée par vidéo, jamais réécrite.

`knowledge/` est découpé **par concept, pas par vidéo**. Quand une nouvelle
vidéo traite un sujet déjà couvert, on modifie la note existante — précision
ajoutée, nuance, ou contradiction signalée avec les deux sources datées. On
n'ajoute pas une section de plus.

C'est cette séparation, et elle seule, qui empêche la documentation de devenir
un empilement de résumés.

## Fiabilité : lire `knowledge/00-lecture-du-corpus.md` en premier

Les sous-titres sont **automatiques**. Trois conséquences non négociables :

**Les nombres sont corrompus.** C'est la défaillance la plus grave parce
qu'elle est silencieuse — un chiffre faux se lit comme un chiffre vrai. Dans la
séance du 19 août, le même montant apparaît sous deux valeurs à quinze secondes
d'intervalle. Aucun montant, prix ou résultat de ce corpus ne se reprend sans
réécoute à l'horodatage.

**Les tickers sont peu fiables** à l'oral. Ceux des titres de vidéos le sont.

**Les référents visuels sont perdus.** Le tutoriel de plateforme est un partage
d'écran bâti sur « vous cliquez *ici* » ; le texte conserve la séquence des
menus, pas les cibles.

Un lexique de décodage (créder → trader, bourousse → bourse, montage → level 2)
est dans la note 00.

## Ce que le corpus apporte malgré cela

Le **raisonnement** passe très bien, et les séances perdantes sont la partie la
plus instructive : l'auteur y nomme les règles qu'il vient d'enfreindre, avec
la perte au bout. C'est de là que vient la règle R13 du niveau 2, la seule dont
la transgression soit documentée avec son coût.

## Actualiser

```powershell
.venv\Scripts\python.exe scripts\youtube\ingest_channel.py --docu-dir docu\academy-germain-trading
```

L'URL est mémorisée. Seules les vidéos publiées depuis le dernier passage sont
récupérées. Après ingestion, la mise à jour de `knowledge/` est un travail de
lecture, pas un traitement automatique — c'est précisément ce qui distingue ce
dossier d'un tas de transcripts.

Consigner les deltas dans `CHANGELOG.md` : ce que les nouvelles vidéos ont
**changé**, pas ce qu'elles racontent.

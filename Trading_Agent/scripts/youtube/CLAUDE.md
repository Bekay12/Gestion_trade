# scripts/youtube/

Outillage d'ingestion. Produit la matière première de
[../../docu/](../../docu/) ; ne produit aucun livrable lui-même.

| Fichier | Rôle |
|---|---|
| `ingest_channel.py` | Chaîne YouTube → transcripts, incrémental |
| `vtt_clean.py` | WebVTT → prose horodatée |
| `split_course.py` | PDF du cours → un fichier par chapitre |

## Pourquoi deux appels yt-dlp par vidéo

Métadonnées et sous-titres sont récupérés **séparément**, et il faut que ça le
reste. Le premier essai les demandait ensemble : yt-dlp interrogeait une langue
absente de la vidéo, prenait un HTTP 429, et n'écrivait jamais le
`.info.json`. Résultat, des transcripts nommés `0000-00-00_<videoid>.md`, sans
titre ni date, avec un statut `ok` mensonger dans le filigrane.

La langue est désormais choisie dans l'**inventaire réel** de la vidéo avant
d'être demandée. Ne pas refusionner ces deux appels pour « économiser une
requête ».

## Les trois défauts des sous-titres automatiques

`vtt_clean.py` les traite dans cet ordre, et l'ordre compte :

1. les balises de timing par mot (`<00:00:01.234><c>mot</c>`) ;
2. la répétition en fenêtre glissante, où chaque cue reprend la queue de la
   précédente ;
3. le **chevauchement partiel** entre cues consécutives, traité au niveau des
   mots.

Sans la troisième passe, le texte duplique le milieu de chaque phrase. C'est le
défaut le moins visible et le plus tenace.

Ce que le nettoyage ne peut pas restituer : la ponctuation et les locuteurs.
C'est l'étape de synthèse humaine qui les reconstruit.

## Le filigrane

`.state/index.json` est réécrit **après chaque vidéo**, de façon atomique. Une
interruption au milieu d'une chaîne ne coûte que la vidéo en cours.

Un statut `no_captions` n'est pas retenté automatiquement : les sous-titres
n'apparaissent pas rétroactivement. `--retry-missing` force le contrôle.

## Limite connue et non contournable

Les nombres, tickers et termes anglais sont corrompus par la transcription.
**Aucun chiffre issu d'un transcript ne doit être repris** — voir
[../../docu/academy-germain-trading/CLAUDE.md](../../docu/academy-germain-trading/CLAUDE.md).

## Encodage

La sortie console Windows mange les accents : les fichiers Python de ce dossier
sont écrits sans accents. Les fichiers produits sont en UTF-8 explicite
(`encoding="utf-8"` partout), ce qui n'est pas le défaut de Windows.

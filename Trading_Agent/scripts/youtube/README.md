# Pipeline d'ingestion YouTube

Transcription incrémentale d'une chaîne YouTube vers une documentation vivante
sous `docu/<nom-de-la-chaine>/`.

## Prérequis

`yt-dlp` est installé dans le venv du projet (`.venv/Scripts/yt-dlp.exe`), pas
sur le PATH global — c'est voulu, la version reste épinglée à ce projet.
`ffmpeg` n'est pas nécessaire : les sous-titres sont servis nativement en WebVTT
et aucune conversion audio n'a lieu.

## Utilisation

Premier passage sur une chaîne (l'URL n'est demandée qu'une fois) :

```powershell
.venv\Scripts\python.exe scripts\youtube\ingest_channel.py "https://www.youtube.com/@la-chaine"
```

Actualisation — ne récupère que les vidéos publiées depuis le dernier passage :

```powershell
.venv\Scripts\python.exe scripts\youtube\ingest_channel.py --docu-dir docu\la-chaine
```

| Option | Effet |
|---|---|
| `--limit N` | S'arrête après N nouvelles vidéos (utile pour un essai) |
| `--sub-langs "fr.*,en.*"` | Sélecteur de langue yt-dlp (défaut) ; `"all"` prend tout |
| `--retry-missing` | Retente les vidéos en échec ou sans sous-titres |
| `--paragraph-seconds N` | Durée de parole par paragraphe horodaté (défaut 90) |

## Ce que produit une exécution

```
docu/<chaine>/
  transcripts/            un .md par vidéo, horodaté, jamais réécrit
  .state/index.json       le filigrane : vidéos vues, statut, dernière traitée
  .state/raw/             scratch yt-dlp, vidé après chaque vidéo
```

Chaque transcript porte un frontmatter (`video_id`, `upload_date`, `url`,
`caption_language`, `caption_manual`) et un corps découpé en paragraphes
préfixés `**[HH:MM:SS]**`, ce qui permet à une note de documentation de citer
un moment précis de la vidéo.

## Le filigrane

`.state/index.json` retient chaque `video_id` déjà traité avec son statut
(`ok`, `no_captions`, `error`), plus `last_video` et `last_run`. L'état est
réécrit après **chaque** vidéo, de façon atomique : une interruption au milieu
d'une chaîne de 300 vidéos ne perd que la vidéo en cours, et la reprise ne
re-télécharge rien.

Une vidéo `no_captions` n'est pas retentée automatiquement — les sous-titres
n'apparaissent pas rétroactivement. `--retry-missing` force le contrôle si la
chaîne a activé les sous-titres après coup.

## Nettoyage des sous-titres

`vtt_clean.py` traite trois défauts propres aux sous-titres automatiques
YouTube, dans cet ordre : les balises de timing par mot
(`<00:00:01.234><c>mot</c>`), la répétition en fenêtre glissante où chaque cue
reprend la queue de la précédente, et le chevauchement partiel entre cues
consécutives, traité au niveau des mots. Sans la troisième passe, le texte
duplique le milieu de chaque phrase.

Ce que le nettoyage ne peut pas restituer : la ponctuation et l'identification
des locuteurs, absentes des sous-titres automatiques. C'est l'étape de synthèse
vers `knowledge/` qui les reconstruit, pas ce script.

# Journal des actualisations

Une entrée par passage d'ingestion. Consigner ce que les nouvelles vidéos ont
**changé** dans les notes, pas ce qu'elles racontent — le résumé, c'est le rôle
de `knowledge/`.

---

## 2026-08-21 — Ingestion initiale

**Vidéos** : 13 nouvelles (intégralité de la chaîne), du 2026-06-26 au
2026-08-20. Aucune erreur, aucune vidéo sans sous-titres.
**Dernière traitée** : `QLBy2SaATkk` — *DayTrade Recap 20 aout 2026*.

**Notes créées** : les sept notes de `knowledge/`, le README et son tableau de
couverture.

**Constat déterminant** — les sous-titres sont automatiques et corrompent
systématiquement les nombres, les tickers et le jargon anglais. Deux montants
contradictoires apparaissent à quinze secondes d'intervalle dans la séance du
19 août. Cette découverte a changé la conception de la documentation : la note
[00](knowledge/00-lecture-du-corpus.md) a été écrite en préalable obligatoire,
et toutes les notes distinguent désormais affirmation de méthode (reprise
telle quelle), affirmation chiffrée (marquée non vérifiée) et affirmation
réglementaire (à vérifier hors corpus).

**Deuxième constat** — le tutoriel DAS Trader est un partage d'écran dont le
transcript perd les cibles visuelles. La note
[05](knowledge/05-plateforme-das-trader.md) documente donc la séquence des
menus en s'annonçant explicitement comme non substituable au visionnage.

**Lacunes ouvertes**, dans l'ordre où elles coûtent le plus cher : hotkeys,
paramétrage du scanner, lecture du level 2 et du tape, warrants, seuils
chiffrés de sélection et de dimensionnement. Détail dans la carte de couverture
du [README](README.md).

**Défauts corrigés dans le pipeline pendant ce passage** : le premier essai
demandait à yt-dlp des langues absentes des vidéos, ce qui déclenchait un
HTTP 429 et empêchait l'écriture du `.info.json` — les transcripts sortaient
sans titre ni date, sous un nom `0000-00-00_<videoid>`, avec un statut `ok`
mensonger dans le filigrane. Métadonnées et sous-titres sont désormais
récupérés par deux appels distincts, et la langue est choisie dans l'inventaire
réel de la vidéo. Les trois fichiers fautifs ont été supprimés et régénérés.

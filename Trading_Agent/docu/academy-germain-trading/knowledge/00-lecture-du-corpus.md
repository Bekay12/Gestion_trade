---
concept: lecture-du-corpus
maj: 2026-08-21
videos_sources: [toutes]
confiance: haute
---

# Comment lire ce corpus

À lire avant toute autre note. Les transcripts viennent des sous-titres
**automatiques** de YouTube, pas d'une transcription humaine. La qualité est
suffisante pour suivre un raisonnement, insuffisante pour citer un chiffre.

## Ce qui est fiable

Les métadonnées (`upload_date`, `title`, `url`, `duration_min`) viennent de
l'API YouTube, pas de l'ASR. Le **fil du raisonnement** passe aussi très bien :
quand l'auteur explique pourquoi il entre ou sort d'une position, l'argument
est lisible.

## Ce qui ne l'est pas

**Les nombres.** C'est la défaillance la plus grave, parce qu'elle est
silencieuse — un chiffre faux se lit comme un chiffre vrai. Dans la séance du
19 août, le même placement privé est transcrit « 10 millions » puis
« 100 millions » à quinze secondes d'intervalle, et un prix apparaît comme
« 2 dollars Kincs » puis « 2 dollars coinc » (vraisemblablement *2,50*, jamais
confirmé). Aucun montant, aucun prix, aucun P&L de ce corpus ne doit être
repris sans réécoute de la vidéo à l'horodatage indiqué.

**Les tickers.** Trois à quatre lettres prononcées à l'oral survivent mal.
Ceux des titres de vidéos (STKH, WETO) sont fiables ; ceux entendus dans le
flux ne le sont pas.

**Les référents visuels.** Le tutoriel DAS Trader est un partage d'écran bâti
sur « vous allez cliquer *ici* », « ça va ressembler à *ça* ». Le texte seul ne
restitue pas la moitié de l'information ; la note correspondante documente la
séquence des menus, pas les gestes.

## Lexique de décodage ASR

Les substitutions systématiques rencontrées, pour que la lecture d'un
transcript brut ne bute pas dessus :

| Transcrit | Lire |
|---|---|
| créder, crédit, créer (sur un stock), crédé | trader, tradé |
| créding, cours de crédit | trading, cours de trading |
| DAS Creader | DAS Trader |
| bourousse | bourse |
| warant, Waran | warrant |
| flottant / flot | float |
| prémché, prém marché | pré-marché (*premarket*) |
| after mac, f market | after market |
| time cellel | time & sales |
| alt (dans « il a eu un alt ») | halt |
| coinc, Kincs (après un montant) | fraction décimale, valeur incertaine |

## Règle d'usage pour les notes

Toute note de `knowledge/` distingue trois statuts. Une affirmation **méthode**
(comment l'auteur raisonne) est reprise telle quelle : l'ASR la restitue bien.
Une affirmation **chiffrée** est marquée `[chiffre non vérifié]` ou omise. Une
affirmation **réglementaire** — les délais légaux, les mécanismes de marché —
est signalée comme *à vérifier hors corpus*, parce que l'auteur les énonce de
mémoire et que le corpus ne peut pas les valider.

Voir [06-journal-des-seances.md](06-journal-des-seances.md) pour retrouver la
vidéo et l'horodatage d'un point précis.

---
niveau: 2
maj: 2026-08-21
source_chiffres: cours Academy Germain (les transcripts vidéo ne sont pas fiables sur les nombres)
---

# Règles exécutables

Toutes les valeurs viennent du **cours**, jamais des transcripts vidéo dont les
nombres sont corrompus. Chaque règle est formulée pour être testable : une
condition, un seuil, une conséquence.

## R1 — Dimensionnement

L'ordre des décisions est imposé et ne se réorganise pas : risque accepté →
niveau d'invalidation → taille. Jamais l'inverse.

```
Risque max par trade = Capital × taux
Taille de position   = Risque max / |Prix entrée − Prix stop|
```

Taux selon l'expérience :

| Profil | Taux par trade |
|---|---|
| Débutant (0-6 mois) | 0,5 % |
| Intermédiaire (6-18 mois, 3 mois positifs) | 1 % |
| Confirmé (18 mois+, 12 mois positifs, journal tenu) | 1 à 2 % |
| Professionnel | 2 à 3 % |

**Contrainte d'exécution** : le calcul se fait avant de passer l'ordre. Une
taille calculée après l'entrée n'est plus un calcul, c'est une justification.

## R2 — Rapport gain/risque

```
R/R = (Cible − Entrée) / (Entrée − Stop)
```

| R/R | Taux de réussite minimal pour être à l'équilibre |
|---|---|
| 1:1 | ~51 % — insuffisant, les frais font basculer |
| 1:2 | ~34 % — plancher acceptable |
| 1:3 | ~25 % — cible recommandée |
| 1:5 | ~17 % — excellent |

**Règle d'abstention** : sous 1:2, le trade ne se prend pas, quelle que soit la
conviction.

## R3 — Bornes de perte

| Borne | Seuil | Conséquence |
|---|---|---|
| Perte journalière max | 2 à 3 % du capital | Fermer tout, arrêter la journée |
| Pertes consécutives | 3 dans la séance | Arrêt de la séance |

## R4 — Paliers de drawdown

| Drawdown depuis le sommet | Action |
|---|---|
| 0 – 5 % | Normal, plan inchangé |
| 5 – 10 % | Réduire les tailles de 25 %, revoir les critères d'entrée |
| 10 – 15 % | Réduire les tailles de 50 %, revoir la stratégie |
| 15 – 20 % | Passage en simulation jusqu'à diagnostic |
| > 20 % | Arrêt complet, reprise en démo uniquement |

## R5 — Stop

- Placé au niveau où **le scénario est invalidé**, pas à une distance
  arbitraire.
- Jamais déplacé dans le sens défavorable en cours de position. Il ne se
  déplace que pour protéger un gain acquis.
- Un stop trop serré est déclenché par le bruit normal ; trop large, il rend la
  taille absurde. C'est le niveau d'invalidation qui arbitre, pas le confort.

Méthodes de placement : sous un support identifié, sous le VWAP (day trading
actif), selon la volatilité moyenne, sous un niveau psychologique, ou en
pourcentage fixe (débutants).

## R6 — Filtres de sélection

| Critère | Valeur retenue |
|---|---|
| Marché | Actions US |
| Capitalisation | Nano à small (< 300 M$) |
| Prix | 0,50 $ à 20 $ |
| Volume moyen | > 500 K (confort : > 1 M) |
| Volume relatif (RVOL) | > 2 |
| Écart d'ouverture | > 5 % |
| Flottant | Vérification obligatoire si < 20 M |
| Intérêt vendeur | > 10 % pour un potentiel de rachat forcé |

Le volume relatif est le filtre décisif : il détecte l'anomalie du jour, pas la
taille de l'entreprise.

## R7 — Mécanique de la vente à découvert

**Locate obligatoire** avant tout ordre. Sans lui, l'opération est illégale.
Le locate est révocable en séance, ce qui peut forcer un rachat.

| État du titre | Conséquence |
|---|---|
| Facile à emprunter | Coût négligeable (< 1 %/an) |
| Difficile à emprunter | Coût significatif, trade nécessairement court |
| Indisponible | Titre non jouable ce jour |

```
Coût journalier d'emprunt = (Valeur position × Taux annuel) / 365
```

Le coût court même si le cours ne bouge pas. Sur un taux élevé et un trade de
quelques jours, il peut absorber le gain espéré — à intégrer au R/R, pas à
découvrir après.

**Restriction Rule 201** : déclenchée à −10 % sous la clôture de la veille,
active jusqu'à la clôture du lendemain. Interdit de vendre au prix acheteur ;
seulement à l'offre ou au-dessus.

## R8 — Indicateurs de tension vendeuse

| Mesure | Formule | Lecture |
|---|---|---|
| Intérêt vendeur % | Actions vendues à découvert / flottant | < 5 % neutre · 15-25 % élevé · > 40 % extrême |
| Jours pour couvrir | Actions vendues / volume moyen | > 5 tendu · > 10 très tendu |
| Rotation du flottant | Volume du jour / flottant | 1-3× journée active · 3-7× exceptionnel · > 7× suspect |

## R9 — Exclusion (règle dérivée du croisement)

**Ne pas prendre de position vendeuse** lorsque plusieurs de ces conditions
sont réunies simultanément :

- restriction Rule 201 active,
- titre difficile à emprunter,
- flottant réduit,
- intérêt vendeur déjà élevé.

Motif : dans cette configuration, les vendeurs en place subissent un coût
quotidien, ne peuvent plus accélérer la baisse, et sont forcés au rachat à la
moindre poussée — ce qui alimente la hausse contre laquelle on se positionne.

Cette règle n'apparaît dans aucune vidéo. Elle est la principale optimisation
issue du croisement — voir
[../synthese/07-chaine-face-au-cours.md](../synthese/07-chaine-face-au-cours.md).

## R10 — Fenêtres horaires (heure de l'Est)

| Plage | Usage |
|---|---|
| 4h00 – 6h30 | Nouvelles de la nuit, vérification des dépôts réglementaires |
| 6h30 – 8h00 | Analyse des candidats — 10 à 15 min par titre, pas plus |
| 8h00 – 9h00 | Rédaction des plans de trade |
| 9h00 – 9h30 | Contrôle final, alertes posées, état mental |
| 9h30 – 9h35 | **Ne pas opérer** sauf plan explicite |
| 10h00 – 11h30 | Fenêtre principale |
| 11h30 – 14h00 | Activité réduite, signaux moins fiables |
| 16h00 – 17h00 | Journal, analyse, mise à jour de la liste |

Le pré-marché sert à analyser, pas à opérer : volume faible, fourchettes
larges.

## R11 — Plan de trade

Sept entrées, remplies **avant** l'ouverture, pour chaque titre retenu :
catalyseur en une phrase · sens · zone d'entrée · niveau d'invalidation ·
objectifs échelonnés · taille calculée par R1 · événement qui annule le trade.

**Test d'admission** : si le catalyseur ne tient pas en une phrase, le titre
sort de la liste.

## R12 — Abstention

Ne pas opérer : après trois pertes dans la séance · en état de fatigue, de
stress ou de pression financière · quand les conditions ne correspondent pas à
la stratégie · quand la motivation est de ne pas rater quelque chose plutôt
qu'un signal · quand la préparation n'est pas faite à l'ouverture.

## R13 — Poste de travail

Décision prise sur un poste affichant simultanément graphique, carnet et
exécutions. Pas depuis un téléphone : une seule source visible à la fois
signifie décider sur une fraction de l'information.

Règle issue de la chaîne (séance du 18 août), et la seule de cette liste dont
la transgression est documentée avec son coût.

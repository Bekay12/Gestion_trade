# Contrôle croisé du salaire net (Tâche 3)

Comparaison du net approché calculé par `scripts/salaire.py` (`netto_jahr`) contre une
table publiée indépendante, pour vérifier l'hypothèse déclarée dans le chapitre 5
(revenu imposable = brut moins part salariale des cotisations sociales moins forfait de
frais professionnels 1 230 EUR, sans correction pour coller à une cible).

## Source de référence

Deutschland-Rechner, table Brutto-Netto 2026, Steuerklasse I
(<https://www.deutschland-rechner.de/brutto-netto-tabelle>, consulté le 30.09.2026),
avec les hypothèses affichées : assurance légale (GKV) au taux de cotisation
supplémentaire moyen de 2,9 %, assujetti à l'assurance retraite, sans enfant à partir de
23 ans (surcharge dépendance de 0,6 %), sans impôt d'Église, sans abattement
supplémentaire. Ce jeu d'hypothèses correspond exactement à `sv_arbeitnehmer` dans
`data/quellen.json` (Steuerklasse I implicite : pas de splitting, pas d'abattement
conjoint) et à l'absence d'impôt d'Église par défaut du projet (`contexte-commun.md`).

La table ne publie que des paliers ronds (pas de ligne à 59 250 EUR, le salaire d'entrée
`einstiegsgehalt_brutto`) ; la ligne la plus proche et directement comparable est
60 000 EUR/an.

## Résultats

| Brut annuel (EUR) | Net publié (Deutschland-Rechner) | Net calculé (`salaire.netto_jahr`, tarif et SV réels 2026) | Écart absolu | Écart relatif |
|---|---|---|---|---|
| 55 000 | 34 978 EUR/an (2 915 EUR/mois) | 35 250,50 EUR/an (2 937,54 EUR/mois) | +272,50 EUR/an | +0,78 % |
| 60 000 | 37 561 EUR/an (3 130 EUR/mois) | 37 874,00 EUR/an (3 156,17 EUR/mois) | +313,00 EUR/an | +0,83 % |

Pour mémoire, au salaire d'entrée réel du projet (`einstiegsgehalt_brutto` = 59 250
EUR/an, non présent dans la table publiée) : net calculé 37 483,12 EUR/an, soit
3 123,59 EUR/mois (63,3 % du brut), cohérent par interpolation entre les deux lignes
ci-dessus.

## Verdict

Écart relatif de 0,78 à 0,83 %, **bien en dessous du seuil de 3 %** fixé par la tâche.
Aucune ligne `LUECKEN.md` n'est nécessaire ; le modèle n'a pas été retouché pour réduire
cet écart (consigne : ne jamais ajuster le modèle pour coller à une cible).

Source probable du petit écart résiduel, systématiquement dans le même sens (le calcul
approché est chaque fois légèrement plus généreux) : le calculateur en ligne applique
vraisemblablement les tables de retenue à la source mensuelle (Lohnsteuertabelle, avec
son propre arrondi et son propre calcul du Vorsorgepauschale), alors que
`salaire.est_32a` applique directement le barème annuel de veranlagte Einkommensteuer
(§32a EStG) sur un revenu imposable simplifié (forfait de frais professionnels fixe de
1 230 EUR au lieu du Vorsorgepauschale complet). Cet écart et son sens sont à publier
tels quels dans le chapitre 5, comme demandé.

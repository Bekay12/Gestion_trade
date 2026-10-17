# Suivi des verdicts — startup-investment-analyzer

But : confronter chaque verdict à ce qui s'est réellement passé, et repérer ce qui manque
systématiquement à la méthode (hurdle = ROE propre, book value tenu plat, pas de prévision de
croissance) plutôt que de juger un rapport une seule fois puis l'oublier. Même logique que
`evaluer_gaps.py` pour le trading de gaps, mais sur un horizon de mois/années au lieu de jours.

Une ligne par rapport, mise à jour quand on revient dessus — jamais réécrite rétroactivement.

**Rempli a posteriori le 07.10.2026** avec les 9 rapports déjà existants dans `Business_Analysen/`
(Alamos, Brown & Brown, Daimler Truck, GEA Group, Neste, Pharma AZN/RDY, Yara, Lakeland) en plus
des deux du jour — je ne les avais pas indexés la première fois, seulement les deux qu'on venait
de produire. Dossiers vides ou hors périmètre (ESAB, EVN, Honeywell, Pool Corp, Prysmian,
Suedzucker, Aumann AG, Dual_Champions, Insider_2026_09, Retraite_Dividendes) ne sont pas des
verdicts actions isolés et restent hors de ce tableau.

| Date | Société | Verdict | Prix au verdict | Hurdle (base) | Seuil d'entrée (10 ans, base) | Cible consensus | Prix constaté (date) | Écart au verdict |
|---|---|---|---|---|---|---|---|---|
| 16.09.2026 | Alamos Gold (AGI) | Ne pas acheter | 35,66 $ | 14,62 % (ROE propre) | 12,97 $ | 46,25 $ (+29,7 %, 13 analystes) | — | — |
| n.d. (modèle de référence) | Brown & Brown (BRO) | Ne pas acheter | 71,39 $ | 11,1 % (ROE propre) | 67,80 $ | n.d. (non relevé dans ce rapport) | — | — |
| ~18.09.2026 | AstraZeneca (AZN) | Ne pas acheter | 166,14 $ | 10,16 % (ETF pharma, pas ROE) | — | 212,9 $ (+28,1 %, 11 estim. S&P Global) | — | — |
| ~18.09.2026 | Dr. Reddy's (RDY) | **Meiden — gap de données, pas un prix jugé trop haut** | 12,19 $ | 10,16 % (même ETF) | **non calculable** (donnée manquante déclarée, pas estimée) | 13,36 $ (+9,6 %, 5 estim.) | — | — |
| 20.09.2026 | GEA Group (G1A.DE) | Ne pas acheter | 64,60 € | 13,56 % / 9,08 % long (indice, pas ROE) | 17,28 € / 22,04 € (long) | 70,26 € (+8,8 %) | — | — |
| ~20.09.2026 | Daimler Truck (DTG.DE) | Ne pas acheter | 43,49 € | 13,56 % / 9,08 % long (indice, pas ROE) | 23,94 € | 51,26 € (+17,9 %, "Outperform") | — | — |
| 27.09.2026 | Neste (NESTE.HE) | Ne pas acheter, tous horizons | 34,01 € | 13,56 % / 9,08 % long (MSCI World, pas ROE) | 7,36 € (base, 3 ans) | 34,26 € (quasi égal au cours — rare accord) | — | — |
| 14.09.2026 | Yara International (YAR.OL) | Ne pas acheter | 460,90 NOK | 11,34 % (ROE propre, hors gain de change) | 338-340 NOK (optimiste, 10 ans) | 474,47 NOK (+2,95 %, 19 analystes, consensus = Halten, **même direction que le verdict**) | — | — |
| 24.09.2026 | Lakeland Industries (LAKE) | **Acheter sous 12,00 $** | 10,22 $ | **5,76 %** (taux du revolver, plancher explicite, pas ROE/indice) | déjà sous le seuil | n.d. | — | — |
| 06.10.2026 | Accenture (ACN) | Ne pas acheter | 195,05 $ | 26,7 % (ROE propre) | 87,5 $ | 223,03 $ (+14 %) | — | — |
| 07.10.2026 | CTS Eventim (EVD.DE) | Ne pas acheter | 55,80 € | 22,3 % (ROE propre) | 5,9 € (pessimiste) / 24,9 € (optimiste) | 85-100 € (consensus Buy) | — | — |
| 11.10.2026 | Nynomic AG (M7U.DE) | Ne pas acheter | 26,00 € | **4,32 % (ROE propre, moyenne 4 ans GJ2022-25 — la plus basse hurdle ROE de la série, Nynomic n'ayant jamais dépassé 9,67 % même en bonne année)** | 16,10 € (base, 15 ans) / 22,20 € (optimiste, 15 ans) | 21,25-30,00 € (consensus Buy, couverture contradictoire 2 vs 6 analystes) | — | — |
| 11.10.2026 | Umicore NV/SA (UMI.BR) | Ne pas acheter | 20,98 € | 12,68 % (ROE sur résultat ajusté, moyenne 3 ans GJ2023-25 — IFRS exclu car GJ2024 donnerait -77 % à cause d'une dépréciation ponctuelle de 1,79 Md€ sur Battery Materials) | 10,60 € (base, 15 ans) / 18,50 € (optimiste, 15 ans) | 23,79-25,63 € (Buy, 13-15 analystes, couverture large et crédible — **consensus le plus fiable de toute la série**) | — | — |

## Nuance apportée par Nynomic (11.10.2026)

Nynomic est le premier cas où la hurdle ROE propre (4,32 %) est **plus basse** que les hurdles
indicielles "douces" des autres rapports (9-13,6 %) — et le verdict reste pourtant "ne pas
acheter", sur les trois scénarios et les trois horizons testés. Ça complique le constat ci-dessous
("le choix de la hurdle décide le verdict") : ici le choix le plus généreux possible ne suffit pas
à faire passer le test, donc au moins dans ce cas le verdict négatif n'est pas un artefact du
choix de hurdle. Raison structurelle : Nynomic elle-même n'a jamais dépassé 9,67 % de ROE sur les
4 années disponibles (deux bonnes, deux de crise) — une entreprise à très forte quote de fonds
propres (55-75 %) et rendement des capitaux propres durablement faible, quelle que soit la hurdle
choisie.

## Premier constat, déjà visible sans attendre des mois

**La hurdle n'a jamais été appliquée de façon cohérente d'un rapport à l'autre.** Trois familles
coexistent : ROE propre de l'entreprise (Alamos, Brown & Brown, Yara, Accenture, CTS Eventim —
14 à 27 %, le plus sévère) ; un indice de marché externe (GEA, Daimler, Neste, AZN/RDY — 9 à
13,6 %, plus doux) ; et une seule fois un plancher de coût de la dette (Lakeland, 5,76 %, le plus
doux de tous). **Les deux seuls verdicts positifs ou "achat possible" de toute la liste
(Lakeland, et Neste où la cible consensus tombe quasiment sur le seuil) sont aussi les deux cas à
hurdle la plus basse ou à cours déjà sous la valeur comptable.** Avec 11 rapports, ce n'est plus
une coïncidence à deux points : **le choix de la hurdle décide le verdict au moins autant que les
fondamentaux de l'entreprise**, et ce choix n'a jamais été standardisé — chaque rapport a pris la
meilleure source sourcée disponible sur le moment, pas la même source. C'est le vrai "manque de
la méthode" qu'on cherchait : pas un biais caché, un biais déjà visible dans les textes, jamais
nommé comme tel avant aujourd'hui.

**Second constat :** sur les 9 verdicts "ne pas acheter" avec un consensus analystes relevé, **8
sur 9 ont un consensus au-dessus du cours** (direction opposée au verdict) — seul Yara a un
consensus dans la même direction ("Halten", pas "Kaufen"). Si cette proportion tient sur les
prochains rapports, ce n'est plus "le marché se trompe parfois", c'est que la méthode (valeur
terminale au book value, pas de croissance) est structurellement plus pessimiste que n'importe
quel consensus de marché — ce qui peut être juste, mais doit être dit explicitement dans chaque
futur rapport plutôt que découvert ici.

## Ce qu'on cherche en confrontant plus tard

- **Le verdict a-t-il eu raison, et pour la bonne raison ?** (voir `rigor-and-assumptions.md` —
  juger contre la promesse exacte, pas contre une hausse générale.)
- **Le consensus analystes avait-il raison plus souvent que la méthode ROE-propre + book value
  plat ?** Les deux rapports actuels divergent du consensus dans la même direction (plus
  pessimiste) — si ça se répète sur 5-10 rapports, c'est un biais systématique de la méthode à
  documenter, pas un hasard.
- **Le seuil d'entrée (break-even) a-t-il été atteint à un moment donné ?** Si oui, le modèle
  aurait dit d'acheter à ce prix précis — vérifiable rétroactivement.
- **La hurdle = ROE propre est-elle trop sévère de façon systématique ?** Les deux cas ici donnent
  des seuils d'entrée très inférieurs au cours constaté sur plusieurs années — si ça se vérifie à
  chaque fois, la méthode a un biais structurel (toute entreprise cotée au-dessus de son book
  value échoue presque mécaniquement ce test), à nommer explicitement plutôt qu'à découvrir à
  chaque rapport.

## Rappel

Chaque rapport complet reste dans son propre dossier (`Business_Analysen/<Société>/`) avec ses
sources primaires et son `data/analysis.json`. Ce fichier n'est qu'un index de confrontation, pas
une copie des rapports.

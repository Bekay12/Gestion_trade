# Design — Hausarbeit « Finanzwirtschaft für Ingenieure » : analyse financière d'Aumann AG

Date : 2026-07-29
Statut : validé par l'utilisateur
Échéance de rendu : **mardi 04.08.2026, 23:59** (envoi PDF à juergen.bott@hs-kl.de et katharina.moor@hs-kl.de)

## 1. Objectif

Produire un document LaTeX compilable, **en allemand**, niveau Master, rendu en PDF, répondant
point par point à l'énoncé `Docu_Hskl/Hausarbeit SoSe2026 Aumann.pdf` (6 parties, 100 points) :

| Partie | Sujet | Points |
|---|---|---|
| 1 | Description de l'entreprise (dirigeants, actionnariat, marchés de capitaux, analystes) | 10 |
| 2 | Analyse du Jahresabschluss 2024 (2.1 → 2.7) | 14 |
| 3 | Mesures stratégiques du Geschäftsbericht 2024 (3.1 → 3.6) | 12 |
| 4 | Interprétation du cours de bourse 2022–2025 | 4 |
| 5 | Évaluation des options A / B / C + synthèse | 40 |
| 6 | Analyse de cash-flow A / B / C + comparaison 2028 | 20 |

Source de base imposée : <https://www.aumann.com/en/investor-relations/>
(Geschäfts- und Quartalsberichte, Investorenpräsentationen).

## 2. Contraintes

### 2.1 Contrainte d'intégrité (centrale)

L'énoncé restreint l'usage de l'IA. L'utilisateur autorise explicitement l'IA à rédiger la prose
argumentative, **à condition qu'elle soit visuellement isolée** pour relecture et réécriture.

Répartition :

- **Noir** — production factuelle : extraction des données, calculs, recherche et citation des
  sources, tableaux chiffrés, structure LaTeX, graphiques.
- **Bleu (`\bk{}`)** — prose évaluative/interprétative : Management-Summary, cellules
  *Einschätzung*, *Statement*, *Empfehlung*, *Fazit*, interprétation du cours (partie 4).

### 2.2 Mécanique du bleu (décision utilisateur)

Option retenue : **prose bleue seule, matériel factuel en commentaires LaTeX.**

```latex
% FAKTENBASIS 5.1.1
%  • Auftragseingang 2024: 213,5 Mio. EUR (−41,1 %) [GB24, S. xx]
%  • Nettoliquidität 31.12.2024: 138,2 Mio. EUR [GB24, S. xx]
\bk{Die Neuausrichtung des vormaligen Segments „Classic“ ... }
```

- `\newcommand{\bk}[1]{{\color{blue}#1}}` défini dans le préambule.
- Le PDF sort propre dès la première compilation ; le matériel de vérification reste dans le `.tex`.
- Avant rendu, l'utilisateur relit/réécrit puis neutralise le bleu (`\renewcommand{\bk}[1]{#1}`).

### 2.3 Autres décisions utilisateur

- **Courbe de cours** : approche **hybride** — série mensuelle réelle d'un fournisseur de marché
  (avec date d'interrogation citée), points clés (clôture 31.12., plus haut/plus bas annuels)
  recoupés contre les Geschäftsberichte Aumann. Repli sur « rapports Aumann seuls » si le
  fournisseur bloque.
- **Citations** : Zitierweise allemande en **notes de bas de page**
  (« Aumann AG, Geschäftsbericht 2024, S. 47. »), + Quellenverzeichnis final scindé en
  (a) Aumann-Berichte, (b) Externe Quellen / Analysten, avec **Abrufdatum** pour tout lien web.

## 3. Sources — disponibilité vérifiée (2026-07-29)

`https://www.aumann.com/en/investor-relations/financial-reports` répond HTTP 200 ; l'archive
complète est téléchargeable. Documents pertinents :

| Rôle | Fichier |
|---|---|
| Source centrale (parties 2, 3, 5, 6) | `2024_aag-geschaeftsbericht-eng-hp.pdf` |
| Fin 2025 (parties 2.7, 4, base de marge 6.3) | `2025_AAG_Geschäftsbericht_FINAL_eng.pdf` |
| Comparatif 2022/2023 | `2022_aag-geschaeftsbericht-hp_englisch.pdf`, `2023_aag-geschaeftsbericht-hp_englisch.pdf` |
| Vérification du Ausblick 2025 (question 3.6) | Quartalsmitteilungen Q1/H1/Q3 2025 + Q1 2026 |

**Limitation à signaler dans le document** : les Geschäftsberichte 2022–2025 ne sont publiés
qu'en **anglais** sur le site. La rédaction est en allemand, les sources citées sont anglaises.

Support de cours de référence pour la terminologie : `Docu_Hskl/FfE_251124_123712.pdf`
(Prof. Dr. Jürgen Bott — Bilanz/GuV, Profitability/Activity/Solvency Ratios, Working Capital,
arbitrage **Liquidität ↔ Rentabilität**), explicitement invoqué par la question 5.1.3.

## 4. Architecture

### 4.1 Principe

Modulaire, avec **couche de données à source unique** : tout chiffre est déclaré une seule fois
comme macro dans `data/kennzahlen.tex` et consommé par le texte, les tableaux et les graphiques.
Une correction se propage partout ; l'incohérence interne entre texte et tableau devient
structurellement impossible.

La partie 6 (20 points, 3 IRR + comparaison) est **calculée par script**, pas à la main.

### 4.2 Arborescence

```
hausarbeit.tex              main, KOMA scrartcl, ngerman, siunitx (virgule décimale)
preamble.tex                \bk, styles de tableaux, Zitierweise en notes de bas de page
sections/
  00-deckblatt.tex          Deckblatt + Inhaltsverzeichnis
  01-unternehmen.tex        Partie 1 (1.1 → 1.5)
  02-jahresabschluss.tex    Partie 2 (2.1 → 2.7) + graphique de cours
  03-strategie.tex          Partie 3 (3.1 → 3.6)
  04-aktienkurs.tex         Partie 4  [bleu]
  05-handlungsoptionen.tex  Partie 5 (5.1 → 5.4)
  06-cashflow.tex           Partie 6 (annahmen, 6.1, 6.2, 6.3)
  90-quellen.tex            Quellenverzeichnis (a) Aumann (b) externes
  91-ki-nutzung.tex         Note KI-Nutzung
  92-erklaerung.tex         Eidesstattliche Erklärung
data/
  kennzahlen.tex            source unique de tous les chiffres (macros)
  aktienkurs.csv            série mensuelle 2022-01 → 2025-12 + date d'Abruf
  cf_a.tex cf_b.tex cf_c.tex vergleich2028.tex   (générés par script)
scripts/
  cashflow.py               tableaux 6.1 A/B/C + 6.3, IRR par bissection (stdlib seule)
  fetch_kurs.py             récupération de la série de cours + horodatage
refs/                       PDF Aumann téléchargés (traçabilité des numéros de page)
build.sh                    latexmk -pdf -outdir=out
out/hausarbeit.pdf
```

`refs/*.pdf` et `out/` sont gitignorés.

### 4.3 Interfaces entre unités

- `data/kennzahlen.tex` → expose des macros LaTeX nommées (`\UmsatzZFV`, `\EbitdaMargeZFV`,
  `\NettoliquiditaetZFIV`…). **Seul** endroit où un chiffre est écrit. Chaque macro porte en
  commentaire sa source et sa page.
- `scripts/cashflow.py` → lit un dict d'hypothèses en tête de fichier, écrit
  `data/cf_{a,b,c}.tex` (corps de tableaux booktabs) et `data/vergleich2028.tex`. Aucune
  dépendance externe. Réexécutable ; sortie déterministe.
- `scripts/fetch_kurs.py` → écrit `data/aktienkurs.csv` (colonnes `date,close`) et l'`Abrufdatum`
  en commentaire d'en-tête. `pgfplots` lit le CSV directement.
- `sections/*.tex` → consomment uniquement des macros et des fichiers `data/`. Aucun chiffre en dur.

### 4.4 Préambule — dépendances

Toutes vérifiées présentes sur la machine : `koma-script`, `pgfplots`, `booktabs`, `siunitx`,
`babel` (ngerman), `csquotes`, `microtype`, `tikz`, `xcolor`, `latexmk`, `pdflatex`.
Aucune dépendance manquante. Python 3 stdlib uniquement.

Réglages : `\sisetup{locale=DE, group-separator={.}, output-decimal-marker={,}}`,
`\usepackage[ngerman]{babel}`, `fontenc T1`, `inputenc utf8`.

## 5. Contenu par partie

### 5.1 Partie 1 — Unternehmen (noir)

1.1 dirigeants (Vorstand, Aufsichtsrat), 1.2 Eigentümerstruktur (actionnaires, free float,
Aktienrückkaufprogramme), 1.3 appréciation par les marchés de capitaux (cours, capitalisation),
1.4 analystes couvrant Aumann, 1.5 messages des analystes. Chaque affirmation porte une note de
bas de page avec source et page.

### 5.2 Partie 2 — Jahresabschluss 2024 (noir)

Pour 2.1 → 2.6, **la formule est posée puis le calcul est déroulé**, pas seulement le résultat :

- 2.1 Umsatz 2024 et Δ vs 2023
- 2.2 Auftragseingang et Δ vs 2023 (l'énoncé annonce −41,1 % ; à confirmer sur le GB)
- 2.3 EBITDA-Marge = operatives EBITDA / Umsatz, en %
- 2.4 operatives EBITDA
- 2.5 Nettogewinn et Δ
- 2.6 Dividende et Dividendenrendite = Dividende je Aktie / Aktienkurs 31.12.2024
- 2.7 évolution du cours 2022 → fin 2025 : graphique `pgfplots` depuis `data/aktienkurs.csv`,
  annotations d'événements sourcées, points de contrôle (clôtures annuelles) recoupés sur les GB.

### 5.3 Partie 3 — Strategische Maßnahmen (noir, factuel, reformulé, sourcé, sans jugement)

3.1 E-Mobility · 3.2 Next Automation · 3.3 Batterie-/Brennstoffzellen-Produktionstechnik et
dernière acquisition · 3.4 Internationalisierung (DE/CN/US) et Kapitalallokation (Nettoliquidität,
rachats d'actions, dividende) · 3.5 effet constaté sur operatives EBITDA / EBIT ·
3.6 Ausblick 2025 **et** confrontation aux Quartalsmitteilungen 2025 + GB 2025.

### 5.4 Partie 4 — Beurteilung Aktienkursverlauf

Entièrement en `\bk{}` (max ½ A4), adossée aux résultats des parties 2 et 3.

### 5.5 Partie 5 — Handlungsoptionen

Structure reproduite à l'identique de l'énoncé (5.1 / 5.2 / 5.3, chacune .1 .2 .3, puis 5.4).
Dans les tableaux *Faktor / Einschätzung* : les ancres factuelles et chiffrées sont pré-remplies
en noir ; la colonne *Einschätzung* qualitative est en `\bk{}`.

### 5.6 Partie 6 — Grobe Flow-Analyse

Hypothèses de l'énoncé :

| Option | Investition (3 ans) | Kapitalbindung | EBITDA additionnel |
|---|---|---|---|
| A — Next Automation | 35 Mio € | 15 Mio € | +16 Mio € à partir de 2028 |
| B — Batterie/Brennstoffzellen | 30 Mio € | 15 Mio € | +14 Mio € à partir de 2027 |
| C — Akquisition | 60 Mio € | 20 Mio € | +10 Mio € à partir de 2027 |

Tableaux 6.1 : colonnes `Jahr / Investition / Operativer CF / Nettoliquidität nach Investition /
Bemerkung`, années 2026–2028, une table par option.

#### Sous-section « Annahmen der Flow-Analyse » (obligatoire)

L'énoncé donne les montants mais pas la mécanique. Les cinq points suivants sont posés
explicitement dans le document :

| # | Point non spécifié | Hypothèse retenue | Justification |
|---|---|---|---|
| 1 | Solde d'ouverture 2026 | **138,2 Mio €** (Nettoliquidität 31.12.2024), comme imposé — note de bas de page donnant la Nettoliquidität réelle au 31.12.2025 (GB 2025) | Respecte la consigne tout en montrant la lecture du GB 2025 |
| 2 | Profil d'investissement | A : 11,7 / 11,7 / 11,7 · B : 10 / 10 / 10 · **C : 45 / 7,5 / 7,5** (Kaufpreis au closing 2026, intégration ensuite) | Un prix d'acquisition n'est pas versé en trois tranches égales ; le creux de liquidité 2026 est l'enseignement de l'option C |
| 3 | Calendrier de la Kapitalbindung | Sortie de trésorerie **l'année précédant** le démarrage de l'EBITDA · A : 2027–2028 · B : 2026–2027 · **C : 2026** (le BFR arrive avec la cible) | Working Capital = préfinancement du chiffre d'affaires (cadre du cours) |
| 4 | Définition d'« Operativer CF » | **Vue incrémentale** : CF de l'option seule (EBITDA additionnel − Δ Kapitalbindung). Note de bas de page chiffrant l'effet de l'ajout du CF opérationnel du cœur de métier | « Grobe Flow-Analyse » ⇒ effet marginal de la décision ; sinon l'effet de la décision est noyé dans le CF du groupe |
| 5 | IRR et EBITDA-Marge 2028 | IRR sur horizon explicite **2026–2035**, EBITDA additionnel constant à partir de son année de démarrage, Kapitalbindung libérée en fin d'horizon, **+ sensibilité 5 / 10 / 15 ans**. Marge 2028 = (operatives EBITDA 2025 + Zuwachs) / Umsatz 2025 maintenu constant, **+ sensibilité +3 %/an** | Un IRR sans horizon déclaré n'a pas de sens ; sur 3 ans les trois options sont négatives |

Tableau 6.3 (Vergleich Ende 2028) : kumulierte Investition, EBITDA-Zuwachs ab Jahr,
EBITDA-Marge Ende 2028, IRR (~), Liquiditätsreserve Ende 2028, Net Debt/EBITDA
(« nicht relevant » pour A et B, calculé pour C).

6.2 (Ergebnis par option) et le Fazit de 6.3 sont en `\bk{}`.

### 5.7 Quellen, KI-Nutzung, Erklärung

- **Quellen** : (a) rapports Aumann — titre, année, page ; (b) sources externes / analystes —
  référence exacte, URL, **Abrufdatum**.
- **KI-Nutzung** : note courte listant les logiciels IA employés (Claude Code + modèles), en
  précisant l'usage — analyse des documents, recherche de sources, et rédaction assistée des
  passages signalés.
- **Eidesstattliche Erklärung**.

## 6. Inventaire des zones `\bk{}` (17, dans l'ordre du document)

| # | Section | Type | Limite de l'énoncé |
|---|---|---|---|
| 1 | §4 | Beurteilung Aktienkursverlauf | max ½ A4 |
| 2 | §5.1.1 | Management-Summary A | max ½ A4 |
| 3 | §5.1.2 | 6 cellules *Einschätzung* A | tableau |
| 4 | §5.1.3 | Statement A | max 1 A4 |
| 5 | §5.2.1 | Management-Summary B | max ½ A4 |
| 6 | §5.2.2 | 6 cellules *Einschätzung* B | tableau |
| 7 | §5.2.3 | Statement B | max 1 A4 |
| 8 | §5.3.1 | Management-Summary C | max ½ A4 |
| 9 | §5.3.2 | 6 cellules *Einschätzung* C | tableau |
| 10 | §5.3.3 | Statement C | max 1 A4 |
| 11 | §5.3.3 | Wachstum vs. Rückgabe an Aktionäre | max ½ A4 |
| 12 | §5.4.1 | cellules qualitatives de la matrice A/B/C | tableau |
| 13 | §5.4.2 | Zusammenfassende Empfehlung | max ¼ A4 |
| 14 | §6.2 | Ergebnis Option A | max ¼ page |
| 15 | §6.2 | Ergebnis Option B | max ¼ page |
| 16 | §6.2 | Ergebnis Option C | max ¼ page |
| 17 | §6.3 | Fazit du comparatif 2028 | max 1 A4 |

## 7. Séquencement (échéance 04.08.2026)

1. **Passe 1** — téléchargement des rapports dans `refs/`, extraction, remplissage de
   `data/kennzahlen.tex`, squelette LaTeX, parties 1 à 3.
2. **Passe 2** — `scripts/cashflow.py`, partie 6 complète, ancres factuelles des tableaux 5.x,
   `scripts/fetch_kurs.py` + graphique de cours.
3. **Passe 3** — rédaction des 17 zones bleues, Quellen, KI-Nutzung, Erklärung, compilation
   finale sans erreur, relecture de cohérence.

## 8. Vérification

- `build.sh` compile sans erreur ; le PDF est produit dans `out/`.
- `scripts/cashflow.py` est réexécutable et déterministe ; les tableaux du PDF proviennent de sa
  sortie, jamais d'une saisie manuelle.
- Contrôle de cohérence croisé : chaque chiffre du texte provient d'une macro de
  `data/kennzahlen.tex` — aucun littéral numérique dans `sections/*.tex`.
- Contrôle de traçabilité : chaque affirmation factuelle porte une note de bas de page avec
  source et page ; chaque source web porte un Abrufdatum.
- Aucune référence, aucun chiffre inventé : toute valeur non trouvée dans une source est signalée
  comme telle plutôt que comblée.

## 9. Éléments à fournir par l'utilisateur

- Deckblatt : nom, Matrikelnummer, Studiengang, semestre. Des placeholders visibles sont posés
  en attendant.
- Relecture et réécriture des 16 zones bleues, puis neutralisation du bleu avant rendu.

## 10. Hors périmètre

- Toute valorisation d'entreprise détaillée (DCF complet, multiples de comparables) : l'énoncé
  demande une analyse « grobe ».
- Toute source non citable exactement.

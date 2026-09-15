# Optimisation de la Zitierweise — regroupement des notes de bas de page

**Date :** 29.07.2026 · **Portée :** `sections/*.tex` (texte noir), `preamble.tex`,
`scripts/check_literals.py`

## Problème

Le document porte **67 notes de bas de page pour 23 emplacements de source distincts**.
La même source réapparaît donc plusieurs fois sur une même page sous des numéros
différents. Page 12 (§ 3.5 et 3.6), le lecteur voit onze notes dont six renvoient au
Geschäftsbericht 2024 :

```
53 Aumann AG, Geschäftsbericht 2024, S. 5.
54 Aumann AG, Geschäftsbericht 2024, S. 2.
55 Aumann AG, Geschäftsbericht 2024, S. 2.
56 Aumann AG, Geschäftsbericht 2024, S. 5.
57 Aumann AG, Geschäftsbericht 2024, S. 13.
58 Aumann AG, Geschäftsbericht 2024, S. 5.
…
```

La cause n'est pas la forme du libellé mais la **densité des appels** : une note après
presque chaque phrase. § 3 concentre 34 des 67 notes.

## Décisions

### Ce qui est corrigé, et ce qui ne l'est pas

Trois remèdes étaient possibles ; le choix est **réduire le nombre d'appels**, pas
raccourcir le libellé ni recycler les numéros.

| Écarté | Raison |
|---|---|
| `Ebd.` / `a. a. O.` | Ne traite que les répétitions *immédiates* : 15 notes sur 67, et seulement 2 des 11 de la page 12. |
| Kurzbeleg après le premier appel | Raccourcit les lignes sans en réduire le nombre ; impose au lecteur d'apprendre des abréviations pour 8 sources seulement. |
| Un numéro unique par emplacement | Supprime le symptôme mais oblige le lecteur à remonter plusieurs pages pour lire la note. |
| `biblatex` | Réoutillage complet pour 8 sources, toutes du même émetteur. Aucun gain. |

Le **Vollbeleg** est conservé partout : `Aumann AG, Geschäftsbericht 2024, S. 2.` Avec
huit sources et un Quellenverzeichnis complet, il n'y a pas d'ambiguïté à lever.

### La traçabilité ne passe pas par la note

Point décisif, propre à ce projet. Une note qui regroupe plusieurs pages
(`S. 5 und 13`) ne dit plus quel chiffre vient de quelle page. Ce n'est **pas** une perte :
`data/kennzahlen.tex` porte, pour chacune de ses 64 macros sourcées, le document et la
page imprimée en commentaire de fin de ligne, et ces attributions ont été vérifiées page
par page le 29.07.2026 (voir `docs/befunde-und-entscheidungen.md`).

La note de bas de page sert le lecteur ; la piste d'audit vit dans la couche de données.
C'est ce qui autorise le regroupement au-delà de la page.

## La règle

**Clause 1 — même document, même paragraphe.**
Des appels de note consécutifs renvoyant au même document fusionnent en une note unique.
Un changement de document rompt le regroupement. Jamais de fusion au-delà d'une frontière
de paragraphe.

**Clause 2 — même phrase, documents différents.**
Quand une seule phrase énumère plusieurs documents, leurs appels fusionnent en une note
unique placée à la fin de la phrase, les sources séparées par des points-virgules. Le
texte de la phrase nomme déjà chaque document, la répartition reste donc lisible.

**Traitement des pages.** Les pages d'une note fusionnée sont **dédoublonnées et triées
par ordre croissant**. Deux sites de l'inventaire l'exigent : `S. 2, 5, 2` devient
`S. 2 und 5`, et `S. 9, 52, 7` devient `S. 7, 9 und 52`. Forme allemande : `S. 2 und 5`
pour deux pages, `S. 2, 5 und 9` à partir de trois. Une page unique reste `S. 2`.

### Exemple — § 3.5

Avant, quatre notes :

```latex
… stieg das operative EBITDA um \Pct{…} auf \MioEUR{…}; das EBIT erreichte
\MioEUR{\EbitZFIV}.\quelleGB{2024}{2} Die EBITDA-Marge verbesserte sich von
\Pct{…} auf \Pct{…}.\quelleGB{2024}{2}

Der Bericht führt die Ergebnisverbesserung … zurück …\quelleGB{2024}{5} Im
Segment Next Automation blieb das EBITDA … unverändert.\quelleGB{2024}{13}
```

Après, deux :

```latex
… stieg das operative EBITDA um \Pct{…} auf \MioEUR{…}; das EBIT erreichte
\MioEUR{\EbitZFIV}. Die EBITDA-Marge verbesserte sich von \Pct{…} auf
\Pct{…}.\quelleGB{2024}{2}

Der Bericht führt die Ergebnisverbesserung … zurück … Im Segment Next
Automation blieb das EBITDA … unverändert.\quellen{\bGB{2024}{5 und 13}}
```

## Les macros

Les trois macros actuelles ne portent qu'une source. Elles sont décomposées en briques
combinables, sans dupliquer le libellé :

```latex
% Bausteine ohne eigenen Fussnotenapparat, damit sie sich kombinieren lassen.
\newcommand{\bGB}[2]{Aumann AG, Gesch\"aftsbericht #1, S.~#2}
\newcommand{\bQM}[2]{Aumann AG, #1, S.~#2}
\newcommand{\bWeb}[3]{#1, \url{#2} (abgerufen am #3)}

% Eine Fussnote aus einem oder mehreren Bausteinen; setzt den Schlusspunkt.
\newcommand{\quellen}[1]{\footnote{#1.}}

% Die bisherigen Makros bleiben im Aufruf unveraendert, werden aber aus
% denselben Bausteinen aufgebaut - der Wortlaut existiert nur noch einmal.
% Die alten \newcommand-Definitionen werden dabei ersetzt, nicht ergaenzt.
\newcommand{\quelleGB}[2]{\quellen{\bGB{#1}{#2}}}
\newcommand{\quelleQM}[2]{\quellen{\bQM{#1}{#2}}}
\newcommand{\quelleWeb}[3]{\quellen{\bWeb{#1}{#2}{#3}}}
```

Trois usages :

| Cas | Écriture |
|---|---|
| Source unique | `\quelleGB{2024}{2}` — inchangé |
| Plusieurs pages | `\quellen{\bGB{2024}{5 und 13}}` |
| Plusieurs documents | `\quellen{\bQM{Quartalsmitteilung Q1 2025}{2}; \bQM{Halbjahresfinanzbericht H1 2025}{3}}` |

Les macros `\quelle*` et `\b*` sont ajoutées à la liste blanche de
`scripts/check_literals.py`. Sans cela, une page à trois chiffres (les rapports vont
jusqu'à 105) serait signalée comme littéral numérique. Un test couvre ce point.

## Inventaire des sites

**Clause 1 — 18 sites, 20 notes supprimées** (seize fusions de deux appels, deux de trois appels : « Der Auftragseingang… » et « *Internationalisierung.* »).

| Fichier | Paragraphe (incipit) | Fusion |
|---|---|---|
| `02` | Der Konzernumsatz belief sich… | GB 2024, S. 2 und 13 |
| `02` | Der Auftragseingang lag 2024 bei… | GB 2024, S. 2 und 5 |
| `02` | Die Aumann AG weist zwei Größen… | GB 2024, S. 2 |
| `02` | Vorstand und Aufsichtsrat schlugen… | GB 2024, S. 8 und 13 |
| `03` | Aumann gliedert das Geschäft… | GB 2024, S. 9 und 10 |
| `03` | Aumann bezeichnet sich als global… | GB 2024, S. 9 |
| `03` | Die Fertigungslösungen adressieren… | GB 2024, S. 10 |
| `03` | Mit der Veröffentlichung der Quartalsmitteilung… | GB 2024, S. 10 |
| `03` | Das Segment wurde in „Next Automation" umbenannt… | GB 2024, S. 10 |
| `03` | Der Bericht ordnet dem Feld der Elektromobilität… | GB 2024, S. 9 und 10 |
| `03` | Die Produktionstechnik für die Brennstoffzellen-… | GB 2024, S. 13 und 14 |
| `03` | Als jüngste Akquisition nennt der Bericht… | GB 2024, S. 6 und 9 |
| `03` | *Internationalisierung.* | GB 2024, S. 7, 9 und 52 |
| `03` | *Kapitalallokation.* | GB 2024, S. 5 |
| `03` | Auf Konzernebene stieg das operative EBITDA… | GB 2024, S. 2 |
| `03` | Der Bericht führt die Ergebnisverbesserung… | GB 2024, S. 5 und 13 |
| `03` | Der Umsatz 2025 lag mit… | GB 2025, S. 2 |
| `05` | Ausgangslage laut Aufgabenstellung… | GB 2024, S. 2 und 5 |

**Clause 2 — 1 site identifié, 2 notes supprimées.** § 3.6, phrase
« Diese Prognose wurde im Jahresverlauf unverändert bestätigt: in der Quartalsmitteilung
zum ersten Quartal 2025, im Halbjahresfinanzbericht und in der Quartalsmitteilung zum
dritten Quartal 2025. » — trois appels deviennent une note listant les trois rapports.

L'implémentation relit les fichiers pour confirmer qu'aucun autre site de clause 2
n'existe ; l'inventaire ci-dessus est le résultat d'une analyse automatique et ne
remplace pas cette relecture.

**Résultat attendu : 67 → 45 notes (−33 %).** Page 12 passe de onze à environ cinq notes.

## Ce qui n'est pas touché

- Aucune zone bleue, aucun chiffre, aucune formulation de fond.
- `data/`, `scripts/cashflow.py`, le Quellenverzeichnis : inchangés.
- `sections/04-aktienkurs.tex` : ne porte aucune note.
- Aucun script de surveillance dédié n'est créé. Le texte noir est terminé ; il n'y a pas
  de régression à surveiller.

## Vérification

1. `python3 scripts/check_literals.py sections/*.tex` → 0.
2. `cd scripts && python3 -m unittest discover -p 'test_*.py'` → tous verts, y compris le
   nouveau test de liste blanche.
3. `./build.sh` → aucune erreur, aucun avertissement.
4. Recompte des **occurrences** (et non des lignes) :
   `grep -oh '\\quelle\(GB\|QM\|Web\)\|\\quellen' sections/*.tex | wc -l` = 45.
5. Aucune source ne doit plus apparaître deux fois sur une même page : contrôle
   automatique page par page sur le PDF produit.
6. Le Quellenverzeichnis doit continuer à couvrir toutes les sources citées.

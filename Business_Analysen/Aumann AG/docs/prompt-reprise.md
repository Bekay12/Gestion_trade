# Prompt de reprise — nouvelle session

À coller tel quel au démarrage d'une nouvelle session dans
`/home/berkam/Projets/Latex/Finanzwirtschaft_Für_Ingenieur`.

---

Reprends une Hausarbeit de Master en LaTeX (allemand) sur l'analyse financière d'Aumann AG,
HS Kaiserslautern. **Échéance : mardi 04.08.2026.** Le travail est déjà avancé et documenté.

**Commence par lire, dans cet ordre, et ne redérive rien qui s'y trouve déjà :**
1. `CLAUDE.md` (racine) — règles, commandes, état d'avancement
2. `docs/befunde-und-entscheidungen.md` — résultats chiffrés, décisions motivées, points ouverts
3. `sections/CLAUDE.md`, `data/CLAUDE.md`, `refs/CLAUDE.md` — conventions locales
4. `docs/superpowers/plans/2026-07-29-hausarbeit-aumann.md` — tâches 9 à 12 (les seules restantes)

Les PDF sources sont gitignorés : lance `./scripts/fetch_reports.sh` avant toute extraction.

**Fait (noir, sourcé, compilé — 13 pages) :** parties 1, 2, 3 ; les chiffres de la partie 6
sont calculés et testés dans `data/cf_*.tex`.

**Reste à faire, dans l'ordre :**
- **Task 9** — `sections/05-handlungsoptionen.tex` : structure des options A/B/C selon
  l'énoncé, tableaux *Faktor / Einschätzung*, 44 emplacements `\bk{}` vides précédés de
  leur bloc `% FAKTENBASIS`
- **Task 10** — `sections/06-cashflow.tex` : sous-section *Annahmen* (les 5 hypothèses),
  `\input` des tableaux générés, labels `tab:cf-a/b/c`, `tab:vergleich2028`, `tab:irr-sens`
- **Task 11** — rédaction des 17 zones bleues en allemand, niveau Master
- **Task 12** — Quellen (repris **verbatim** de `refs/MANIFEST.md`), note KI-Nutzung,
  Eidesstattliche Erklärung, `README.md`, vérification finale

**Règles non négociables :**
1. Noir = factuel et sourcé. Bleu (`\bk{}`) = toute prose évaluative, avec la base factuelle
   en commentaires `%` juste au-dessus, jamais visible dans le PDF.
2. Aucun littéral numérique dans `sections/*.tex` — tout passe par une macro de
   `data/kennzahlen.tex`. `python3 scripts/check_literals.py sections/*.tex` doit sortir 0.
3. Ne rien inventer. Une valeur introuvable est signalée dans le document, pas comblée.
4. Les numéros de page cités sont les pages **imprimées**, vérifiées avec le script de
   `refs/CLAUDE.md` — jamais déduites des numéros de ligne de `pdftotext`.
5. `./build.sh` doit passer sans erreur après chaque tâche.

**Deux points ouverts à traiter explicitement, pas à enterrer :**
- La marge EBITDA fin 2028 (A 21,2 % / B 20,2 % / C 18,3 %) repose sur le CA 2025 maintenu
  constant. Or ce CA s'était effondré de 34,7 % : le petit dénominateur flatte la marge.
  La sous-section *Annahmen* doit le dire et montrer la sensibilité à +3 %/an.
- Les tâches 4 à 8 n'ont pas eu de revue indépendante (limite de dépense du compte). Si tu
  fais une revue finale, couvre-les en priorité : chiffres de `data/kennzahlen.tex` contre
  les rapports, et exactitude des pages citées.

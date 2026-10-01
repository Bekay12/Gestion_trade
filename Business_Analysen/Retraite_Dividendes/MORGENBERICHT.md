# Rapport du matin — Retraite anticipée par les dividendes

Document prêt : `Business_Analysen/Retraite_Dividendes/out/plan.pdf` (65 pages, 31 figures).
Reconstruit par `./build.sh` depuis les sources ; rien n'est tapé à la main dans le PDF.

## À ta relecture en priorité

- **Écarts assumés au plan initial** (décidés cette nuit, à valider ou révoquer) :
  - Rendement réel de marché = **moyenne géométrique** (TCAC) 1950-2022, **7,3 %/an**, et non
    la médiane annuelle prévue par le plan (12,1 %, exposée pour comparaison sous
    `\RenditeReelleMediane`). La médiane surestime la capitalisation composée sur 45 ans.
  - Retenue américaine calculée avec **formulaire W-8BEN déposé** (15 %, pas 30 %) : hypothèse
    que le courtier européen l'a fait souscrire automatiquement, à vérifier sur ton propre
    compte-titres.
  - Les seuils de l'assurance maladie (KV, plancher et plafond) sont désormais **constants en
    euros réels** (revalorisés chaque année avec les salaires), et non figés en nominal comme la
    première version le faisait par erreur : ce bug a été trouvé et corrigé pendant la nuit
    (commit `be97a8df`) ; il flattait fortement le plan (cotisation maximale simulée
    1 226 → 513 €/mois en euros 2026 en 2071 avant correction).
  - Le **scénario de référence a été déplacé de 20 % à 30 % d'épargne**, parce qu'aucune
    stratégie de dividendes n'atteint jamais l'objectif à 20 % d'épargne dans l'horizon de
    45 ans (décision du contrôleur, commit `a1633bf7`).
  - Un **test historique de la règle des 4 %** (promis au chapitre 4 par la spécification mais
    absent du code) a été ajouté au chapitre 8 : la règle des 4 % tient plus souvent que le
    revenu de dividendes sur les mêmes départs historiques (83,8 % contre 62,2 % sur 40 ans).
  - La **Günstigerprüfung n'est pas modélisée** (économie estimée à ~415 €/mois à la retraite,
    `\GuenstigerEconomieMois`) : c'est le principal levier non pris en compte, et son absence
    rend tous les capitaux requis **prudents** (plutôt surestimés que sous-estimés).
  - **Poche ETF des stratégies à dividendes (dont la référence 70/30) : capitalisante pendant
    l'épargne, traitée comme distribuante dès la retraite**, sans modéliser le coût fiscal du
    changement de catégorie de parts (nouvelle macro `\HypotheseBasculeReference`, citée au
    chapitre 5 ; même angle mort que `\HypotheseUmschichtung`, qui ne couvrait que la variante
    ETF distribuant). Le plafond annuel de la Vorabpauschale (Basisertrag) est calculé en euros
    réels plutôt qu'en euros nominaux comme le prévoit le droit, et la Vorabpauschale n'est plus
    modélisée après le départ en retraite pour la référence en retrait de 4 % : les deux
    favorisent légèrement l'ETF, dans le même sens que la plupart des simplifications déjà
    listées ici.
- **Littérature académique vide** : `data/literatur.bib` ne contient aucune référence. Le skill
  `scopus` a échoué faute de `SCOPUS_API_KEY` / accès campus, et l'API publique Semantic
  Scholar a renvoyé `HTTP 429` sur presque toutes les requêtes tentées (deux essais, à des
  heures différentes). À reprendre le jour avec un accès campus/VPN ou une clé `S2_API_KEY`.
- **Portefeuille-exemple : 23 titres retenus, pas 30** (chapitre 6). Sur 142 tickers examinés
  (3 indices), 44 passaient le filtre rendement/baisse, et seuls 23 survivaient à tous les
  paliers de relâchement essayés. Le chapitre présente les 23 titres réels obtenus ; la méthode
  reste valable même sous la cible de 30.
- **Trois interruptions par limite de session API** pendant la nuit (vers 01h, puis à nouveau
  pendant la tâche 10), sans aucune perte de travail : l'arbre restait propre à chaque reprise
  et le travail a continué où il s'était arrêté.

## Les trois résultats principaux

Lus depuis `data/zusammenfassung.json` et `data/kennzahlen.tex`, jamais recalculés à la main.

1. **Âge de départ, répartition 70/30 (maison/ETF monde) dividendes seuls** :
   - à 20 % d'épargne : **non atteint** dans l'horizon du plan (`\AlterMaisonSiebzigZwanzig`) ;
   - à 30 % d'épargne (scénario de référence) : **67 ans** (`\AlterMaisonSiebzigDreissig`), soit
     l'âge légal de la retraite : aucune retraite anticipée à ce taux.
2. **Écart de capital, dividendes seuls (répartition 70/30) contre retrait de 4 % (référence
   100 % ETF monde)** : 2 974 126 € contre 1 562 746 € (`\KapitalMaisonSiebzig` /
   `\KapitalEtfReferenz`), soit **1 411 381 € (90,3 %) de capital en plus** pour vivre des
   seuls dividendes (`\KapitalDifferenzSiebzigEuro` / `\KapitalDifferenzSiebzigProzent`).
   `\KapitalEtfReferenz` est 100 % ETF monde, pas 70/30 : l'écart compare deux portefeuilles
   différents, pas deux retraits sur le même portefeuille.
3. **Probabilité de réussite Monte Carlo avec 2 ans de réserve de liquidités** : **89,2 %**
   (`\ErfolgsquoteMitPuffer`, `\PufferJahreMit` = 2 ans), contre 63,5 % sans réserve
   (`\ErfolgsquoteBasis`). Modèle sans impôt ni cotisation KV/PV, tiré sur l'historique du
   marché américain (S&P 500, Shiller) ; réussite = objectif de revenu atteint puis revenu en
   dividendes ≥ 80 % de l'objectif chaque année de retraite, la réserve comblant les années en
   dessous de ce seuil.

Pour contraste, la stratégie de référence **ETF monde + retrait de 4 %** : **60 ans** à 20 %
d'épargne, **55 ans** à 30 % (`\AlterEtfReferenzZwanzig` / `\AlterEtfReferenzDreissig`). L'ETF
monde fait mieux sur chaque indicateur du tableau de bord du résumé (moins de capital, départ
plus tôt, meilleure tenue historique).

Ajout de la nuit : variante à **2 500 €/mois** (section 5.4, demande explicite), même capital de
départ et mêmes répartitions. À 30 % d'épargne, répartition 70/30 **62 ans**
(`\AlterMaisonSiebzigDreissigNiedrig`) contre ETF retrait 4 % **51 ans**
(`\AlterEtfReferenzDreissigNiedrig`).

## Ce qui manque (résumé de `LUECKEN.md`)

- Littérature académique (voir ci-dessus).
- `basiszins_2026` (3,20 %) : confirmé par deux comptes-rendus secondaires du même
  BMF-Schreiben, pas par le PDF officiel lui-même (fetch a échoué).
- `einstiegsgehalt_brutto` : pas de grille IG Metall spécifique trouvée pour Jungheinrich
  Hambourg ; repli sur la médiane StepStone 2026 (59 250 €).
- `gehaltssteigerung_real` (~1,8 %/an) : dérivé par approximation CAGR, pas une série Destatis
  directe.
- `quellensteuer` : tableau BZSt au 1ᵉʳ janvier 2025 (édition 2026 pas encore publiée) ; l'écart
  probable est faible mais non nul.
- `etf_welt_rendite_div` (1,8 %) : repli imposé, faute de fiche exploitable pour le plus grand
  ETF monde distribuant ; à revoir si une source fiable apparaît.
- Écart de révision sur l'inflation allemande 2022 (7,9 % provisoire contre 6,9 % rétrospectif,
  changement de base du VPI) : non arbitré, `inflation_2022_de` n'a pas été modifiée.
- Nom tronqué hérité (`"McCormick & Company, Incorporat"` dans `portefeuille.csv`, colonne
  `name` du ticker MKC) : cosmétique, non corrigé car hors périmètre de la tâche qui l'a trouvé.
- Deux sources de presse sur huit prévues n'ont pas pu être exploitées (paywall JS de test.de
  et de Handelsblatt) et un domaine (capital.de) bloqué au crawler ; remplacées par deux autres
  blogs FIRE germanophones pour rester dans la fourchette demandée.

## À vérifier de ton côté

- Hypothèses secondaires listées ci-dessus (salaire d'entrée, taux de retenue, base
  d'inflation 2022).
- Écart du salaire net : le modèle approché (`salaire.py`) est systématiquement
  **0,78 à 0,83 % au-dessus** d'un calculateur public indépendant (Deutschland-Rechner), écart
  documenté et publié tel quel au chapitre 5 (`docs/salaire_controle.md`), jamais corrigé pour
  coller à une cible.
- Les relâchements de règles du portefeuille (paliers essayés au chapitre 6 : strict, croissance
  ≥ 2 %, payout ≤ 90 %) donnent tous 23 titres ; vérifier que cela te convient comme échantillon
  de méthode plutôt que comme recommandation d'achat.
- Toutes les vérifications automatiques ont été rejouées ce matin : suites de tests unitaires
  OK, `./build.sh` complet OK (0 « Overfull » dans `out/analyse.log`, vérifié après une
  recompilation réelle de latexmk et pas seulement après un « Nothing to do » lu dans
  `out/build.log` ; 0 référence non définie ; gardes de notes de bas de page désormais
  bloquantes, pas seulement silencieuses : `check_footnote_pages.py` et
  `check_footnote_groups.py` arrêtent le build en cas de défaut, comme `README.md`
  l'annonce), 65 pages, 31 figures confirmées par `pdftotext`. Relecture
  visuelle de 12 pages représentatives (titre, résumé, un chapitre sur deux avec figure, le
  tableau des 23 titres, les annexes) : aucun défaut trouvé (légendes, axes, tableaux tous
  lisibles et bien cadrés).

## Commits de la nuit

```
git log --oneline master..retraite-dividendes
```

Liste complète de `7aab8556` (squelette du projet) à ce rapport ; le compte exact dépend des
corrections apportées après cette phrase et n'est donc pas figé ici. Détail tâche par tâche
dans `PROGRESS.md` et le journal complet du contrôleur
(`.superpowers/sdd/2026-09-29-retraite-dividendes/progress.md`, hors dépôt).

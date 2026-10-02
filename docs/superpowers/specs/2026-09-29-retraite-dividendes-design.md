# Retraite anticipée par les dividendes : spécification du document

Date : 29.09.2026. Livrable : un PDF LaTeX en français, `Business_Analysen/Retraite_Dividendes/out/plan.pdf`.

## 1. But

Répondre, pour un profil précis, à la question : **à quel âge et avec quelle épargne peut-on
vivre de 3 500 € par mois en pouvoir d'achat de 2026, nets d'impôts et d'assurance maladie,
grâce aux dividendes ?** Le document doit être illustré (une trentaine de figures), honnête
(il montre aussi ce qui joue contre la stratégie dividendes) et sourcé (chaque chiffre renvoie
à une source datée).

Ce n'est pas un conseil en investissement réglementé ; le document le dit en première page,
et signale chaque point fiscal à faire valider par un conseiller.

## 2. Profil et paramètres fixés avec l'utilisateur

| Paramètre | Valeur retenue |
|---|---|
| Situation | Stagiaire chez Jungheinrich (Allemagne), fin du master prévue en août 2027, ~26 ans |
| Résidence fiscale, aujourd'hui et à la retraite | Allemagne |
| Capital de départ | 2 000 à 10 000 € (calcul sur une fourchette, bornes et milieu) |
| Épargne | À partir de septembre 2027, au moins 200 €/mois, puis taux d'épargne sur un salaire d'ingénieur sourcé (10, 20, 30, 50 %) |
| Objectif | 3 500 € par mois en euros de 2026, **nets d'impôts ET de l'assurance maladie et dépendance** (~5 000 € nominaux vers 2044 à 2 % d'inflation) |
| Départ | Le plus tôt possible ; l'âge est un résultat, pas une donnée |
| Impôt d'Église (Kirchensteuer) | Non retenu par défaut ; variante chiffrée à 8 et 9 % de l'impôt dans le chapitre 3 (à confirmer par l'utilisateur) |
| Tolérance au risque | Forte |
| Situation personnelle | Célibataire, sans immobilier |
| Construction du portefeuille | « ETF maison » d'environ 30 actions, trois répartitions comparées : **50/50, 70/30, 100 %** maison contre ETF |
| Hors périmètre | Startups (Companisto), trading intraday, paris sur titres isolés : exclus du calcul et du document (une ligne en annexe le signale) |

## 3. Stratégies comparées

- **Répartitions** 50/50, 70/30 et 100 % maison, avec dividendes réinvestis pendant l'accumulation.
- **Référence** : 100 % ETF monde capitalisant, puis retrait total (« règle des 4 % »), et une
  variante qui bascule vers des ETF distribuants trois à cinq ans avant le départ.
- Chaque stratégie est calculée avec les mêmes hypothèses de marché ; seules la fiscalité, la
  composition et la règle de revenu changent.

## 4. Plan du document

1. Résumé en une page
2. Point de départ et cible (euros constants et nominaux, brut → disponible)
3. Du dividende brut à ton compte en Allemagne (Abgeltungsteuer, Soli, Sparerpauschbetrag,
   Teilfreistellung des ETF, retenues à la source par pays, Günstigerprüfung, assurance maladie
   volontaire)
4. Combien de capital faut-il ?
5. La phase d'accumulation (salaire, taux d'épargne, âge de départ)
6. Ton ETF maison : répartitions comparées, diversification, méthode de sélection,
   **portefeuille-exemple de 30 titres** construit par règles (illustration, pas recommandation)
7. Ce qui peut mal tourner (baisses de dividendes historiques, inflation, séquence des
   rendements, Monte Carlo)
8. La retraite elle-même (phase pont jusqu'à 67 ans, rente légale réduite, réserve de liquidités,
   règles de retrait)
9. Ce que disent les magazines et la recherche
10. Plan d'action daté

Annexes : hypothèses déclarées, données non disponibles, sources.

## 5. Architecture

Projet dans `Business_Analysen/Retraite_Dividendes/`, sur le gabarit LaTeX des analyses
précédentes (`~/.claude/skills/startup-investment-analyzer/assets/scaffold/`).

| Élément | Rôle | Dépend de |
|---|---|---|
| `scripts/hypotheses.py` | Toutes les hypothèses, chacune avec source et date | – |
| `scripts/fiscalite.py` | Fonction pure : dividende brut → disponible selon pays et enveloppe ; cotisation maladie | hypotheses |
| `scripts/salaire.py` | Trajectoire brute et nette du salaire | hypotheses |
| `scripts/projection.py` | Accumulation mensuelle par stratégie, impôt annuel et Vorabpauschale compris | fiscalite, salaire |
| `scripts/histoire.py` | Rejeu sur les données Shiller (S&P 500 depuis 1871 : cours, dividendes, IPC) | données Shiller |
| `scripts/montecarlo.py` | Tirage par blocs dans l'historique, graine fixe, percentiles et probabilité de réussite | histoire |
| `scripts/portefeuille_exemple.py` | Filtre de l'univers et sélection des 30 titres par règles chiffrées | données de marché |
| `scripts/graphiques.py` | Écrit les CSV des figures ; rendu en pgfplots dans `preamble.tex` | tous |
| `scripts/rechnung_retraite.py` | Macros LaTeX (`data/kennzahlen.tex`) et corps de tableaux | tous |

**Règles héritées des rapports précédents :** aucun chiffre tapé dans `sections/*.tex`
(vérifié par `check_literals.py`) ; les sources officielles sont téléchargées dans `refs/` avec
page imprimée vérifiée ; les sources web sont dans un bloc séparé avec URL et date de
consultation ; le calcul est déterministe (même graine, mêmes CSV).

**Données de marché :** univers d'environ 150 à 200 valeurs de dividende (aristocrates
américains et européens, grandes valeurs allemandes, sorties des screeners du dépôt). Cours et
dividendes arrivent en **un seul appel groupé** (`yf.download(..., actions=True)`), jamais en
boucle par titre (contrainte de quota du dépôt).

## 6. Sources

1. **Officielles** : BMF (Abgeltungsteuer, Teilfreistellung, Vorabpauschale, Basiszins),
   Deutsche Rentenversicherung (rente selon la durée de cotisation), GKV-Spitzenverband
   (cotisations des assurés volontaires, plancher de revenu), Destatis (inflation), grille
   IG Metall ou rapport de salaires pour le salaire d'entrée d'ingénieur.
2. **Presse et sites spécialisés**, marqués comme secondaires : Finanztip, Stiftung
   Warentest/Finanztest, Börse Online, Handelsblatt, justETF, Extra-Magazin, communauté FIRE
   germanophone.
3. **Recherche** : 6 à 10 études, chaque référence vérifiée via le skill `scopus` (taux de
   retrait soutenables, diversification selon le nombre de titres, performance des stratégies à
   haut dividende, préférence des investisseurs pour les dividendes). Aucune référence ni DOI
   inventés.

## 7. Figures

Une trentaine, par chapitre (liste validée le 29.09.2026) :

- **Ch. 1–2 :** tableau de bord, 3 500 € d'aujourd'hui en euros nominaux jusqu'en 2060,
  cascade du brut au disponible.
- **Ch. 3 :** « 100 € de dividende, combien pour toi ? » par pays et pour un ETF ; coût fiscal
  cumulé sur 18 ans.
- **Ch. 4 :** capital requis selon le rendement ; dividendes seuls contre règle des 4 %.
- **Ch. 5 :** salaire ; années jusqu'à la retraite selon le taux d'épargne ; capital dans le
  temps ; part des intérêts composés ; épargne nécessaire pour 40, 45, 50 ans ; effet d'un an
  d'avance ou de retard.
- **Ch. 6 :** trois répartitions comparées ; risque selon le nombre de titres ; secteurs et
  pays du portefeuille-exemple ; tableau des 30 titres ; rendement contre croissance du
  dividende ; historique réel de deux dividendes (l'un coupé, l'autre en hausse continue).
- **Ch. 7 :** dividendes réels du S&P 500 depuis 1871 ; baisses pendant les crises ; rejeu par
  année de départ ; éventail Monte Carlo ; probabilité de réussite ; inflation 2022.
- **Ch. 8 :** revenu pendant la phase pont ; rente selon l'âge d'arrêt ; effet de la réserve
  de liquidités.
- **Ch. 9–10 :** tableau d'accord et de désaccord ; frise du plan d'action.

Chaque chapitre contient un exemple chiffré « Yann en 2044 », calculé et non rédigé à la main.

## 8. Contrôles

- Tests unitaires hors ligne : `fiscalite` (cas écrits à la main pour chaque pays et pour l'ETF),
  `projection` (retrouver la valeur finale d'un taux fixe calculée à la main), `salaire`,
  `montecarlo` (graine fixe → résultat stable).
- Contrôle croisé : le capital requis du chapitre 4 et l'âge de départ du chapitre 5 doivent
  concorder dans le résumé (même macro, jamais deux calculs).
- Vérifications de la compilation : chiffres hors couche de données, notes de bas de page
  dupliquées, références non définies.

## 9. Honnêteté : ce que le document doit dire même si cela dessert la thèse

- En Allemagne, une action en direct est imposée plus lourdement qu'un ETF actions
  (Teilfreistellung) ; le coût est chiffré sur 18 ans.
- Viser les seuls dividendes demande en général plus de capital qu'un retrait total ; l'écart
  est chiffré.
- Les dividendes ont baissé fortement dans plusieurs crises ; les baisses sont montrées avec
  leurs dates et leurs sources.
- Le portefeuille-exemple est une illustration de méthode, avec ses biais déclarés (biais du
  survivant, données courantes appliquées au passé).

## 10. Données non disponibles connues d'avance

- Capital de départ exact (fourchette 2 000–10 000 €).
- Salaire réel après le master (remplacé par une référence sourcée et des variantes).
- Évolution future de la fiscalité allemande (règles 2026 figées, sensibilité si le taux change).

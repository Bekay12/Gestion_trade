# Lacunes (cherché, non trouvé, repli utilisé)

## Tâche 1 (hypothèses sourcées, 2026-09-29)

- `basiszins_2026` : le BMF-Schreiben du 13.01.2026 (IV C 1 - S 1980/00230/012/001) a été localisé
  (bundesfinanzministerium.de), mais son contenu texte n'a pas pu être extrait par l'outil de
  fetch (page de renvoi vers un PDF sans texte lisible retourné). Valeur 3,20 % confirmée par
  deux comptes-rendus indépendants du même BMF-Schreiben (stollfuss.de, ecovis-kso.com), marquée
  `primaer: true` car elle cite explicitement la même source primaire datée ; à revérifier
  directement sur le PDF du BMF si l'occasion se présente.
- `einstiegsgehalt_brutto` : aucune grille salariale IG Metall spécifique à Jungheinrich /
  région côtière (Hamburg) trouvée pour un poste d'ingénieur diplômé master en 2026 ; seule une
  fourchette générale d'entrée en gruppe E9-E11 a été trouvée (igmetall.de), sans montant net.
  Repli sur le StepStone Gehaltsreport 2026 (médiane ingénieurs < 1 an d'expérience, 59 250 €),
  marqué `primaer: false`.
- `gehaltssteigerung_real` : aucune série Destatis Verdiensterhebung donnant directement une
  progression réelle annuelle moyenne sur une carrière d'ingénieur n'a été trouvée. Valeur
  dérivée par approximation d'un taux de croissance annuel composé (CAGR) à partir de la coupe
  transversale StepStone par expérience (59 250 € à l'entrée contre 93 250 € à 25 ans
  d'expérience, soit environ 1,8 %/an), marquée `primaer: false`.
- `quellensteuer` : le tableau officiel BZSt « Anrechenbarkeit der Quellensteuer » disponible au
  moment de la recherche est celui au 1er janvier 2025 (Redaktionsschluss Juin 2025) ; l'édition
  2026 n'était pas encore publiée sur bzst.de. Les taux DBA ne changent qu'en cas de
  renégociation de convention, donc le risque d'écart est faible, mais à revérifier quand
  l'édition 2026 paraît. La ligne DE n'est pas une ligne du tableau BZSt (elle ne concerne que
  les dividendes étrangers) ; elle a été construite à partir de `abgeltungsteuer_satz` +
  `soli_satz` pour satisfaire le test qui exige la clé DE.

## Tâche 8 (presse et littérature, 2026-09-30)

- Littérature académique non vérifiée cette nuit. Le skill `scopus` a été essayé en premier
  (`scopus_api.py search 'TITLE-ABS-KEY("safe withdrawal rate" AND retirement)'`) : échec
  immédiat, `SCOPUS_API_KEY` n'est pas défini dans l'environnement et
  `.claude/skills/scopus/.scopus_key` n'existe pas (vérifié par existence de fichier
  seulement, sans lire son contenu) ; attendu, le réseau campus est de toute façon absent
  la nuit. Repli sur `semantic_scholar_api.py` (API publique Semantic Scholar) : l'endpoint
  `/paper/search` a renvoyé `HTTP 429 Too Many Requests` sur les 6 requêtes tentées
  (`safe withdrawal rate retirement`), espacées de 10 s à 90 s sur environ 12 minutes ;
  aucune clé `S2_API_KEY`/`SEMANTIC_SCHOLAR_API_KEY` n'est configurée, donc le pool public
  partagé (fortement throttlé) a été utilisé et n'a jamais répondu. Conformément à la
  tâche 8, `data/literatur.bib` est laissé vide (commentaire d'en-tête expliquant l'état)
  et le chapitre 9 n'aura pas de référence académique cette nuit. Aucune référence n'a été
  écrite de mémoire. À reprendre le jour avec un accès campus/VPN pour Scopus, ou une clé
  Semantic Scholar, sur les 6 requêtes prévues dans le brief de la tâche 8.
- Presse : 2 des 8 sources nommées dans le brief n'ont pas pu fournir de texte exploitable.
  `test.de` (Stiftung Warentest / Finanztest, page
  `Dividendenrendite-von-ETF-...-5963530-0/`) ne renvoie que la navigation du site sans
  aucun corps d'article (paywall/JS, contenu réel non servi côté serveur) ; abandonné, pas
  d'entrée dans `presse_roh.json`. `handelsblatt.com` (commentaire
  `.../100179817.html`) ne renvoie qu'un teaser de deux phrases sans citation
  spécifique aux dividendes (paywall) ; abandonné pour la même raison. `capital.de` n'a
  pas pu être recherché : le crawler WebSearch d'Anthropic est explicitement bloqué sur ce
  domaine (« not accessible to our user agent »); substitué par un second texte de
  blog FIRE germanophone (`dividende-statt-rente.de`) en plus de `getmad.de`, ce qui reste
  dans la fourchette 10-14 articles demandée. `rente-mit-dividende.de` a échoué au niveau
  TLS (`SSL routines::tlsv1 alert internal error`) et a été remplacé par
  `dividende-statt-rente.de`.
- portefeuille: 23/30 titres retenus (univers 142 tickers sur 3 indices Wikipedia, 134 avec >= 8 dividendes annuels sur 15 ans, 44 apres filtre rendement 2-8 %/0 baisse et fusion .info reussie); paliers essayes (strict: 23; croissance >= 2 %: 23; payout <= 90 %: 23); retenu strict avec 23 titres. Le chapitre 6 presente les 23 titres reels obtenus, la methode restant illustree meme sous 30 lignes.

## Tâche 9 partie A (couche de données, chapitres 2 à 5, 2026-09-30)

- `etf_welt_rendite_div` : cherché sur la fiche justETF du plus grand ETF MSCI World
  (iShares Core MSCI World UCITS ETF USD Acc, IE00B4L5Y983, encours EUR 129 840 m) : aucun
  rendement de distribution affiché, car cette part est thésaurisante (vérifié par WebFetch
  le 30.09.2026, page lue en entier). Repli explicite 0,018 conservé tel qu'imposé par le
  contrôleur. À titre de comparaison seulement (non retenu, car ce n'est pas le plus grand
  ETF) : la part distribuante iShares MSCI World UCITS ETF (Dist), IE00B0M62Q58, encours
  EUR 8 303 m, affiche un rendement de dividende courant de 0,85 % (1,01 % sur 1 an) :
  sensiblement sous le repli de 1,8 % retenu, à revoir de jour si une fiche exploitable
  pour le plus grand ETF distribuant apparaît (recherche non poussée plus loin cette nuit
  faute de confirmation fiable de « quel est le plus grand ETF distribuant »).
- Série d'inflation Destatis 2019-2025 (pour `fig_inflation_2022`, chapitre 7, partie B de
  la tâche 9) : sourcée par avance sous `inflation_destatis_2019_2025` dans quellen.json,
  pour éviter une deuxième recherche identique. Écart de révision détecté et non résolu :
  la première annonce provisoire de 2022 (janvier 2023, PD23_022_611, déjà citée sous
  `inflation_2022_de` = 7,9 %) diffère de la valeur rétrospective citée uniformément dans
  les communiqués 2024 et 2025 (6,9 %), vraisemblablement à cause du changement de base du
  VPI (nouvelle base 2020=100 en 2024). `inflation_2022_de` n'a pas été modifiée (hors
  périmètre de cette tâche) ; la série ajoutée utilise la version rétrospective cohérente
  (6,9 %). À arbitrer de jour : lequel des deux chiffres pour 2022 le document doit citer,
  et si `inflation_2022_de` doit être mise à jour en conséquence.
- **Retenue américaine : corrigé (remplace l'entrée précédente ci-dessus).** Le premier
  passage de la tâche 9 partie A signalait ici que tout le moteur (`projection.py`,
  `fiscalite.py`) retenait implicitement la retenue américaine standard sans formulaire
  (30 %), jamais le taux W-8BEN (15 %). Sur revue du contrôleur (post-commit 89a4e659),
  décision explicite : le W-8BEN est supposé déjà déposé (pratique quasi automatique chez
  les courtiers européens). `fiscalite.py` et `projection.py` **ne sont pas modifiés**
  (déjà testés, hors périmètre) ; `rechnung_retraite.py` construit désormais sa propre
  copie de `hypotheses.wert("quellensteuer")` avec l'entrée US substituée au taux
  `mit_antrag` (15 %), passée partout via `_sim_kwargs()`/`_sq_de_base()`. La Suisse et la
  France restent sans remboursement par défaut (conservateur), le cas "avec
  remboursement"/"avec formulaire" restant modélisé séparément dans `fig_steuer_100`, qui
  affiche aussi désormais "US_sans_w8ben" (30 %) en comparaison. Effet numérique :
  `\NettoUs` passe de 59,45 à 74,45 (pour 100 EUR bruts), `\BruttoNoetig` de 87 595 à
  76 050 EUR, `\KapitalMaisonSiebzig` de 3 310 368 à 2 973 818 EUR (détail complet dans
  task-9-report.md). Toujours signalé pour arbitrage de jour : si l'hypothèse W-8BEN est
  jugée trop optimiste pour un lecteur donné, une révision de `projection.py`
  (tâche 4) permettrait de paramétrer `antrag` par pays plutôt que de le simuler en aval
  dans la tâche 9.
- **Rendement réel de marché : corrigé (remplace l'entrée initiale de markt_basis).** La
  première version utilisait la médiane des rendements réels annuels Shiller 1950-2022
  (12,1 %), conforme au plan initial mais statistiquement invalide comme taux de
  capitalisation composé sur 45 ans (la médiane/moyenne arithmétique d'une série annuelle
  ignore l'effet multiplicatif des années de krach). Remplacé par le TCAC (moyenne
  géométrique, `prod(1+r)^(1/n) - 1`) = 7,3 %, conservé en écart documenté au plan décidé
  par le contrôleur (voir docstring de `markt_basis` dans `rechnung_retraite.py`). La
  médiane (12,1 %) reste exposée sous `\RenditeReelleMediane` à titre de comparaison.
  Effet numérique majeur : plusieurs combinaisons stratégie x taux d'épargne à 10 %
  n'atteignent plus jamais l'objectif dans l'horizon de 45 ans (`ZUS["alter"]` = null),
  ce qui n'était le cas d'aucune des 20 combinaisons avec la médiane (détail dans
  task-9-report.md).

## Tâche 9 partie B (couche de données, chapitres 6, 7, 8, 10, 2026-09-30)

- **Nom tronqué hérité de la tâche 7/8** : `data/portefeuille.csv`, colonne `name` de
  MKC, porte `"McCormick & Company, Incorporat"` (le "ed" final manque, troncature de
  scraping antérieure à cette tâche). Utilisé tel quel dans `\BeispielHausse` (chapitre 6,
  fig_div_beispiele) après échappement ; non corrigé ici car `portefeuille.csv` est un
  livrable de la tâche 8, hors périmètre de la tâche 9, et compléter le nom de mémoire
  serait une valeur inventée. À corriger de jour en régénérant la colonne `name` depuis
  `.info` pour ce ticker, ou en tronquant proprement à la virgule dans `rechnung_retraite.py`
  si la tâche 8 n'est pas rejouée.
- **Défaut de conception détecté et corrigé avant publication (`fig_mc_erfolg`)** : la
  première implémentation forçait le rendement réel de l'année CALENDAIRE 0 de chaque
  trajectoire Monte Carlo (`pfade[:, 0, 0]`) sur une grille -40 % à +40 %, et produisait un
  taux de réussite rigoureusement identique sur toute la grille (implausible, détecté par
  relecture des CSV avant commit, pas ignoré). Cause : `montecarlo.erfolg()` démarre le
  capital à 0.0, donc `0 * (1 + rendement forcé) = 0` quel que soit le rendement ; de plus la
  réussite post-retraite ne dépend jamais de `rendite_real`, seulement de
  `div_wachstum_real`. Corrigé par une réimplémentation locale documentée
  (`_erfolg_avec_choc_retraite` dans `rechnung_retraite.py`, `montecarlo.py` non modifié,
  déjà testé) qui force la croissance du dividende de la PREMIÈRE ANNÉE DE RETRAITE de
  chaque trajectoire (indice variable d'une trajectoire à l'autre selon la date
  d'atteinte de l'objectif). Les valeurs publiées dans `fig_mc_erfolg.csv` varient
  maintenant de façon monotone et plausible avec `rendite_start` (0 % de réussite à -40/-30 %,
  87,6 % à +40 %, tous horizons de retraite confondus) ; test de non-régression ajouté
  (`test_fig_mc_erfolg_varie_avec_le_choc`).

## Tâche 10, chapitres 09-10 (littérature, second essai, 2026-09-30 vers 18 h 35)

- **Littérature académique : second essai, échec à nouveau.** `SCOPUS_API_KEY` toujours absent
  (`scopus_api.py search` refuse immédiatement : « SCOPUS_API_KEY is not set »), pas de
  `.scopus_key`. Repli sur l'API publique Semantic Scholar (`/paper/search` via
  `semantic_scholar_api._request`, un seul essai par requête, avec le réessai unique intégré au
  client) : **HTTP 429** sur 5 des 6 requêtes (`safe withdrawal rate retirement`,
  `sequence of returns retirement`, `dividend retirement income`,
  `number of stocks diversification`, `high dividend yield performance`) ; la sixième
  (`dividend disconnect preference for dividends`) a répondu, mais avec deux résultats hors
  sujet (une étude de santé publique en Afrique subsaharienne, un article sans DOI sur la
  valorisation des banques), donc inutilisables. `data/literatur.bib` reste vide ; le chapitre 9
  le dit en une phrase (section « Et la recherche ? ») et ne cite aucune étude de mémoire.
  À reprendre de jour avec un accès campus/VPN (Scopus) ou une clé `S2_API_KEY`.
- **Chapitre 10, réserve d'urgence** : fixée à 3 salaires nets (`RESERVE_URGENCE_MOIS`,
  convention du document, présentée comme telle dans un `\annahme`) ; aucune source dans
  `quellen.json`. Les modalités pratiques (Freistellungsauftrag, prélèvement de la
  Vorabpauschale, validité du W-8BEN) ne sont pas sourcées non plus et sont renvoyées au
  courtier et à un conseiller fiscal. Le cas « capital de départ nul » (capital de départ
  entièrement mis en réserve) n'est pas simulé ; le texte le dit.

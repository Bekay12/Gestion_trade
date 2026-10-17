# Progress — Umicore Investitionsanalyse

## Cadrage (11.10.2026)

- Sujet : Umicore NV/SA (ISIN BE0974320526, ticker UMI, Euronext Bruxelles), groupe industriel
  diversifie : Catalysis, Recycling, Battery Materials Solutions, Specialty Materials.
- n = 0. Decision : acheter / ne pas acheter. Annees couvertes : GJ2021-GJ2025.

## Domaines (orchestration.md) — tous complets

- [x] domain1_governance : 68 faits, tres fiable (toutes les citations spot-checkees se sont
      reverees exactes). Une correction importante trouvee par rapport a une recherche web
      anterieure (hors skill) : GBL detient 7,96 % et non 16,49 % au 31.12.2025 (source
      primaire GB2025 p.30, web secondaire perimee/fausse).
- [x] domain2_financials : 45 faits, VALEURS fiables mais ~40% des citations de PAGE fausses —
      pas le meme piege que Nynomic (confusion ligne/page), ici confusion entre plusieurs
      tableaux structurellement similaires du meme document (p.ex. "Consolidated statement of
      comprehensive income" confondu avec "Consolidated income statement", ou un tableau
      d'ajustements par segment confondu avec le tableau de chiffre d'affaires). Corrige a la
      main avant redaction — carte de pages definitive ci-dessous.
- [x] domain3_market : 24 faits, le plus fiable des trois (citations spot-checkees exactes,
      0 contradiction). Correction importante vs une recherche web anterieure : Catalysis
      (450 M€ EBITDA) est le premier contributeur en absolu, pas Recycling (371 M€) —
      contrairement a la narrative de presse secondaire ("Recycling = ~80% du CA").

## Piege methodologique decouvert (different de Nynomic)

Nynomic : confusion numero-de-ligne-pdftotext vs numero-de-page-imprimee.
Umicore : confusion entre deux tableaux du MEME document ayant une structure visuellement
similaire (les trois etats financiers consolides se suivent sur 2-3 pages consecutives dans un
rapport annuel de 200+ pages, et un sous-agent qui lit en diagonale peut facilement prendre la
page suivante/precedente pour la bonne). Lecon pour la prochaine analyse d'un grand groupe
cote : demander explicitement de citer le TITRE exact de la table/section lu en plus du numero
de page (pas seulement le numero), ce qui rend l'erreur immediatement visible a la relecture.

Deuxieme piege, plus subtil : Umicore publie DEUX notions de chiffre d'affaires sous le nom
"Turnover" (IFRS, avec passage des metaux, tres grand ~15-25 Md€) et "Group revenue excluding
metals" (la metrique preferee par l'entreprise, ~3,5-4,2 Md€) — melanger les deux sur une serie
multi-annee cree une fausse impression d'"effondrement du CA" qui n'existe pas operationnellement.
Toujours verifier QUELLE definition de chiffre d'affaires est utilisee avant de construire une
serie multi-annee, surtout pour un groupe avec activite de negoce/raffinage de metaux precieux.

## Carte de pages definitive (verifiee personnellement, PDF page = page imprimee)

| Document | Bilan | Compte de resultat | Cash-flow |
|---|---|---|---|
| GB2022 (GJ2021/2022) | p.154 | p.152 | (non requis directement) |
| GB2023 (GJ2022/2023) | p.201 (domain2) | p.198 | p.203 (domain2) |
| GB2024 (GJ2023/2024) | p.61 | p.32 (domain2, non revérifié ligne par ligne) | p.67 (domain2) |
| GB2025 (GJ2024/2025) | p.64 | GuV consolidee (section "Consolidated income statement", pres de p.65-66, non numerotee exactement dans mes notes) | p. Kapitalflussrechnung (domain2) |
| GB2024 remuneration table "Company's performance" (ROCE/EBIT 2020-2024) | — | — | p.50 |

## Rapport final

`Umicore_Investitionsanalyse.md` — termine 11.10.2026. Verdict : ne pas acheter a 20,98 €.
Hurdle ROE sur resultat ajuste 12,68 % (moyenne 3 ans GJ2023-25, IFRS exclu a cause de la
depreciation ponctuelle GJ2024 qui donnerait -77%). Seuil d'entree 10,60 € (base, 15 ans) /
18,50 € (optimiste, 15 ans). Signal le plus fort contre ce verdict de toute la serie : consensus
analystes large et credible (13-15 analystes, cible 23,8-25,6 €), contrairement aux small-caps
precedents de la serie a couverture fine. Entree ajoutee dans `Suivi_Verdicts.md`.

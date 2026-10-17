# Progress — Nynomic AG Investitionsanalyse

## Cadrage (11.10.2026)

- Sujet : Nynomic AG (ISIN DE000A0MSN11, WKN A0MSN1, Ticker Xetra "M7U"), photonique/optique
  (spectroscopie, capteurs, automatisation labo, technologie médicale), Scale-Segment Deutsche
  Börse, siège Wedel.
- n = 0 (decision binaire achat/non-achat, pas de comparaison d'options strategiques).
- Annees couvertes : GJ2022-GJ2025 (serie 4 ans, crise 2025 incluse), + interim S1 2026.
- Decision : acheter / ne pas acheter au cours actuel, et si achat jusqu'a quel prix.
- Sources primaires telechargees localement dans quellen/ :
  - NynomicAG_AR2022_EN.pdf (76 p., page PDF = page imprimee verifie p.5)
  - NynomicAG_AR2023_EN.pdf (80 p., idem)
  - NynomicAG_AR2024_EN.pdf (80 p., idem)
  - NynomicAG_AR2025_EN.pdf (80 p., idem, verifie p.10/40/80)
  - NynomicAG_GB_2025.pdf (version allemande, sauvegarde/croisement)
  - HJB_2026.pdf (rapport semestriel S1 2026, interim)
  - Chaque PDF a son .txt (pdftotext -layout) pour recherche rapide.
- Marche (secondaire, yfinance M7U.DE, 11.10.2026) : cours 26,00 EUR, capi 170,8 M EUR,
  6 568 820 actions, ROE ttm yfinance 3,94 % (biaise par l'annee de crise 2025 - a ne PAS
  utiliser comme hurdle sans sensibilite), book value/action 14,91 EUR, P/B 1,74, beta 1,02,
  52 sem. 8,22-28,40 EUR (ecart enorme : effondrement puis reprise), pas de dividende.
  Couverture analystes contradictoire selon la source (2 vs 6 analystes, cible moyenne 21,25
  vs 29,00 EUR) -> a declarer comme contradiction, les deux citees.

## Domaines (orchestration.md)

- [x] domain1_governance : complete (57 faits, 4 gaps) — citations de pages globalement fiables
- [x] domain2_financials : complete (86 faits, contradictions 0) — VALEURS fiables (verifiees par
      recoupement direct), mais ~50% des citations de PAGE etaient fausses (confusion
      numero-de-ligne pdftotext / numero-de-page-imprimee). Corrige a la main avant redaction.
- [x] domain3_market : complete (65 faits) — meme defaut de citation que domain2, plus marque
      (ex. "p.749", "p.1032" dans un document de 80 pages = impossible). Corrige a la main.

## Correction methodologique (11.10.2026)

Les trois sous-agents de recherche (Haiku) ont correctement trouve et rapporte les VALEURS
(revenus, EBIT, parts, etc. — tout verifie et confirme exact par recoupement direct avec les PDF),
mais ont frequemment confondu le numero de ligne de leur grep/pdftotext avec le numero de page
imprimee, surtout pour toute recherche de texte au-dela des toutes premieres pages. Un outil
`/tmp/pagegrep.py` a ete ecrit pour retrouver la vraie page (suit les en-tetes "Annual Report of
Nynomic AG | N" / "N | Annual Report..." qui different entre GB2022-2023 et GB2024-2025). Carte
de pages verifiee et definitive :

| Document | Bilan | Compte de resultat | Flux de tresorerie | Nombre d'actions |
|---|---|---|---|---|
| GB2022 (GJ2022, comp. 2021) | p.26-27 | p.28 | p.56 | p.24 |
| GB2023 (GJ2023, comp. 2022) | p.24-25 | p.26 | p.56 | p.22 |
| GB2024 (GJ2024, comp. 2023) | p.30-31 | p.32 | p.60 | p.28 |
| GB2025 (GJ2025, comp. 2024) | p.30-31 | p.32 | p.60 | p.29 |
| HJB2026 (S1 2026, comp. S1 2025) | p.16-17 | p.18 (resume p.2) | — (non requis) | — |

Lecon pour la prochaine analyse de ce type : demander explicitement aux sous-agents de ne JAMAIS
citer un numero de ligne de leur outil de recherche comme page, et de valider chaque page citee
contre le total de pages du document (`pdfinfo | grep Pages`) avant de l'ecrire dans le digest —
une page citee superieure au nombre total de pages du PDF est un signal immediat et automatique
d'erreur.

## Rapport final

`Nynomic_Investitionsanalyse.md` — termine 11.10.2026. Verdict : ne pas acheter a 26,00 €.
Hurdle ROE propre 4,32 % (moyenne 4 ans, la plus basse de la serie car Nynomic n'a jamais depasse
9,67 % de ROE), seuil d'entree 16,10 € (base, 15 ans) / 22,20 € (optimiste, 15 ans). Entree
ajoutee dans `Suivi_Verdicts.md`.

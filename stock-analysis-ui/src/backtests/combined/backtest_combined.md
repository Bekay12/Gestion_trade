# Backtest point-in-time du Combined scan

Dates T : 2024-04-01, 2024-07-01, 2024-10-01, 2025-01-02, 2025-04-01, 2025-07-01, 2025-10-01, 2026-01-02. Univers : 2536 titres du store, cours jusqu'au 2026-09-19.

Méthode : à chaque date T, les 12 critères du scanner (core/scan_fondamentaux.py) sur les seules données connues avant T (exercice publié 90 j après clôture, trimestre 45 j). Rendement total (dividendes réinvestis) de la première séance après T à l'échéance. Écarts : à l'indice local, et à la médiane de l'univers à la même date. Rendements et écarts écrêtés aux 1er/99e centiles par date.
Rendements écartés pour saut de cours invraisemblable (x4 ou /4 en une séance) : 92 à 6 mois, 185 à 12 mois.
Tableaux principaux : dates T >= 2025-04-01 seulement (avant, S6/S7 quasi incalculables). Limites : univers survivant, PEG rétrospectif, nombre d'actions actuel, bêta hebdomadaire 52 sem.

## Par profil (T >= 2025-04-01)

| profil | horizon | titres | dates | perf_moy_% | perf_med_% | ecart_med_univers_pts | ecart_indice_pts | battent_indice_% | gagnants_% |
|---|---|---|---|---|---|---|---|---|---|
| 💎 Dual Champion* | 12 mois | 35 | 2 | +55.7 | +29.0 | +38.3 | +34.9 | +77.1 | +91.4 |
| 💎 Dual Champion | 12 mois | 85 | 2 | +32.7 | +30.8 | +15.5 | +17.2 | +63.5 | +67.1 |
| 🛡️  Pure Safe | 12 mois | 514 | 2 | +17.4 | +7.4 | -0.1 | -0.8 | +39.1 | +59.1 |
| 🚀 Pure Growth | 12 mois | 8 | 1 | +40.7 | +12.9 | +22.2 | +20.0 | +37.5 | +62.5 |
| ⚖️  Balanced | 12 mois | 131 | 2 | +36.6 | +26.2 | +19.3 | +18.9 | +55.0 | +69.5 |
| ⚪ Below | 12 mois | 1832 | 2 | +39.4 | +19.9 | +21.8 | +20.7 | +51.3 | +65.8 |
| 💎 Dual Champion* | 6 mois | 64 | 4 | +20.8 | +13.8 | +13.7 | +10.8 | +57.8 | +70.3 |
| 💎 Dual Champion | 6 mois | 169 | 4 | +14.3 | +11.9 | +7.4 | +6.4 | +52.1 | +61.5 |
| 🛡️  Pure Safe | 6 mois | 1380 | 4 | +7.4 | +4.5 | +1.6 | -0.0 | +43.5 | +57.7 |
| 🚀 Pure Growth | 6 mois | 28 | 3 | -0.8 | +2.1 | -4.5 | -6.2 | +42.9 | +50.0 |
| ⚖️  Balanced | 6 mois | 295 | 4 | +15.3 | +8.2 | +8.7 | +6.6 | +48.1 | +61.7 |
| ⚪ Below | 6 mois | 5484 | 4 | +12.4 | +5.6 | +6.9 | +5.0 | +47.9 | +58.2 |

## Par score total (T >= 2025-04-01)

| score_bande | horizon | titres | dates | perf_moy_% | perf_med_% | ecart_med_univers_pts | ecart_indice_pts | battent_indice_% | gagnants_% |
|---|---|---|---|---|---|---|---|---|---|
| 0-3 | 12 mois | 992 | 2 | +43.2 | +20.7 | +25.6 | +24.5 | +51.4 | +63.0 |
| 4-5 | 12 mois | 741 | 2 | +33.7 | +17.3 | +16.1 | +15.1 | +48.9 | +69.0 |
| 6-7 | 12 mois | 595 | 2 | +27.7 | +16.3 | +10.2 | +9.1 | +47.7 | +65.4 |
| 8-9 | 12 mois | 248 | 2 | +21.0 | +9.6 | +3.6 | +3.9 | +47.2 | +58.5 |
| 10-12 | 12 mois | 29 | 2 | +48.7 | +32.5 | +31.7 | +32.2 | +82.8 | +86.2 |
| 0-3 | 6 mois | 3044 | 4 | +13.6 | +4.7 | +8.0 | +6.2 | +48.2 | +56.4 |
| 4-5 | 6 mois | 2188 | 4 | +11.6 | +6.8 | +6.1 | +4.1 | +48.2 | +61.9 |
| 6-7 | 6 mois | 1595 | 4 | +8.7 | +4.6 | +3.0 | +1.3 | +44.4 | +57.6 |
| 8-9 | 6 mois | 545 | 4 | +9.1 | +4.5 | +2.7 | +1.0 | +45.1 | +56.1 |
| 10-12 | 6 mois | 48 | 4 | +21.0 | +17.2 | +13.2 | +11.8 | +62.5 | +72.9 |

## Dual Champions par date (toutes dates ; avant avril 2025 : non comparable)

| date_T | horizon | dual_champions | dual_perf_med_% | univers_perf_med_% | dual_ecart_med_univers_pts | dual_ecart_indice_pts | dual_battent_indice_% |
|---|---|---|---|---|---|---|---|
| 2024-04-01 | 12 mois | 2 | -1.6 | -0.4 | -1.3 | -38.7 | +0.0 |
| 2024-07-01 | 12 mois | 5 | +20.7 | +9.7 | -8.6 | -11.5 | +40.0 |
| 2024-10-01 | 12 mois | 8 | +26.4 | +7.1 | +19.8 | +8.9 | +50.0 |
| 2025-01-02 | 12 mois | 6 | +30.1 | +10.6 | +13.1 | +6.9 | +50.0 |
| 2025-04-01 | 12 mois | 78 | +31.1 | +16.5 | +25.7 | +27.6 | +70.5 |
| 2025-07-01 | 12 mois | 50 | +29.0 | +18.5 | +15.7 | +12.6 | +52.0 |
| 2024-04-01 | 6 mois | 2 | +28.8 | +3.3 | +25.5 | +3.9 | +50.0 |
| 2024-07-01 | 6 mois | 5 | +1.0 | +2.5 | -5.9 | -10.5 | +40.0 |
| 2024-10-01 | 6 mois | 8 | +0.2 | -4.8 | +9.9 | +5.2 | +50.0 |
| 2025-01-02 | 6 mois | 6 | +17.5 | +4.8 | +7.4 | +6.1 | +83.3 |
| 2025-04-01 | 6 mois | 78 | +17.3 | +13.3 | +9.8 | +8.3 | +57.7 |
| 2025-07-01 | 6 mois | 50 | +19.4 | +5.6 | +17.8 | +12.6 | +60.0 |
| 2025-10-01 | 6 mois | 69 | +2.1 | -0.0 | +7.9 | +8.8 | +55.1 |
| 2026-01-02 | 6 mois | 49 | -1.3 | +8.6 | -1.9 | -3.0 | +24.5 |

## Dual Champions (T >= 2025-04-01, détail)

| date_T | ticker | nom | score_total | perf6_% | perf12_% | ecart6_indice_pts | ecart12_indice_pts |
|---|---|---|---|---|---|---|---|
| 2025-04-01 | AEM | Agnico Eagle Mines Limited | 11 | +58.2 | +96.1 | +39.1 | +79.4 |
| 2025-04-01 | AU | AngloGold Ashanti PLC | 11 | +94.6 | +188.8 | +75.5 | +172.0 |
| 2025-04-01 | ENX.PA | EURONEXT | 11 | -6.3 | +6.6 | -7.5 | +5.3 |
| 2025-04-01 | NVS | Novartis AG | 11 | +20.2 | +45.8 | +1.0 | +29.1 |
| 2025-04-01 | TLX.DE | Talanx AG                     N | 11 | +16.9 | +14.9 | +9.9 | +11.5 |
| 2025-04-01 | VICI | VICI Properties Inc. | 11 | +4.3 | -10.2 | -14.9 | -26.9 |
| 2025-04-01 | AGI | Alamos Gold Inc. | 10 | +31.2 | +70.8 | +12.1 | +54.0 |
| 2025-04-01 | ALC | Alcon Inc. | 10 | -18.7 | -18.3 | -37.8 | -35.0 |
| 2025-04-01 | ALV.DE | Allianz SE                    v | 10 | +5.8 | +7.4 | -1.2 | +4.0 |
| 2025-04-01 | B | Barrick Mining Corporation | 10 | +73.4 | +119.8 | +54.2 | +103.1 |
| 2025-04-01 | BA.L | BAE SYSTEMS PLC ORD 2.5P | 10 | +31.2 | +45.8 | +21.8 | +25.7 |
| 2025-04-01 | BABA | Alibaba Group Holding Limited | 10 | +40.1 | -5.2 | +20.9 | -21.9 |
| 2025-04-01 | GFI | Gold Fields Limited | 10 | +83.4 | +117.6 | +64.3 | +100.9 |
| 2025-04-01 | GTT.PA | GAZTRANSPORT & TECHNIGAZ | 10 | +13.0 | +50.7 | +11.9 | +49.3 |
| 2025-04-01 | NOVN.SW | NOVARTIS N | 10 | +7.4 | +31.1 | +10.0 | +28.7 |
| 2025-04-01 | RHM.DE | RHEINMETALL AG                I | 10 | +53.9 | +21.7 | +46.9 | +18.3 |
| 2025-04-01 | SBS | Companhia de saneamento Basico  | 10 | +41.4 | +74.4 | +22.2 | +57.7 |
| 2025-04-01 | SCCO | Southern Copper Corporation | 10 | +35.3 | +101.7 | +16.1 | +85.0 |
| 2025-04-01 | TW | Tradeweb Markets Inc. | 10 | -27.1 | -19.5 | -46.3 | -36.3 |
| 2025-04-01 | UTHR | United Therapeutics Corporation | 10 | +42.8 | +85.8 | +23.6 | +69.1 |
| 2025-04-01 | WELL | Welltower Inc. | 10 | +17.4 | +31.1 | -1.7 | +14.4 |
| 2025-04-01 | 2020.HK | ANTA SPORTS | 9 | +10.9 | -4.0 | -4.8 | -13.0 |
| 2025-04-01 | 9988.HK | BABA-W | 9 | +41.1 | -8.7 | +25.4 | -17.6 |
| 2025-04-01 | ABBN.SW | ABB LTD N | 9 | +27.6 | +46.1 | +30.1 | +43.7 |
| 2025-04-01 | ACS.MC | ACS,ACTIVIDADES DE CONSTRUCCION | 9 | +34.0 | +113.5 | +17.1 | +81.3 |
| 2025-04-01 | AGI.TO | ALAMOS GOLD INC CLS A | 9 | +28.0 | +66.1 | +7.7 | +34.5 |
| 2025-04-01 | APAM | Artisan Partners Asset Manageme | 9 | +10.4 | +0.8 | -8.7 | -15.9 |
| 2025-04-01 | AXIA | AXIA Energia | 9 | +46.9 | +76.4 | +27.7 | +59.7 |
| 2025-04-01 | CWC.DE | CEWE Stiftung & Co. KGaA      I | 9 | +1.3 | -8.1 | -5.7 | -11.4 |
| 2025-04-01 | DB1.DE | DEUTSCHE BOERSE AG            N | 9 | -15.3 | -6.3 | -22.3 | -9.7 |
| 2025-04-01 | DTE.DE | DEUTSCHE TELEKOM AG           N | 9 | -11.9 | -3.9 | -18.8 | -7.2 |
| 2025-04-01 | FHI | Federated Hermes, Inc. | 9 | +25.3 | +45.0 | +6.1 | +28.3 |
| 2025-04-01 | G24.DE | Scout24 SE                    N | 9 | +7.7 | -31.3 | +0.8 | -34.6 |
| 2025-04-01 | GMAB | Genmab A/S | 9 | +70.3 | +45.9 | +51.1 | +29.1 |
| 2025-04-01 | KGC | Kinross Gold Corporation | 9 | +102.7 | +155.4 | +83.6 | +138.6 |
| 2025-04-01 | NBIX | Neurocrine Biosciences, Inc. | 9 | +28.0 | +23.9 | +8.9 | +7.2 |
| 2025-04-01 | NOVO-B.CO | Novo Nordisk B A/S | 9 | -19.5 | -46.9 | -19.3 | -48.9 |
| 2025-04-01 | NOW | ServiceNow, Inc. | 9 | +12.4 | -35.9 | -6.7 | -52.6 |
| 2025-04-01 | OGC | OceanaGold Corporation | 9 | +124.9 | +234.5 | +105.8 | +217.8 |
| 2025-04-01 | PAYC | Paycom Software, Inc. | 9 | -9.5 | -44.9 | -28.7 | -61.6 |
| 2025-04-01 | UHS | Universal Health Services, Inc. | 9 | +8.6 | -4.9 | -10.5 | -21.6 |
| 2025-04-01 | YOU | Clear Secure, Inc. | 9 | +24.0 | +94.0 | +4.9 | +77.3 |
| 2025-04-01 | 005930.KS | SamsungElec | 8 | +54.1 | +208.4 | +17.0 | +91.1 |
| 2025-04-01 | 2388.HK | BOC HONG KONG | 8 | +23.5 | +46.1 | +7.7 | +37.1 |
| 2025-04-01 | 2628.HK | CHINA LIFE | 8 | +45.7 | +66.8 | +29.9 | +57.8 |
| 2025-04-01 | ABT | Abbott Laboratories | 8 | +1.6 | -21.4 | -17.6 | -38.1 |
| 2025-04-01 | ADC | Agree Realty Corporation | 8 | -5.4 | +2.2 | -24.5 | -14.5 |
| 2025-04-01 | ADUS | Addus HomeCare Corporation | 8 | +17.1 | -5.1 | -2.0 | -21.9 |
| 2025-04-01 | AGS.BR | AGEAS | 8 | +9.4 | +24.7 | -3.0 | +5.1 |
| 2025-04-01 | AKAM | Akamai Technologies, Inc. | 8 | -5.9 | +43.6 | -25.0 | +26.9 |
| 2025-04-01 | BFSA.DE | BEFESA S.A.                   A | 8 | +14.9 | +15.7 | +7.9 | +12.3 |
| 2025-04-01 | BTG | B2Gold Corp | 8 | +73.4 | +68.5 | +54.2 | +51.7 |
| 2025-04-01 | COK.DE | CANCOM SE                     I | 8 | +11.6 | +7.8 | +4.7 | +4.4 |
| 2025-04-01 | DEME.BR | DEME GROUP | 8 | -0.4 | +49.0 | -12.8 | +29.4 |
| 2025-04-01 | EQNR | Equinor ASA | 8 | -5.8 | +59.3 | -24.9 | +42.5 |
| 2025-04-01 | ESE | ESCO Technologies Inc. | 8 | +34.5 | +85.1 | +15.3 | +68.3 |
| 2025-04-01 | ESI | Element Solutions Inc. | 8 | +14.3 | +55.3 | -4.8 | +38.6 |
| 2025-04-01 | EZPW | EZCORP, Inc. | 8 | +16.8 | +67.7 | -2.3 | +50.9 |
| 2025-04-01 | G1A.DE | GEA Group AG                  I | 8 | +16.0 | +13.4 | +9.0 | +10.0 |
| 2025-04-01 | GILD | Gilead Sciences, Inc. | 8 | +1.3 | +29.4 | -17.8 | +12.7 |
| 2025-04-01 | HEI.DE | Heidelberg Materials AG       I | 8 | +17.6 | +9.6 | +10.6 | +6.3 |
| 2025-04-01 | HST | Host Hotels & Resorts, Inc. | 8 | +21.0 | +39.6 | +1.8 | +22.9 |
| 2025-04-01 | JEN.DE | JENOPTIK AG                   N | 8 | +1.3 | +57.6 | -5.7 | +54.2 |
| 2025-04-01 | KBX.DE | Knorr-Bremse AG               I | 8 | -1.6 | +21.8 | -8.6 | +18.4 |
| 2025-04-01 | META | Meta Platforms, Inc. | 8 | +22.6 | -0.8 | +3.5 | -17.6 |
| 2025-04-01 | NTST | NetSTREIT Corp. | 8 | +17.4 | +24.3 | -1.7 | +7.6 |
| 2025-04-01 | OVV | Ovintiv Inc. (DE) | 8 | -6.3 | +34.4 | -25.5 | +17.7 |
| 2025-04-01 | PLMR | Palomar Holdings, Inc. | 8 | -20.7 | -15.8 | -39.8 | -32.5 |
| 2025-04-01 | PRGO | Perrigo Company plc | 8 | -16.8 | -58.5 | -36.0 | -75.2 |
| 2025-04-01 | PRY.MI | PRYSMIAN | 8 | +76.9 | +110.5 | +65.2 | +91.9 |
| 2025-04-01 | SIE.DE | SIEMENS AG                    N | 8 | +12.3 | +1.5 | +5.3 | -1.9 |
| 2025-04-01 | SKWD | Skyward Specialty Insurance Gro | 8 | -16.9 | -20.2 | -36.0 | -37.0 |
| 2025-04-01 | TER | Teradyne, Inc. | 8 | +71.3 | +279.3 | +52.1 | +262.6 |
| 2025-04-01 | TFPM | Triple Flag Precious Metals Cor | 8 | +56.0 | +86.9 | +36.9 | +70.2 |
| 2025-04-01 | VH2.DE | Friedrich Vorwerk Group SE    I | 8 | +39.4 | +11.1 | +32.4 | +7.8 |
| 2025-04-01 | VIE.PA | VEOLIA ENVIRON. | 8 | -6.4 | +7.2 | -7.5 | +5.9 |
| 2025-04-01 | VOD | Vodafone Group Plc | 8 | +27.5 | +71.0 | +8.4 | +54.3 |
| 2025-04-01 | Z74.SI | Singtel | 8 | +22.7 | +48.3 | +13.8 | +22.9 |
| 2025-07-01 | AU | AngloGold Ashanti PLC | 11 | +90.9 | +86.4 | +80.5 | +65.6 |
| 2025-07-01 | 9022.T | CENTRAL JAPAN RAILWAY CO | 10 | +34.9 | +7.8 | +9.1 | -68.4 |
| 2025-07-01 | AEM | Agnico Eagle Mines Limited | 10 | +44.4 | +32.5 | +34.0 | +11.8 |
| 2025-07-01 | GTT.PA | GAZTRANSPORT & TECHNIGAZ | 10 | -3.0 | +18.0 | -9.3 | +9.2 |
| 2025-07-01 | MPWR | Monolithic Power Systems, Inc. | 10 | +21.8 | +79.5 | +11.3 | +58.8 |
| 2025-07-01 | NVS | Novartis AG | 10 | +12.0 | +29.0 | +1.6 | +8.3 |
| 2025-07-01 | OGC | OceanaGold Corporation | 10 | +94.2 | +72.6 | +83.7 | +51.8 |
| 2025-07-01 | SBS | Companhia de saneamento Basico  | 10 | +7.9 | +30.8 | -2.5 | +10.1 |
| 2025-07-01 | 4507.T | SHIONOGI & CO | 9 | +12.8 | +11.6 | -13.1 | -64.6 |
| 2025-07-01 | APH | Amphenol Corporation | 9 | +39.2 | +78.0 | +28.8 | +57.3 |
| 2025-07-01 | FEIM | Frequency Electronics, Inc. | 9 | +159.5 | +217.1 | +149.0 | +196.3 |
| 2025-07-01 | FHI | Federated Hermes, Inc. | 9 | +17.9 | +26.9 | +7.4 | +6.2 |
| 2025-07-01 | FUTU | Futu Holdings Limited | 9 | +35.0 | -16.5 | +24.5 | -37.3 |
| 2025-07-01 | GMAB | Genmab A/S | 9 | +48.5 | +33.3 | +38.1 | +12.5 |
| 2025-07-01 | GOOG | Alphabet Inc. | 9 | +77.7 | +102.9 | +67.2 | +82.1 |
| 2025-07-01 | GOOGL | Alphabet Inc. | 9 | +78.3 | +106.0 | +67.8 | +85.3 |
| 2025-07-01 | HEI | Heico Corporation | 9 | +0.6 | n.d. | -9.8 | n.d. |
| 2025-07-01 | KGC | Kinross Gold Corporation | 9 | +81.6 | +51.8 | +71.2 | +31.0 |
| 2025-07-01 | NBIX | Neurocrine Biosciences, Inc. | 9 | +10.8 | +31.1 | +0.4 | +10.3 |
| 2025-07-01 | NOVO-B.CO | Novo Nordisk B A/S | 9 | -24.3 | -21.3 | -31.3 | -29.7 |
| 2025-07-01 | TFPM | Triple Flag Precious Metals Cor | 9 | +40.5 | +28.1 | +30.1 | +7.3 |
| 2025-07-01 | TLX.DE | Talanx AG                     N | 9 | +4.4 | +9.5 | +0.9 | +3.7 |
| 2025-07-01 | TSM | Taiwan Semiconductor Manufactur | 9 | +36.0 | +99.9 | +25.6 | +79.2 |
| 2025-07-01 | TTD | The Trade Desk, Inc. | 9 | -48.3 | n.d. | -58.8 | n.d. |
| 2025-07-01 | UTHR | United Therapeutics Corporation | 9 | +67.5 | +88.1 | +57.0 | +67.3 |
| 2025-07-01 | 1177.HK | SBP GROUP | 8 | +20.1 | -11.8 | +14.3 | -6.2 |
| 2025-07-01 | 2628.HK | CHINA LIFE | 8 | +58.1 | +58.0 | +52.3 | +63.5 |
| 2025-07-01 | AMAT | Applied Materials, Inc. | 8 | +40.5 | n.d. | +30.1 | n.d. |
| 2025-07-01 | APAM | Artisan Partners Asset Manageme | 8 | -7.2 | n.d. | -17.6 | n.d. |
| 2025-07-01 | AXON | Axon Enterprise, Inc. | 8 | -26.8 | -23.4 | -37.2 | -44.1 |
| 2025-07-01 | BF-B | Brown Forman Inc | 8 | -5.0 | -4.1 | -15.4 | -24.8 |
| 2025-07-01 | DEME.BR | DEME GROUP | 8 | +10.5 | +37.6 | -2.4 | +10.8 |
| 2025-07-01 | HLNE | Hamilton Lane Incorporated | 8 | -6.5 | n.d. | -17.0 | n.d. |
| 2025-07-01 | HPE | Hewlett Packard Enterprise Comp | 8 | +18.6 | +119.1 | +8.2 | +98.3 |
| 2025-07-01 | KCR.HE | Konecranes Plc | 8 | +38.8 | +27.4 | +19.2 | -1.4 |
| 2025-07-01 | LULU | lululemon athletica inc. | 8 | -15.2 | -52.5 | -25.7 | -73.2 |
| 2025-07-01 | NDAQ | Nasdaq, Inc. | 8 | +9.8 | n.d. | -0.6 | n.d. |
| 2025-07-01 | NEM | Newmont Corporation | 8 | +70.8 | +60.3 | +60.3 | +39.5 |
| 2025-07-01 | NOW | ServiceNow, Inc. | 8 | -24.3 | -47.7 | -34.8 | -68.5 |
| 2025-07-01 | NVDA | NVIDIA Corporation | 8 | +21.7 | +29.1 | +11.2 | +8.3 |
| 2025-07-01 | OPHC | OptimumBank Holdings, Inc. | 8 | -5.3 | +31.0 | -15.8 | +10.2 |
| 2025-07-01 | ORCL | Oracle Corporation | 8 | -10.6 | -34.3 | -21.1 | -55.0 |
| 2025-07-01 | PINS | Pinterest, Inc. | 8 | -27.4 | -38.6 | -37.9 | -59.4 |
| 2025-07-01 | PYPL | PayPal Holdings, Inc. | 8 | -22.3 | -41.0 | -32.7 | -61.7 |
| 2025-07-01 | RCI | Rogers Communication, Inc. | 8 | +25.7 | +9.0 | +15.3 | -11.7 |
| 2025-07-01 | TDG | Transdigm Group Incorporated | 8 | -6.0 | -7.5 | -16.5 | -28.2 |
| 2025-07-01 | TEL | TE Connectivity plc | 8 | +34.1 | +19.6 | +23.6 | -1.2 |
| 2025-07-01 | V72.MU | Betsson AB                    N | 8 | -21.9 | n.d. | -32.4 | n.d. |
| 2025-07-01 | WPM | Wheaton Precious Metals Corp | 8 | +31.5 | n.d. | +21.0 | n.d. |
| 2025-07-01 | YOU | Clear Secure, Inc. | 8 | +26.3 | +102.4 | +15.8 | +81.7 |
| 2025-10-01 | UTHR | United Therapeutics Corporation | 11 | +30.1 | n.d. | +32.2 | n.d. |
| 2025-10-01 | 9022.T | CENTRAL JAPAN RAILWAY CO | 10 | -0.1 | n.d. | -20.7 | n.d. |
| 2025-10-01 | AEM | Agnico Eagle Mines Limited | 10 | +23.9 | n.d. | +26.0 | n.d. |
| 2025-10-01 | ALC | Alcon Inc. | 10 | +0.5 | n.d. | +2.6 | n.d. |
| 2025-10-01 | AU | AngloGold Ashanti PLC | 10 | +48.4 | n.d. | +50.4 | n.d. |
| 2025-10-01 | AXIA | AXIA Energia | 10 | +20.1 | n.d. | +22.2 | n.d. |
| 2025-10-01 | B | Barrick Mining Corporation | 10 | +26.8 | n.d. | +28.8 | n.d. |
| 2025-10-01 | FOX | Fox Corporation | 10 | -5.0 | n.d. | -3.0 | n.d. |
| 2025-10-01 | GMAB | Genmab A/S | 10 | -14.3 | n.d. | -12.3 | n.d. |
| 2025-10-01 | RHM.DE | RHEINMETALL AG                I | 10 | -20.9 | n.d. | -17.5 | n.d. |
| 2025-10-01 | SBS | Companhia de saneamento Basico  | 10 | +23.4 | n.d. | +25.4 | n.d. |
| 2025-10-01 | VICI | VICI Properties Inc. | 10 | -13.8 | n.d. | -11.8 | n.d. |
| 2025-10-01 | WELL | Welltower Inc. | 10 | +11.7 | n.d. | +13.7 | n.d. |
| 2025-10-01 | AGI | Alamos Gold Inc. | 9 | +30.1 | n.d. | +32.1 | n.d. |
| 2025-10-01 | AGI.TO | ALAMOS GOLD INC CLS A | 9 | +29.8 | n.d. | +20.4 | n.d. |
| 2025-10-01 | AOF.DE | ATOSS Software SE             I | 9 | -31.1 | n.d. | -27.7 | n.d. |
| 2025-10-01 | APH | Amphenol Corporation | 9 | +2.8 | n.d. | +4.9 | n.d. |
| 2025-10-01 | ARCAD.AS | ARCADIS | 9 | -39.3 | n.d. | -42.3 | n.d. |
| 2025-10-01 | AZN.L | ASTRAZENECA PLC ORD SHS $0.25 | 9 | +23.0 | n.d. | +13.3 | n.d. |
| 2025-10-01 | CWC.DE | CEWE Stiftung & Co. KGaA      I | 9 | -9.2 | n.d. | -5.9 | n.d. |
| 2025-10-01 | DEME.BR | DEME GROUP | 9 | +49.6 | n.d. | +43.2 | n.d. |
| 2025-10-01 | FDS | FactSet Research Systems Inc. | 9 | -20.9 | n.d. | -18.8 | n.d. |
| 2025-10-01 | FER | Ferrovial SE | 9 | +15.7 | n.d. | +17.7 | n.d. |
| 2025-10-01 | FUTU | Futu Holdings Limited | 9 | -20.6 | n.d. | -18.6 | n.d. |
| 2025-10-01 | GTT.PA | GAZTRANSPORT & TECHNIGAZ | 9 | +33.3 | n.d. | +33.1 | n.d. |
| 2025-10-01 | HMY | Harmony Gold Mining Company Lim | 9 | -12.4 | n.d. | -10.3 | n.d. |
| 2025-10-01 | IBKR | Interactive Brokers Group, Inc. | 9 | -1.0 | n.d. | +1.0 | n.d. |
| 2025-10-01 | IDCC | InterDigital, Inc. | 9 | -11.7 | n.d. | -9.7 | n.d. |
| 2025-10-01 | INSW | International Seaways, Inc. | 9 | +64.2 | n.d. | +66.2 | n.d. |
| 2025-10-01 | KGC | Kinross Gold Corporation | 9 | +26.0 | n.d. | +28.0 | n.d. |
| 2025-10-01 | MNSO | MINISO Group Holding Limited | 9 | -27.5 | n.d. | -25.4 | n.d. |
| 2025-10-01 | MPWR | Monolithic Power Systems, Inc. | 9 | +22.7 | n.d. | +24.7 | n.d. |
| 2025-10-01 | NBIX | Neurocrine Biosciences, Inc. | 9 | -3.2 | n.d. | -1.2 | n.d. |
| 2025-10-01 | NEM | Newmont Corporation | 9 | +33.0 | n.d. | +35.1 | n.d. |
| 2025-10-01 | NOVO-B.CO | Novo Nordisk B A/S | 9 | -34.0 | n.d. | -36.2 | n.d. |
| 2025-10-01 | OGC | OceanaGold Corporation | 9 | +48.7 | n.d. | +50.8 | n.d. |
| 2025-10-01 | REY.MI | REPLY | 9 | -33.8 | n.d. | -39.9 | n.d. |
| 2025-10-01 | TSM | Taiwan Semiconductor Manufactur | 9 | +19.0 | n.d. | +21.1 | n.d. |
| 2025-10-01 | UHS | Universal Health Services, Inc. | 9 | -12.4 | n.d. | -10.4 | n.d. |
| 2025-10-01 | WPM | Wheaton Precious Metals Corp | 9 | +22.9 | n.d. | +24.9 | n.d. |
| 2025-10-01 | 0027.HK | GALAXY ENT | 8 | -16.2 | n.d. | -8.9 | n.d. |
| 2025-10-01 | 9988.HK | BABA-W | 8 | -35.3 | n.d. | -28.0 | n.d. |
| 2025-10-01 | ACN | Accenture plc | 8 | -19.1 | n.d. | -17.1 | n.d. |
| 2025-10-01 | ACS.MC | ACS,ACTIVIDADES DE CONSTRUCCION | 8 | +59.4 | n.d. | +46.2 | n.d. |
| 2025-10-01 | AVGO | Broadcom Inc. | 8 | -5.6 | n.d. | -3.6 | n.d. |
| 2025-10-01 | BRC | Brady Corporation | 8 | +6.3 | n.d. | +8.4 | n.d. |
| 2025-10-01 | CALM | Cal-Maine Foods, Inc. | 8 | -8.1 | n.d. | -6.1 | n.d. |
| 2025-10-01 | CVE | Cenovus Energy Inc | 8 | +55.1 | n.d. | +57.1 | n.d. |
| 2025-10-01 | DCI | Donaldson Company, Inc. | 8 | +5.5 | n.d. | +7.5 | n.d. |
| 2025-10-01 | DRE2.F | DEUTSCHE REAL ESTATE AG       I | 8 | +91.0 | n.d. | +93.1 | n.d. |
| 2025-10-01 | EZPW | EZCORP, Inc. | 8 | +43.5 | n.d. | +45.5 | n.d. |
| 2025-10-01 | GATX | GATX Corporation | 8 | -0.9 | n.d. | +1.1 | n.d. |
| 2025-10-01 | HLNE | Hamilton Lane Incorporated | 8 | -24.0 | n.d. | -21.9 | n.d. |
| 2025-10-01 | INFY | Infosys Limited | 8 | -17.5 | n.d. | -15.5 | n.d. |
| 2025-10-01 | IRDM | Iridium Communications Inc | 8 | +64.0 | n.d. | +66.0 | n.d. |
| 2025-10-01 | IT | Gartner, Inc. | 8 | -38.3 | n.d. | -36.3 | n.d. |
| 2025-10-01 | KDP | Keurig Dr Pepper Inc. | 8 | +2.1 | n.d. | +4.1 | n.d. |
| 2025-10-01 | KLAC | KLA Corporation | 8 | +35.0 | n.d. | +37.1 | n.d. |
| 2025-10-01 | LRCX | Lam Research Corporation | 8 | +55.9 | n.d. | +57.9 | n.d. |
| 2025-10-01 | LTR.SG | Loews Corp. | 8 | +7.7 | n.d. | +9.7 | n.d. |
| 2025-10-01 | LULU | lululemon athletica inc. | 8 | -10.6 | n.d. | -8.5 | n.d. |
| 2025-10-01 | MC.PA | LVMH | 8 | -12.5 | n.d. | -12.7 | n.d. |
| 2025-10-01 | MO | Altria Group, Inc. | 8 | +2.3 | n.d. | +4.3 | n.d. |
| 2025-10-01 | MORN | Morningstar, Inc. | 8 | -24.9 | n.d. | -22.9 | n.d. |
| 2025-10-01 | NVDA | NVIDIA Corporation | 8 | -6.1 | n.d. | -4.1 | n.d. |
| 2025-10-01 | SII | Sprott Inc. | 8 | +78.7 | n.d. | +80.7 | n.d. |
| 2025-10-01 | TEL | TE Connectivity plc | 8 | -4.2 | n.d. | -2.2 | n.d. |
| 2025-10-01 | TFPM | Triple Flag Precious Metals Cor | 8 | +19.8 | n.d. | +21.8 | n.d. |
| 2025-10-01 | TTD | The Trade Desk, Inc. | 8 | -55.4 | n.d. | -53.4 | n.d. |
| 2026-01-02 | AU | AngloGold Ashanti PLC | 11 | +1.8 | n.d. | -7.3 | n.d. |
| 2026-01-02 | ACN | Accenture plc | 10 | -46.7 | n.d. | -55.8 | n.d. |
| 2026-01-02 | AEM | Agnico Eagle Mines Limited | 10 | -9.3 | n.d. | -18.5 | n.d. |
| 2026-01-02 | TW | Tradeweb Markets Inc. | 10 | -3.0 | n.d. | -12.1 | n.d. |
| 2026-01-02 | UTHR | United Therapeutics Corporation | 10 | +11.9 | n.d. | +2.8 | n.d. |
| 2026-01-02 | VICI | VICI Properties Inc. | 10 | -0.2 | n.d. | -9.3 | n.d. |
| 2026-01-02 | 4507.T | SHIONOGI & CO | 9 | +1.2 | n.d. | -31.4 | n.d. |
| 2026-01-02 | 8725.T | MS&AD INS GP HLDGS | 9 | +20.3 | n.d. | -12.3 | n.d. |
| 2026-01-02 | AGI | Alamos Gold Inc. | 9 | -17.6 | n.d. | -26.7 | n.d. |
| 2026-01-02 | CRM | Salesforce, Inc. | 9 | -34.2 | n.d. | -43.3 | n.d. |
| 2026-01-02 | GMAB | Genmab A/S | 9 | -10.4 | n.d. | -19.5 | n.d. |
| 2026-01-02 | GTT.PA | GAZTRANSPORT & TECHNIGAZ | 9 | +23.0 | n.d. | +19.6 | n.d. |
| 2026-01-02 | INFY | Infosys Limited | 9 | -37.2 | n.d. | -46.3 | n.d. |
| 2026-01-02 | KGC | Kinross Gold Corporation | 9 | -12.5 | n.d. | -21.6 | n.d. |
| 2026-01-02 | MPWR | Monolithic Power Systems, Inc. | 9 | +38.1 | n.d. | +29.0 | n.d. |
| 2026-01-02 | NEM | Newmont Corporation | 9 | -3.7 | n.d. | -12.8 | n.d. |
| 2026-01-02 | NOVN.SW | NOVARTIS N | 9 | +21.5 | n.d. | +13.1 | n.d. |
| 2026-01-02 | NOVO-B.CO | Novo Nordisk B A/S | 9 | -2.5 | n.d. | -6.2 | n.d. |
| 2026-01-02 | ZTS | Zoetis Inc. | 9 | -40.1 | n.d. | -49.2 | n.d. |
| 2026-01-02 | 2318.HK | PING AN | 8 | -21.5 | n.d. | -9.0 | n.d. |
| 2026-01-02 | ACS.MC | ACS,ACTIVIDADES DE CONSTRUCCION | 8 | +45.4 | n.d. | +33.0 | n.d. |
| 2026-01-02 | AGI.TO | ALAMOS GOLD INC CLS A | 8 | -14.8 | n.d. | -24.4 | n.d. |
| 2026-01-02 | AKAM | Akamai Technologies, Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | ALEX | Alexander & Baldwin, Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | AVGO | Broadcom Inc. | 8 | +4.1 | n.d. | -5.0 | n.d. |
| 2026-01-02 | BMRN | BioMarin Pharmaceutical Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | CALM | Cal-Maine Foods, Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | CX | Cemex, S.A.B. de C.V. Sponsored | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | EZPW | EZCORP, Inc. | 8 | +77.4 | n.d. | +68.3 | n.d. |
| 2026-01-02 | FFIV | F5, Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | HLNE | Hamilton Lane Incorporated | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | IIPR | Innovative Industrial Propertie | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | JEN.DE | JENOPTIK AG                   N | 8 | +116.4 | n.d. | +112.2 | n.d. |
| 2026-01-02 | KCR.HE | Konecranes Plc | 8 | -4.5 | n.d. | -12.1 | n.d. |
| 2026-01-02 | KVUE | Kenvue Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | LRCX | Lam Research Corporation | 8 | +90.3 | n.d. | +81.1 | n.d. |
| 2026-01-02 | LULU | lululemon athletica inc. | 8 | -43.8 | n.d. | -52.9 | n.d. |
| 2026-01-02 | N9B.MU | BANDAI NAMCO Holdings Inc.    R | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | NDAQ | Nasdaq, Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | NOW | ServiceNow, Inc. | 8 | -27.9 | n.d. | -37.0 | n.d. |
| 2026-01-02 | OPHC | OptimumBank Holdings, Inc. | 8 | +35.1 | n.d. | +26.0 | n.d. |
| 2026-01-02 | PSN | Parsons Corporation | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | SII | Sprott Inc. | 8 | +14.2 | n.d. | +5.1 | n.d. |
| 2026-01-02 | SKWD | Skyward Specialty Insurance Gro | 8 | +26.3 | n.d. | +17.2 | n.d. |
| 2026-01-02 | TFPM | Triple Flag Precious Metals Cor | 8 | -3.6 | n.d. | -12.7 | n.d. |
| 2026-01-02 | TTD | The Trade Desk, Inc. | 8 | n.d. | n.d. | n.d. | n.d. |
| 2026-01-02 | UQA.VI | UNIQA Insurance Group AG | 8 | +20.4 | n.d. | -1.0 | n.d. |
| 2026-01-02 | WING | Wingstop Inc. | 8 | -30.4 | n.d. | -39.5 | n.d. |
| 2026-01-02 | YOU | Clear Secure, Inc. | 8 | +58.1 | n.d. | +49.0 | n.d. |

## Filtres séparés : sélections (T >= 2025-04-01)

`ecart_indice_*` : rendement moins indice local. `dates_positives` : dates où la médiane de la sélection bat l'indice. `med_sans_materiaux` : médiane hors secteur Basic Materials (mines d'or). `p_hasard` : part des tirages au hasard de même taille, date par date, qui font au moins aussi bien.

| selection | horizon | titres | par_date | perf_med_% | ecart_indice_moy_pts | ecart_indice_med_pts | battent_indice_% | dates_positives | med_sans_materiaux_pts | p_hasard |
|---|---|---|---|---|---|---|---|---|---|---|
| Big Growth G>=3 (seuil Big_Growth_scan) | 6 mois | 828 | 207 | +8.0 | +6.7 | -1.6 | +48.3 | 1/4 | -3.6 | 0.601 |
| Big Growth G>=4 | 6 mois | 97 | 24 | +12.4 | +8.3 | +3.7 | +54.6 | 3/4 | -0.4 | 0.168 |
| Big Growth G=5 | 6 mois | 3 | 3 | -7.6 | -5.6 | -5.6 | +33.3 | 0/1 | -5.6 | 0.652 |
| Sichere S>=5 (seuil Pure Safe / Dual) | 6 mois | 1609 | 402 | +5.0 | +1.1 | -2.8 | +45.1 | 1/4 | -3.4 | 0.959 |
| Sichere S>=6 | 6 mois | 718 | 180 | +3.9 | -0.8 | -3.6 | +43.6 | 0/4 | -4.5 | 0.986 |
| Sichere S=7 | 6 mois | 187 | 47 | -0.1 | -3.9 | -6.2 | +38.5 | 0/4 | -7.5 | 0.995 |
| Dual Champion (G>=3 et S>=5) | 6 mois | 233 | 58 | +12.3 | +7.6 | +2.6 | +53.6 | 3/4 | -1.0 | 0.070 |
| Dual Champion* (Dual + G3 + G4 + S4) | 6 mois | 64 | 16 | +13.8 | +10.8 | +5.9 | +57.8 | 3/4 | +1.3 | 0.177 |
| Dual sans etoile | 6 mois | 169 | 42 | +11.9 | +6.4 | +1.8 | +52.1 | 2/4 | -2.0 | 0.295 |
| Univers | 6 mois | 7393 | 1848 | +5.5 | +4.2 | -1.9 | +47.4 | 2/4 | -2.7 | n.d. |
| Big Growth G>=3 (seuil Big_Growth_scan) | 12 mois | 361 | 180 | +26.3 | +22.3 | +7.6 | +57.1 | 2/2 | +4.0 | 0.003 |
| Big Growth G>=4 | 12 mois | 47 | 24 | +26.8 | +34.0 | +11.5 | +66.0 | 2/2 | +5.7 | 0.082 |
| Big Growth G=5 | 12 mois | 0 | 0 | n.d. | n.d. | n.d. | n.d. | 0/0 | n.d. | n.d. |
| Sichere S>=5 (seuil Pure Safe / Dual) | 12 mois | 632 | 316 | +10.5 | +3.6 | -5.8 | +44.6 | 0/2 | -6.7 | 0.996 |
| Sichere S>=6 | 12 mois | 295 | 148 | +7.7 | +0.1 | -8.3 | +44.4 | 0/2 | -12.4 | 0.993 |
| Sichere S=7 | 12 mois | 93 | 46 | +3.2 | -7.1 | -15.6 | +37.6 | 0/2 | -22.0 | 0.997 |
| Dual Champion (G>=3 et S>=5) | 12 mois | 120 | 60 | +30.1 | +22.4 | +11.6 | +67.5 | 2/2 | +8.3 | 0.007 |
| Dual Champion* (Dual + G3 + G4 + S4) | 12 mois | 35 | 18 | +29.0 | +34.9 | +12.3 | +77.1 | 2/2 | +8.7 | 0.028 |
| Dual sans etoile | 12 mois | 85 | 42 | +30.8 | +17.2 | +10.8 | +63.5 | 2/2 | +8.3 | 0.016 |
| Univers | 12 mois | 2594 | 1297 | +17.7 | +16.4 | +0.0 | +50.0 | 1/2 | -2.2 | n.d. |

## Big Growth seul : par score G sur 5 (T >= 2025-04-01)

| score_G | horizon | titres | dates | perf_moy_% | perf_med_% | ecart_med_univers_pts | ecart_indice_pts | battent_indice_% | gagnants_% |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 mois | 442 | 2 | +36.4 | +19.6 | +18.9 | +18.0 | +51.8 | +68.3 |
| 1 | 12 mois | 991 | 2 | +31.8 | +16.4 | +14.2 | +12.9 | +47.2 | +63.9 |
| 2 | 12 mois | 810 | 2 | +35.4 | +16.2 | +17.9 | +17.2 | +48.6 | +63.8 |
| 3 | 12 mois | 315 | 2 | +38.9 | +26.2 | +21.5 | +20.5 | +55.6 | +66.7 |
| 4 | 12 mois | 47 | 2 | +50.0 | +26.8 | +32.7 | +34.0 | +66.0 | +70.2 |
| 0 | 6 mois | 1382 | 4 | +13.4 | +6.8 | +7.6 | +5.9 | +50.4 | +63.7 |
| 1 | 6 mois | 2952 | 4 | +10.3 | +5.3 | +4.8 | +3.0 | +45.8 | +58.0 |
| 2 | 6 mois | 2257 | 4 | +11.3 | +4.6 | +5.7 | +3.8 | +46.8 | +55.8 |
| 3 | 6 mois | 732 | 4 | +14.7 | +6.4 | +8.6 | +6.5 | +47.4 | +56.8 |
| 4 | 6 mois | 94 | 4 | +17.2 | +12.7 | +10.4 | +8.7 | +55.3 | +67.0 |
| 5 | 6 mois | 3 | 1 | -7.7 | -7.6 | -7.6 | -5.6 | +33.3 | +33.3 |

## Sichere seul : par score S sur 7 (T >= 2025-04-01)

| score_S | horizon | titres | dates | perf_moy_% | perf_med_% | ecart_med_univers_pts | ecart_indice_pts | battent_indice_% | gagnants_% |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 mois | 244 | 2 | +45.5 | +8.6 | +27.9 | +27.1 | +46.7 | +56.1 |
| 1 | 12 mois | 499 | 2 | +44.9 | +22.2 | +27.3 | +25.7 | +51.5 | +60.3 |
| 2 | 12 mois | 406 | 2 | +36.6 | +18.0 | +19.1 | +18.3 | +49.3 | +68.2 |
| 3 | 12 mois | 402 | 2 | +41.3 | +24.7 | +23.7 | +22.4 | +57.2 | +75.1 |
| 4 | 12 mois | 420 | 2 | +29.2 | +18.3 | +11.7 | +11.2 | +51.0 | +67.9 |
| 5 | 12 mois | 339 | 2 | +24.7 | +11.2 | +7.3 | +6.6 | +44.5 | +66.4 |
| 6 | 12 mois | 202 | 2 | +21.2 | +11.8 | +3.7 | +3.4 | +47.5 | +58.9 |
| 7 | 12 mois | 93 | 2 | +10.6 | +3.2 | -6.7 | -7.1 | +37.6 | +52.7 |
| 0 | 6 mois | 737 | 4 | +14.6 | -0.9 | +9.0 | +7.1 | +45.3 | +49.5 |
| 1 | 6 mois | 1493 | 4 | +13.5 | +3.4 | +8.0 | +6.1 | +47.1 | +54.3 |
| 2 | 6 mois | 1200 | 4 | +12.1 | +5.6 | +6.4 | +4.6 | +47.5 | +58.3 |
| 3 | 6 mois | 1239 | 4 | +13.6 | +8.9 | +8.2 | +6.0 | +51.9 | +66.7 |
| 4 | 6 mois | 1138 | 4 | +9.1 | +5.5 | +3.4 | +1.7 | +46.6 | +60.2 |
| 5 | 6 mois | 895 | 4 | +10.0 | +5.9 | +4.1 | +2.6 | +46.0 | +60.8 |
| 6 | 6 mois | 531 | 4 | +7.9 | +4.7 | +2.1 | +0.4 | +45.4 | +58.0 |
| 7 | 6 mois | 187 | 4 | +4.5 | -0.1 | -2.2 | -3.9 | +38.5 | +49.7 |

## Seuils des scanners, date par date (écart médian à l'indice)

| date_T | horizon | Big Growth G>=3 n | Big Growth G>=3 med_pts | Sichere S>=5 n | Sichere S>=5 med_pts | Dual Champion n | Dual Champion med_pts |
|---|---|---|---|---|---|---|---|
| 2025-04-01 | 12 mois | 197 | +10.0 | 326 | -1.7 | 78 | +18.4 |
| 2025-07-01 | 12 mois | 164 | +4.7 | 306 | -8.2 | 42 | +8.8 |
| 2025-04-01 | 6 mois | 198 | +2.1 | 327 | -7.2 | 78 | +5.1 |
| 2025-07-01 | 6 mois | 256 | -0.4 | 425 | -6.3 | 50 | +8.6 |
| 2025-10-01 | 6 mois | 254 | -3.5 | 524 | +2.3 | 69 | +4.1 |
| 2026-01-02 | 6 mois | 120 | -6.7 | 333 | -1.4 | 36 | -10.7 |

## Annexe : par profil, dates antérieures à 2025-04-01

| profil | horizon | titres | dates | perf_moy_% | perf_med_% | ecart_med_univers_pts | ecart_indice_pts | battent_indice_% | gagnants_% |
|---|---|---|---|---|---|---|---|---|---|
| 💎 Dual Champion* | 12 mois | 2 | 1 | -1.6 | -1.6 | -1.3 | -38.7 | +0.0 | +50.0 |
| 💎 Dual Champion | 12 mois | 19 | 3 | +19.1 | +20.7 | +10.2 | +2.9 | +47.4 | +63.2 |
| 🛡️  Pure Safe | 12 mois | 323 | 4 | +18.4 | +12.7 | +9.8 | +0.9 | +42.7 | +71.2 |
| 🚀 Pure Growth | 12 mois | 40 | 4 | +7.0 | -11.1 | +1.2 | -4.9 | +45.0 | +45.0 |
| ⚖️  Balanced | 12 mois | 170 | 4 | +15.9 | +6.3 | +9.1 | +0.1 | +39.4 | +57.6 |
| ⚪ Below | 12 mois | 4157 | 4 | +16.8 | +5.9 | +10.0 | +2.6 | +41.3 | +57.1 |
| 💎 Dual Champion* | 6 mois | 2 | 1 | +28.8 | +28.8 | +25.5 | +3.9 | +50.0 | +100.0 |
| 💎 Dual Champion | 6 mois | 19 | 3 | +5.1 | +8.3 | +4.9 | +1.3 | +57.9 | +63.2 |
| 🛡️  Pure Safe | 6 mois | 323 | 4 | +7.9 | +5.4 | +6.9 | +2.3 | +54.5 | +65.6 |
| 🚀 Pure Growth | 6 mois | 42 | 4 | -1.4 | -5.6 | -3.7 | -7.5 | +33.3 | +33.3 |
| ⚖️  Balanced | 6 mois | 171 | 4 | +4.4 | +2.7 | +2.8 | -1.8 | +42.7 | +53.8 |
| ⚪ Below | 6 mois | 4189 | 4 | +3.1 | +1.0 | +1.7 | -2.4 | +42.4 | +51.2 |

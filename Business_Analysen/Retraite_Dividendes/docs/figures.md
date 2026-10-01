# Inventaire figures et macros (tache 10, infrastructure)

Ce document sert aux passages "chapitres" (agent `latex-writer`) qui suivent la tache 10.
Il n'est pas lu par le build (`build.sh` ne le consomme pas); c'est une carte de reference.

## Contrat des macros de figure (`preamble.tex`)

```latex
\linienfigur{fichier}[cles pgfplots]{colonne x}{liste colonnes y}{liste legende}{caption+label}
\balkenfigur{fichier}[cles pgfplots]{colonne x (etiquettes)}{colonne y}{caption+label}
```

- `fichier` : nom du CSV dans `data/`, **sans** extension ni prefixe `data/` (ex. `fig_gehalt`, pas
  `data/fig_gehalt.csv`).
- `[cles pgfplots]` (**correctif infrastructure, tache 10, second passage**) : argument
  OPTIONNEL, place juste apres `{fichier}` (jamais avant : voir la remarque de compatibilite
  ci-dessous). Absent, l'appel a 5 (`\linienfigur`) ou 4 (`\balkenfigur`) arguments obligatoires
  compile exactement comme avant ce correctif (retro-compatible). Present, c'est une liste de
  cles pgfplots standard separees par des virgules, appliquee APRES le style `retraite` et apres
  les valeurs par defaut de la macro - une cle repetee ici l'emporte. C'est la voie normale pour
  donner un titre et une unite aux axes, ex. :
  ```latex
  \linienfigur{fig_ziel_nominal}[xlabel={Ann\'ee}, ylabel={Cible mensuelle (euros nominaux)}]%
    {jahr}{inflation_1,inflation_2,inflation_3}{L\'egende 1,L\'egende 2,L\'egende 3}%
    {Texte de la legende.\label{fig:exemple}}
  ```
  Remplace le contournement `{\pgfplotsset{every axis post/.append style={...}}} \linienfigur{...}}`
  utilise par les chapitres 2-3 avant ce correctif (converti vers la nouvelle syntaxe, texte
  inchange). `{fichier}` reste TOUJOURS le premier argument, avant `[cles pgfplots]` (et non
  apres, malgre la convention LaTeX habituelle du `\sqrt[n]{x}`) : cela le garde sur la MEME
  ligne source que le nom de macro, y compris quand le bloc d'options s'etend sur plusieurs
  lignes, ce dont depend `scripts/check_literals.py` (son exemption du nom de CSV, qui contient
  souvent un nombre, ex. `fig_x_250`, fonctionne ligne par ligne).
- `\linienfigur` : une ou plusieurs colonnes y, listees comme `{brutto,netto}` (pas d'espace apres
  la virgule), avec la legende dans le MEME ORDRE `{Brut,Net}`.
- `\balkenfigur` : NE PEUT PAS utiliser une cle "symbolic x coords" native (elle n'existe pas dans
  pgfplots 1.18.1, installee sur cette machine - verifie par lecture directe de
  `pgfplots.code.tex`, absente de tout resultat de recherche). La macro utilise donc en interne le
  repli prevu par le brief (`\pgfplotstableread` + `xticklabels from table` + `x expr=\coordindex`).
  Les etiquettes de graduation sont posees sur la colonne `<colonne x>_disp`, PAS `<colonne x>`
  directement (**correctif infrastructure, tache 10, second passage**) : `scripts/rechnung_retraite.py`
  (`schreiben()`) ecrit automatiquement une colonne `<col>_disp` echappee pour LaTeX pour CHAQUE
  colonne texte d'un CSV de figure, afin qu'une valeur de colonne contenant un underscore
  (ex. `brut_necessaire`, `capital_10000` - catcode 8 hors mode mathematique, ferait planter
  LaTeX telle quelle) compile TOUJOURS, meme sans intervention de la redaction. Pour des
  etiquettes plus lisibles que la colonne brute echappee (ex. "Dividende brut" plutot que
  "brut\_necessaire"), passer une cle `xticklabels={...}` explicite dans `[cles pgfplots]` (dans
  l'ORDRE DES LIGNES du CSV) : elle remplace la valeur par defaut, c'est la voie normale pour des
  etiquettes en prose (voir `sections/02-depart.tex`, figure de la cascade, pour un exemple reel).
- `caption+label` : texte de `\caption{}` suivi de `\label{fig:trois-mots}` (convention
  `code-style.md`), ex. `Salaire brut et net simule au fil des annees.\label{fig:salaire}`.
- Les deux macros compilent (verifie par un test de compilation reel sur `fig_gehalt.csv` et
  `fig_laender.csv`, tache 10; voir `task-10-report.md` pour le detail du bogue pgfplots+babel
  rencontre et corrige dans `preamble.tex`, et sa section "Correctif d'infrastructure" pour
  l'argument optionnel et la colonne `_disp`).
- Chaque figure doit etre suivie, dans la prose, d'une phrase "ce que montre ce graphique" (regle
  du brief); ce fichier n'inclut pas cette phrase, seulement la matiere pour l'ecrire.

## Contrat des macros de citation (`preamble.tex`)

- `\QH{cle}` : note de bas de page complete pour une entree de `data/quellen.json` (titre,
  reference/page si presente, lien, date). Ex. `\QH{kv_satz_ermaessigt}`. `\QHtexte{cle}` donne le
  texte seul (sans `\footnote`), pour un ancrage manuel `\footnote{\label{fn:x}\QHtexte{cle}}` +
  `\QR{x}` si la meme source revient sur la meme page imprimee. **Correctif infrastructure (tache
  10, second passage)** : la note affiche un lien COURT (le domaine, ex. "gesetze-im-internet.de")
  via `\href`, pas l'URL complete via `\url` - footmisc en mode `[para]` compose chaque note dans
  un `\hbox` non brisable ou aucun point de coupure de `\url` n'est insere (Overfull `\hbox` de
  152 a 461 pt constate sur les notes a URL longue avant ce correctif, meme avec `xurl`). L'URL
  complete reste la cible cliquable de la note ET s'affiche en clair dans l'annexe "Sources"
  (`data/quellen_annexe.tex`, pas composee dans ce `\hbox`). Rien a changer cote redaction : le
  correctif est dans `quellen_tex()`/`presse_notes_tex()` (`scripts/rechnung_retraite.py`).
- `\QP{id}` : meme mecanique pour `data/presse.json`, par id de SOURCE (pas par citation; plusieurs
  lignes de `presse.json` peuvent partager le meme id). Ne reprend jamais la citation elle-meme
  (`zitat`/`aussage`), qui reste dans le tableau `data/tab_presse.tex` (voir plus bas, chapitre 9).
  Meme correctif lien court/`\href` que `\QH` ci-dessus.
- `\Q{Dokument}{Seite}` / `\QL{label}{Dokument}{Seite}` / `\QR{label}` / `\QW{Titre}{cle}{Date}` :
  mecanique generique heritee de Neste, gardee pour une citation ponctuelle hors
  `data/quellen.json`/`data/presse.json` (ex. une page precise du classeur Shiller dans `refs/`).
  `\Q`/`\QL` exigent un `\kurz<Dokument>` defini par l'appelant (aucun predefini ici).
- Annexe "Sources" (chapitre 99) : `\input{data/quellen_annexe}` (sources officielles/secondaires,
  labels `\label{src:<cle>}`) et `\input{data/presse_annexe}` (presse, labels
  `\label{presse:<id>}`), tous deux generes par `scripts/rechnung_retraite.py`
  (`quellen_tex()`/`presse_notes_tex()`), a regenerer via `./build.sh` si `data/quellen.json` ou
  `data/presse.json` changent.
- Presse (chapitre 9) : `\tabellenkoerper{data/tab_presse}` dans un tableau (colonnes source,
  position, theme, citation courte) - deja genere, ne pas retaper les citations.
- Litterature academique (chapitre 9) : `\cite{...}` sur `data/literatur.bib`, **vide cette nuit**
  (voir `LUECKEN.md`, tache 8); le dire en une phrase plutot que d'inventer une reference.

## Figures (`data/fig_*.csv`, 28 fichiers)

### Chapitre 2-3 (point de depart, fiscalite) - `sections/02-depart.tex`, `03-fiscalite.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_ziel_nominal` | `jahr, inflation_1, inflation_2, inflation_3` | 3 500 EUR de 2026 convertis en euros NOMINAUX annee par annee, sous 3 hypotheses d'inflation | `\linienfigur{fig_ziel_nominal}{jahr}{inflation_1,inflation_2,inflation_3}{...}{...}` |
| `fig_kaskade` | `posten, betrag` | Cascade du dividende BRUT au montant disponible (postes: impot, soli, assurance maladie, etc.) | `\balkenfigur{fig_kaskade}{posten}{betrag}{...}` |
| `fig_steuer_100` | `art, netto` | "100 EUR de dividende brut, combien pour toi ?" par pays/enveloppe (DE, US avec/sans W-8BEN, CH, ETF) | `\balkenfigur{fig_steuer_100}{art}{netto}{...}` |
| `fig_steuer_kumuliert` | `jahr, maison, etf_thes, etf_aus` | Cout fiscal CUMULE sur 18 ans: actions en direct vs ETF thesaurisant vs ETF distribuant | `\linienfigur{fig_steuer_kumuliert}{jahr}{maison,etf_thes,etf_aus}{...}{...}` |

### Chapitre 4 (capital requis) - `sections/04-capital.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_kapital_rendite` | `rendite, maison50, maison70, maison100` | Capital requis selon le rendement de marche retenu, pour les 3 repartitions maison | `\linienfigur{fig_kapital_rendite}{rendite}{maison50,maison70,maison100}{...}{...}` |
| `fig_div_vs_4` | `strategie, kapital` | Capital requis: dividendes seuls vs regle des 4 % (retrait total) | `\balkenfigur{fig_div_vs_4}{strategie}{kapital}{...}` |

### Chapitre 5 (accumulation) - `sections/05-accumulation.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_gehalt` | `jahr, brutto, netto` | Trajectoire du salaire brut/net simule (testee dans le smoke-test de la tache 10) | `\linienfigur{fig_gehalt}{jahr}{brutto,netto}{Brut,Net}{...}` |
| `fig_jahre_sparquote` | `quote, maison50, maison70, maison100, etf_ref` | Annees jusqu'a la retraite selon le taux d'epargne (10/20/30/50 %), par strategie | `\linienfigur{fig_jahre_sparquote}{quote}{maison50,maison70,maison100,etf_ref}{...}{...}` |
| `fig_kapital_zeit` | `jahr, q10, q20, q30, q50, ziel_kapital` | Capital dans le temps selon le taux d'epargne, avec la ligne cible | `\linienfigur{fig_kapital_zeit}{jahr}{q10,q20,q30,q50,ziel_kapital}{...}{...}` |
| `fig_zinseszins` | `jahr, eingezahlt, gewinn` | Part versee vs part "interets composes" dans le capital final | `\linienfigur{fig_zinseszins}{jahr}{eingezahlt,gewinn}{...}{...}` |
| `fig_sparbedarf` | `alter, monatlich` | Epargne mensuelle necessaire selon l'age de depart vise (40/45/50 ans) | `\linienfigur{fig_sparbedarf}{alter}{monatlich}{...}{...}` |
| `fig_start_verzoegerung` | `verzoegerung_jahre, ziel_alter` | Effet d'un an d'avance ou de retard au demarrage sur l'age de depart | `\linienfigur{fig_start_verzoegerung}{verzoegerung_jahre}{ziel_alter}{...}{...}` |

### Chapitre 6 (portefeuille-exemple) - `sections/06-portefeuille.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_aufteilungen` | `strategie, kapital_xlv, netto_monat_xlv, ziel_alter` | Comparaison des 5 strategies a l'annee 45 de l'horizon (capital, revenu net mensuel, age cible) | `\balkenfigur{fig_aufteilungen}{strategie}{kapital_xlv}{...}` (+ une seconde figure pour `netto_monat_xlv`) |
| `fig_anzahl_titel` | `n, risiko` | Risque (ecart-type annualise) d'un portefeuille equipondere selon le nombre de titres tires | `\linienfigur{fig_anzahl_titel}{n}{risiko}{...}{...}` |
| `fig_sektoren` | `kategorie, anteil` | Repartition sectorielle du portefeuille-exemple | `\balkenfigur{fig_sektoren}{kategorie}{anteil}{...}` |
| `fig_laender` | `kategorie, anteil` | Repartition par pays du portefeuille-exemple (teste dans le smoke-test de la tache 10) | `\balkenfigur{fig_laender}{kategorie}{anteil}{...}` |
| `fig_rendite_wachstum` | `ticker, rendite, wachstum` | Rendement du dividende vs croissance du dividende, un point par titre retenu | nuage de points, hors `\linienfigur`/`\balkenfigur` (utiliser `\addplot[only marks]` directement) |
| `fig_div_beispiele` | `jahr, titre_hausse, titre_baisse` | Historique reel de deux dividendes: le titre en plus longue hausse continue vs le titre en plus forte baisse (tout l'univers de 142 tickers) | `\linienfigur{fig_div_beispiele}{jahr}{titre_hausse,titre_baisse}{...}{...}` |
| `data/tab_portefeuille.tex` | ticker, pays, secteur, rendement, rendement net, croissance, payout | Tableau des titres retenus (`\NombreTitres` = 23, pas 30: voir `LUECKEN.md`) | `\tabellenkoerper{data/tab_portefeuille}` dans un tableau |

### Chapitre 7 (risques) - `sections/07-risques.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_shiller_div` | `jahr, div_real` | Dividende reel du S&P 500 depuis 1871 (donnees Shiller) | `\linienfigur{fig_shiller_div}{jahr}{div_real}{...}{...}` |
| `fig_einbrueche` | `episode, rueckgang_prozent` | Baisses de dividende pendant les crises historiques (par episode nomme) | `\balkenfigur{fig_einbrueche}{episode}{rueckgang_prozent}{...}` |
| `fig_rueckspiel` | `start, jahre_bis_ziel, ueberlebt, ans_tenu, ans_echec, ans_non_juge` | Rejeu historique: annees jusqu'a l'objectif selon l'annee de depart, succes/echec. Les trois colonnes `ans_*` (chapitre 07) repetent `jahre_bis_ziel` pour une seule issue (vides ailleurs) et servent au nuage a trois classes | `\linienfigur{fig_rueckspiel}[... table/empty cells with={nan}, only marks ...]{start}{ans_tenu,ans_echec,ans_non_juge}{...}{...}` (voir `sections/07-risques.tex`) |
| `fig_mc_faecher` | `jahr, p10, p50, p90, seuil_mc, capital_requis` | Eventail Monte Carlo (percentiles 10/50/90) du capital dans le temps; `seuil_mc` = capital a partir duquel le Monte Carlo (SANS impot) juge l'objectif atteint, `capital_requis` = capital requis du chapitre 4 pour la repartition de reference (AVEC impot), deux constantes (chapitre 07) | `\linienfigur{fig_mc_faecher}{jahr}{p10,p50,p90,seuil_mc,capital_requis}{...}{...}` |
| `fig_mc_erfolg` | `div_schock, erfolg_30, erfolg_40, erfolg_50` | Probabilite de reussite selon un choc de croissance du dividende la 1re annee de retraite (voir `\HypotheseDivSchock`, PAS un rendement de depart), pour 3 horizons de retraite | `\linienfigur{fig_mc_erfolg}{div_schock}{erfolg_30,erfolg_40,erfolg_50}{...}{...}` |
| `fig_inflation_2022` | `jahr, indexiert, nicht_indexiert` | Effet de l'inflation 2022 sur un revenu indexe vs non indexe | `\linienfigur{fig_inflation_2022}{jahr}{indexiert,nicht_indexiert}{...}{...}` |

### Chapitre 8 (la retraite elle-meme) - `sections/08-retraite.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_rente_alter` | `ausstiegsalter, rente_monat` | Rente legale mensuelle BRUTE selon l'age d'arret de cotisation (points plafonnes au plafond de cotisation RV depuis le chapitre 08) | `\linienfigur{fig_rente_alter}{ausstiegsalter}{rente_monat}{...}{...}` |
| `fig_bruecke` | `alter, dividende, rente` | Revenu pendant la phase pont (dividendes seuls) jusqu'a la rente legale a 67 ans (vide si `\BrueckeJahre` = "non atteint") | `\linienfigur{fig_bruecke}{alter}{dividende,rente}{...}{...}` |
| `fig_puffer` | `puffer_jahre, erfolgsquote` | Effet d'une reserve de liquidites (en annees de depenses) sur le taux de reussite Monte Carlo (memes trajectoires que `\ErfolgsquoteBasis`/`\ErfolgsquoteMitPuffer`, chapitre 7: meme macro, jamais deux calculs) | `\linienfigur{fig_puffer}{puffer_jahre}{erfolgsquote}{...}{...}` |
| `fig_bruecke_anticipee` | `alter, dividende, rente, cible` | (chapitre 08) Pont pour un depart anticipe: repartition de reference au taux `BRUECKE_QUOTE_NOM` (50 %), dividendes nets de l'age d'objectif (`\AlterMaisonSiebzigFuenfzig`) a 90 ans, rente brute des 67 ans, cible constante | `\linienfigur{fig_bruecke_anticipee}[... const plot]{alter}{dividende,rente,cible}{...}{...}` |
| `fig_regle_quatre_histoire` | `start, fin_dreissig, ans_dreissig, fin_vierzig, ans_vierzig` | (chapitre 08) Test historique du retrait constant reel de `TAUX_RETRAIT` du capital initial (Shiller, 1re ligne exclue): capital reel final en multiple du capital initial (0 si epuise) et annees de retraits couverts, pour 30 et 40 ans; vide si la fenetre depasse la serie | `\linienfigur{fig_regle_quatre_histoire}[... only marks, empty cells with={nan}]{start}{fin_dreissig,fin_vierzig}{...}{...}` |

### Chapitre 10 (plan d'action) - `sections/10-plan.tex`

| CSV | Colonnes | Sens | Macro suggeree |
|---|---|---|---|
| `fig_zeitplan` | `jahr, ereignis, kapital` | Frise: annees ou des seuils de capital sont franchis (10k/50k/100k/250k/500k EUR) et annee-objectif | `\balkenfigur{fig_zeitplan}{ereignis}{kapital}{...}` ou une frise TikZ dediee |

## Macros de `data/kennzahlen.tex` (54, groupees par chapitre)

Valeurs au 30.09.2026 (regenerees par `./build.sh`; ne pas les recopier de memoire dans une
section, toujours passer par la macro).

### Chapitre 2-3 (marche/fiscalite, poses par `markt_basis()` + `kapitel_2_und_3()`)

| Macro | Valeur | Sens |
|---|---|---|
| `\RenditeReal` | 7,3 | Rendement reel de marche retenu (TCAC 1950-2022, PAS la mediane - voir `\RenditeReelleConvention`) |
| `\RenditeReelleConvention` | texte | Rappelle que c'est le TCAC (moyenne geometrique), pas la mediane |
| `\RenditeReelleMediane` | 12,1 | Mediane des rendements annuels reels 1950-2022, exposee a titre de comparaison seulement |
| `\RenditeDivMaison` | 2,8 | Rendement du dividende du portefeuille-exemple (mediane) |
| `\RenditeDivEtf` | 1,8 | Rendement de distribution retenu pour l'ETF monde (repli documente, voir `LUECKEN.md`) |
| `\InflationZiel` | 2,0 | Hypothese d'inflation a long terme |
| `\HypotheseUsFormulaire` | texte | Retenue americaine au taux W-8BEN (15 %), formulaire suppose deja depose |
| `\HypotheseUmschichtung` | texte | `EtfUmschichtung` modelisee comme un ETF distribuant des le debut (ecart au plan section 3) |
| `\ZielNominalXLIV` | 4\,999 | Cible en euros NOMINAUX a l'annee 2044 (XLIV en romain) |
| `\BruttoNoetig` | 76\,058 | Dividende brut annuel necessaire pour atteindre la cible nette |
| `\NettoDe` / `\NettoUs` / `\NettoUsSansFormulaire` / `\NettoCh` / `\NettoEtf` | 73,62 / 74,45 / 59,45 / 54,45 / 81,54 | Net disponible pour 100 EUR de dividende brut, par pays/enveloppe |
| `\SteuerDifferenzAchtzehn` | 7\,405 | Ecart de cout fiscal cumule sur 18 ans, actions en direct vs ETF |

### Chapitre 4 (capital requis, `kapitel_4()`)

| Macro | Valeur | Sens |
|---|---|---|
| `\KapitalMaisonSiebzig` | 2\,974\,126 | Capital requis, repartition maison 70 % |
| `\KapitalEtfReferenz` | 1\,562\,746 | Capital requis, reference 100 % ETF + regle des 4 % |
| `\KapitalDifferenzProzent` | 73,6 | Ecart en % entre maison 100 % et la reference ETF |

### Chapitre 5 (accumulation, `kapitel_5()`) - 20 macros `Alter<Strategie><Quote>`

Strategies : `MaisonFuenfzig` (50 %), `MaisonSiebzig` (70 %), `MaisonHundert` (100 %),
`EtfReferenz`, `EtfUmschichtung`. Quotes : `Zehn` (10 %), `Zwanzig` (20 %), `Dreissig` (30 %),
`Fuenfzig` (50 %). Valeur = age de depart simule, ou le texte "non atteint" si l'objectif n'est
jamais atteint dans l'horizon de 45 ans (a publier tel quel, resultat honnete, pas une erreur).
Ex. `\AlterMaisonSiebzigZwanzig` = non atteint (repartition maison 70 %, taux d'epargne 20 %).

### Chapitre 6 (portefeuille-exemple, `kapitel_6()`)

| Macro | Valeur | Sens |
|---|---|---|
| `\NombreTitres` | 23 | Titres retenus (23, pas 30: voir `LUECKEN.md`, methode illustree malgre tout) |
| `\BeispielHausse` | ticker | Titre a la plus longue serie de hausses continues du dividende |
| `\BeispielBaisse` | `ADS.DE` | Titre a la plus forte baisse de dividende (tout l'univers, 142 tickers) |

### Chapitre 7 (risques, `kapitel_7()`)

| Macro | Valeur | Sens |
|---|---|---|
| `\GroessterEinbruch` | 48,5 | Plus forte baisse de dividende reel observee (%) |
| `\GroessterEinbruchJahr` | 1920 | Annee de cette baisse |
| `\ErfolgsquoteBasis` | 63,5 | Taux de reussite Monte Carlo, scenario de reference (`\ReferenzStrategie` a `\ReferenzSparquote` % d'epargne, 40 ans de retraite, sans reserve, SANS impot: voir le tableau `tab:projection-contre-simulation` du chapitre 7) |
| `\ErfolgsquoteMitPuffer` | 89,2 | Meme scenario avec une reserve de liquidites de 2 ans |
| `\HypotheseDivSchock` | texte | Definit le sens de la colonne `div_schock` de `fig_mc_erfolg` (choc sur la croissance du dividende de la 1re annee de retraite, PAS un rendement de depart) |

Macros de redaction du chapitre 7 (`_macros_redaction_kapitel_7`, tache 10, chapitre 07), non detaillees ici: `\Shiller*` (bornes, croissance reelle du dividende, rendement de l'indice), `\CroissanceModeleRapport`, `\FenetreMarche*`, episodes de baisse (`\GroessterEinbruch*`, `\BaisseLongue*`, `\EinbruchRecent*`, `\EinbruchInflation*`, `\SeuilBaisseHistorique`, `\NombreBaisses`), rejeu (`\Rueckspiel*`), Monte Carlo (`\Mc*`, `\SeuilReussite`, `\ChocGrille*`, `\KapitalRequisReference`), reserve (`\Puffer*`), inflation (`\InflationPic*`, `\InflationSerie*`, `\Rente*`, `\PrixHausseCumulee`) et `\YannBrutApresPireBaisse`. Les ages et capitaux `\Mc*` sont ceux du modele SANS impot: ne jamais les reprendre comme ages ou capitaux de Yann.

### Chapitre 8 (la retraite elle-meme, `kapitel_8()`)

| Macro | Valeur | Sens |
|---|---|---|
| `\RenteBeiZielalter` | 2\,820 | Rente legale mensuelle a l'age cible (ou "non atteint") |
| `\BrueckeJahre` | 0 | Duree de la phase pont avant la rente legale (ou "non atteint") |
| `\KapitalYannBasis` | 3\,108\,376 | Capital de Yann (scenario de base) a l'annee-objectif |
| `\RevenuNetYannBasis` | 3\,618 | Revenu net mensuel de Yann a l'annee-objectif |
| `\AnneeYannBasis` | 2068 | Annee ou Yann atteint son objectif (scenario de base: 70 % maison, 20 % d'epargne, capital de depart "Mitte"). A UTILISER TELLE QUELLE dans l'encadre `\Yannbox{\AnneeYannBasis}{...}`, jamais retaper "2044" ou toute autre annee a la main |
| `\KapitalDepartMitte` | 6\,000 | Capital de depart retenu pour le scenario de base (milieu de la fourchette 2 000-10 000 EUR) |

Macros de redaction du chapitre 8 (`_macros_redaction_kapitel_8`, tache 10, chapitre 08), non detaillees ici: rente legale (`\Durchschnittsentgelt`, `\Rentenwert`, `\BbgRvJahr`, `\Points*`, `\AgePlafondRv`, `\AgeDebutCotisation`, `\RenteArret<Age>`, `\RenteDerniereAnnee`, `\RenteLegalePartCible`), pont anticipe (`\Bruecke*Fuenfzig`, `\RenteBrueckeFuenfzig`, `\RevenuNetBrueckeFuenfzig`, `\KapitalBrueckeFuenfzig`, `\RenteBruecke*`, `\BrueckeJahreAnticipee`), reserve (`\ErfolgsquotePuffer*`, `\PufferJahreMax`, `\PufferMontantMax`, `\PufferGain*`, lus dans fig_puffer) et test historique de la regle des 4 % (`\RetraitHist*<Duree>` pour Dreissig/Vierzig, `\RetraitHistReussitePrudent*`, comparaison aux memes departs que le rejeu du chapitre 7: `\RetraitHistMemesDeparts*`, `\RetraitHistLesDeuxTiennent`, `\RetraitHistSeul*Tient`, `\RetraitHistAucunNeTient`, `\RetraitHistAnnees*`). `\RenteBeiZielalter` est BRUT et, depuis le chapitre 08, plafonne au plafond de cotisation RV (2\,905 -> 2\,820).

### Chapitre 10 (plan d'action, `kapitel_10()`)

| Macro | Valeur | Sens |
|---|---|---|
| `\ZielJahrBasis` | 2068 | Annee-objectif du scenario 70 % maison, 20 % d'epargne (meme calcul que le resume, jamais recalcule deux fois - controle croise spec section 8) |

## Encadre "Yann"

`\Yannbox{\AnneeYannBasis}{texte}` : encadre "Yann en <annee>" (un par chapitre, texte
exclusivement en macros/annees). L'annee est TOUJOURS `\AnneeYannBasis` (ou une autre macro
d'annee de `data/kennzahlen.tex` si le chapitre a un scenario different), jamais un chiffre tape.

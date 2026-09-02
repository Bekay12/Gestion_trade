# Partie 12 / Ch.01 — Outils Selection Screeners Scanners
<!-- source: Academy_Germain_COMPLET.pdf p.322-328 -->

PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 1 —
Academy Germain | PARTIE 12 — Section I — Chapitre 1
Les Screeners et Scanners
Finviz, TradeIdeas, Benzinga Pro, Floatchecker — Trouver les opportunités chaque matin
Un screener est ton radar. Sans lui, tu cherches manuellement parmi des milliers d'actions celle qui
bouge ce matin — une tâche impossible. Avec un bon screener correctement paramétré, les
meilleures opportunités viennent à toi. Ce chapitre te donne les outils et les paramètres exacts pour
chaque situation.
Screener vs Scanner — Quelle Différence ?
Screener (statique)
Scanner (dynamique en temps réel)
Filtre une base de données à un instant T
Scanne le marché en continu, en temps réel
Résultats figés jusqu'à la prochaine actualisation
Alertes instantanées dès qu'une condition est remplie
Idéal pour la préparation la veille ou le matin
Idéal pendant la session pour ne rater aucun
mouvement
Généralement gratuit (Finviz)
Souvent payant (TradeIdeas, Benzinga Pro)
Utilisation : construire la watchlist du jour
Utilisation : détecter les gappers et movers en live
I Bon à savoir
Les deux sont complémentaires — un bon workflow combine screener la veille + scanner en temps
réel le matin.
Pour un débutant avec un budget limité : commencer par Finviz gratuit + Benzinga Pre-Market
(version gratuite).
Pour un trader actif qui veut des alertes en temps réel : TradeIdeas est la référence professionnelle.
Finviz — Le Screener Gratuit de Référence
I Définition
Finviz (Financial Visualizations) est le screener boursier gratuit le plus utilisé par les traders retail.
Il permet de filtrer les 8 000+ actions cotées aux USA selon des dizaines de critères fondamentaux et
techniques.
URL : finviz.com → onglet 'Screener'
Les filtres essentiels pour le day trader — configuration recommandée :
PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 2 —
Filtre Finviz
Valeur
recommandée
Pourquoi
Country
USA
Se concentrer sur les actions américaines
Market Cap
Nano à Small (<
300M$)
Les gros mouvements viennent des small/nano caps
Price
Entre 0,50$ et 20$
Zone optimale pour les day traders — volatilité + accessibilité
Average Volume
> 500K / > 1M
Liquidité minimale pour trader sans problème d'exécution
Relative Volume
> 2 (Over 2x)
Actions avec volume anormal ce matin = quelque chose se
passe
Float Short
> 10%
Short interest élevé = potentiel de short squeeze si catalyseur
Gap
Up 5%+
Actions qui gappent à la hausse ce matin sur catalyseur
EPS Growth this
Year
Positif
Filtre optionnel pour actions avec fondamentaux en
amélioration
I Stratégie
Configuration de base pour trouver les gappers du matin :
1. Aller sur finviz.com/screener.ashx
2. Filters → Market Cap: Nano + Micro + Small | Price: $0.5 to $20 | Avg Volume: Over 500K
3. Ajouter : Relative Volume: Over 2 | Gap: Up 5%+
4. Trier par 'Change' (colonne %) pour voir les plus gros mouvements en tête
5. Pour chaque action : cliquer → vérifier le catalyseur sur EDGAR ou les news
I Bon à savoir
Finviz Elite (payant ~$25/mois) ajoute les alertes en temps réel et les données pre-market.
La version gratuite est suffisante pour débuter et préparer la watchlist.
Finviz a aussi une 'heatmap' visuelle qui montre d'un coup d'oeil les secteurs qui bougent.
Le screener Finviz peut être sauvegardé et partagé via une URL — pratique pour créer ses propres
configurations.
PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 3 —
TradeIdeas — Le Scanner Professionnel en Temps Réel
I Définition
TradeIdeas est le scanner professionnel le plus utilisé par les day traders actifs et les prop firms.
Il scanne le marché en temps réel et envoie des alertes instantanées dès qu'une action remplit des
critères définis.
Il inclut 'Holly' — une IA qui génère automatiquement des idées de trades basées sur des algorithmes.
Les alertes TradeIdeas les plus utiles pour le day trader :
Alerte
TradeIdeas
Paramètre type
Ce qu'elle détecte
High of Day
Momentum
Volume > 1M | Gap > 5% |
RVOL > 3
Action qui casse son plus haut du jour sur volume —
signal d'entrée potentiel
Gapper Alert
Gap > 10% | Volume
pre-market > 100K
Action qui gappe fortement avant l'ouverture sur
catalyseur
SSR Triggered
Prix en baisse > 10% depuis
J-1
Action qui vient de déclencher la SSR — potentiel rebond
New High
(52-week)
Prix = plus haut 52 semaines |
Volume > ADV
Breakout sur plus haut annuel — signal de continuation
haussière
Halted Stock
T1 ou T12 halt
Action suspendue — news importante en attente
Low Float
Mover
Float < 10M | RVOL > 5 |
Change > 15%
Low float en mouvement explosif — configuration de
short squeeze possible
II Attention
TradeIdeas est payant (~$100-180/mois) — investissement justifié uniquement pour un trader qui
trade quotidiennement.
La quantité d'alertes peut être écrasante au début. Commencer avec 3-5 alertes max et les affiner
progressivement.
Les alertes TradeIdeas sont un point de départ — toujours vérifier le catalyseur et les fondamentaux
avant d'agir.
PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 4 —
Benzinga Pro — Les News en Temps Réel
I Définition
Benzinga Pro est un service de news financières en temps réel spécialisé pour les traders actifs.
Sa valeur principale : les news arrivent souvent AVANT qu'elles apparaissent sur les sites classiques
(CNBC, Bloomberg).
La vitesse des news est cruciale en day trading — quelques secondes peuvent faire la différence sur
un gap.
Ce que Benzinga Pro offre :
• Newsfeed en temps réel avec filtres par action, secteur, type de news
• Alertes sur les résultats d'earnings (EPS beat/miss en temps réel)
• Alertes sur les décisions FDA et approbations de médicaments
• Calendrier pre-market avec les gappers et leurs raisons chaque matin
• Alertes sur les dépôts SEC (8-K, S-3, 424B) en temps quasi-réel
Plan Benzinga Pro
Prix approximatif
Ce qui est inclus
Basic (gratuit)
0$/mois
News avec délai de 15 min. Calendrier économique.
Essential
~30$/mois
News en temps réel. Calendrier earnings. Alertes de base.
Pro (~standard)
~50$/mois
Tout Essential + alertes personnalisées + audio squawk.
Options Mentoring
~100$/mois
Pro + suivi des options flow en temps réel.
I Bon à savoir
La version gratuite de Benzinga.com/pre-market est suffisante pour la préparation du matin.
Pour la version payante : négocier lors des périodes promotionnelles — des réductions de 50% sont
fréquentes.
Alternative gratuite : StockAnalysis.com (movers du matin) + SEC EDGAR (8-K récents).
Floatchecker — Analyser le Float en Profondeur
I Définition
Floatchecker est un outil spécialisé dans l'analyse du float des actions.
Il donne des informations précises sur le float, les shares outstanding, le short interest et les actions
bloquées (restricted shares).
Indispensable pour les traders de low floats qui ont besoin de données précises avant de trader.
Ce que Floatchecker fournit :
PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 5 —
• Float exact en temps quasi-réel (mis à jour plus fréquemment que Finviz)
• Restricted shares (actions bloquées, souvent d'insiders) — pour calculer le float réel
• Short interest récent avec Days to Cover
• Historique du float — identifier si l'entreprise a dilué récemment
• Alertes si le float change significativement (nouveau registre SEC déposé)
I Exemple concret
ARAI (notre cas d'étude) : Floatchecker confirme le float de 9,37M avec le détail des shares bloquées.
Sans Floatchecker, un trader peut confondre les 36,43M de Shares Outstanding avec le float
disponible — erreur qui fausse tous les calculs.
Règle : pour tout trade sur un low float (< 20M), vérifier Floatchecker avant d'entrer en position.
I Bon à savoir
Floatchecker dispose d'une version gratuite avec les données de base.
La version premium ajoute les alertes et les données historiques.
Alternative : Finviz pour les données de float + EDGAR pour les données détaillées sur les restricted
shares.
Tableau Comparatif — Quel Outil pour Quel Usage ?
Outil
Prix
Usage principal
Moment optimal
Niveau
recommandé
Finviz (gratuit)
0$
Screener statique,
watchlist, heatmap
La veille + matin
avant 8h30
Débutant → Expert
Finviz Elite
~25$/mois
Alertes temps réel +
données pre-market
Continu pendant la
session
Intermédiaire →
Expert
Benzinga Pro
(gratuit)
0$
News pre-market du
matin
6h30-9h30 EST
Débutant → Expert
Benzinga Pro
(payant)
~50$/mois
News ultra-rapides +
alertes earnings/FDA
Continu pendant la
session
Intermédiaire →
Expert
TradeIdeas
~100$/mois
Scanner temps réel +
alertes + IA Holly
6h30-16h00 EST
Intermédiaire →
Expert
Floatchecker
Gratuit/Premium
Analyse précise du
float des low floats
Avant chaque trade
low float
Débutant → Expert
StockAnalysis.com
0$
Movers du matin +
données
fondamentales
6h30-9h30 EST
Débutant → Expert
PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 6 —
I À retenir
Screener = filtre statique (Finviz). Scanner = alertes temps réel (TradeIdeas).
Finviz gratuit est suffisant pour débuter. Paramétrer : Cap Nano/Small + Price $0,50-20 + RVOL > 2 +
Gap Up.
Benzinga Pro gratuit couvre les news pre-market essentielles.
Floatchecker est indispensable pour les traders de low floats — float exact vs Shares Outstanding.
Workflow optimal : Finviz (veille/matin) + Benzinga (news) + TradeIdeas (temps réel pendant session).
I Pour mémoriser
Screener = statique | Scanner = temps réel | Les deux sont complémentaires
Finviz : RVOL > 2 + Gap Up + Float < 20M + Vol > 500K = gappers du matin
TradeIdeas : High of Day Momentum + Gapper Alert + Low Float Mover = 3 alertes de base
Floatchecker : float réel ≠ Shares Outstanding. Toujours vérifier avant un trade low float
PARTIE 12  ·  Ch.01
Outils de Sélection — Screeners et Scanners
Academy Germain — Formation aux Marchés Financiers
— 7 —
I GLOSSAIRE — Termes Techniques du Chapitre
Vocabulaire anglais essentiel à maîtriser
Terme anglais
Traduction
Définition
Screener
Filtre d'actions
Outil qui filtre une base de données d'actions selon des
critères définis (prix, volume, float, etc.). Résultat statique à
un instant T.
Scanner
Scanner en temps réel
Outil qui surveille le marché en continu et envoie des
alertes instantanées quand une condition est remplie.
Gapper
Action en gap
Action qui ouvre significativement au-dessus ou en
dessous de sa clôture précédente suite à une annonce
overnight.
Relative Volume (RVOL)
Volume relatif
Volume actuel / Volume moyen à la même heure. RVOL >
2 = activité anormale. Premier filtre du screener du matin.
Float
Flottant disponible
Nombre d'actions réellement disponibles au trading public.
Différent des Shares Outstanding qui incluent les actions
bloquées.
Restricted shares
Actions bloquées
Actions non disponibles au trading public (détenues par
insiders sous lock-up, actions émises pour
compensation...).
High of Day (HOD)
Plus haut du jour
Prix le plus élevé atteint par une action depuis l'ouverture
de la session. Casser le HOD sur volume = signal haussier
fort.
Audio squawk
Fil audio de news
Service de Benzinga Pro diffusant les news à voix haute en
temps réel — les traders peuvent suivre les news sans
regarder l'écran.
# Partie 7 / Ch.02 — Vente Decouvert Locate Borrow Rate SSR Short Squeeze
<!-- source: Academy_Germain_COMPLET.pdf p.206-216 -->

PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 1 —
Academy Germain | PARTIE 7 — Chapitre 2
Les Mécanismes Avancés
Short Selling, Locate, Borrow Rate, SSR, Short Interest, Short Squeeze, Float Rotation, Gaps,
VWAP, Tape — ARAI
Dans ce chapitre, on approfondit les mécanismes avancés du marché. Le cas ARAI continue — tu
vas voir comment le graphique, le Time & Sales et le VWAP racontent tous la même histoire en
temps réel.
Le Tape Reading sur ARAI — Time & Sales en Direct
Le Time & Sales (ou 'tape') est le registre de chaque transaction exécutée, en temps réel. C'est le
document le plus honnête du marché.
Image réelle — Time & Sales ARAI : transactions à 0,8333$ sur NASD et FADF en after-hours
(04h48)
I Analyse de l'image
Heure : 04:48 → c'est le PRE-MARKET (avant l'ouverture à 9h30 EST)
Prix dominant : 0,8333$ répété de nombreuses fois → le marché 'cherche' ce niveau
Exchanges : NASD (NASDAQ) et FADF (dark pool / autre venue)
Quantités : 1 à 8 actions par transaction → petites transactions de retail traders
Uniformité des prix : 0,8333$ répété = le marché est en équilibre à ce niveau
Comment lire ce Time & Sales d'ARAI :
• Les transactions à 0,8333$ répétées = le prix colle à ce niveau — ni hausse ni baisse marquée
• Volume de 1-8 actions = retail traders actifs en pre-market, pas d'institutionnels
• FADF = dark pool ou venue alternative — les ordres institutionnels transitent là
• Pas de grosse transaction visible = pas de gros acheteur/vendeur en ce moment
I Bon à savoir
En pre-market (04h00-09h30), le volume est très faible — les petites transactions dominent.
Une transaction de 1 action sur ARAI = quelqu'un qui teste le marché ou un algorithme.
Pour détecter un gros acheteur institutionnel, chercher des transactions de 10 000+ actions d'un coup.
L'absence de gros blocs ici indique que ce mouvement est principalement retail.
Le Graphique ARAI — Le Pump et la Chute
Voici le graphique 2 minutes d'ARAI sur les journées du 8 et 9 avril. Ce graphique illustre
visuellement PLUSIEURS concepts en même temps.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 2 —
Image réelle — Graphique ARAI 2min : Pump de 0,70$ à 1,25$, puis chute rapide. VWAP en rouge
pointillé. Volume explosif.
I Analyse de l'image
Période couverte : 8 avril (16h30) au 9 avril (04h30+) — after-hours + pre-market
Mouvement principal : montée de ~0,70$ vers 1,25$ = +79% en quelques heures
Chute rapide : de 1,25$ vers 0,85$ = -32% depuis le sommet
VWAP (ligne rouge pointillée) : monte avec le prix puis reste en dessous lors de la chute
Volume (bas du graphique) : explosion massive au sommet → c'est là que les vendeurs sortent
Le VWAP — Lecture sur le Graphique ARAI
I Définition
Le VWAP d'ARAI ce jour = 1,0241$ (visible dans le Level 2). Le cours actuel = 0,82$.
Cours < VWAP → les vendeurs dominent → la tendance intraday est baissière.
I Formule
VWAP = Somme (Prix × Volume) / Volume total de la session
ARAI VWAP = 1,0241$ | Cours actuel 0,82$ → cours 20% sous le VWAP
Position cours vs VWAP
Signal sur ARAI
Comportement institutionnel
Cours au-dessus du VWAP
Période de force — début de journée
pour ARAI
Les institutionnels achètent sous
le VWAP
Cours = VWAP
Zone de transition — décision
imminente
Zone neutre — attendre la
direction
Cours en dessous du VWAP
(situation actuelle)
Pression vendeuse domine — ARAI
en phase corrective
Les institutionnels vendent au
VWAP ou au-dessus
Cours loin sous le VWAP
(>15%)
Action en forte correction
Signal de sortie pour les longs
Le Short Selling — Mécanisme Complet
Vendre des actions qu'on ne possède pas, empruntées auprès d'un broker. On profite quand le prix
BAISSE. Sur ARAI : HTB = Hard To Borrow → très peu d'actions disponibles pour emprunter.
• Emprunt des actions : ton broker emprunte des actions auprès d'autres clients ou d'un prime broker.
• Vente immédiate : tu vends ces actions empruntées sur le marché au prix actuel. Ex sur ARAI : 100
actions × 0,82$ = 82$ reçus.
• Attente de la baisse : tu attends que le cours baisse. Si ARAI tombe à 0,60$, tu rachètes pour 60$.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 3 —
• Rachat et remboursement : profit = 82$ − 60$ = 22$. Mais si ARAI monte à 1,50$, tu perds 68$.
Le Locate — Trouver des Actions à Emprunter
I Définition
Avant de shorter une action, le trader doit obtenir un 'locate' auprès de son broker.
Un locate est la confirmation que le broker peut trouver les actions nécessaires à emprunter pour
exécuter le short.
Sans locate confirmé, la vente à découvert est interdite — c'est ce qu'on appelle un 'naked short',
illégal aux USA.
Le processus de locate en pratique :
• Tu identifies une action que tu veux shorter (ex : ARAI)
• Tu demandes un locate à ton broker AVANT de passer l'ordre
• Le broker vérifie si des actions sont disponibles dans son inventaire ou auprès de partenaires
• Si disponible → le locate est accordé → tu peux passer ton ordre short
• Si non disponible (HTB sans locate) → impossible de shorter cette action ce jour-là
Type d'action
Disponibilité du
locate
Ce que ça signifie
Easy to Borrow
(ETB)
Disponible
automatiquement
Actions liquides et largement détenues. Pas de démarche
nécessaire.
Hard to Borrow
(HTB)
Disponible mais limité
Peu d'actions disponibles. Souvent coûteux. Visible dans le
Level 2 d'ARAI.
No Borrow (NB)
Indisponible
Impossible de shorter. Le broker n'a pas d'actions à prêter ce
jour.
Short Exempt
Règles spéciales
Certains market makers sont exemptés de certaines
restrictions.
I Bon à savoir
Le locate est disponible via ta plateforme de trading (bouton 'Locate' chez Interactive Brokers, DAS
Trader, etc.).
Les locates sont limités — si 1 000 traders veulent shorter la même action HTB, certains n'obtiendront
pas de locate.
Un locate peut être annulé par le broker en cours de journée si les actions deviennent indisponibles →
'buy-in forcé'.
Sur ARAI : le HTB visible dans le Level 2 signifie que peu de locates sont disponibles → difficile de
shorter.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 4 —
Le Borrow Rate — Le Coût du Short Selling
I Définition
Le borrow rate (taux d'emprunt) est le coût annualisé pour emprunter des actions afin de les shorter.
Il est exprimé en pourcentage annuel et débité quotidiennement sur la position short.
Sur les actions Easy to Borrow, le borrow rate est souvent < 1%/an. Sur les HTB, il peut dépasser
100%/an.
I Formule
Coût journalier du borrow = (Valeur de la position × Borrow Rate annuel) / 365
Exemple : position short de 10 000$ sur une action HTB avec borrow rate de 50%/an
Coût journalier = (10 000$ × 50%) / 365 = 13,70$/jour
Sur 30 jours = 411$ de frais de borrow, indépendamment du résultat du trade
Niveau de
Borrow Rate
Catégorie
Impact sur la stratégie short
0% à 1%/an
Easy to Borrow (ETB)
Coût négligeable. Short selling économique.
1% à 10%/an
Légèrement difficile
Coût modéré. Tenir compte dans le calcul du P&L.;
10% à 50%/an
Hard to Borrow (HTB)
Coût significatif. Trade doit être rapide ou très profitable.
50% à 200%/an
HTB extrême
Très coûteux. Seulement pour des trades intraday rapides.
> 200%/an
Situation
exceptionnelle
GameStop jan. 2021 : borrow rate > 200%. Quasiment
impossible à shorter.
I Exemple concret
ARAI avec HTB : si le borrow rate est de 80%/an et que tu shortes 5 000$ d'ARAI pendant 2 jours :
Coût = (5 000$ × 80%) / 365 × 2 = 21,92$ de frais de borrow pour 2 jours
Si tu gagnes seulement 20$ sur le trade → le borrow rate a effacé ton gain
GameStop (janvier 2021) : borrow rate > 200%/an → les shorts payaient une fortune par jour →
pression supplémentaire pour couvrir
II Attention
Le borrow rate s'applique même si l'action ne bouge pas — c'est un coût fixe par jour.
Sur les HTB, le borrow rate peut changer d'un jour à l'autre sans prévenance.
Toujours calculer le coût total du borrow avant d'entrer un short sur une HTB.
Un borrow rate élevé est une des raisons pour lesquelles les shorts sont forcés de couvrir rapidement
→ alimente les short squeezes.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 5 —
Le SSR — Short Sale Restriction (Rule 201)
I Définition
Le SSR (Short Sale Restriction), aussi appelé Rule 201 de la SEC, est une restriction automatique sur
les ventes à découvert.
Elle se déclenche quand une action chute de 10% ou plus par rapport à son cours de clôture de la
veille.
Une fois activée, elle reste en vigueur pour le reste de la journée ET la journée suivante entière.
I Formule
Déclenchement : cours actuel ≤ cours de clôture J-1 × 90% (baisse de 10% ou plus)
Exemple : ARAI a clôturé hier à 1,00$. Si aujourd'hui ARAI tombe à 0,90$ ou moins → SSR activé
Durée : reste actif jusqu'à la clôture du lendemain
Ce que le SSR change concrètement :
• SANS SSR : tu peux shorter à n'importe quel prix — au bid, à l'ask, entre les deux
• AVEC SSR actif : tu peux SEULEMENT shorter au prix de l'ask ou AU-DESSUS
• En pratique : impossible de shorter au bid (prix d'achat) — tu dois attendre que quelqu'un te vende
• Résultat : les shorts ne peuvent plus accélérer la baisse en shortant au bid — la pression baissière
est réduite
Situation
Sans SSR
Avec SSR actif
Action à bid 10,00$ /
ask 10,05$
Peut shorter à 10,00$ (au bid)
Doit shorter à 10,05$ minimum (à l'ask ou
au-dessus)
Impact sur la chute
Les shorts accélèrent la baisse en
frappant le bid
La baisse est freinée — les shorts ne
peuvent pas accélérer
Opportunité pour le
trader
Short simple, exécution facile
Short plus difficile à exécuter, souvent moins
intéressant
Pour les longs
La chute peut être très brutale
Protection partielle — la chute est ralentie
Comment détecter le SSR sur ta plateforme :
• Finviz : sur la page d'une action, chercher 'SSR' dans les données
• Interactive Brokers : le SSR est affiché dans la fenêtre de trading
• DAS Trader : indicateur SSR visible dans la fenêtre Level 2
• Short-selling.com ou fintel.io : liste des actions avec SSR actif
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 6 —
I Exemple concret
ARAI a chuté de -32% dans notre graphique (de 1,25$ à 0,85$). Si cette chute dépasse -10% depuis
la clôture J-1 → SSR activé.
Avec SSR actif sur ARAI, tout trader voulant shorter doit placer son ordre à l'ask ou au-dessus.
En pratique : si le bid est 0,84$ et l'ask 0,85$, le short doit se faire à 0,85$ minimum.
GameStop (2021) : SSR activé très rapidement → les shorts ne pouvaient plus frapper le bid →
alimentait encore plus la montée.
SSR = Opportunité pour le LONG
SSR = Contrainte pour le SHORT
La baisse est freinée mécaniquement
Impossible de shorter au bid — ordre must be above
ask
Les shorts ont du mal à entrer ou accélérer la
baisse
Exécution plus difficile et plus coûteuse
Possible rebond plus facile si SSR actif
Le short squeeze est encore plus probable avec SSR
Signal que l'action a déjà beaucoup baissé
Chercher une action sans SSR pour shorter plus
facilement
I Stratégie
Utiliser le SSR intelligemment :
Si tu veux shorter : éviter les actions avec SSR actif. Chercher des actions sans SSR.
Si tu es long : le SSR sur ton action = protection partielle. La baisse est ralentie.
Si tu vois un SSR se déclencher : anticiper un rebond potentiel — les shorts ne peuvent plus accélérer
la baisse.
SSR + HTB + Low Float = configuration très défavorable pour les shorts → potentiel de rebond violent.
Short Interest % et Days to Cover
I Définition
Le Short Interest % est le pourcentage du float vendu à découvert. C'est la mesure du pessimisme
institutionnel.
Les Days to Cover indiquent combien de jours il faudrait aux shorts pour racheter toutes leurs
positions.
I Formule
Short Interest % = (Nombre d'actions shortées / Float) × 100
Days to Cover = Nombre d'actions shortées / Average Daily Volume
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 7 —
Short Interest %
Signal
Interprétation pratique
< 5%
Neutre
Peu de pessimisme. Pas de potentiel de short squeeze
significatif.
5% à 15%
Modéré
Quelques shorts en position. Surveiller si catalyseur positif.
15% à 25%
Élevé
Sentiment baissier significatif. Potentiel de short squeeze si bon
catalyseur.
25% à 40%
Très élevé
Risque fort de short squeeze ET de confirmation baissière.
> 40%
Extrême
Short squeeze de grande ampleur possible OU l'entreprise a de
sérieux problèmes.
I Bon à savoir
Où trouver le Short Interest : Finviz.com, Shortsqueeze.com, MarketBeat.com.
Mise à jour : 2 fois par mois par la FINRA (vers le 9 et le 26).
Days to Cover > 5 = situation tendue. > 10 = très tendue.
Combiner : Short Interest % élevé + SSR actif + HTB + Low Float = configuration explosive.
Le Short Squeeze — Le Graphique ARAI l'Illustre
I Définition
Un short squeeze est un mouvement haussier violent causé par les short sellers forcés de racheter
leurs positions.
ARAI avec son HTB et son low float est un candidat classique au short squeeze.
• Float très bas (9,37M) → peu d'actions disponibles → mouvement amplifié
• HTB → difficult de shorter → peu de pression short naturelle
• Borrow rate élevé → les shorts existants sont pressés de couvrir pour limiter les coûts
• SSR potentiellement actif → les nouveaux shorts ne peuvent pas frapper le bid
• Volume explosif au sommet → les vendeurs sortent massivement
I Exemple concret
GameStop (janvier 2021) : Short Interest > 130% du float + HTB + borrow rate > 200% + SSR activé
→ short squeeze légendaire → cours ×20.
AMC (2021) : Short Interest > 20% + catalyseur Reddit → short squeeze important.
ARAI : HTB + Low Float + RVOL > 7 = configuration compatible avec un short squeeze ou un pump.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 8 —
Le Float Rotation
I Formule
Float Rotation = Volume quotidien / Float
ARAI : 23 649 207 / 9 370 000 = 2,52× → le float a changé de mains 2,52 fois ce jour
Float Rotation
Signal
Cas ARAI
< 0,5×
Journée calme
ARAI en temps calme
0,5× - 1×
Activité modérée
ARAI journée normale
1× - 3×
Fort mouvement — journée active
ARAI ce jour : 2,52× 
3× - 7×
Mouvement exceptionnel
ARAI lors d'un vrai catalyseur
> 7×
Explosion — possible manipulation
Les grands pump and dump
Les Gaps
Sur le graphique ARAI, on voit clairement un gap entre la clôture du 8 avril et l'ouverture du 9 avril.
Type de gap
Description
Ce qu'on voit sur ARAI
Gap Up
Ouverture bien au-dessus de la clôture
ARAI a un gap up en after-hours après le pic
Gap Down
Ouverture en dessous de la clôture
Possible gap down sur ARAI après la chute
nocturne
Gap Fill
Retour dans la zone du gap
À surveiller sur ARAI — 70% des gaps sont
comblés
Breakaway Gap
Gap hors d'une consolidation avec fort
volume
Le gap initial d'ARAI avec volume massif
La Dilution en Temps Réel — Signal à Surveiller sur ARAI
II Attention
ARAI = nano cap avec FCF par action faible → risque élevé de dilution future via ATM ou PIPE.
Si après une montée forte ARAI annonce une levée de fonds → action peut chuter de 30-50%.
Vérification sur EDGAR : y a-t-il un S-3 actif ? Un ATM program autorisé ?
C'est souvent la vraie raison pour laquelle un pump se transforme en dump sur les small caps.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 9 —
I À retenir
Le Locate = confirmation d'emprunt AVANT de shorter. Sans locate = naked short = illégal.
Borrow Rate = coût annualisé de l'emprunt. HTB peut dépasser 100%/an → coût journalier significatif.
SSR (Rule 201) = déclenché à -10% depuis J-1. Interdit de shorter au bid — seulement à l'ask ou
au-dessus.
SSR + HTB + Low Float + Short Interest élevé = configuration explosive pour les shorts.
Short squeeze : Short Interest élevé + borrow rate élevé + SSR actif = les shorts sont pressés de
couvrir.
I Pour mémoriser
Locate = permission d'emprunter | HTB = difficile à trouver | NB = impossible
Borrow Rate élevé = les shorts perdent de l'argent chaque jour → pression pour couvrir
SSR = -10% depuis J-1 → interdit de shorter au bid → baisse ralentie → rebond possible
SSR + HTB + Low Float = piège à shorts = potentiel explosif haussier
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 10 —
I GLOSSAIRE — Termes Techniques du Chapitre
Vocabulaire anglais essentiel à maîtriser
Terme anglais
Traduction
Définition
Locate
Autorisation d'emprunt
Confirmation par le broker que des actions sont disponibles
à emprunter pour une vente à découvert. Obligatoire avant
tout short.
Easy to Borrow (ETB)
Facile à emprunter
Action disponible en abondance pour le short selling.
Borrow rate faible (< 1%/an).
Hard to Borrow (HTB)
Difficile à emprunter
Action avec peu de titres disponibles pour le short. Borrow
rate élevé. Visible dans le Level 2 d'ARAI.
No Borrow (NB)
Indisponible à
l'emprunt
Action impossible à shorter ce jour. Le broker n'a pas
d'actions à prêter.
Borrow Rate
Taux d'emprunt
Coût annualisé pour emprunter des actions. HTB peut
dépasser 100%/an. Débité quotidiennement sur la position
short.
Naked short
Vente à découvert nue
Shorter sans avoir obtenu de locate préalable. Illégal aux
USA depuis la réglementation post-2008.
SSR (Short Sale
Restriction)
Restriction de vente à
découvert
Rule 201 de la SEC. Se déclenche à -10% depuis la clôture
J-1. Interdit de shorter au bid — seulement à l'ask ou
au-dessus. Valable jusqu'à la clôture du lendemain.
Buy-in
Rachat forcé
Le broker force le rachat d'une position short si les actions
empruntées sont rappelées par leur propriétaire.
Short Interest %
Intérêt à découvert %
(Actions shortées / Float) × 100. Mesure le pessimisme du
marché et le potentiel de short squeeze.
Days to Cover
Jours pour couvrir
Actions shortées / Volume quotidien moyen. Combien de
jours il faudrait aux shorts pour racheter toutes leurs
positions.
Short squeeze
Compression des
positions short
Mouvement haussier violent causé par les shorts forcés
d'acheter. Amplifié par le low float, HTB, borrow rate élevé
et SSR.
Float rotation
Rotation du flottant
Volume / Float. ARAI = 23,6M / 9,37M = 2,52×. Mesure
l'intensité du mouvement.
VWAP
Prix moyen pondéré
par le volume
Prix moyen de la session. Cours sous VWAP = vendeurs
dominent.
Gap
Écart de prix
Zone sans transaction entre deux séances.
PARTIE 7  ·  Ch.02
Vente à Découvert — Locate, Borrow Rate, SSR...
Academy Germain — Formation aux Marchés Financiers
— 11 —
Terme anglais
Traduction
Définition
Time & Sales
Tape / Ruban de
transactions
Registre de chaque transaction exécutée en temps réel.
Montre prix, quantité, exchange et heure.
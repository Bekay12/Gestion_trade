# Partie 7 / Ch.01 — Formation Prix Order Book Volume Liquidite
<!-- source: Academy_Germain_COMPLET.pdf p.198-205 -->

PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 1 —
Academy Germain | PARTIE 7 — Chapitre 1
Les Fondamentaux du Prix
Order Book, Bid/Ask, Spread, Types d'ordres, Liquidité, Volatilité, Volume, RVOL — Cas réel
ARAI
Ce chapitre couvre les mécanismes fondamentaux qui déterminent le prix d'une action à chaque
instant. Pour te montrer ces concepts sur du CONCRET, nous allons utiliser l'action ARAI (Arrive AI)
comme fil conducteur tout au long de ce chapitre.
Cas Réel — L'Action ARAI (Arrive AI) : Présentation
Avant de plonger dans les mécanismes, prenons connaissance de l'action que nous allons analyser
tout au long de ce chapitre. Toutes les captures d'écran qui suivent sont RÉELLES et ont été prises
en séance.
Image réelle — Fiche ARAI : Float 9,37M / Shares Outstanding 36,43M / Market Cap 30,8M
I Analyse de l'image
Ticker : ARAI — Secteur : Technologie / Logiciels — Bourse : NASDAQ
Capitalisation boursière : 30,8M$ → Nano Cap (< 50M$)
Float : 9,37 MILLIONS d'actions seulement → LOW FLOAT EXTRÊME
Shares Outstanding : 36,43M → Float = 9,37M / 36,43M = 25,7% des actions disponibles au trading
Possession interne (insiders) : 1,6% → reste = 98,4% en circulation publique
Estimation FCF par action : 0,33$ → entreprise qui brûle du cash
I Formule
Float % = Float / Shares Outstanding × 100
ARAI : 9,37M / 36,43M × 100 = 25,7%
Float Rotation potentiel : avec 9,37M d'actions seulement, il suffit de 9,37M$ d'achats pour faire
tourner tout le float une fois
I Bon à savoir
Avec seulement 9,37 millions d'actions disponibles au trading, ARAI est une action LOW FLOAT.
Un low float signifie qu'une petite quantité d'achats peut faire exploser le cours.
C'est exactement pour cette raison qu'ARAI peut faire +100% en une journée... et redescendre tout
aussi vite.
Ce type d'action est le terrain de jeu des day traders actifs — et des manipulateurs.
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 2 —
L'Order Book et le Bid/Ask — Analyse du Level 2 ARAI
Maintenant que tu connais ARAI, regardons son Level 2 en temps réel. Le Level 2 est la fenêtre sur
le carnet d'ordres — il montre exactement qui veut acheter et qui veut vendre, à quel prix.
Image réelle — Level 2 ARAI : Bid 0,8092$ / Ask 0,81$ / VWAP 1,0241$ / HTB visible
I Analyse de l'image
Prix actuel (Last) : 0,8204$ — en hausse de +0,15$ (+22,4%) sur la journée
Meilleur BID (achat) : 0,8092$ — côté gauche (vert) — les acheteurs
Meilleur ASK (vente) : 0,81$ — côté droit (rouge) — les vendeurs
SPREAD = 0,81$ - 0,8092$ = 0,0008$ → très serré ici car forte activité
VWAP = 1,0241$ → le cours (0,8204$) est EN DESSOUS du VWAP → les vendeurs dominent
HTB = Hard To Borrow → difficile de shorter ARAI — le borrow rate est élevé
Volume total : 23 649 207 actions → Float rotation = 23,6M / 9,37M = 2,5× le float !
Exchange
Côté BID (Achat)
Prix BID
Côté ASK (Vente)
Prix ASK
ARCA
38 actions
0,820$
NASD
0,820$ — 1 action
EDGX
31 actions
0,820$
MEMX
0,821$ — 30 actions
MEMX
12 actions
0,820$
EDGX
0,823$ — 1 action
24X
7 actions
0,820$
ARCA
0,824$ — 3 actions
NASD
2 actions
0,818$
24X
0,855$ — 1 action
MIAX
1 action
0,814$
BATS
0,861$ — 2 actions
BATS
1 action
0,813$
MIAX
0,877$ — 1 action
NYSE
4 actions
0,646$
—
—
I Analyse de l'image
Les ordres côté BID et ASK sont très petits (1 à 38 actions) → typique d'un low float en fin de
mouvement.
Plusieurs exchanges présents simultanément (NASD, ARCA, EDGX, MEMX, BATS, MIAX) →
fragmentation normale du marché US.
Le prix NYSE à 0,646$ est anormalement bas → cet ordre est vieux ou un artefact — ignoré par le
marché.
HTB visible = Hard To Borrow → impossible de shorter ARAI facilement = risque de short squeeze.
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 3 —
Bid, Ask, Spread — Le Moteur de Chaque Transaction
I Formule
Spread = Ask − Bid
ARAI sur ce Level 2 : Spread = 0,81$ - 0,8092$ = 0,0008$ = 0,1%
Spread en % = (Ask − Bid) / Ask × 100
Type d'action
Spread typique
Exemple réel
Apple (AAPL), mega cap
< 0,01$
0,01$ en session normale
Large cap standard
0,01$ - 0,05$
Google, Amazon : ~0,02$
Mid cap
0,05$ - 0,20$
Action à 50$ : 0,10$ de spread
Small cap
0,10$ - 0,50$
Action à 5$ : 0,25$ de spread
ARAI (low float actif)
0,001$ - 0,05$
0,0008$ en plein mouvement
Penny stock peu liquide
0,50$ - 2,00$+
Action à 0,50$ avec spread de 0,10$ = 20% !
II Attention
Le spread d'ARAI est serré ICI parce que l'action est en fort mouvement avec beaucoup de traders
actifs.
Le même ARAI en temps calme peut avoir un spread de 0,05$ à 0,10$ — soit 5 à 12% du prix.
Toujours vérifier le spread AVANT de placer un ordre sur une small cap ou low float.
En pre-market et after-hours, les spreads sont 3 à 5× plus larges.
Les Types d'Ordres — Application sur ARAI
Avec un prix de 0,82$ sur ARAI, voici comment chaque type d'ordre se comporterait :
Type
d'ordre
Définition
Exemple avec ARAI à 0,82$
Risque
Market Order
Exécution immédiate au
meilleur prix
Achat exécuté à 0,81$ (meilleur ask)
Slippage si peu de liquidité
Limit Order
Buy
Achat au prix fixé ou
moins
Buy limit à 0,80$ → attend que le cours
descende
Peut ne pas s'exécuter si ARAI
continue à monter
Stop Loss
Vente auto quand le prix
tombe à X$
Stop à 0,75$ → vente auto si ARAI
chute
Peut s'exécuter sur un spike
baissier temporaire
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 4 —
Type
d'ordre
Définition
Exemple avec ARAI à 0,82$
Risque
Stop-Limit
Stop converti en limit à
X$
Stop à 0,75$, limit à 0,73$ → vente
entre 0,73$ et 0,75$
Peut ne pas s'exécuter si gap
down sous 0,73$
II Attention
Sur une action low float comme ARAI, les market orders peuvent avoir un slippage important.
Toujours utiliser des LIMIT orders sur les small caps et low floats.
Un stop loss sur ARAI doit tenir compte de la volatilité — un stop trop serré sera déclenché par le
'bruit' normal.
La Liquidité et la Volatilité d'ARAI
I Avantages du low float (ARAI)
I Risques du low float (ARAI)
Mouvements forts possibles (+100% en une
journée)
Retournements violents (-50% en quelques minutes)
Volume × float = float rotation élevé le jour J
Peu de liquidité en temps calme — spread s'élargit
Opportunités pour les day traders réactifs
Stop loss déclenché facilement par la volatilité
HTB = difficile à shorter = protection partielle
Manipulation plus facile avec peu d'actions disponibles
Le Volume — Ce que Dit le Volume d'ARAI
Le volume d'ARAI ce jour : 23 649 207 actions. Pour une action avec un float de 9,37M, c'est
énorme.
I Analyse de l'image
Volume quotidien : 23,6 millions d'actions
Float : 9,37 millions d'actions
Float Rotation = 23,6M / 9,37M = 2,52× → le float a changé de mains 2,5 fois
Cela représente un volume anormalement élevé → un catalyseur important a déclenché ce
mouvement
Sans un catalyseur réel (news, FDA, contrat...), un volume aussi élevé sur une nano cap est suspect
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 5 —
Average Daily Volume — La Référence de Base
I Définition
L'Average Daily Volume (ADV) est le volume moyen d'actions échangées chaque jour sur une période
de référence (30 ou 90 jours).
C'est la mesure de liquidité la plus fondamentale — elle te dit si une action est facile ou dangereuse à
trader.
Le volume d'un jour donné n'a de sens que comparé à ce volume moyen.
I Formule
Average Daily Volume = Somme des volumes quotidiens sur N jours / N jours
Exemple ARAI : si l'ADV sur 30 jours = 3 millions, et qu'aujourd'hui il y a 23,6 millions → volume
anormal × 7,8
Niveau d'ADV
Liquidité
Utilisation pratique
< 100 000
actions/jour
Très faible —
dangereux
À éviter. Spreads larges, exécution difficile, manipulation
possible.
100K — 500K/jour
Faible — spéculatif
Microcaps actives. Volatilité élevée. Attention aux spreads.
500K — 5M/jour
Modérée — tradable
Small et mid caps actives. La majorité des day traders opèrent
ici.
5M — 50M/jour
Bonne
Large caps. Liquidité suffisante pour la plupart des stratégies.
> 50M/jour
Excellente
Blue chips (Apple, Tesla, NVIDIA...). Aucun problème de
liquidité.
I Bon à savoir
Où trouver l'ADV : Finviz.com (colonne 'Avg Volume'), Yahoo Finance, ta plateforme de trading.
L'ADV est le premier filtre à vérifier avant de mettre une action sur ta watchlist.
Règle : ne jamais prendre une position qui représente > 1% du volume quotidien moyen.
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 6 —
Volume Relatif (RVOL) — Le Premier Filtre du Day Trader
I Définition
Le Volume Relatif (RVOL) compare le volume actuel au volume moyen habituel à la même heure de la
journée.
C'est l'indicateur le plus important pour identifier les actions en mouvement anormal chaque matin.
Un RVOL élevé signifie qu'il se passe quelque chose d'inhabituel sur cette action.
I Formule
RVOL = Volume actuel (depuis 9h30) / Volume moyen à cette même heure sur les 30 derniers jours
ARAI ce jour : si l'ADV habituel est 3M et qu'il y a déjà 23,6M tradés → RVOL ≈ 7,8
Niveau RVOL
Signal
Que faire
< 0,5
Volume très faible — action
inactive
Ignorer. Pas d'intérêt pour le trading actif ce jour.
0,5 à 1
Volume normal
Pas de signal particulier. Marché neutre.
1 à 2
Activité légèrement élevée
À surveiller. Possibilité d'un mouvement en cours.
2 à 5
Volume significativement élevé
Fort intérêt. Chercher le catalyseur.
> 5
Volume anormal
Signal fort. Quelque chose se passe. News ? Catalyseur
? Manipulation ?
> 10
Volume explosif
Vérifier IMMÉDIATEMENT : news, 8-K, rumeur, short
squeeze en cours.
I Stratégie
Routine RVOL chaque matin (avant 9h30) :
1. Screener Finviz ou ta plateforme : filtrer les actions avec RVOL > 3 en pre-market
2. Pour chaque action sur ta watchlist : quel est son RVOL actuel ?
3. Si RVOL > 2 → chercher le catalyseur sur EDGAR (8-K récent ?) et les sites de news
4. RVOL élevé sans catalyseur fondamental = suspect → possible manipulation
5. RVOL élevé avec catalyseur réel = opportunité potentielle à analyser
Les Niveaux Psychologiques sur ARAI
• 0,80$ : niveau psychologique majeur — beaucoup de stops placés juste en dessous
• 0,82$ : prix actuel — zone de décision
• 0,85$ : résistance visible dans le Level 2 (24X à 0,855$)
• 1,00$ : niveau psychologique fort au-dessus — si ARAI y retourne, forte résistance
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 7 —
I À retenir
ARAI = Nano cap (30,8M$) + Low float (9,37M) = volatile + manipulable.
Level 2 montre le bid/ask en temps réel : ARAI bid 0,8092$ / ask 0,81$ = spread 0,1%.
Average Daily Volume = mesure de liquidité de base. ADV < 500K = risqué pour le trading actif.
RVOL = Volume actuel / Volume moyen à la même heure. RVOL > 2 = activité anormale. Premier filtre
chaque matin.
VWAP 1,0241$ vs cours 0,82$ → cours en dessous du VWAP → pression vendeuse domine.
I Pour mémoriser
Float ARAI = 9,37M = LOW FLOAT = mouvements violents possibles dans les 2 sens
Level 2 : BID (gauche/vert) = acheteurs | ASK (droite/rouge) = vendeurs
ADV = volume moyen sur 30 jours | RVOL = Volume actuel / ADV à la même heure
RVOL > 2 = surveiller | RVOL > 5 = signal fort | RVOL > 10 = explosif → chercher le catalyseur
PARTIE 7  ·  Ch.01
Formation des Prix — Order Book, Volume et L...
Academy Germain — Formation aux Marchés Financiers
— 8 —
I GLOSSAIRE — Termes Techniques du Chapitre
Vocabulaire anglais essentiel à maîtriser
Terme anglais
Traduction
Définition
Order book / Level 2
Carnet d'ordres /
Niveau 2
Liste en temps réel de tous les ordres d'achat et vente. Le
Level 2 montre les 5-10 meilleurs prix de chaque côté.
Bid price
Prix d'achat
Meilleur prix qu'un acheteur est prêt à payer. Sur ARAI :
0,8092$.
Ask price
Prix de vente
Meilleur prix auquel un vendeur accepte de vendre. Sur
ARAI : 0,81$.
Spread
Écart bid-ask
Ask − Bid. Sur ARAI en mouvement : 0,0008$ soit 0,1%.
Peut être 10× plus large en temps calme.
Low float
Faible flottant
Action avec peu d'actions disponibles au trading (< 20M).
ARAI = 9,37M = très low float.
Float rotation
Rotation du flottant
Volume / Float. ARAI = 23,6M / 9,37M = 2,52×. Mesure
l'intensité de l'activité sur la journée.
Hard to Borrow (HTB)
Difficile à emprunter
Action dont les titres sont rares pour le short selling. Borrow
rate très élevé. Visible dans le Level 2.
Average Daily Volume
(ADV)
Volume quotidien
moyen
Volume moyen d'actions échangées chaque jour sur 30 ou
90 jours. Mesure de liquidité fondamentale.
Relative Volume (RVOL)
Volume relatif
Volume actuel / Volume moyen à la même heure sur 30
jours. RVOL > 2 = activité anormale. Premier filtre du day
trader chaque matin.
VWAP
Prix moyen pondéré
par le volume
Prix moyen de toutes les transactions de la journée. Cours
sous VWAP = pression vendeuse.
Market order
Ordre au marché
Exécution immédiate au meilleur prix disponible. Risque de
slippage sur les low floats.
Limit order
Ordre à cours limité
Exécution au prix fixé ou mieux. Recommandé sur les
small caps et low floats.
Stop loss
Ordre stop /
Coupe-perte
Vente automatique quand le prix atteint un seuil. Essentiel
sur les low floats très volatils.
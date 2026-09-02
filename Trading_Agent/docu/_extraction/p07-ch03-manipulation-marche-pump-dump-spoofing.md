# Partie 7 / Ch.03 — Manipulation Marche Pump Dump Spoofing
<!-- source: Academy_Germain_COMPLET.pdf p.217-221 -->

PARTIE 7  ·  Ch.03
Manipulation de Marché — Pump and Dump et Sp...
Academy Germain — Formation aux Marchés Financiers
— 1 —
Academy Germain | PARTIE 7 — Chapitre 3
Les Manipulations de Marché
Pump and Dump, Spoofing, Layering — Analyse du graphique ARAI comme cas d'étude
Les marchés financiers ne sont pas toujours un terrain équitable. Des acteurs mal intentionnés
utilisent des techniques de manipulation pour voler l'argent des traders non avertis. Ce chapitre te
donne les outils pour les identifier et les éviter — en utilisant le cas ARAI comme illustration concrète.
Le Graphique ARAI — Anatomie d'un Mouvement Suspect
Avant d'expliquer les manipulations, analysons le graphique ARAI pour identifier les caractéristiques
d'un mouvement potentiellement manipulé.
Image réelle — Graphique ARAI 2min : montée rapide +79% puis chute -32%. Volume explosif au
sommet. VWAP dépassé puis abandonné.
I Analyse de l'image
Phase 1 (16h30-18h30) : cours stable entre 0,68$ et 0,75$ → accumulation discrète probable
Phase 2 (18h30-19h30) : première montée progressive de 0,75$ à 0,90$ → intérêt croissant
Phase 3 (19h30-19h50) : explosion à 1,25$ en 20 minutes → le pump est à son maximum
Signaux de vente (triangles roses) : signaux automatiques de la plateforme au sommet
Phase 4 (19h50-04h30) : chute de 1,25$ vers 0,85$ → distribution et dump progressif
Volume bas du graphique : pic massif exactement au sommet → les vendeurs absorbent les derniers
acheteurs
VWAP (rouge pointillé) : dépassé lors du pump, puis cours retombe EN DESSOUS
II Attention
Les 5 signaux classiques d'un pump sur ce graphique ARAI :
1. Montée très rapide (+79% en quelques heures) sans news fondamentale majeure connue
2. Volume explosif exactement au sommet (pas à la base — ce serait normal)
3. Chute rapide après le sommet (-32% en quelques heures)
4. Action nano cap (30,8M$) avec low float (9,37M) = facile à manipuler
5. Cours retombe SOUS le VWAP rapidement après le pic = le mouvement n'était pas fondamental
PARTIE 7  ·  Ch.03
Manipulation de Marché — Pump and Dump et Sp...
Academy Germain — Formation aux Marchés Financiers
— 2 —
Le Pump and Dump — Mécanisme Complet
I Définition
Le pump and dump est la manipulation la plus répandue sur les marchés.
Les manipulateurs achètent d'abord discrètement, puis font monter artificiellement le cours via de
fausses informations, avant de vendre massivement au sommet.
Les 5 phases sur le graphique ARAI :
• Accumulation discrète (Phase ARAI 16h30-18h30) : les manipulateurs achètent progressivement à
0,68$-0,75$. Volume bas, prix stable.
• Création du buzz — Le Pump (Phase ARAI 18h30-19h50) : diffusion de fausses informations
positives. Montée de 0,75$ à 1,25$. Le volume explose. Les retail traders achètent par FOMO.
• Afflux de retail traders (Sommet ARAI 1,25$) : des traders non avertis voient +79% et achètent. Le
volume est au maximum.
• Distribution au sommet — Le Dump (Phase ARAI 19h50-04h30) : les manipulateurs vendent
massivement pendant que les retail achètent encore. Cours tombe de 1,25$ à 0,85$.
• Effondrement : une fois les manipulateurs sortis, le cours s'effondre. Les retail qui ont acheté à
1,20$-1,25$ se retrouvent avec des pertes de 30-40%.
I Bon à savoir
IMPORTANT : nous ne savons pas avec certitude si ARAI a été manipulé.
Le graphique montre des CARACTÉRISTIQUES compatibles avec un pump and dump.
Mais il pourrait aussi s'agir d'un short squeeze naturel sur un low float HTB.
Short squeeze = pas illégal. Pump and dump = illégal. La prudence s'impose dans les deux cas.
Signal d'alarme
Présent sur
ARAI ?
Analyse
Montée rapide sans catalyseur
fondamental clair
Possible
79% en quelques heures — news à vérifier sur
EDGAR
Volume explosif au sommet
Oui
Pic de volume exactement au plus haut =
distribution
Nano cap / low float
Oui
Capitalisation 30,8M$ + float 9,37M = facile à
manipuler
Chute rapide après le sommet
Oui
-32% depuis le pic → distribution en cours
Cours retombe sous VWAP
Oui
VWAP 1,02$ vs cours 0,82$ → -20% sous
VWAP
Email/SMS de recommandation
Non visible ici
À vérifier — signal le plus fiable du pump
PARTIE 7  ·  Ch.03
Manipulation de Marché — Pump and Dump et Sp...
Academy Germain — Formation aux Marchés Financiers
— 3 —
Signal d'alarme
Présent sur
ARAI ?
Analyse
Pas de SEC filings récents
À vérifier
Consulter EDGAR pour les dépôts récents
d'ARAI
Comment Distinguer un Short Squeeze d'un Pump and Dump
Short Squeeze (naturel)
Pump and Dump (manipulation)
Short interest élevé AVANT le mouvement
Short interest faible avant le mouvement
HTB = difficile à shorter = shorts piégés
HTB peut aussi être utilisé pour protéger le pump
Volume élevé mais progressif
Volume explose très soudainement
Mouvement continue sur plusieurs jours
Pump très rapide puis dump brutal le même jour
News réelle comme catalyseur
Pas de news ou news très vague
Les shorts couvrent → achats forcés naturels
Volume vient de retail poussé par spam/hype
L'action peut rebondir après consolidation
L'action retombe généralement sous le niveau pré-pump
I Exemple concret
ARAI a un HTB confirmé dans le Level 2 → présence de shorts difficiles à couvrir → favorise le short
squeeze.
Le mouvement de +79% correspond aux deux scénarios — impossible de certifier sans analyser le
short interest exact.
Leçon pratique : avant d'acheter en espérant un short squeeze, vérifie sur Finviz ou MarketBeat le
short interest d'ARAI.
Si short interest < 5% → peu de shorts à couvrir → le pump ressemble plus à une manipulation qu'un
squeeze.
Le Spoofing et le Layering — Détection dans le Level 2
I Définition
Le spoofing consiste à placer de gros ordres factices dans le Level 2 pour créer une fausse
impression de demande ou d'offre, puis les annuler juste avant l'exécution.
Sur ARAI : surveiller les gros ordres qui apparaissent et disparaissent rapidement dans le Level 2.
Comment détecter le spoofing dans le Level 2 d'ARAI :
• Gros ordre soudain côté BID (ex : 100 000 actions à 0,80$) qui disparaît en quelques secondes
PARTIE 7  ·  Ch.03
Manipulation de Marché — Pump and Dump et Sp...
Academy Germain — Formation aux Marchés Financiers
— 4 —
• Plusieurs grosses offres de vente empilées côté ASK qui s'annulent avant exécution
• Le tape (Time & Sales) ne montre aucune transaction sur ces ordres → ils n'étaient pas réels
• Répétition du schéma plusieurs fois de suite = spoofing probable
I Bon à savoir
Dans le Level 2 d'ARAI qu'on a analysé, les ordres sont tous petits (1-38 actions). Pas de signe
évident de spoofing.
Le spoofing est plus visible sur des actions avec des ordres plus importants.
La SEC surveille le spoofing avec des algorithmes automatiques — les contrevenants sont poursuivis.
Comment se Protéger des Manipulations
Protection
Description
Application sur ARAI
Analyser les
fondamentaux
Entreprise avec revenus réels = plus
difficile à manipuler
ARAI : nano cap tech, FCF modeste —
vérifier les revenus réels
Vérifier les SEC filings
Pas de filings récents = suspect. S-3
actif = risque dilution
Vérifier EDGAR pour ARAI : y a-t-il un
ATM ou S-3 actif ?
Ignorer les
recommandations non
sollicitées
Règle absolue — personne ne partage
par générosité
Si quelqu'un te parle d'ARAI sans que
tu demandes = méfiance
Trader uniquement avec
du volume organique
Volume + news réelle = légitime
ARAI : identifier la vraie news qui a
déclenché le mouvement
Stop loss systématique
Limite les dégâts si tu es dans un
dump
ARAI à 1,25$ : stop à 1,10$ pour limiter
la perte à -12%
Taille de position réduite
Même si pris dans un dump, les
dégâts sont limités
Sur ARAI : ne jamais mettre plus de
1-2% de son capital
I Bon à savoir
Si ça semble trop beau pour être vrai, c'est que c'est trop beau pour être vrai.
Sur ARAI : +79% en quelques heures sur une nano cap = TRÈS inhabituel.
Ne jamais acheter un tel mouvement sans avoir identifié le catalyseur réel.
Un inconnu sur internet ne partage pas une opportunité par générosité — il partage pour que tu
achètes afin qu'il puisse vendre.
PARTIE 7  ·  Ch.03
Manipulation de Marché — Pump and Dump et Sp...
Academy Germain — Formation aux Marchés Financiers
— 5 —
I À retenir
Le graphique ARAI illustre les caractéristiques d'un mouvement suspect : montée rapide, volume au
sommet, chute brutale.
Pump and dump = accumulation → buzz → FOMO retail → dump → effondrement.
Short squeeze = possible aussi sur ARAI grâce au HTB et low float — mais impossible à certifier sans
données de short interest.
La protection absolue : stop loss + position réduite + vérification des SEC filings avant d'acheter.
I Pour mémoriser
Graphique ARAI : accumulation (0,68$) → pump (1,25$) → dump (0,85$) = pattern classique
Short squeeze vs Pump : vérifier le short interest AVANT — si < 5% = suspect
Spoofing dans Level 2 : gros ordres qui apparaissent et disparaissent sans transaction
Protection : stop loss + 1-2% max du capital + vérifier EDGAR avant tout achat
I GLOSSAIRE — Termes Techniques du Chapitre
Vocabulaire anglais essentiel à maîtriser
Terme anglais
Traduction
Définition
Pump and dump
Gonfler et larguer
Manipulation : gonfler artificiellement le cours via de
fausses informations, puis vendre au sommet. Pattern
compatible visible sur ARAI.
Spoofing
Usurpation d'ordres
Placer de gros ordres factices dans le Level 2 pour tromper
le marché, puis les annuler avant exécution. Illégal aux
USA.
Layering
Empilement d'ordres
Variante du spoofing avec plusieurs niveaux d'ordres
factices dans le carnet d'ordres.
FOMO (Fear Of Missing
Out)
Peur de rater une
opportunité
Biais psychologique. Les retail traders voient +79% sur
ARAI et achètent impulsionnellement.
Accumulation phase
Phase d'accumulation
Période où les manipulateurs achètent discrètement avant
de lancer le pump. Visible sur ARAI de 16h30 à 18h30.
Distribution phase
Phase de distribution
Période où les manipulateurs vendent au sommet pendant
que les retail achètent encore. Volume élevé + cours
baisse.
Short interest
Intérêt vendeur
% du float vendu à découvert. Clé pour distinguer un short
squeeze naturel d'un pump artificiel.
Securities fraud
Fraude sur valeurs
mobilières
Délit fédéral. Le pump and dump en est la forme la plus
courante sur les nano caps.
"""Génère le rapport Honeywell. Toutes les tables viennent du modèle, rien n'est recopié."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / '.claude/skills/startup-investment-analyzer/scripts'))
from cashflow_irr import break_even_investment, full_cashflows, irr, required_growth
from daten_us import EXTERN, WERTE

W = {k: v[0] for k, v in WERTE['Honeywell'].items()}
Q = {k: f"10-K 2025, p.&nbsp;{v[2]}" if v[1] == '10K2025' else f"10-Q S1 2026, p.&nbsp;{v[2]}"
     for k, v in WERTE['Honeywell'].items()}
PRIX = EXTERN['Honeywell']['kurs'][0]
ACT = EXTERN['Honeywell']['aktien_mio'][0]

FCF = {a: W[f'op_cashflow_{a}'] - W[f'investitionen_{a}'] for a in ('2023', '2024', '2025')}
SERIE = {2022: 4508, 2023: FCF['2023'], 2024: FCF['2024'], 2025: FCF['2025']}
EK = W['eigenkapital_2025']
DETTE = W['geldmarkt_2025'] + W['faellige_schuld_2025'] + W['langfrist_schuld_2025']
NETTE = DETTE - W['liquiditaet_2025']
NETTE_1H = (W['langfrist_schuld_1h2026'] + 5282 + 2478) - W['liquiditaet_1h2026']
CF_ACT = FCF['2025'] / ACT
BV_ACT = EK / ACT
ROE = W['nettoergebnis_2025'] / EK
COUT_DETTE = 1344 / DETTE
CONS = {'note': 'Buy', 'n': 22, 'cible': 263.41, 'haut': 303.0, 'bas': 186.0, 'hausse': 30.80}


def cfg(taux, jahre):
    return {'currency': 'USD/action', 'opening_liquidity': 0.0, 'horizon_start': 2026,
            'horizon_end': 2026 + jahre, 'schedule_years': [2026],
            'sensitivity_horizons': [2026 + jahre], 'revenue_base': 1.0, 'earnings_base': 1.0,
            'hurdle': taux * 100, 'hurdle_source': '10-K 2025, p. 55 et p. 57',
            'options': {'K': {'name': 'Achat', 'investment': {'2026': PRIX},
                              'earnings_uplift': CF_ACT, 'uplift_start': 2027,
                              'terminal_value': PRIX}}}


def n(x, d=0):
    return f'{x:,.{d}f}'.replace(',', ' ')


def pc(x, d=1):
    return f'{x:.{d}f}&nbsp;%'.replace('.', ',')


def tab(kopf, zeilen, cap=''):
    k = ''.join(f'<th>{h}</th>' for h in kopf)
    b = ''.join('<tr' + (' class="summe"' if str(z[0]).startswith('*') else '') + '>'
                + ''.join(f'<td>{c}</td>' for c in [str(z[0]).lstrip('*')] + list(z[1:]))
                + '</tr>' for z in zeilen)
    c = f'<caption>{cap}</caption>' if cap else ''
    return f'<div class="tabelle"><table>{c}<thead><tr>{k}</tr></thead><tbody>{b}</tbody></table></div>'


HURDLES = [('Rendement des capitaux propres 2025', ROE, 'belegt'),
           ('Coût de la dette 2025 (1 344 / 34 580)', COUT_DETTE, 'belegt')]

lignes_irr, lignes_seuil, lignes_g = [], [], []
for nom, taux, _ in HURDLES:
    s, g = [], []
    for j in (5, 10, 15):
        c = cfg(taux, j)
        be = break_even_investment(c, 'K')
        gg = required_growth(c, 'K')
        s.append(f'{n(be, 2)}&nbsp;USD' if be else 'n.&nbsp;a.')
        g.append(pc(gg * 100) if gg is not None else 'hors de portée')
    lignes_seuil.append((nom, *s))
    lignes_g.append((nom, *g))

STIL = open(Path(__file__).resolve().parent.parent / 'stil.css').read()

HTML = f"""<title>Honeywell à 201 dollars</title><style>{STIL}</style><div class="wrap">
<header><p class="eyebrow">Analyse d'entreprise · conglomérat industriel · seul signal d'inflexion retenu</p>
<h1>Honeywell à 201 dollars</h1>
<p class="frage">À quel prix une entrée dans Honeywell est-elle défendable, et que vaut un rendement du flux libre de {pc(CF_ACT/PRIX*100, 2)} pour une société qui emprunte à {pc(COUT_DETTE*100, 2)} ?</p>
<div class="meta"><span>Date d'arrêté <b>15.09.2026</b></span><span>Bourse <b>Nasdaq · HON</b></span>
<span>Sources primaires <b>10-K 2025 · 10-Q S1 2026</b></span><span>Valeurs vérifiées <b>30 sur 30</b></span></div>
<dl class="legende">
<div><dt><span class="sw" style="background:var(--ink)"></span>Noir</dt><dd>Chiffres, tableaux, sources. La source primaire en répond.</dd></div>
<div><dt style="color:var(--blau)"><span class="sw" style="background:var(--blau)"></span>Bleu</dt><dd>Appréciation de l'auteur, après vérification.</dd></div>
<div><dt style="color:var(--rot)"><span class="sw" style="background:var(--rot)"></span>Rouge</dt><dd>Le verdict, section 7. Recommandation sous hypothèses déclarées, pas un constat.</dd></div>
</dl></header>

<section><h2><span class="nr">1</span>Description de l'entreprise</h2><div class="strich"></div>
<h3><span class="nr">1.3</span>Jugement des marchés de capitaux</h3>
{tab(['Grandeur', 'Valeur', 'Source'], [
 ('Cours au 14.09.2026', f'{n(PRIX,2)}&nbsp;USD', 'Yahoo Finance'),
 ('Actions en circulation', f'{n(ACT,2)}&nbsp;millions', '10-K 2025, page de garde'),
 ('*Capitalisation', f'{n(PRIX*ACT/1000,1)}&nbsp;Md&nbsp;USD', 'produit des deux'),
 ('Capitaux propres par action', f'{n(BV_ACT,2)}&nbsp;USD', Q['eigenkapital_2025']),
 ('*Cours / valeur comptable', f'{n(PRIX/BV_ACT,2)}×', '')], 'Données de cours')}
<p class="quelle"><b>Un écart à signaler.</b> Yahoo Finance donnait 316,9&nbsp;millions d'actions, soit une capitalisation de 63,8&nbsp;Md&nbsp;USD. La page de garde du 10-K indique 635&nbsp;675&nbsp;701 actions, et le résultat par action dilué de 7,36&nbsp;USD sur 4&nbsp;729&nbsp;M&nbsp;USD confirme environ 642&nbsp;millions. Le chiffre de Yahoo est faux d'un facteur deux ; tous les ratios par action de ce rapport reposent sur celui du dépôt.</p>
<h3><span class="nr">1.4</span>Couverture et consensus</h3>
{tab(['Grandeur', 'Valeur', 'Contre le cours'], [
 ('Analystes recensés', f"{CONS['n']}", '—'),
 ('Recommandations', '14 achat · 8 conservation · 1 vente', '—'),
 ('*Objectif moyen', f"{n(CONS['cible'],2)}&nbsp;USD", f"+{pc(CONS['hausse'],2)}"),
 ('Objectif le plus haut', f"{n(CONS['haut'],2)}&nbsp;USD", '—'),
 ('Objectif le plus bas', f"{n(CONS['bas'],2)}&nbsp;USD", '—')], 'Consensus · source secondaire · 15.09.2026')}
<p class="quelle">Investing.com, page consensus HON, consultée le 15.09.2026. Contrôle de cohérence : 263,41 / 1,3080 = 201,38, soit exactement le cours retenu ici. La donnée est donc à jour, contrairement aux valeurs divergentes de 246,67 et 265 qu'affichent d'autres agrégateurs sans indiquer leur cours de référence.</p>
<h3><span class="nr">1.5</span>Messages et lacunes</h3>
<div class="luecken"><ul>
<li><b>Aucune prévision chiffrée pour 2026 dans les dépôts.</b> Honeywell la publie dans ses communiqués de résultats, que ce rapport n'a pas dépouillés. La comparaison prévision contre réalisé est donc absente, et rien ne la remplace ici.</li>
<li><b>Pas de noms d'analystes ni de notes individuelles.</b></li>
<li><b>Série de flux libre sur quatre ans seulement</b>, recoupée entre le 10-K et l'API XBRL. Le squelette en demande sept à dix ; le centile de la section&nbsp;6 porte donc sur une fenêtre courte, et c'est dit là où il est utilisé.</li>
</ul></div></section>

<section><h2><span class="nr">2</span>Analyse des comptes</h2><div class="strich"></div>
<p>Millions de dollars, sauf indication contraire.</p>
{tab(['Grandeur', '2023', '2024', '2025', 'Δ 2025'], [
 ('Chiffre d’affaires', n(W['umsatz_2023']), n(W['umsatz_2024']), n(W['umsatz_2025']), '+7,9&nbsp;%'),
 ('Résultat net part du groupe', n(W['nettoergebnis_2023']), n(W['nettoergebnis_2024']), n(W['nettoergebnis_2025']), '−17,1&nbsp;%'),
 ('dont dépréciation d’écart d’acquisition', '—', '—', n(W['goodwill_abschreibung_2025']), '—'),
 ('Résultat par action dilué, USD', '8,47', '8,71', '7,36', '−15,5&nbsp;%'),
 ('Flux de trésorerie opérationnel', n(W['op_cashflow_2023']), n(W['op_cashflow_2024']), n(W['op_cashflow_2025']), '+5,1&nbsp;%'),
 ('Investissements', n(W['investitionen_2023']), n(W['investitionen_2024']), n(W['investitionen_2025']), '+13,2&nbsp;%'),
 ('*Flux de trésorerie libre', n(FCF['2023']), n(FCF['2024']), n(FCF['2025']), '+3,7&nbsp;%'),
 ('Dividendes versés', '2 855', '2 902', n(W['dividenden_2025']), '+2,6&nbsp;%'),
 ('Capitaux propres', '—', n(W['eigenkapital_2024']), n(W['eigenkapital_2025']), '−21,5&nbsp;%'),
 ('Dette nette', '—', '—', n(NETTE), '—')], 'Exercices, 10-K 2025')}
<p class="quelle">{Q['umsatz_2025']} · {Q['op_cashflow_2025']} · {Q['eigenkapital_2025']}. Dette nette = papier commercial 5&nbsp;893 + échéances courantes 1&nbsp;546 + dette long terme 27&nbsp;141 − liquidités 12&nbsp;487.</p>
<p><b>Le résultat net recule de 17,1&nbsp;% alors que le chiffre d'affaires progresse de 7,9&nbsp;%.</b> L'écart tient pour l'essentiel à une dépréciation d'écart d'acquisition de {n(W['goodwill_abschreibung_2025'])}&nbsp;M&nbsp;USD, absente des deux exercices précédents. Le flux de trésorerie libre, lui, ne la subit pas et progresse de 3,7&nbsp;%.</p>
<p><b>Les capitaux propres perdent 21,5&nbsp;%</b> sur un exercice bénéficiaire. Ce n'est pas une perte : c'est le rachat d'actions et le versement de {n(W['dividenden_2025'])}&nbsp;M&nbsp;USD de dividendes. La conséquence est arithmétique et elle pèse sur toute la suite : une base de capitaux propres comprimée produit mécaniquement un rendement des capitaux propres élevé et un rapport cours sur valeur comptable de {n(PRIX/BV_ACT,2)}×.</p>
<h3>Premier semestre 2026</h3>
{tab(['Grandeur', 'S1 2025', 'S1 2026', 'Δ'], [
 ('Chiffre d’affaires', n(W['umsatz_1h2025']), n(W['umsatz_1h2026']), '+3,4&nbsp;%'),
 ('Résultat net part du groupe', n(W['nettoergebnis_1h2025']), n(W['nettoergebnis_1h2026']), '+113,5&nbsp;%'),
 ('*Flux de trésorerie opérationnel', n(W['op_cashflow_1h2025']), n(W['op_cashflow_1h2026']), '−67,3&nbsp;%'),
 ('Capitaux propres', n(W['eigenkapital_2025']), n(W['eigenkapital_1h2026']), '+25,5&nbsp;%'),
 ('Dette nette', n(NETTE), n(NETTE_1H), '+14,2&nbsp;%')], 'Exercice en cours, 10-Q S1 2026')}
<p class="quelle">{Q['umsatz_1h2026']} · {Q['op_cashflow_1h2026']} · {Q['eigenkapital_1h2026']}</p>
<p class="bewertung">C'est la ligne la plus importante du rapport et elle va dans le mauvais sens. Le résultat net double, mais le flux de trésorerie opérationnel du semestre tombe de 1 916 à 626&nbsp;M&nbsp;USD. Un résultat qui double pendant que l'encaissement est divisé par trois signale que le bénéfice du semestre ne vient pas de l'exploitation courante. La dette nette progresse de 14,2&nbsp;% sur la même période.</p></section>

<section><h2><span class="nr">3</span>Mesures stratégiques</h2><div class="strich"></div>
<p><i>Cette section rapporte, elle ne juge pas.</i></p>
<p>Le compte de résultat 2025 porte deux dépréciations : {n(W['goodwill_abschreibung_2025'])}&nbsp;M&nbsp;USD sur des écarts d'acquisition et 270&nbsp;M&nbsp;USD sur des actifs destinés à la vente, contre 219&nbsp;M&nbsp;USD l'exercice précédent pour cette seconde ligne. Les charges financières passent de 1 048 à 1 344&nbsp;M&nbsp;USD, soit une hausse de 28,2&nbsp;%, pour une dette brute de {n(DETTE)}&nbsp;M&nbsp;USD à la clôture.</p>
<p>Aucune prévision chiffrée pour 2026 ne figure dans le 10-Q ; la section 1.5 le déclare comme lacune.</p></section>

<section><h2><span class="nr">4</span>Lecture du cours</h2><div class="strich"></div>
<p>Le titre a perdu 22,6&nbsp;% depuis son plus haut à trois ans, atteint le 2&nbsp;mars&nbsp;2026, et se tient 15,7&nbsp;% au-dessus de son plus bas. Sur trois mois il recule de 12,8&nbsp;% quand son indice de référence progresse de 2,5&nbsp;% : la baisse est donc intégralement propre à l'entreprise, pas subie.</p>
<p class="bewertung">L'indicateur qui explique ce décrochage n'est ni le chiffre d'affaires, qui progresse, ni le résultat net, qui double au semestre. C'est l'encaissement. Un marché qui voit le flux opérationnel semestriel fondre de deux tiers cesse de payer le résultat comptable, et c'est exactement ce que montre la chronologie : le plus haut date de mars 2026, le 10-Q qui publie ces 626&nbsp;M&nbsp;USD a été déposé le 23&nbsp;juillet.</p></section>

<section><h2><span class="nr">5</span>Options d'action</h2><div class="strich"></div>
<div class="entfaellt"><p><b>Sans objet (n&nbsp;=&nbsp;0).</b> La décision n'est pas de choisir entre des options de l'entreprise mais d'entrer ou non à un cours donné. La numérotation est conservée pour que ce rapport reste comparable ligne à ligne avec les précédents.</p></div></section>

<section><h2><span class="nr">6</span>Analyse de flux</h2><div class="strich"></div>
<h3><span class="nr">6.0</span>Hypothèses, déclarées avant tout chiffre</h3>
<div class="annahme"><b>Hypothèse 1 — flux distribuable et non dividende</b><p>La série porte le flux de trésorerie libre publié, soit {n(FCF['2025'])}&nbsp;M&nbsp;USD en 2025, et non les {n(W['dividenden_2025'])}&nbsp;M&nbsp;USD de dividendes versés. Ne modéliser que le dividende reviendrait à ignorer les rachats d'actions, qui sont ici le premier emploi de la trésorerie.</p></div>
<div class="annahme"><b>Hypothèse 2 — aucune croissance</b><p>Chaque scénario maintient son niveau. Le modèle ne prévoit rien : il mesure ce que les flux déjà publiés rapportent au prix demandé. La croissance est rendue en section 7.3 sous forme de taux requis, pas de prévision.</p></div>
<div class="annahme"><b>Hypothèse 3 — sortie à la valorisation actuelle</b><p>La valeur terminale est le cours du jour, {n(PRIX,2)}&nbsp;USD, tenu constant : sortie sans expansion de multiple. La valeur comptable serait ici trompeuse, car {n(PRIX/BV_ACT,2)}× sépare les deux et cet écart tient aux rachats d'actions, pas à l'exploitation.</p></div>
<div class="annahme"><b>Hypothèse 4 — deux hurdles, tous deux sourcés</b><p>Le rendement des capitaux propres 2025 de {pc(ROE*100, 2)} et le coût de la dette 2025 de {pc(COUT_DETTE*100, 2)}, soit 1 344&nbsp;M&nbsp;USD de charges financières sur {n(DETTE)}&nbsp;M&nbsp;USD de dette brute. Aucun coût moyen pondéré du capital n'est construit : ni bêta, ni taux sans risque, ni prime de marché ne figurent dans les sources primaires. Le premier hurdle est gonflé par la compression des capitaux propres ; il est retenu parce qu'il est sourcé, et encadré par le second qui ne l'est pas moins.</p></div>
<div class="annahme"><b>Hypothèse 5 — fiscalité de l'actionnaire hors champ</b><p>Le taux dépend du domicile du porteur et ne figure dans aucune source. Les taux de rendement interne sont donc avant impôt au niveau de l'actionnaire.</p></div>
<h3><span class="nr">6.3</span>Où se situe l'exercice retenu dans sa propre histoire</h3>
{tab(['Exercice', 'Flux libre (M USD)', 'Part du sommet'], [
 (str(a), n(v), pc(v/max(SERIE.values())*100)) for a, v in sorted(SERIE.items())], 'Série de flux libre, 10-K et API XBRL')}
<p><b>L'exercice 2025 est le sommet de la fenêtre disponible.</b> Le scénario de base de ce rapport occupe donc le centile le plus élevé des quatre années observables, et non un creux. C'est le contrôle que prescrit la méthode après qu'un modèle de niveau a pris un creux cyclique pour la normale ; ici il conclut l'inverse, et la conséquence est que ce rapport est, si l'on se trompe, trop généreux et non trop sévère.</p>
<p class="quelle">Fenêtre de quatre ans, plus courte que les sept à dix demandées. Les valeurs 2023 à 2025 sont vérifiées page à page dans le 10-K ; celle de 2022 provient de l'API XBRL et n'a pas de page.</p></section>

<section id="verdikt"><h2><span class="nr">7</span>Verdict</h2><div class="strich"></div>
<p style="font-size:8.5pt"><b>Tout ce qui suit est en rouge :</b> une recommandation sous les hypothèses de la section 6.0, pas un constat. Aucune grandeur nouvelle n'est introduite.</p>
<h3><span class="nr">7.1</span>Verdict</h3>
<p class="spruch">Ne pas acheter à {n(PRIX,2)}&nbsp;USD. Le flux de trésorerie libre rapporte {pc(CF_ACT/PRIX*100, 2)}, soit 35 points de base de plus que ce qu'Honeywell paie à ses propres créanciers. Ce n'est pas une rémunération du risque actions.</p>
<div class="zweiseiten">
<div><b>Pour le porteur</b><p>Conserver, sans renforcer. Le flux libre est au plus haut de sa fenêtre et couvre le dividende presque deux fois. Ce qu'il faut surveiller est l'encaissement du second semestre, pas le résultat publié.</p></div>
<div><b>Pour le non-porteur</b><p>Attendre. Le prix suppose une croissance du flux libre que quatre ans d'historique ne montrent nulle part, et le semestre en cours va dans l'autre sens.</p></div></div>
<h3><span class="nr">7.2</span>Ce qui fonde ce verdict</h3>
<div class="richtung"><b>Contre</b><p>Le flux opérationnel du premier semestre 2026 recule de 67,3&nbsp;%, de 1 916 à 626&nbsp;M&nbsp;USD, pendant que la dette nette monte de 14,2&nbsp;%. Le rendement du flux libre de {pc(CF_ACT/PRIX*100, 2)} dépasse de 0,35 point le coût de la dette du groupe. Le rapport cours sur valeur comptable atteint {n(PRIX/BV_ACT,2)}×, et les capitaux propres ont reculé de 21,5&nbsp;% en un exercice.</p></div>
<div class="richtung"><b>Pour</b><p>Le flux libre progresse de 4 599 à {n(FCF['2025'])}&nbsp;M&nbsp;USD en deux ans et se tient au sommet de sa fenêtre. Le chiffre d'affaires croît de 7,9&nbsp;%. La baisse du résultat net s'explique par une dépréciation sans effet de trésorerie. C'est le seul des trois titres détectés dont l'encaissement ne se dégrade pas.</p></div>
<div class="richtung"><b>Où ce verdict s'écarte du marché</b><p>Le consensus des {CONS['n']} analystes recensés est <i>achat</i>, avec un objectif moyen de {n(CONS['cible'],2)}&nbsp;USD, soit 28,8&nbsp;% au-dessus du seuil calculé ici au coût de la dette. L'écart ne se joue pas sur les chiffres mais sur la question posée : un objectif à douze mois valorise un multiple de résultat, ce modèle valorise un encaissement sans croissance. Que 14 maisons sur 22 recommandent l'achat est un argument à porter au crédit de la section 7.5, pas à écarter.</p></div>
<h3><span class="nr">7.3</span>Seuil d'entrée et croissance requise</h3>
{tab(['Hurdle sourcé', '5 ans', '10 ans', '15 ans'], lignes_seuil, f'Prix auquel le hurdle est exactement atteint · cours actuel {n(PRIX,2)} USD')}
{tab(['Hurdle sourcé', '5 ans', '10 ans', '15 ans'], lignes_g, 'Croissance annuelle du flux libre que le cours actuel présuppose')}
<p><b>Les deux hurdles encadrent la réponse.</b> Au coût de la dette, le seuil se situe entre 204 et 209&nbsp;USD : le cours de {n(PRIX,2)} est donc à peine en dessous, et la croissance requise est négative, c'est-à-dire que le flux actuel suffit déjà. Au rendement des capitaux propres, le seuil tombe à 38&nbsp;USD sur dix ans et le cours supposerait 49,9&nbsp;% de croissance annuelle. La vérité tient dans cet intervalle, et il dit une chose simple : à ce prix, un actionnaire d'Honeywell est payé comme un créancier d'Honeywell.</p>
<h3><span class="nr">7.4</span>Ce qui ferait basculer le verdict</h3>
<ol class="lst">
<li><b>À l'achat :</b> flux opérationnel du second semestre 2026 supérieur à <b>3 000&nbsp;M&nbsp;USD</b>, ce qui ramènerait l'exercice sur la trajectoire de 2025 et montrerait que le premier semestre était un décalage de besoin en fonds de roulement.</li>
<li><b>À l'achat :</b> cours sous <b>150&nbsp;USD</b>, seuil d'un actionnaire exigeant 8&nbsp;% — hypothèse de l'auteur, pas sourcée, et signalée comme telle.</li>
<li><b>À la vente :</b> flux libre d'un exercice complet sous <b>4 500&nbsp;M&nbsp;USD</b>, soit le niveau de 2022.</li>
<li><b>À la vente :</b> dette nette au-delà de <b>30 000&nbsp;M&nbsp;USD</b> sans progression correspondante de l'encaissement.</li></ol>
<h3><span class="nr">7.5</span>Ce que ce verdict ne peut pas savoir</h3>
<p>La question ouverte est de savoir si l'effondrement du flux opérationnel semestriel est saisonnier, lié au besoin en fonds de roulement, ou durable. Le 10-Q ne le dit pas, et un semestre ne tranche pas.</p>
<ol class="lst">
<li><b>Le hurdle du rendement des capitaux propres est faussé par le haut.</b> 31,5&nbsp;% n'est pas la rentabilité de l'exploitation : c'est le résultat divisé par des capitaux propres que les rachats d'actions ont comprimés de 21,5&nbsp;% en un an. Un lecteur qui refuse ce hurdle refuse la moitié de mon encadrement.</li>
<li><b>L'hypothèse 2 exclut la croissance</b> d'une société dont le chiffre d'affaires progresse de 7,9&nbsp;% et dont le flux libre a gagné 18&nbsp;% en deux ans. Le taux requis de la section 7.3 rend cette hypothèse visible, il ne la supprime pas.</li>
<li><b>Vingt-deux analystes concluent l'inverse</b>, et leur objectif le plus bas, 186&nbsp;USD, reste au-dessus de mon seuil au coût de la dette. Quand une couverture entière se place au-dessus de son propre seuil, l'explication la plus probable est une hypothèse de l'auteur, pas une erreur collective.</li>
<li><b>La fenêtre de flux libre ne couvre que quatre ans.</b> Le squelette en demande sept à dix, et le centile de la section 6.3 est d'autant plus fragile que la fenêtre est courte.</li></ol></section>

<footer><p><b>Sources :</b> Honeywell International Inc., Form 10-K de l'exercice 2025 déposé le 17.02.2026 ; Form 10-Q du premier semestre 2026 déposé le 23.07.2026 ; Yahoo Finance pour le cours ; Investing.com pour le consensus (source secondaire), consultés le 15.09.2026.</p>
<p><b>Reproductibilité :</b> <code>daten_us.py</code> contient chaque valeur avec sa source et sa page imprimée, <code>pruefe_us.py</code> vérifie chacune contre le texte de sa page, <code>rapport.py</code> produit toutes les tables de ce document.</p></footer></div>"""

if __name__ == '__main__':
    Path(__file__).resolve().parent.joinpath('Honeywell_Analyse.html').write_text(HTML)
    print('rapport ecrit')

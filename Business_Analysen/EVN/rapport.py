"""Rapport EVN AG. Toutes les tables viennent du modèle et du fichier de valeurs."""
import sys
from pathlib import Path
B = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(Path.home()/'.claude/skills/startup-investment-analyzer/scripts'))
from cashflow_irr import break_even_investment, full_cashflows, irr, required_growth
from daten_evn import EXTERN, WERTE

W = {k: v[0] for k, v in WERTE.items()}
SRC = {'JFB': 'Rapport financier annuel 2024/25', 'IR': 'Communiqué de résultats 2024/25',
       'AB': 'Lettre aux actionnaires 9 mois 2025/26'}
Q = {k: f"{SRC[v[1]]}, p.&nbsp;{v[2]}" for k, v in WERTE.items()}
P, A = EXTERN['kurs'][0], EXTERN['aktien_mio'][0]
MCAP = P*A
EK, RES, DIV = W['ek_aktionaere_2425'], W['konzernergebnis_2425'], W['dividende_2425']
BW, EPS = EK/A, RES/A
ROE = RES/EK
ND, EBITDA = W['nettoverschuldung_2425'], W['ebitda_2425']
CONS = dict(note='Buy', n=4, cible=33.25, haut=36.50, bas=29.50, hausse=15.65, cours=28.75)

def n(x, d=1): return f'{x:,.{d}f}'.replace(',', ' ').replace('.', ',')
def pc(x, d=1): return f'{n(x,d)}&nbsp;%'
def dv(a, b): return f'{(b/a-1)*100:+.1f}'.replace('.', ',') + '&nbsp;%'

def tab(kopf, zeilen, cap=''):
    k = ''.join(f'<th>{h}</th>' for h in kopf)
    b = ''.join('<tr' + (' class="summe"' if str(z[0]).startswith('*') else '') + '>'
        + ''.join(f'<td>{c}</td>' for c in [str(z[0]).lstrip('*')]+list(z[1:])) + '</tr>' for z in zeilen)
    c = f'<caption>{cap}</caption>' if cap else ''
    return f'<div class="tabelle"><table>{c}<thead><tr>{k}</tr></thead><tbody>{b}</tbody></table></div>'

def cfg(cf, j, ew):
    return {'currency':'EUR/action','opening_liquidity':0.0,'horizon_start':2026,
      'horizon_end':2026+j,'schedule_years':[2026],'sensitivity_horizons':[2026+j],
      'revenue_base':1.0,'earnings_base':1.0,'hurdle':ROE*100,
      'hurdle_source':'ROE 2024/25, rapport annuel p. 23 et 24',
      'options':{'K':{'name':'Achat','investment':{'2026':P},'earnings_uplift':cf,
                      'uplift_start':2027,'terminal_value':ew}}}

SERIES = [('Dividende versé (0,90&nbsp;€)', DIV), ('Résultat net (2,45&nbsp;€)', EPS)]
SORTIES = [('A — valeur comptable 35,49&nbsp;€', BW), ('B — cours actuel 28,45&nbsp;€', P)]
l_irr, l_seuil, l_g = [], [], []
for ns, cf in SERIES:
    for ne, ew in SORTIES:
        i, s, g = [], [], []
        for j in (5, 10, 15):
            c = cfg(cf, j, ew)
            i.append(pc(irr(full_cashflows(c,'K'))*100, 2))
            be = break_even_investment(c,'K'); s.append(f'{n(be,2)}&nbsp;€' if be else 'n.&nbsp;a.')
            gg = required_growth(c,'K'); g.append(pc(gg*100) if gg is not None else 'hors de portée')
        l_irr.append((f'{ns} · {ne}', *i)); l_seuil.append((f'{ns} · {ne}', *s)); l_g.append((f'{ns} · {ne}', *g))

STIL = (B/'stil_ecran.css').read_text()
HTML = f"""<title>EVN sous sa valeur comptable</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600;700&display=swap">
<style>{STIL}</style><div class="wrap">
<header><p class="eyebrow">Analyse d'entreprise · réseau régulé · Basse-Autriche, Bulgarie, Macédoine du Nord</p>
<h1>EVN sous sa valeur comptable</h1>
<p class="frage">À quel prix une entrée dans EVN est-elle défendable, quand l'action cote 20&nbsp;% sous ses capitaux propres et que les investissements viennent de dépasser 900&nbsp;millions d'euros pour la première fois&nbsp;?</p>
<div class="meta"><span>Date d'arrêté <b>15.09.2026</b></span><span>Bourse <b>Vienne · EVN.VI</b></span>
<span>Exercice <b>clos au 30 septembre</b></span><span>Valeurs vérifiées <b>51 sur 51</b></span></div>
<dl class="legende">
<div><dt><span class="sw" style="background:var(--ink)"></span>Noir</dt><dd>Chiffres, tableaux, sources. La source primaire en répond.</dd></div>
<div><dt style="color:var(--blau)"><span class="sw" style="background:var(--blau)"></span>Bleu</dt><dd>Appréciation de l'auteur, après vérification.</dd></div>
<div><dt style="color:var(--rot)"><span class="sw" style="background:var(--rot)"></span>Rouge</dt><dd>Le verdict, section 7. Recommandation sous hypothèses déclarées.</dd></div>
</dl></header>

<section><h2><span class="nr">1</span>Description de l'entreprise</h2><div class="strich"></div>
<p>EVN AG exploite des réseaux de distribution d'électricité et de gaz, produit de l'électricité et distribue de l'eau potable, sur trois marchés : la Basse-Autriche, la Bulgarie et la Macédoine du Nord. L'essentiel du résultat vient d'activités régulées.</p>
{tab(['Grandeur','Valeur','Source'],[
 ('Cours au 14.09.2026', f'{n(P,2)}&nbsp;€', 'Yahoo Finance'),
 ('Actions', f'{n(A,1)}&nbsp;millions', 'Yahoo Finance'),
 ('Capitalisation', f'{n(MCAP,0)}&nbsp;M€', 'produit des deux'),
 ('Capitaux propres part du groupe', f'{n(EK,1)}&nbsp;M€', Q['ek_aktionaere_2425']),
 ('Capitaux propres par action', f'{n(BW,2)}&nbsp;€', 'quotient'),
 ('*Cours / valeur comptable', f'{n(P/BW,2)}×', ''),
 ('Bêta', n(EXTERN['beta'][0],2), 'Yahoo Finance')], 'Données de cours et de bilan')}
<h3><span class="nr">1.4</span>Couverture et consensus</h3>
{tab(['Grandeur','Valeur','Contre le cours de la page'],[
 ('Analystes recensés', f"{CONS['n']}", '—'),
 ('Consensus', CONS['note'], '—'),
 ('*Objectif moyen', f"{n(CONS['cible'],2)}&nbsp;€", f"+{pc(CONS['hausse'],2)}"),
 ('Objectif haut / bas', f"{n(CONS['haut'],2)} / {n(CONS['bas'],2)}&nbsp;€", '—')],
 'Consensus · source secondaire · 15.09.2026')}
<p class="quelle">MarketScreener, page consensus EVN, consultée le 15.09.2026. Contrôle de cohérence : 33,25 / 1,1565 = 28,75, soit exactement le cours affiché par cette page. <b>Quatre analystes seulement</b> : la couverture est mince, et un consensus à quatre voix ne pèse pas comme un consensus à vingt-deux.</p>
<h3><span class="nr">1.5</span>Lacunes</h3>
<div class="luecken"><ul>
<li><b>Aucun historique de prévision contre réalisé sur cinq ans.</b> Je n'ai dépouillé que l'exercice en cours ; la fiabilité des prévisions d'EVN sur la durée n'est donc pas établie ici.</li>
<li><b>Série de flux de trésorerie sur deux exercices seulement</b> dans les documents téléchargés, au lieu des sept à dix que demande la méthode.</li>
<li><b>Le prix de cession du projet international à STRABAG n'est pas chiffré</b> dans les documents consultés.</li>
</ul></div></section>

<section><h2><span class="nr">2</span>Analyse des comptes</h2><div class="strich"></div>
<p>Exercice 2024/25 : 1er octobre 2024 au 30 septembre 2025. Millions d'euros.</p>
{tab(['Grandeur','2023/24','2024/25','Δ'],[
 ('Chiffre d’affaires', n(W['umsatz_2324'],1), n(W['umsatz_2425'],1), dv(W['umsatz_2324'],W['umsatz_2425'])),
 ('*EBITDA', n(W['ebitda_2324'],1), n(W['ebitda_2425'],1), dv(W['ebitda_2324'],W['ebitda_2425'])),
 ('Amortissements', f"−{n(W['abschreibungen_2425'],1)}", f"−{n(W['abschreibungen_2425'],1)}", '—'),
 ('Dépréciations', '−24,9', f"−{n(W['wertminderung_2425'],1)}", '—'),
 ('*Résultat d’exploitation (EBIT)', n(W['ebit_2324'],1), n(W['ebit_2425'],1), dv(W['ebit_2324'],W['ebit_2425'])),
 ('Résultat financier', n(W['finanzergebnis_2324'],1), n(W['finanzergebnis_2425'],1), dv(W['finanzergebnis_2324'],W['finanzergebnis_2425'])),
 ('Résultat avant impôts', n(W['ebt_2324'],1), n(W['ebt_2425'],1), dv(W['ebt_2324'],W['ebt_2425'])),
 ('*Résultat net part du groupe', '471,6', n(RES,1), '−7,4&nbsp;%'),
 ('Flux de trésorerie opérationnel', n(W['op_cashflow_2324'],1), n(W['op_cashflow_2425'],1), dv(W['op_cashflow_2324'],W['op_cashflow_2425'])),
 ('Dette nette', n(W['nettoverschuldung_2324'],1), n(ND,1), dv(W['nettoverschuldung_2324'],ND)),
 ('Capitaux propres part du groupe', n(W['ek_aktionaere_2324'],1), n(EK,1), dv(W['ek_aktionaere_2324'],EK))],
 'Exercice 2024/25')}
<p class="quelle">{Q['umsatz_2425']} · {Q['nettoverschuldung_2425']} · {Q['op_cashflow_2425']}</p>
<p><b>EBITDA en hausse de 19,2&nbsp;%, résultat net en baisse de 7,4&nbsp;%.</b> L'écart tient à deux postes situés sous l'EBITDA. Les dépréciations passent de 24,9 à {n(W['wertminderung_2425'],1)}&nbsp;M€, liées aux dégâts des crues de 2024 sur les centrales thermiques. Et surtout le résultat financier recule de {n(W['finanzergebnis_2324'],1)} à {n(W['finanzergebnis_2425'],1)}&nbsp;M€, parce que le dividende de Verbund AG est tombé de {n(W['verbund_div_2023'],2)} à {n(W['verbund_div_2024'],2)}&nbsp;€ par action.</p>
<p class="bewertung">Cette structure mérite d'être comprise avant toute conclusion : une part notable du résultat d'EVN ne vient pas de son exploitation mais de sa participation dans Verbund. Un investisseur qui achète EVN achète aussi, indirectement, un dividende hydroélectrique autrichien dont la variation d'une année sur l'autre a suffi à faire reculer le résultat net d'un exercice pourtant meilleur au niveau opérationnel.</p>
<h3>Neuf premiers mois de 2025/26</h3>
{tab(['Grandeur','9M 2024/25','9M 2025/26','Δ'],[
 ('Chiffre d’affaires', n(W['umsatz_9m2425'],1), n(W['umsatz_9m2526'],1), '+3,1&nbsp;%'),
 ('EBITDA', n(W['ebitda_9m2425'],1), n(W['ebitda_9m2526'],1), '+4,9&nbsp;%'),
 ('*Résultat d’exploitation (EBIT)', n(W['ebit_9m2425'],1), n(W['ebit_9m2526'],1), '+3,0&nbsp;%'),
 ('dont 3ᵉ trimestre seul', n(W['ebit_q3_2425'],1), n(W['ebit_q3_2526'],1), '−12,7&nbsp;%'),
 ('Résultat avant impôts', n(W['ebt_9m2425'],1), n(W['ebt_9m2526'],1), '+6,6&nbsp;%'),
 ('*Résultat net part du groupe', n(W['konzernergebnis_9m2425'],1), n(W['konzernergebnis_9m2526'],1), '+20,8&nbsp;%'),
 ('Résultat par action', f"{n(W['eps_9m2425'],2)}&nbsp;€", f"{n(W['eps_9m2526'],2)}&nbsp;€", '+20,8&nbsp;%')],
 'Neuf mois au 30 juin 2026')}
<p class="quelle">{Q['konzernergebnis_9m2526']}</p>
<p class="bewertung">Le bond de 20,8&nbsp;% du résultat net n'est pas opérationnel, et EVN le dit elle-même : « effets ponctuels positifs non monétaires dans le résultat de l'activité abandonnée, conséquence de la déconsolidation ». L'activité abandonnée apporte 32,5&nbsp;M€, et le résultat financier bénéficie d'un dividende Verbund plus élevé. Au niveau qui compte, l'EBIT ne progresse que de 3,0&nbsp;% sur neuf mois et <b>recule de 12,7&nbsp;% au seul troisième trimestre</b>. C'est exactement le motif qui m'avait alerté chez Honeywell, et il faut le nommer ici aussi.</p></section>

<section><h2><span class="nr">3</span>Mesures stratégiques</h2><div class="strich"></div>
<p><i>Cette section rapporte, elle ne juge pas.</i></p>
<p><b>Programme d'investissement.</b> Les investissements ont dépassé 900&nbsp;M€ pour la première fois sur l'exercice 2024/25, dont 89,1&nbsp;% classés conformes à la taxonomie européenne. La stratégie 2030 prévoit environ <b>1&nbsp;milliard d'euros par an jusqu'en 2030</b>, dont quatre cinquièmes en Basse-Autriche : réseaux, production renouvelable, batteries de grande capacité, bornes de recharge, eau potable.</p>
<p><b>Capacités renouvelables.</b> {n(W['erneuerbar_mw'],0)}&nbsp;MW installés au 30 septembre 2025. Objectifs 2030 : 770&nbsp;MW d'éolien, 300&nbsp;MWc de photovoltaïque, 300&nbsp;MW de stockage. La production d'électricité de l'exercice s'établit à {n(W['stromerzeugung_gwh'],0)}&nbsp;GWh, en recul de 12,2&nbsp;% faute de vent et d'eau.</p>
<p><b>Cession du projet international.</b> Contrat signé avec STRABAG en juin 2025 pour la vente de l'activité de projets internationaux, closing attendu début 2026. Le segment est comptabilisé selon IFRS&nbsp;5 et les comparatifs du compte de résultat ont été retraités.</p>
<p><b>Prévision pour 2025/26</b>, donnée le 18 décembre 2025 : EBITDA et résultat net « à peu près au niveau de l'exercice précédent », avec un résultat net attendu dans une fourchette de <b>{n(W['prognose_unten'],0)} à {n(W['prognose_oben'],0)}&nbsp;M€</b>.</p>
<p class="quelle">{Q['prognose_unten']} · {Q['erneuerbar_mw']}</p></section>

<section><h2><span class="nr">4</span>Lecture du cours</h2><div class="strich"></div>
<p>Le titre a perdu 10,7&nbsp;% depuis son sommet à trois ans et se tient 41,2&nbsp;% au-dessus de son plus bas. Sur trois mois il recule de 2,9&nbsp;% quand son indice de référence perd 0,9&nbsp;%. Le bêta ressort à {n(EXTERN['beta'][0],2)}, le plus faible de toutes les valeurs examinées dans cette série.</p>
<p class="bewertung">Il n'y a pas de décrochage à expliquer : EVN ne fait rien de spectaculaire, ni à la hausse ni à la baisse, ce qui est le comportement attendu d'un réseau régulé. L'anomalie n'est pas dans le mouvement, elle est dans le niveau : <b>le marché valorise 178,3 millions d'actions à {n(MCAP,0)}&nbsp;M€ alors que les capitaux propres part du groupe s'élèvent à {n(EK,1)}&nbsp;M€</b>. Il manque un cinquième. Pour une société dont l'actif est un réseau régulé dont la base d'actifs porte un rendement fixé par le régulateur, cette décote est le fait marquant du dossier.</p></section>

<section><h2><span class="nr">5</span>Options d'action</h2><div class="strich"></div>
<div class="entfaellt"><p><b>Sans objet (n&nbsp;=&nbsp;0).</b> La décision porte sur une entrée à un cours donné, non sur un choix entre options de l'entreprise. Numérotation conservée pour rester comparable aux rapports précédents.</p></div></section>

<section><h2><span class="nr">6</span>Analyse de flux</h2><div class="strich"></div>
<h3><span class="nr">6.0</span>Hypothèses</h3>
<div class="annahme"><b>Hypothèse 1 — le flux de trésorerie libre n'est pas la bonne mesure ici</b><p>Le flux libre d'EVN est tombé à 26&nbsp;M€ sur le dernier exercice, non parce que l'exploitation faiblit — le flux opérationnel reste à {n(W['op_cashflow_2425'],1)}&nbsp;M€ — mais parce que les investissements ont dépassé 900&nbsp;M€. Pour un réseau régulé en phase d'investissement, cette compression est le fonctionnement normal du modèle. Mon propre détecteur classe d'ailleurs EVN en « dégradation » sur ce critère, et il a tort, de la même façon qu'il a tort sur une banque. La série retenue ici est donc le <b>résultat</b>, pas le flux libre.</p></div>
<div class="annahme"><b>Hypothèse 2 — deux séries, parce que l'écart entre elles est la question</b><p>Le dividende versé est de {n(DIV,2)}&nbsp;€ par action, le résultat net de {n(EPS,2)}&nbsp;€. La différence de {n(EPS-DIV,2)}&nbsp;€ est retenue pour financer le programme d'investissement. Retenir le dividende seul suppose que le bénéfice conservé ne crée aucune valeur ; retenir le résultat entier suppose qu'il en crée autant qu'un euro distribué. La vérité est entre les deux, et les deux sont calculées.</p></div>
<div class="annahme"><b>Hypothèse 3 — deux conventions de sortie</b><p>A : sortie à la valeur comptable de {n(BW,2)}&nbsp;€ par action. B : sortie au cours actuel de {n(P,2)}&nbsp;€, sans réévaluation. Fait remarquable et inverse des dossiers précédents : <b>ici la convention comptable est la plus généreuse</b>, puisque le titre cote sous ses fonds propres.</p></div>
<div class="annahme"><b>Hypothèse 4 — exigence</b><p>Le rendement des capitaux propres 2024/25, soit {n(RES,1)} / {n(EK,1)} = <b>{pc(ROE*100,2)}</b>. Sourcé, et modeste — c'est le propre d'un actif régulé. Aucun coût moyen pondéré du capital n'est construit.</p></div>
<div class="annahme"><b>Hypothèse 5 — avant impôt de l'actionnaire</b><p>La retenue à la source autrichienne et l'imposition du porteur ne sont pas prises en compte : elles dépendent de son domicile et ne figurent dans aucune source.</p></div>
<h3><span class="nr">6.3</span>Ce que rapporte une entrée au cours actuel</h3>
{tab(['Série · sortie','5 ans','10 ans','15 ans'], l_irr, f'Taux de rendement interne à {n(P,2)} € · exigence {pc(ROE*100,2)}')}
{tab(['Série · sortie','5 ans','10 ans','15 ans'], l_seuil, 'Prix auquel l’exigence est exactement atteinte')}
{tab(['Série · sortie','5 ans','10 ans','15 ans'], l_g, 'Croissance annuelle que le cours actuel présuppose')}
<p><b>Le résultat tient en une ligne.</b> Si l'on retient le résultat net entier et une sortie sans réévaluation, le rendement interne ressort à {pc(8.61,2)} contre une exigence de {pc(ROE*100,2)}, le seuil d'entrée se situe entre {n(30.45,2)} et {n(32.90,2)}&nbsp;€ — <b>au-dessus du cours</b> — et la croissance requise est <b>négative</b> : EVN peut décliner de 3 à 8&nbsp;% par an et satisfaire encore son propre rendement des capitaux propres.</p>
<p>Si l'on ne retient que le dividende versé, tout s'inverse : rendement interne de {pc(3.16,2)}, seuil entre {n(18.70,2)} et {n(24.08,2)}&nbsp;€, croissance requise de {pc(10.7)} à {pc(28.4)}.</p></section>

<section id="verdikt"><h2><span class="nr">7</span>Verdict</h2><div class="strich"></div>
<p style="font-size:14px"><b>Tout ce qui suit est en rouge :</b> recommandation sous les hypothèses de la section 6.0, pas un constat.</p>
<h3><span class="nr">7.1</span>Verdict</h3>
<p class="spruch">Achetable à {n(P,2)}&nbsp;€, et c'est le premier titre de cette série dont je peux le dire. Le seuil d'entrée se situe autour de {n(31,0)}&nbsp;€ si l'on crédite le bénéfice conservé, soit au-dessus du cours actuel. Horizon dix ans.</p>
<div class="zweiseiten">
<div><b>Pour le porteur</b><p>Conserver. Dividende de {n(DIV,2)}&nbsp;€ couvert {n(EPS/DIV,2)} fois, dette nette à {n(ND/EBITDA,2)} fois l'EBITDA, ratio de fonds propres de 60,4&nbsp;%. Rien n'oblige à vendre.</p></div>
<div><b>Pour le non-porteur</b><p>Entrée défendable, à condition d'accepter la thèse centrale : que le milliard d'euros investi chaque année jusqu'en 2030 rejoigne une base d'actifs régulée et y porte un rendement fixé. C'est un pari sur le régulateur, pas sur le marché.</p></div></div>
<h3><span class="nr">7.2</span>Ce qui fonde ce verdict</h3>
<div class="richtung"><b>Pour</b><p>Le titre cote {n(P/BW,2)} fois ses fonds propres, soit une décote de {pc((1-P/BW)*100)} sur des actifs de réseau régulé. Le PER ressort à {n(P/EPS,1)} et le rendement bénéficiaire à {pc(EPS/P*100)}. La dette nette de {n(ND,1)}&nbsp;M€ ne représente que {n(ND/EBITDA,2)} fois l'EBITDA. Après neuf mois, le résultat net de {n(W['konzernergebnis_9m2526'],1)}&nbsp;M€ dépasse déjà le haut de la fourchette annuelle annoncée, {n(W['prognose_oben'],0)}&nbsp;M€.</p></div>
<div class="richtung"><b>Contre</b><p>Ce dépassement de prévision est en grande partie non monétaire : 32,5&nbsp;M€ viennent de la déconsolidation et le reste d'un dividende Verbund plus favorable. L'EBIT ne progresse que de 3,0&nbsp;% sur neuf mois et recule de 12,7&nbsp;% au troisième trimestre seul. Le rendement des capitaux propres n'est que de {pc(ROE*100,2)}, et le flux libre est tombé à 26&nbsp;M€ : tant que le programme d'investissement tourne à un milliard par an, l'actionnaire ne touchera que le dividende.</p></div>
<div class="richtung"><b>Où ce verdict rejoint le marché</b><p>Le consensus des quatre analystes recensés est <i>achat</i>, objectif moyen {n(CONS['cible'],2)}&nbsp;€. C'est, pour une fois, du même ordre que mon seuil d'environ {n(31,0)}&nbsp;€ — l'écart tient dans les arrondis. Après six dossiers où ma méthode concluait systématiquement plus sévèrement que la couverture, cette convergence mérite d'être relevée : elle vient de ce que la décote sur fonds propres est un fait, pas une hypothèse.</p></div>
<h3><span class="nr">7.3</span>Seuil d'entrée</h3>
{tab(['Série · sortie','5 ans','10 ans','15 ans'], l_seuil, f'Cours actuel : {n(P,2)} €')}
<h3><span class="nr">7.4</span>Ce qui ferait basculer le verdict</h3>
<ol class="lst">
<li><b>À la vente :</b> résultat net d'un exercice complet sous <b>400&nbsp;M€</b>, soit en dessous du bas de la fourchette annoncée.</li>
<li><b>À la vente :</b> dette nette au-delà de <b>2,5 fois l'EBITDA</b>, signe que le programme d'investissement dépasse la capacité de financement.</li>
<li><b>À la vente :</b> coupe du dividende sous <b>0,90&nbsp;€</b>, qui signalerait que le milliard annuel n'est plus finançable en interne.</li>
<li><b>À l'achat renforcé :</b> cours sous <b>25&nbsp;€</b>, soit une décote de 30&nbsp;% sur les fonds propres.</li>
<li><b>À surveiller :</b> le dividende de Verbund AG, qui a fait à lui seul reculer le résultat financier de {n(W['finanzergebnis_2324']-W['finanzergebnis_2425'],1)}&nbsp;M€ en un exercice.</li></ol>
<h3><span class="nr">7.5</span>Ce que ce verdict ne peut pas savoir</h3>
<p>La question ouverte est celle du <b>rendement régulé</b> : la valeur d'EVN dépend du taux que les régulateurs autrichien, bulgare et macédonien accorderont sur une base d'actifs en forte croissance. Aucune source consultée ne donne ce taux, et c'est pourtant lui qui décide si le milliard investi chaque année crée ou détruit de la valeur.</p>
<ol class="lst">
<li><b>Mon verdict repose sur le crédit accordé au bénéfice conservé.</b> Sur le seul dividende, le seuil tombe à {n(20.95,2)}&nbsp;€ sur dix ans et le titre devient cher. Tout l'écart entre « achetable » et « cher » tient dans cette hypothèse, et elle n'est pas démontrée ici.</li>
<li><b>La décote sur fonds propres peut être justifiée.</b> Un marché qui valorise un réseau à 0,8 fois ses fonds propres peut anticiper une révision réglementaire défavorable, ou juger que les actifs sont comptabilisés trop haut. Je n'ai aucun élément pour trancher.</li>
<li><b>Quatre analystes ne font pas un consensus.</b> La convergence relevée en 7.2 s'appuie sur une couverture très mince.</li>
<li><b>L'exposition à Verbund est un actif que je n'ai pas analysé.</b> Elle influence sensiblement le résultat et sort du périmètre de ce rapport.</li>
<li><b>Une série de deux exercices</b> ne permet pas de situer l'exercice retenu dans son cycle, comme la méthode l'exige depuis le cas Neste.</li></ol></section>

<footer><p><b>Sources :</b> EVN AG, rapport financier annuel 2024/25 (175 p.) ; communiqué de résultats du 18.12.2025 ; lettre aux actionnaires des neuf premiers mois 2025/26 ; Yahoo Finance pour le cours ; MarketScreener pour le consensus (source secondaire), consultés le 15.09.2026.</p>
<p><b>Reproductibilité :</b> <code>daten_evn.py</code> contient les 51 valeurs avec leur page imprimée, <code>pruefe_evn.py</code> vérifie chacune contre le texte de sa page, <code>rapport.py</code> produit toutes les tables.</p></footer></div>"""

if __name__ == '__main__':
    (Path(__file__).resolve().parent/'EVN_Analyse.html').write_text(HTML)
    print('rapport ecrit')

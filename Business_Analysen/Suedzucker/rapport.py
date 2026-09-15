import sys
from pathlib import Path
B = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(Path.home()/'.claude/skills/startup-investment-analyzer/scripts'))
from cashflow_irr import break_even_investment, full_cashflows, irr, required_growth
from daten_szu import EXTERN, WERTE
W = {k: v[0] for k, v in WERTE.items()}
SRC = {'GB': 'Rapport annuel 2025/26', 'Q1': 'Rapport du 1er trimestre 2026/27'}
Q = {k: f"{SRC[v[1]]}, p.&nbsp;{v[2]}" for k, v in WERTE.items()}
P, A = EXTERN['kurs'][0], EXTERN['aktien_mio'][0]
MC = P*A; EK = W['ek_aktionaere_2526']; BW = EK/A
FCF = {a: W[f'op_cashflow_{a}'] - W[f'investitionen_{a}'] for a in ('2223','2324','2425','2526')}
EBITDA = {a: W[f'op_ebitda_{a}'] for a in ('2223','2324','2425','2526')}
HURDLE = 0.099
def n(x, d=1): return f'{x:,.{d}f}'.replace(',', ' ').replace('.', ',')
def pc(x, d=1): return f'{n(x,d)}&nbsp;%'
def tab(kopf, zeilen, cap=''):
    k = ''.join(f'<th>{h}</th>' for h in kopf)
    b = ''.join('<tr' + (' class="summe"' if str(z[0]).startswith('*') else '') + '>'
        + ''.join(f'<td>{c}</td>' for c in [str(z[0]).lstrip('*')]+list(z[1:])) + '</tr>' for z in zeilen)
    return f'<div class="tabelle"><table>{f"<caption>{cap}</caption>" if cap else ""}<thead><tr>{k}</tr></thead><tbody>{b}</tbody></table></div>'
SC = [('Creux — exercice 2025/26', FCF['2526']), ('Médian — exercice 2024/25', FCF['2425']),
      ('Haut — exercice 2023/24', FCF['2324'])]
def cfg(cf, taux, j):
    return {'currency':'EUR/action','opening_liquidity':0.0,'horizon_start':2026,'horizon_end':2026+j,
      'schedule_years':[2026],'sensitivity_horizons':[2026+j],'revenue_base':1.0,'earnings_base':1.0,
      'hurdle':taux*100,'hurdle_source':'ROCE, rapport annuel 2025/26 p. 4',
      'options':{'K':{'name':'Achat','investment':{'2026':P},'earnings_uplift':cf,
                      'uplift_start':2027,'terminal_value':P}}}
l_irr, l_seuil, l_g = [], [], []
for nom, v in SC:
    cf = v/A; i=[]; s=[]; g=[]
    for j in (5,10,15):
        c = cfg(cf, HURDLE, j)
        i.append(pc(irr(full_cashflows(c,'K'))*100,2))
        be = break_even_investment(c,'K'); s.append(f'{n(be,2)}&nbsp;€' if be else 'n.&nbsp;a.')
        gg = required_growth(c,'K'); g.append(pc(gg*100) if gg is not None else 'hors de portée')
    l_irr.append((f'{nom} · {n(cf,2)} €/action', *i)); l_seuil.append((nom, *s)); l_g.append((nom, *g))

STIL = (B/'stil_ecran.css').read_text()
HTML = f"""<title>Südzucker au creux du sucre</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600;700&display=swap">
<style>{STIL}</style><div class="wrap">
<header><p class="eyebrow">Analyse d'entreprise · sucre, amidon, bioéthanol, fruits · exercice clos fin février</p>
<h1>Südzucker au creux du sucre</h1>
<p class="frage">La perte de 378&nbsp;millions d'euros est-elle cyclique ou structurelle&nbsp;? De la réponse dépend tout&nbsp;: au niveau de trésorerie de l'exercice écoulé, le titre vaut 5&nbsp;€&nbsp;; à celui de l'exercice précédent, il en vaut 15.</p>
<div class="meta"><span>Date d'arrêté <b>15.09.2026</b></span><span>Bourse <b>XETRA · SZU.DE</b></span>
<span>Valeurs vérifiées <b>65 sur 65</b></span><span>Pagination <b>imprimée = PDF − 2</b></span></div>
<dl class="legende">
<div><dt><span class="sw" style="background:var(--ink)"></span>Noir</dt><dd>Chiffres, tableaux, sources.</dd></div>
<div><dt style="color:var(--blau)"><span class="sw" style="background:var(--blau)"></span>Bleu</dt><dd>Appréciation de l'auteur, après vérification.</dd></div>
<div><dt style="color:var(--rot)"><span class="sw" style="background:var(--rot)"></span>Rouge</dt><dd>Le verdict, section 7.</dd></div>
</dl></header>

<section><h2><span class="nr">1</span>Description de l'entreprise</h2><div class="strich"></div>
<p>Südzucker est le premier sucrier européen, avec quatre autres métiers : produits spéciaux, amidon, bioéthanol et fruits. L'exercice se clôt fin février. {n(W['mitarbeiter_2526'],0)} salariés en équivalent temps plein au 28 février 2026.</p>
{tab(['Grandeur','Valeur','Source'],[
 ('Cours au 15.09.2026', f'{n(P,2)}&nbsp;€', 'Yahoo Finance'),
 ('Actions', f'{n(A,1)}&nbsp;millions', EXTERN['aktien_mio'][1]),
 ('Capitalisation', f'{n(MC,0)}&nbsp;M€', 'produit des deux'),
 ('*Capitaux propres part du groupe', f'{n(EK,0)}&nbsp;M€', Q['ek_aktionaere_2526']),
 ('Capital hybride', f"{n(W['hybridkapital_2526'],0)}&nbsp;M€", Q['hybridkapital_2526']),
 ('Intérêts minoritaires', f"{n(W['minderheiten_2526'],0)}&nbsp;M€", Q['minderheiten_2526']),
 ('Capitaux propres totaux', f"{n(W['ek_gesamt_2526'],0)}&nbsp;M€", Q['ek_gesamt_2526']),
 ('*Cours / valeur comptable part du groupe', f'{n(P/BW,2)}×', '')], 'Structure du capital au 28 février 2026')}
<p class="bewertung"><b>Un piège à signaler d'emblée.</b> Le tableau de bord du rapport annuel intitule « Shareholders' equity » la somme de {n(W['ek_gesamt_2526'],0)}&nbsp;M€. Le bilan montre que cette somme comprend {n(W['hybridkapital_2526'],0)}&nbsp;M€ de capital hybride et {n(W['minderheiten_2526'],0)}&nbsp;M€ de minoritaires. La part réellement attribuable aux actionnaires de Südzucker AG est de {n(EK,0)}&nbsp;M€. Calculé sur le total, le rapport cours/valeur comptable ressort à 0,73× et le titre paraît décoté ; calculé correctement, il vaut <b>{n(P/BW,2)}×</b> et la décote disparaît.</p>
<h3><span class="nr">1.5</span>Lacunes</h3>
<div class="luecken"><ul>
<li><b>Fenêtre de quatre exercices seulement.</b> Le tableau de bord du rapport annuel n'en couvre pas davantage, alors que la méthode en demande sept à dix. Pour un cycle sucrier, c'est court.</li>
<li><b>Aucun consensus d'analystes recherché</b> pour ce dossier.</li>
<li><b>Le prix du sucre lui-même n'est pas dans ce rapport.</b> C'est pourtant la variable qui décide de tout, et elle est exogène.</li>
</ul></div></section>

<section><h2><span class="nr">2</span>Analyse des comptes</h2><div class="strich"></div>
{tab(['Millions d’euros','2022/23','2023/24','2024/25','2025/26'],[
 ('Chiffre d’affaires', n(W['umsatz_2223'],0), n(W['umsatz_2324'],0), n(W['umsatz_2425'],0), n(W['umsatz_2526'],0)),
 ('*EBITDA opérationnel', n(EBITDA['2223'],0), n(EBITDA['2324'],0), n(EBITDA['2425'],0), n(EBITDA['2526'],0)),
 ('Marge d’EBITDA opérationnel', '11,3&nbsp;%', pc(W['op_marge_2324']), '7,5&nbsp;%', pc(W['op_marge_2526'])),
 ('Résultat opérationnel', '704', n(W['op_ergebnis_2324'],0), n(W['op_ergebnis_2425'],0), n(W['op_ergebnis_2526'],0)),
 ('*Résultat net', n(W['nettoergebnis_2223'],0), n(W['nettoergebnis_2324'],0), n(W['nettoergebnis_2425'],0), n(W['nettoergebnis_2526'],0)),
 ('Résultat par action, €', '1,93', n(W['eps_2324'],2), n(W['eps_2425'],2), n(W['eps_2526'],2)),
 ('Flux de trésorerie opérationnel', n(W['op_cashflow_2223'],0), n(W['op_cashflow_2324'],0), n(W['op_cashflow_2425'],0), n(W['op_cashflow_2526'],0)),
 ('Investissements corporels', n(W['investitionen_2223'],0), n(W['investitionen_2324'],0), n(W['investitionen_2425'],0), n(W['investitionen_2526'],0)),
 ('*Flux de trésorerie libre', n(FCF['2223'],0), n(FCF['2324'],0), n(FCF['2425'],0), n(FCF['2526'],0)),
 ('Dette financière nette', '1 864', n(W['nettoschuld_2324'],0), n(W['nettoschuld_2425'],0), n(W['nettoschuld_2526'],0)),
 ('*Dette nette / EBITDA opérationnel', '1,7×', f"{n(W['verschuldungsgrad_2324'],1)}×", '2,3×', f"{n(W['verschuldungsgrad_2526'],1)}×"),
 ('Rendement des capitaux employés', pc(W['roce_2223']), pc(W['roce_2324']), pc(W['roce_2425']), pc(W['roce_2526'])),
 ('*Dividende par action, €', '0,70', n(W['dividende_2324'],2), n(W['dividende_2425'],2), n(W['dividende_2526'],2))],
 'Quatre exercices · rapport annuel 2025/26, page 4')}
<p class="quelle">{Q['umsatz_2526']}</p>
<p><b>L'effondrement est réel et il porte sur tout.</b> L'EBITDA opérationnel passe de {n(EBITDA['2324'],0)} à {n(EBITDA['2526'],0)}&nbsp;M€, soit une division par 2,5 en deux exercices. La marge tombe de {pc(W['op_marge_2324'])} à {pc(W['op_marge_2526'])}. Le résultat net enchaîne deux pertes, {n(W['nettoergebnis_2425'],0)} puis {n(W['nettoergebnis_2526'],0)}&nbsp;M€. Le rendement des capitaux employés descend de {pc(W['roce_2324'])} à {pc(W['roce_2526'])}.</p>
<p><b>Et le dividende a été supprimé</b>, de 0,90&nbsp;€ à 0,20 puis à zéro. C'est le signal le plus lourd du dossier : un conseil qui coupe entièrement le dividende d'un groupe coopératif ne le fait pas par prudence rhétorique.</p>
<p class="bewertung">Deux choses résistent pourtant. Le flux de trésorerie opérationnel reste positif à {n(W['op_cashflow_2526'],0)}&nbsp;M€, ce qui n'est pas le cas des sociétés réellement en perdition — Baidu, à titre de comparaison, est passé en flux opérationnel négatif. Et le ratio de fonds propres tient à {pc(W['ek_quote_2526'])}. Südzucker perd de l'argent, mais elle ne brûle pas de trésorerie d'exploitation.</p>
<h3>Premier trimestre 2026/27, clos le 31 mai 2026</h3>
{tab(['Millions d’euros','T1 2025/26','T1 2026/27','Δ'],[
 ('Chiffre d’affaires', n(W['umsatz_q1_2526'],0), n(W['umsatz_q1_2627'],0), '−4,4&nbsp;%'),
 ('*EBITDA opérationnel', n(W['op_ebitda_q1_2526'],0), n(W['op_ebitda_q1_2627'],0), '+40,6&nbsp;%'),
 ('Marge d’EBITDA opérationnel', pc(W['op_marge_q1_2526']), pc(W['op_marge_q1_2627']), '+2,1&nbsp;pt'),
 ('Résultat opérationnel', n(W['op_ergebnis_q1_2526'],0), n(W['op_ergebnis_q1_2627'],0), '+182&nbsp;%'),
 ('*Résultat net', n(W['nettoergebnis_q1_2526'],0), n(W['nettoergebnis_q1_2627'],0), 'retour au bénéfice'),
 ('Dette financière nette', '1 755', n(W['nettoschuld_q1_2627'],0), '+5,7&nbsp;%')],
 'Premier trimestre · rapport trimestriel, page 3')}
<p class="quelle">{Q['op_ebitda_q1_2627']}</p>
<p class="bewertung">C'est le fait nouveau du dossier, et il va dans le bon sens : la marge remonte de {pc(W['op_marge_q1_2526'])} à {pc(W['op_marge_q1_2627'])}, le résultat opérationnel presque triple, et le résultat net repasse de {n(W['nettoergebnis_q1_2526'],0)} à <b>+{n(W['nettoergebnis_q1_2627'],0)}&nbsp;M€</b>. Le chiffre d'affaires continue de reculer, ce qui est cohérent avec des prix de vente encore bas : ce n'est pas le volume qui redresse la marge, c'est le coût.</p></section>

<section><h2><span class="nr">3</span>Mesures stratégiques et prévision</h2><div class="strich"></div>
<p><i>Cette section rapporte, elle ne juge pas.</i></p>
<p>Pour l'exercice 2026/27, Südzucker prévoit un chiffre d'affaires compris entre 8,1 et 8,5&nbsp;milliards d'euros et un <b>EBITDA opérationnel entre {n(W['prognose_ebitda_unten'],0)} et {n(W['prognose_ebitda_oben'],0)}&nbsp;M€</b>, contre {n(EBITDA['2526'],0)}&nbsp;M€ réalisés.</p>
<p>La fourchette est large : le bas est inférieur de 10&nbsp;% à l'exercice écoulé, le haut supérieur de 27&nbsp;%. Le milieu, 580&nbsp;M€, représente une progression de 8&nbsp;%. Après un trimestre, l'EBITDA opérationnel s'établit à {n(W['op_ebitda_q1_2627'],0)}&nbsp;M€, soit 28&nbsp;% du bas de la fourchette et 20&nbsp;% du haut.</p>
<p class="quelle">{Q['prognose_ebitda_unten']}</p></section>

<section><h2><span class="nr">4</span>Lecture du cours</h2><div class="strich"></div>
<p>L'action cotait {n(W['kurs_ende_2526'],2)}&nbsp;€ au 28 février 2026, contre 16,06&nbsp;€ deux exercices plus tôt. Elle vaut aujourd'hui {n(P,2)}&nbsp;€, en hausse de {pc((P/W['kurs_ende_2526']-1)*100)} depuis la clôture, et se tient 40&nbsp;% au-dessus de son plus bas à trois ans. Le bêta ressort à {n(EXTERN['beta'][0],2)}, le plus faible de toutes les valeurs de cette série.</p>
<p class="bewertung">Le cours ne suit ni le chiffre d'affaires, qui recule encore, ni le résultat net, qui est négatif. Il suit <b>la marge</b>. La remontée de 4,5 à 6,6&nbsp;% au premier trimestre est le seul chiffre publié qui explique un titre en hausse de 24&nbsp;% depuis la clôture d'un exercice en perte. Pour un transformateur de matière première, c'est la lecture attendue : le marché achète l'écart entre le prix de la betterave et celui du sucre, pas le volume.</p></section>

<section><h2><span class="nr">5</span>Options d'action</h2><div class="strich"></div>
<div class="entfaellt"><p><b>Sans objet (n&nbsp;=&nbsp;0).</b> La décision porte sur une entrée à un cours donné.</p></div></section>

<section><h2><span class="nr">6</span>Analyse de flux</h2><div class="strich"></div>
<h3><span class="nr">6.0</span>Hypothèses</h3>
<div class="annahme"><b>Hypothèse 1 — trois niveaux réalisés, aucun niveau inventé</b><p>Les trois scénarios sont les flux de trésorerie libres des trois derniers exercices : {n(FCF['2526'],0)}, {n(FCF['2425'],0)} et {n(FCF['2324'],0)}&nbsp;M€. Aucun n'est une prévision. L'exercice 2022/23, négatif à {n(FCF['2223'],0)}&nbsp;M€, est écarté parce qu'il précède le pic et gonflerait artificiellement l'amplitude ; ce retrait flatte la fourchette et doit être su.</p></div>
<div class="annahme"><b>Hypothèse 2 — où se situe l'exercice retenu dans son propre cycle</b><p>C'est le contrôle décisif pour un cyclique, et il est sans appel : l'EBITDA opérationnel de 2025/26, {n(EBITDA['2526'],0)}&nbsp;M€, est le <b>minimum des quatre exercices connus</b>, soit {pc(EBITDA['2526']/max(EBITDA.values())*100,0)} du sommet. Le scénario de base occupe donc le centile zéro de la fenêtre disponible. Un modèle de niveau qui prendrait cet exercice pour la normale conclurait à la faillite, et il aurait tort de la même façon qu'il aurait eu tort sur Neste en février 2025.</p></div>
<div class="annahme"><b>Hypothèse 3 — sortie sans réévaluation</b><p>Valeur terminale égale au cours du jour, {n(P,2)}&nbsp;€, tenue constante. La valeur comptable part du groupe, {n(BW,2)}&nbsp;€ par action, serait <i>moins</i> généreuse ici puisqu'elle est inférieure au cours.</p></div>
<div class="annahme"><b>Hypothèse 4 — exigence à mi-cycle</b><p>Le rendement des capitaux employés de {pc(W['roce_2526'])} de l'exercice écoulé est un chiffre de creux, inutilisable comme exigence. Celui de 2022/23, <b>{pc(W['roce_2223'])}</b>, est retenu : c'est un exercice ni au sommet ni au creux, et il est sourcé. Aucun coût moyen pondéré du capital n'est construit.</p></div>
<div class="annahme"><b>Hypothèse 5 — dividende nul</b><p>Aucun dividende n'est versé au titre de 2025/26. La série porte donc le flux libre, pas une distribution.</p></div>
<h3><span class="nr">6.3</span>Ce que rapporte une entrée au cours actuel</h3>
{tab(['Scénario · flux par action','5 ans','10 ans','15 ans'], l_irr, f'Taux de rendement interne à {n(P,2)} € · exigence {pc(HURDLE*100,1)}')}
{tab(['Scénario','5 ans','10 ans','15 ans'], l_seuil, 'Prix auquel l’exigence est exactement atteinte')}
{tab(['Scénario','5 ans','10 ans','15 ans'], l_g, 'Croissance annuelle que le cours actuel présuppose')}
<p><b>Tout le dossier tient dans l'écart entre les deux premières lignes.</b> Si Südzucker reste au niveau de trésorerie de l'exercice écoulé, le titre vaut {n(5.26,2)}&nbsp;€ et le cours actuel en suppose 60&nbsp;% de croissance annuelle. S'il retrouve le niveau de l'exercice précédent — pas le sommet, l'exercice <i>précédent</i> — il vaut {n(14.93,2)}&nbsp;€, soit 19&nbsp;% au-dessus du cours, et peut même décliner de 5,7&nbsp;% par an en satisfaisant encore l'exigence.</p></section>

<section id="verdikt"><h2><span class="nr">7</span>Verdict</h2><div class="strich"></div>
<p style="font-size:14px"><b>Tout ce qui suit est en rouge :</b> recommandation sous les hypothèses de la section 6.0.</p>
<h3><span class="nr">7.1</span>Verdict</h3>
<p class="spruch">Achat spéculatif à {n(P,2)}&nbsp;€, pour qui accepte de parier sur le retour du cycle sucrier — pas un achat de qualité. Le seuil bascule entre 5 et 15&nbsp;€ selon le seul niveau de marge, et aucune analyse de bilan ne tranche cette question.</p>
<div class="zweiseiten">
<div><b>Pour le porteur</b><p>Conserver. Le premier trimestre montre une marge qui remonte et un retour au bénéfice ; vendre maintenant, c'est réaliser la perte au moment précis où l'indicateur se retourne.</p></div>
<div><b>Pour le non-porteur</b><p>Entrée défendable en petite taille seulement. La thèse est binaire et le dividende est nul : vous ne serez pas payé pour attendre.</p></div></div>
<h3><span class="nr">7.2</span>Ce qui fonde ce verdict</h3>
<div class="richtung"><b>Pour</b><p>Le premier trimestre 2026/27 montre un EBITDA opérationnel en hausse de 40,6&nbsp;%, une marge de {pc(W['op_marge_q1_2526'])} à {pc(W['op_marge_q1_2627'])} et un retour au bénéfice à +{n(W['nettoergebnis_q1_2627'],0)}&nbsp;M€. La prévision de l'entreprise, {n(W['prognose_ebitda_unten'],0)} à {n(W['prognose_ebitda_oben'],0)}&nbsp;M€, place son milieu 8&nbsp;% au-dessus de l'exercice écoulé. Le flux de trésorerie opérationnel n'a jamais cessé d'être positif, et le ratio de fonds propres tient à {pc(W['ek_quote_2526'])}. Au niveau de trésorerie de 2024/25, le titre vaut {n(14.93,2)}&nbsp;€.</p></div>
<div class="richtung"><b>Contre</b><p>Deux exercices de pertes consécutives, {n(W['nettoergebnis_2425'],0)} puis {n(W['nettoergebnis_2526'],0)}&nbsp;M€. <b>Dividende supprimé</b>, de 0,90 à zéro. Dette nette à {n(W['verschuldungsgrad_2526'],1)} fois l'EBITDA opérationnel, contre {n(W['verschuldungsgrad_2324'],1)} fois deux ans plus tôt. Et le titre ne cote pas sous sa valeur comptable : à {n(P/BW,2)}× les capitaux propres part du groupe, il est au-dessus. Au niveau de trésorerie actuel, il vaut {n(5.26,2)}&nbsp;€.</p></div>
<h3><span class="nr">7.3</span>Seuil d'entrée</h3>
{tab(['Scénario','5 ans','10 ans','15 ans'], l_seuil, f'Cours actuel : {n(P,2)} €')}
<p>La fourchette de 5,26 à 20,82&nbsp;€ n'est pas une imprécision : c'est la mesure de l'amplitude cyclique de cette entreprise. Le cours de {n(P,2)}&nbsp;€ se situe au premier tiers.</p>
<h3><span class="nr">7.4</span>Ce qui ferait basculer le verdict</h3>
<ol class="lst">
<li><b>À l'achat renforcé :</b> EBITDA opérationnel semestriel supérieur à <b>300&nbsp;M€</b>, qui placerait l'exercice au-dessus du haut de la fourchette annoncée.</li>
<li><b>À l'achat renforcé :</b> rétablissement d'un dividende, qui signalerait que le conseil tient la reprise pour acquise.</li>
<li><b>À la vente :</b> marge d'EBITDA opérationnel de nouveau sous <b>5&nbsp;%</b> sur deux trimestres consécutifs.</li>
<li><b>À la vente :</b> dette nette au-delà de <b>4 fois</b> l'EBITDA opérationnel.</li>
<li><b>À la vente :</b> troisième exercice de perte, qui ferait passer le dossier du cyclique au structurel.</li></ol>
<h3><span class="nr">7.5</span>Ce que ce verdict ne peut pas savoir</h3>
<p>La variable qui décide de tout est <b>le prix européen du sucre</b>, et elle est exogène : elle dépend des récoltes, des importations ukrainiennes, de la politique commerciale et des surfaces plantées. Aucun document de Südzucker ne la prévoit, et je n'ai aucun moyen de le faire.</p>
<ol class="lst">
<li><b>La fenêtre ne couvre que quatre exercices.</b> Pour un cycle sucrier qui se compte en cinq à sept ans, c'est insuffisant pour affirmer que 2025/26 est un creux plutôt qu'un nouveau palier.</li>
<li><b>J'ai écarté l'exercice 2022/23</b>, dont le flux libre était négatif à {n(FCF['2223'],0)}&nbsp;M€. L'inclure élargirait la fourchette vers le bas et rendrait le verdict plus prudent.</li>
<li><b>Un trimestre ne fait pas un retournement.</b> Le premier trimestre 2026/27 est encourageant, mais il représente 28&nbsp;% du bas de la fourchette annuelle : rien n'est acquis.</li>
<li><b>Le dividende nul supprime la rémunération de l'attente.</b> Si le cycle met trois ans à revenir, le porteur n'aura rien touché entre-temps, ce qu'aucun taux de rendement interne calculé ici ne reflète.</li>
<li><b>Je n'ai pas cherché le consensus des analystes</b> sur ce dossier, contrairement aux précédents.</li></ol></section>

<footer><p><b>Sources :</b> Südzucker AG, rapport annuel 2025/26 (278 pages, publié le 20 mai 2026) ; rapport du premier trimestre 2026/27 (20 pages, publié le 20 juillet 2026) ; Yahoo Finance pour le cours, consulté le 15.09.2026.</p>
<p><b>Reproductibilité :</b> <code>daten_szu.py</code> contient les 65 valeurs avec leur page imprimée, <code>pruefe_szu.py</code> vérifie chacune contre le texte de sa page, <code>rapport.py</code> produit toutes les tables.</p></footer></div>"""
if __name__ == '__main__':
    (Path(__file__).resolve().parent/'Suedzucker_Analyse.html').write_text(HTML)
    print('rapport ecrit')

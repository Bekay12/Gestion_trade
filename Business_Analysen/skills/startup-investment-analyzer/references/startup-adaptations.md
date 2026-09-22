# Startup adaptations

The skeleton assumes a listed company with published reports. A private startup has no
share price, no analyst coverage, no dividend, and no audited annual report — but the same
structure holds once each input is substituted. **The rigor requirements do not relax.**
A startup's data is thinner, which makes declared gaps more important, not less.

---

## Substitution map

| Listed-company input | Startup substitute | Where to get it |
|---|---|---|
| Aktienkurs, Kursverlauf | Valuation history across rounds (post-money per round) | Term sheets, cap table, Crunchbase with retrieval date |
| Marktkapitalisierung | Last post-money valuation, **with its date** | Round documents |
| Dividendenrendite | Not applicable — state it, don't invent a proxy | — |
| Analystenabdeckung | Existing investors, board observers, notable angels | Cap table, press releases |
| Analystenbotschaften | Investor letters, board minutes, prior diligence memos | Data room |
| Umsatz + Δ | ARR / MRR + growth rate, **net vs. gross stated** | Financial statements, billing system export |
| Auftragseingang | Bookings, signed pipeline, LOIs | CRM export, sales report |
| EBITDA-Marge | Gross margin and contribution margin | P&L, unit economics model |
| Operatives EBITDA | Burn (net and gross), **and runway in months** | Bank statements, financial plan |
| Nettoliquidität | Cash on hand + committed undrawn facilities | Bank statements |
| Eigentümerstruktur | Cap table: founders, employee pool, investors by round, **fully diluted** | Cap table |
| Geschäftsbericht | Data room, board decks, monthly investor updates | Data room |

---

## Section-by-section changes

### 1. Company description

1.1 unchanged in spirit: founders, key hires, board, advisors — with prior track record.
1.2 becomes the **cap table**: shares by class, employee option pool, dilution per round,
liquidation preferences, and any founder vesting still running. Preferences matter more
than percentages — a 1x participating preference changes who gets paid.

1.3 becomes **traction**: ARR/MRR trajectory, growth rate, and the retention numbers.
1.4/1.5 become **existing investors and what they signalled** — did the lead follow on in
the last round? Non-participation by an insider is information; note it factually. This
is also where the listed company's directors' dealings (1.2) land: a private company has
no trade register, so insider behaviour shows only as follow-on or its absence.

1.6 keeps its table and the rule that each claimed advantage names its figure; the
evidence shifts to retention cohorts, gross margin against named competitors, and the
patents or licences themselves. Section 5's **Competitive advantage** criterion then
judges what 1.6 established, rather than restating it.

The 7.3 multiple-in-context table has no own history to draw on; it becomes the round's
valuation multiple against comparable rounds, which section 5 already requires under
**Round terms**. Do not build it twice.

### 2. Financial analysis

Same equation-then-calculation discipline, on startup metrics:

- **ARR / MRR** with growth rate, and whether it is net or gross of churn
- **Gross margin** and contribution margin per unit
- **Burn**: gross and net, monthly
- **Runway** = cash on hand / net monthly burn — state the burn window used
- **Net revenue retention** and logo churn
- **CAC payback** and LTV/CAC, with the discount and churn assumptions declared

**The startup equivalent of the reported/adjusted trap** is ARR definition. "ARR"
including pilots, LOIs, or non-recurring services is not ARR. State the definition you are
using, show both if the company reports both, and reconcile the difference.

Section 2.7 becomes the **valuation and round history** chart: post-money by round against
ARR at the time, which shows the multiple trajectory.

### 3. Strategic measures

Unchanged in method: report factually what the company says it is doing — product roadmap,
go-to-market shifts, hiring plan, geographic expansion — with no evaluative adjectives.

The **Ausblick vs. Ist** table is the highest-value part here. Compare each prior plan
against what actually happened:

| Metric | Plan (date) | Actual | Position |
|---|---|---|---|
| ARR end of year | target | actual | below / within / above |
| Headcount | target | actual | |
| Burn | target | actual | |

A founder who consistently hits plan is a different investment from one who consistently
misses by 40 %, and this table is the only place that shows it. If prior plans are not in
the data room, that absence is itself a finding — record it.

### 4. Explaining the trajectory

The listed-company insight was that price tracked the **leading indicator** (order intake),
not the income statement. The startup equivalents: bookings and pipeline lead ARR; net
revenue retention leads long-run growth far more than new logo count.

Explain the ARR curve through the leading indicator, not by restating the ARR curve.

### 5. Options → investment criteria

Replace the corporate options grid with investment criteria. **Keep the six-factor grid
shape** so reports stay comparable, and evaluate each criterion with the same rows:

| Criterion | What to assess |
|---|---|
| **Market size** | TAM/SAM/SOM with the derivation shown — a TAM without a bottom-up cross-check is marketing |
| **Team** | Founder-market fit, prior outcomes, key-person risk, gaps in the org |
| **Competitive advantage** | What is actually defensible: data, distribution, switching costs, regulatory position. Being "first" is not a moat |
| **Use of funds** | What this round buys, and which milestone it reaches. Map it to runway |
| **Round terms** | Valuation vs. comparable multiples, preferences, pro-rata, board composition |

Where the corporate version compares n strategic options, the startup version compares
**n scenarios**: invest at the proposed terms / invest at revised terms / pass / wait for a
named milestone. The Empfehlung then names one, and — as in the corporate case — a
staged answer (a smaller cheque now with pro-rata reserved for the next round) is often
better than a binary.

### 6. Flow analysis

Directly reusable, with the inputs renamed:

- **Anfangsbestand** → cash on hand at the analysis date
- **Investition** → the cheque, tranched if milestone-based
- **Kapitalbindung** → working capital, which for SaaS is usually small and for
  hardware or inventory businesses is decisive
- **EBITDA-Zuwachs** → the modelled path to contribution positive
- **Renditehorizont** → exit horizon, and **it must be stated**

The horizon sensitivity matters more here than for a listed company, because a venture
return depends almost entirely on exit timing and multiple. Run at minimum 5 / 7 / 10
years, and state the exit assumption as an explicit `Annahme:`.

`scripts/cashflow_irr.py` takes these inputs unchanged — the field names are generic.

### 7. Verdict

Same structure, and the short/long split carries more weight: a startup can be an
excellent seven-year bet and a poor eighteen-month one, and the round you are being offered
has a specific horizon attached.

Confidence must reference the gaps section explicitly. For a startup, the gaps are usually
**cohort retention, true churn definition, and pipeline quality** — the three things most
often absent from a data room and most determinant of the outcome.

---

## Data-room gaps worth naming by default

Check for these; when absent, list them rather than working around them silently:

- Cohort retention curves (not blended retention — cohorts)
- Churn definition in writing, and gross vs. net revenue retention
- Monthly burn split into gross and net
- Fully diluted cap table including the unallocated option pool
- Customer concentration (revenue share of the top 5)
- Prior plans against actuals
- Liquidation preferences and any structure on prior rounds

The last one is regularly omitted and regularly decisive. A headline valuation says little
without the preference stack behind it.

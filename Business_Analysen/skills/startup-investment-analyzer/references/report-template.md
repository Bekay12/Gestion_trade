# Report skeleton

Seven sections, fixed order. Sections 1–6 come from the Aufgabenstellung of Prof. Dr.
Jürgen Bott (HS Kaiserslautern, *Finanzwirtschaft für Ingenieure*); section 7 is the
investment verdict added for non-academic use.

Parameters: **n** = number of strategic options compared in section 5 (reference case: 3),
**m** = number of segments or initiatives in section 3 (reference case: 6 sub-questions).
Every table below is mandatory in its stated column order.

**n may be zero.** When the decision has already been taken — the acquisition is
closed, the round is raised, the money is spent — there is nothing to compare and
section 5 disappears. Section 6 then changes shape; see "The no-options variant" below.
Section 7 is unaffected and is where such a report earns its keep.

---

## 1. Beschreibung des Unternehmens — Company description

| Sub-section | Content |
|---|---|
| 1.1 | Handelnde Personen — board, management, mandates, ties to the anchor shareholder |
| 1.2 | Eigentümerstruktur — shareholders with percentages, free float, share count history |
| 1.3 | Beurteilung durch die Kapitalmärkte — listing, market cap, price development |
| 1.4 | Analystenabdeckung **und Konsens** — covering houses, and the aggregated consensus when one is obtainable |
| 1.5 | **Botschaften und Lücken** — what analysts say *and* what is not obtainable |

**1.4 has two halves, and the second is easy to skip.** The company's own reports give at
most the *number* of covering houses; they never give ratings or targets. That absence is
not a gap - it is the normal shape of the document, and the consensus is obtainable from
aggregators. A report on a covered listed company that carries no consensus table has an
unfinished search in it, not a declared gap. Retrieve it, label it as a secondary source,
and run the staleness and false-independence checks in `rigor-and-assumptions.md` rule 5
before quoting a single figure. A genuinely uncovered subject is the exception, and then
the emptiness of the search is itself worth one sentence.

Sub-section 1.5 is where the report earns trust. When ratings and price targets are not
published *and not obtainable*, say so with the retrieval date and claim nothing. When the ownership
percentages do not sum to 100 %, show the residual as its own table row and state that the
company does not explain it — do not round it away.

**Governance signal worth a sentence:** overlapping mandates between the supervisory board
and the controlling shareholder, and any technical expertise deliberately seated on the
board.

---

## 2. Auswertung des Jahresabschlusses — Financial statement analysis

One sub-section per metric. **Each states the equation, then walks the calculation** — a
result without its formula is not checkable.

| Sub-section | Metric |
|---|---|
| 2.1 | Umsatz + Δ vs. prior year |
| 2.2 | Auftragseingang (order intake) + Δ — the leading indicator |
| 2.3 | EBITDA-Marge, **reported and adjusted, side by side** |
| 2.4 | Operatives EBITDA — state which of the two you carry, and why |
| 2.5 | Nettogewinn + Δ |
| 2.6 | Dividende and Dividendenrendite at the year-end price |
| 2.7 | Aktienkursverlauf over several years, with a chart |

Then, **unnumbered** (it answers no assignment question — numbering it shifts every
following sub-section):

> **Kennzahlenüberblick** — a summary table, prior year vs. current year vs. Δ.

### The reported/adjusted trap

Companies often publish both EBITDA and adjusted EBITDA. Carry one as the *operatives
EBITDA*, show the other beside it, and reconcile the difference with its cause and page.
Check whether the relationship inverts across years — in the reference case adjusted
exceeded reported in one year and fell below it the next. That inversion is a finding.

### Collect the multi-year series, not two years

Two years is enough to state what changed and nothing else. It is not enough to know
whether the latest figure is normal, and that is the question the verdict actually turns
on. **Collect seven to ten years of the metric section 6 will use** - the free cash flow,
the adjusted EBITDA, whichever carries the owner's series - and publish two things beside
the current value:

| | |
|---|---|
| **Through-cycle level** | the median of the collected years, held constant |
| **Percentile of the latest** | where the current figure sits in its own history |

This costs one extra table and removes the single worst failure mode of this method. A
scenario ladder built from "last full year" and "last twelve months" is exactly wrong at a
cyclical trough: it reads the bottom as the base case. In the reference run, a subject's
free cash flow of -341 in the trough year would have entered the ladder as the pessimistic
*scenario* when it was in truth the 0th percentile of that company's own decade - and the
share went on to quintuple from the low. Say which percentile the base case occupies, and
a reader can discount the verdict accordingly without re-deriving it.

The same series is what makes section 7.3's required growth rate an argument rather than a
number: the rate the price demands only means something against the rates this company has
actually delivered.

### Precision

Store figures at full published precision and round only on output. Percentage changes
computed from pre-rounded values diverge from the company's own published deltas.

---

## 3. Strategische Maßnahmen — Strategic measures

Purely factual. **No evaluative adjectives** — no "successful", "promising",
"disappointing", no forward-looking conditionals. This section reports; section 5 judges.
State that explicitly in the opening line.

One sub-section per segment or initiative (m of them), then:

- **Wirkung auf EBITDA/EBIT** — what the report says the measures have already produced,
  split by segment. Name which segment carries the improvement.
- **Ausblick vs. Ist** — guidance from the report against actual outcome:

| Kennzahl | Guidance | Actual | Position |
|---|---|---|---|
| Umsatz | range | value | below / within / above |
| EBITDA-Marge | range | value | below / within / above |

**Also check whether guidance was revised during the year.** Guidance held unchanged
through every interim report and then missed is a materially different fact from guidance
lowered in Q3. Cite each interim report separately.

### The guidance track record — five years, guidance against actual

One year of guidance-vs-actual says whether the company hit its number. Five years say
whether its guidance is worth anything, and that is a different and more useful fact:

| Year | Guidance (as first given) | Actual | Position |
|---|---|---|---|
| Y-4 … Y | range | value | below / within / **above** |

Build it from the prior years' reports, which are on the same IR page as the ones already
downloaded, and it costs one table. It earns its place the first time the market punishes a
cautious forecast. In the reference run a subject fell **12.2 % in a single session** on
guidance of 2,250-2,350 for adjusted EBITDA, and the year closed at 2,398 - above the top of
the range it had been punished for. A report carrying the track record can say that the
company had beaten its own range before; one carrying only the current year cannot, and has
nothing to set against the market's reaction.

Record the guidance **as first given**, not as later revised, or the table measures nothing.

---

## 4. Beurteilung des Kursverlaufs — Explaining the price

Half a page. The requirement is causal explanation, not description.

The reference case: the share price tracked **order intake**, not the income statement.
The 2024 collapse (−42.9 %) coincided with order intake falling 41.1 %, in a year of
record revenue and record EBITDA. A price that falls on record earnings is only explicable
through the leading indicator.

Find the equivalent leading indicator for your subject — backlog, order intake, net
revenue retention, pipeline — and tie the price path to it, using figures already
established in sections 2 and 3.

---

## 5. Beurteilung ausgewählter Handlungsoptionen — Strategic options

For **each** of the n options, three sub-sections in this order:

### 5.x.1 Relevanz (Management-Summary)

Half a page, written for a non-specialist. Why this option matters to this company now.
Economic logic, not technical description.

### 5.x.2 Finanzwirtschaftliche Bewertung

**Fixed six-factor grid. Same rows, same order, for every option:**

| Faktor | Einschätzung |
|---|---|
| Investitionsbedarf | |
| Kapitalbindung | |
| Liquiditätsbedarf (jährlich) | |
| Zeithorizont bis Return | |
| Risiko | |
| Potenzial | |

Each cell: one to two sentences, anchored to a figure wherever the data allows.

### 5.x.3 Statement

One page, covering four points in this order:

1. **Dauer der Umsetzung** — how long until it works
2. **Rentabilität** — effect on equity
3. **Liquidität** — effect on working capital / current assets
4. **Wirkung auf Kerngeschäft und Marge**

Points 2 and 3 are the Liquidität ↔ Rentabilität trade-off. Treat them as a tension, not
as two separate observations.

### 5.n+1 Übersichtstabelle

| Option | Investitionsbedarf | Kapitalbindung | Liquiditätsbedarf | Risiko | Zeithorizont | Chancen |
|---|---|---|---|---|---|---|

Use one consistent qualitative scale (gering / mittel / hoch) plus the anchoring figure, so
the rows are comparable.

**This table summarises section 5, so it carries section 5's figures.** If your per-option
grids hold your own estimates while the flow analysis in section 6 uses the brief's
prescribed amounts, do not silently substitute the latter here — it reads as a
contradiction with the grids two pages earlier. Say per column which track it shows. And
fill each cell with the quantity its heading names: *Liquiditätsbedarf (jährlich)* takes an
annual requirement, not the closing reserve. See `rigor-and-assumptions.md` rule 10.

### 5.n+2 Empfehlung

Quarter page. **A ranked recommendation, and usually a combination rather than a single
pick** — sequencing one option while another matures is a legitimate and often better
answer than choosing one. What is never acceptable is a recommendation that does not
follow from the preceding tables.

---

## 6. Grobe Flow-Analyse — Cash-flow analysis

### 6.0 Annahmen — declared before any number

**This sub-section comes first.** Amounts may be given; their mechanics never are. Declare
at minimum:

1. **Anfangsbestand** — opening liquidity, and its source date
2. **Investitionsprofil** — how the total is spread across years, per option
3. **Kapitalbindung-Timing** — when working capital is tied up
4. **Abgrenzung des operativen CF** — incremental (option only) or total (group)
5. **Steuerliche Betrachtung** — an EBITDA uplift is pre-tax and carries no depreciation
   on the spend; say so, or the IRRs will be read as after-tax returns
6. **Renditehorizont** — the IRR horizon, explicitly

Each gets a justification, not just a statement. See `rigor-and-assumptions.md`, and
rule 9 there for the wording of the tax assumption and for anchoring a hurdle rate.

**Two traps worth naming now.** A purchase price falls due at closing, not in equal
thirds — an acquisition option modelled in thirds hides its own liquidity trough. And
working capital is revenue pre-financing: it is tied up *before* the earnings contribution
starts, not with it.

### 6.1 Per-option tables

One table per option:

| Jahr | Investition | Operativer CF | Liquidität nach Investition | Bemerkung |
|---|---|---|---|---|

The *Bemerkung* column is factual (market entry, closing of the acquisition, first earnings
contribution), not evaluative.

### 6.2 Ergebnis per option

Quarter page each: capital requirement, expected return, and the two set against the
liquidity effect.

### 6.3 Vergleich und Fazit

| Kennzahl | Option A | … | Option n |
|---|---|---|---|
| Kumulierte Investition | | | |
| EBITDA-Zuwachs ab Jahr | | | |
| EBITDA-Marge (Endjahr) | | | |
| IRR (~) | | | |
| Liquiditätsreserve (Endjahr) | | | |
| Net Debt/EBITDA | | | |

Plus **the IRR sensitivity table across at least three horizons** — never a bare IRR:

| Horizont | Option A | … | Option n |
|---|---|---|---|
| 5 Jahre | | | |
| 10 Jahre | | | |
| 15 Jahre | | | |

The Fazit must **check whether the ranking is stable across scenarios**. A ranking that
holds while the levels move is a usable result; a ranking that flips means the metric
cannot decide the question, and saying so is the honest finding.

Two things this table must not leave implicit. The IRR row is **pre-tax** if the uplift was
given as EBITDA, and the Annahmen in 6.0 must already have said so. And an IRR is only an
argument against a hurdle: name it, source it (the subject's current return on equity is
usually the defensible choice), and if none can be sourced, restrict the Fazit to the
ranking rather than asserting the returns are "clearly sufficient". Where a row carries a
term a lay reader will not know — *Net Debt/EBITDA* is the usual one — gloss it correctly:
net debt against operating result **before depreciation**, never against "profit".

---

### The no-options variant

With n = 0 there is no incremental view to take, so section 6 asks two questions of the
subject as a whole. They are **alternatives, not additions**, and the report must say so:

| 6.x | Question | Capital allocation assumed |
|---|---|---|
| 6.1 | Can it carry what it borrowed, and how fast? | Every free euro retires debt; liquidity held flat |
| 6.3 | What does an owner earn buying in today? | Only scheduled amortisation is served; the rest is distributed |

**The same cash cannot be spent twice.** In the reference case the first model let cash
accumulate with no use assumed, and the closing liquidity ran to 5,492 against an opening
1,079 — arithmetically correct and substantively absurd, because a broker that generates
that much cash either repays debt, buys companies or pays it out. Declare which of the
three each table assumes. A deleveraging path is a **capacity** calculation, never a
forecast: label it as such, especially when the subject's business model is acquisitions
and "no further acquisitions" is therefore the least realistic assumption in the report.

### The owner's series needs a terminal value - and it decides the result

6.3 asks what an owner earns buying in today, so the series is: the entry price at t = 0,
the distributable cash flow per share each year, and **the value of the stake at the end
of the horizon**. Leave that last term out and the model silently answers a different
question - how long the distributions alone take to repay the price at the hurdle - whose
break-even always lands far below any market price, for every company. The verdict then
writes itself, and it writes itself wrong.

Set `terminal_value` on the option in `cashflow_irr.py` (per share, same unit as the
price). It is added in the final year and, deliberately, does **not** scale with the
investment: an exit value does not grow because the entry price did, and if it scaled,
`break_even_investment()` would move both sides of its own comparison.

Choose it from a **sourced** figure and declare it, because it dominates the answer. Book
equity per share is the defensible default: it is on the balance sheet, it is independent
of the entry price, and it is conservative for anything trading above book. A valuation
multiple is exactly the plausible-looking placeholder rule 5 forbids - unless it is the
subject's own current multiple, labelled as the assumption it is.

Then publish the inversion as a counter-check. `required_terminal_value()` answers what
the *current* price already presupposes; as a multiple of book it states plainly how much
of today's price rests on the exit rather than on the distributions:

| Scenario | 5 years | 10 years | 15 years |
|---|---|---|---|
| pessimistic | 2.3x book | 3.8x book | 6.4x book |
| base | 1.8x book | 2.3x book | 3.2x book |
| optimistic | 1.7x book | 2.1x book | 2.9x book |

with today's multiple stated beside it (1.44x in the reference run). This is the table that
makes a break-even far below the market price intelligible instead of merely alarming, and
it is what 7.2's divergence block argues from.

---

## 7. Verdict — Investment potential

Not in the original assignment. Required for any investment use, and the section a
returning reader turns to first. **Five fixed sub-sections, always in this order**, so
two reports on two companies can be compared line by line.

Set this section in its own colour. The reference case runs three: black for what is
sourced, blue for the author's assessments inside the running text, red for the verdict.
See `rigor-and-assumptions.md` rule 7.

### 7.1 Verdict

**One sentence**, naming the action, the price and the horizon: buy / watch / avoid.
Then one sentence for the holder and one for the non-holder — they face different
decisions and often get different answers. A verdict that reads the same for both has not
been thought through.

### 7.2 Reasoning from the findings

**Introduces no new figure.** Every quantity is one already established in sections 2 to
6. Group the argument by direction and say so out loud: what argues against, what argues
for, what argues for waiting rather than walking away. Three short blocks beat one long
paragraph.

**Add a fourth block whenever a consensus was obtainable in 1.4: where this verdict
diverges from the street, and why.** Give the direction (does the consensus agree on
buy/hold/avoid?) and the level (where does the mean target sit against your entry
threshold from 7.3?). Both halves matter, and they routinely disagree: in the reference
run the consensus rating matched the verdict exactly - *hold* - while the mean target sat
40.4 % above the entry threshold, and even the single lowest target of the whole coverage
sat 12.4 % above it.

Then read that gap honestly. **When an entire coverage sits above your threshold, the
likeliest explanation is an assumption of yours, not a collective error by nineteen
houses** - and it is almost always the terminal value. Name the mechanism rather than
declaring a winner: a twelve-month target values the company on earnings multiples with an
implicit going-concern value, a distributable-cash-flow model with a book-value exit does
not. The difference is the valuation premium, not the underlying figures. A verdict that
quietly ignores a consensus pointing the other way is not defensible, and one that folds
the moment it meets a consensus was never a verdict. The same comparison then belongs in
7.5 as an argument against your own verdict.

### 7.3 Entry threshold

The table that turns an analysis into a decision:

| Scenario | Growth | 5 years | 10 years | 15 years |
|---|---|---|---|---|
| pessimistic (latest measured) | | | | |
| base (last full year) | | | | |
| optimistic (best recent year) | | | | |

Each cell is **the price at which the IRR exactly meets the hurdle** — from
`break_even_investment()` in `cashflow_irr.py`. State the current price beside the table.

Two things make this table the strongest part of the report. It answers "up to what
price" instead of "is it good", which is the question a reader can act on. And it exposes
an asymmetry a single IRR hides: in the reference case the price sat 5.3 % above the base
scenario's threshold but 58.9 % above the threshold of the scenario the *latest published
figures actually support*. Say which row the current data supports — that is the finding.

**Build the scenarios out of the subject's own published rates**, held constant. Three
figures it has reported itself need no defending; three growth rates you chose do.

### The required growth rate — give back the term the model dropped

Holding the cash flow flat is what keeps the analysis honest, and it is also what makes
every growing company look expensive: the case sits entirely in the term the model removed.
Do not repair that by forecasting. Repair it by **inverting the price into the growth rate
it already presupposes** — `required_growth()` in `cashflow_irr.py`:

| Scenario | 5 years | 10 years | 15 years |
|---|---|---|---|
| base | % p.a. | % p.a. | % p.a. |

Then set each rate against the realized distribution from the multi-year series in section
2, and say plainly whether the company has ever delivered it over a comparable stretch.

This is what turns a static verdict into a testable claim. "Do not buy at 63.85" invites the
reader to disagree on taste; "this price contains 26.7 % annual growth of free cash flow for
ten years, and the company has delivered 1.4 % over the last two" invites them to check.
Three reference values from one run, all at each subject's own sourced hurdle: 2.0 %, 26.7 %
and 42.0 % for ten years. The first is a modest claim, the third is not, and no further
argument is needed to separate them.

Where the function returns None, say so: the price is then not justifiable by growth of the
earnings stream at any rate inside the bracket, and widening the bracket until a number
appears is how a report starts lying.

### 7.4 Triggers that flip the verdict

Concrete and observable, in both directions, with the number attached. Prefer triggers
that cost nothing to watch: a figure the company publishes quarterly, or the share price.
A trigger no one will check is decoration.

### 7.5 What the verdict cannot know

Name the variable the case turns on and state plainly that it is not calculable — then
give **the arguments against your own verdict**. In the reference case: the model excludes
further acquisitions although they are the business model, and the share had already run
32.7 % off its low, so part of what the verdict waits for may be priced in.

A verdict that cannot be argued against is not a verdict. This sub-section is also what
separates a defensible recommendation from a confident one.

## Optional academic blocks

Include only for coursework. Omit for an investment memo.

- **Quellenverzeichnis** — split into (a) company reports, (b) external sources with
  retrieval dates.
- **Hinweis zur Nutzung von KI-Werkzeugen** — tools used and for what. Must match reality;
  if the assignment restricts AI use, the disclosure is the author's decision to make.
- **Eidesstattliche Erklärung** — declaration of independent authorship.

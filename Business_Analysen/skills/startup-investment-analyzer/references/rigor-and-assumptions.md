# Rigor and assumptions

The discipline that makes the report survive a reader who checks. Each rule below exists
because its absence produced a real defect in the reference case.

---

## 1. Every derived figure carries report + page, or URL + retrieval date

Not "according to the annual report" — the page.

### Verify the printed page, never infer it

Page numbers derived from line numbers of a full-text `pdftotext` dump are wrong. In the
reference case this produced **five incorrect citations** before it was caught. The PDF
page and the printed page also diverge systematically: in that case the annual reports
matched, but all three interim reports ran PDF = printed + 1.

Extract page by page and read the footer:

```python
import subprocess, re

def find_printed_page(pdf: str, needle: str):
    """Return (pdf_page, printed_page) of the first page containing needle."""
    n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(
        ['pdfinfo', pdf], capture_output=True, text=True).stdout).group(1))
    for p in range(1, n + 1):
        t = subprocess.run(['pdftotext', '-layout', '-f', str(p), '-l', str(p), pdf, '-'],
                           capture_output=True, text=True).stdout
        if needle in t:
            printed = re.findall(r'Page (\d+)', t)
            return p, (printed[-1] if printed else None)
    return None, None
```

### Make the check a guard, not a habit

The helper above finds a page. It does not stop you from citing the wrong one. In the
reference case **23 of 176 page citations were wrong on the first pass** — without
exception the ones inferred from context rather than read off the page. Judgement fails at
that scale; a script does not.

So store every raw value once, with its source and printed page, and add a script that
loops over all of them and fails when a value does not occur in the text of the page it
cites:

```python
def check(values):            # values: {name: (value, "10-K 2025, p. 49")}
    for name, (val, cite) in values.items():
        doc, page = parse(cite)
        text = pages(doc)[page]
        if not any(f in text for f in spellings(val)):
            yield f"{name} = {val}: not on {doc}, p. {page}"
```

`spellings()` is where this earns its keep: filings write `5,763` for 5763 and `(54` for
−54, and a naive `str(value) in text` reports false failures on every grouped amount and
every negative. Generate the variants, do not weaken the check.

Run it before writing a single sentence of prose. In the other order you build paragraphs
around citations that later turn out to be wrong, and the rewrite costs more than the
check ever would.

### Check attribution, not just page number

A figure can be on a page you never read. In the reference case the ISIN was attributed to
the annual report; it appears in **none of the eight reports** — it came from the investor
relations website. Before publishing, confirm each cited value actually occurs on the
cited page.

### Web sources need a retrieval date

Company IR pages change without notice. `Source, URL, retrieved DD.MM.YYYY`.

---

## 2. Every non-deducible assumption gets a declared paragraph

A brief that specifies amounts almost never specifies mechanics. Each gap becomes an
explicit paragraph **before** the calculation that consumes it:

> **Annahme:** Ausgewiesen wird der *zusätzliche* operative Cash-Flow der Option, also der
> EBITDA-Zuwachs abzüglich des Aufbaus der Kapitalbindung. Der operative Cash-Flow des
> bestehenden Kerngeschäfts bleibt unberücksichtigt, damit die Wirkung der Entscheidung
> sichtbar bleibt.

Statement plus justification. An assumption without a reason is a preference in disguise.

The six that always need declaring in a flow analysis: opening balance, investment
profile, working-capital timing, cash-flow definition (incremental vs. total), tax
treatment (rule 9), and the return horizon.

**Describe the convention the way the numbers actually encode it.** In the reference case
the working-capital assumption read "in the year(s) immediately before the earnings
contribution", while the tables tied up half the amount in the contribution year itself.
The tables were internally correct; only the prose was wrong. That is the harder defect to
catch, because every figure checks out. After writing an assumption paragraph, read it
back against the generated table row by row.

---

## 3. Any metric resting on a non-neutral base gets a sensitivity table

The most dangerous number in a report is one that is arithmetically correct and
substantively misleading.

**Reference case.** The 2028 EBITDA margin was computed on the 2025 revenue held constant.
That revenue had collapsed 34.7 % — so the small denominator inflated the margin. The
figure was right; presented alone it would have been deceptive.

The fix is a table with alternative bases:

| Revenue base | Option A | Option B | Option C |
|---|---|---|---|
| 2025 held constant (base case) | 21.2 % | 20.2 % | 18.3 % |
| +3 % p.a. | 19.4 % | 18.5 % | 16.7 % |
| 2024 level | 13.9 % | 13.2 % | 11.9 % |

Then say what the table shows: **the ranking held across all three; the level did not.**
The metric compares options; it does not forecast.

Ask of every headline metric: what happens to it if the base year were normal?

---

## 4. IRR is meaningless without a horizon

Over three years virtually every investment case is negative — the outflows land first.
An IRR published without its horizon is a number chosen by an undeclared assumption.

Always publish the horizon, always publish a sensitivity across at least three:

| Horizon | A | B | C |
|---|---|---|---|
| 5 years | 11.7 % | 27.4 % | **−9.9 %** |
| 10 years | 29.6 % | 38.3 % | 6.3 % |
| 15 years | 32.0 % | 39.6 % | 10.2 % |

Option C is value-destroying on five years and acceptable on fifteen. That reversal is the
finding — a single-horizon IRR would have concealed it.

Compute by bisection over an explicit bracket, and state the terminal convention (does
working capital release at the end?). `scripts/cashflow_irr.py` does both.

---

## 5. Missing data is declared, never estimated

Maintain an explicit **"Nicht verfügbare Angaben" / "Data not available"** section, and
name gaps inline where they bite.

Two real examples:

- **Ownership that does not sum.** Anchor shareholder 44.49 % + free float 45.5 % =
  89.99 %. The residual 10.01 % is unexplained by the company, and treasury shares do not
  explain it (none held at year-end, cited). Shown as its own table row, not rounded away.
- **No analyst opinions obtainable.** Three covering houses named with their analysts, but
  no rating and no price target published *by the company, and none carried by the
  aggregators either* — a genuinely uncovered small cap. So none is claimed, and the price
  explanation is built instead on the guidance-vs-actual comparison, which *is* verifiable.
  Note the qualifier: this is a gap only because the search below came back empty too. For a
  widely covered subject the same sentence would be an unfinished search, not a gap.

A declared gap costs a sentence. An invented figure costs the report.

### But first: "not in the primary documents" is not "not obtainable"

This rule forbids inventing a figure. It does not license skipping the search.
The failure mode is quiet and it looks like diligence: you read the annual report,
the figure is not there, you write "the company does not publish this", and you
move on - while the figure sits on half a dozen public pages.

Whole categories of decision-relevant data are *never* in a company's own reports,
by design, and are always obtainable elsewhere:

| Never in the filing | Where it lives |
|---|---|
| Analyst consensus, ratings, price targets | consensus aggregators |
| Number of covering analysts with their estimates | same, and it often differs from the company's own count |
| Short interest, index membership, free-float factor | exchange and index provider |
| Current price, FX rate, market capitalisation today | market data |
| Peer multiples | screeners; same definition and same date as the subject's (rule 11) |
| Directors' dealings (board and management trades) | issuer's MAR Art. 19 notices and the national regulator's register (BaFin, AMF, FCA); US: SEC Form 4 on EDGAR |
| The subject's own historical multiples | computed: year-end price history over the section 2 series, never a screener's single current value |

So before any line of the "Data not available" section, ask the second question:
*is it unobtainable, or merely not in this document?* Only the first is a gap. The
second is an unfinished search, and a reader who finds the number in thirty seconds
stops trusting the rest of the report.

### Using a secondary source without contaminating the primary ones

Secondary data is admissible with `Source, URL, retrieved DD.MM.YYYY`, under four
conditions:

1. **Label it as secondary, in the table it appears in.** It does not carry the
   evidential weight of a page-verified filing, and the reader must be able to see
   which figures would survive an audit.
2. **Keep it out of the page-verified value store.** The guard in rule 1 checks
   values against the text of a cited page; a web figure has no page. Give it its
   own block in the data file so the two never blur.
3. **Test the figure against itself.** A target price published with an upside
   percentage implies the reference price it was computed from:

   ```
   implied reference price = target / (1 + upside)
   ```

   If that reference does not match the price you are using, the figure is stale
   and must be discarded, not averaged in. In the reference run this test separated
   a current consensus (474.47 at +2.95 % against a last close of 460.90 - consistent)
   from two stale ones (543.38 at +24.29 % implies a reference of 437.19; 538.84 at
   +6.79 % implies 504.58, a price the share never traded at recently). Publish the
   check, not just its conclusion: it tells the reader why you chose one aggregator
   over another, which otherwise looks arbitrary.
4. **Two aggregators agreeing to two decimals are one source.** Consensus pages
   overwhelmingly resell the same provider's feed. Quoting both as independent
   confirmation overstates the evidence; say plainly that it is one feed seen twice.

One more thing worth stating rather than smoothing over: when the company reports a
different coverage count than the aggregator does - 23 covering houses in the annual
report against 19 estimates collected - that difference is a fact about the data, and
neither source explains it. Report it and leave it unexplained.

---

## 6. Enforce the discipline mechanically

Judgement fails at scale; a script does not.

**Single source of truth for numbers.** Every figure declared once, in one data file, with
its source and page in a trailing comment. Prose, tables, and charts read the same
declaration, so text can never contradict a table.

**A guard that forbids literals in prose.** In the reference case this caught real defects
and two genuine false positives that were fixed properly rather than by weakening the
guard. Watch for locale-grouped numbers — `12.345`, `246.800`, `1.234,5` initially slipped
through a naive pattern, which is precisely the format a German finance document uses for
every amount above a thousand.

**Generated tables are generated.** Cash-flow tables come from the script's output, never
retyped. Re-running must be deterministic.

**Verify length limits in the rendered output, not in the source.** A brief that caps a
section at half a page states a requirement, not a preference. Source lines and word
counts do not predict the typeset result. Measure the rendered page: take text positions
(`pdftotext -bbox-layout`), the first and last baseline of the section, and divide by the
height of one full text block. In the reference case thirteen sections carried a limit,
twelve sat comfortably inside, and the one that overran did so by 22 % unnoticed until it
was measured. Two traps in your own measuring script: a section beginning and ending on
the same page needs `end - start`, not the across-pages formula, and a floated table
occupying an intervening page is not prose and must be excluded.

---

## 7. Keep factual and evaluative text separated

The descriptive sections report; the assessment sections judge. Ban evaluative adjectives
from the factual sections — *successful*, *promising*, *disappointing* — and forward-looking
conditionals with them. State the separation in the opening line of the factual section so
a reader knows it is deliberate.

When an AI drafts evaluative prose that a human must own, mark those passages visually
(a colour macro) and keep the supporting facts in comments beside them, so the human can
verify and rewrite before the marking is removed.

**Three colours, three responsibilities.** Once the report carries a verdict (section 7),
two levels are no longer enough:

| Colour | Content | Who stands behind it |
|---|---|---|
| black | figures, tables, sources, pages | the primary source |
| blue | assessments inside the running text | the author, after checking |
| red | the verdict in section 7 | a recommendation under declared assumptions — never a finding |

State the scheme in the document itself, once, at the head of section 7 and again in the
methodology note. The test of the scheme: a reader who reads only the black text gets the
verified position with no judgement mixed in. If that is not true, something evaluative is
sitting in black, which is the failure this rule exists to prevent.

Keep the colours switchable in one place (`\newcommand{\vd}[1]{#1}`) so a version for
third parties can be produced without touching the text. Two traps, both met in the
reference case: `\textcolor` is not `long`, so a multi-paragraph verdict aborts the run —
use `\color` inside a group. And a colour macro must never wrap a figure, or the number
stops being traceable to the data layer.

---

## 8. Local models never source a cited figure

Measured, not assumed: a 14B local model stopped honouring a strict output format at
roughly 14k prompt tokens, discarded the required page anchors, injected banned evaluative
wording, and **returned two different employee counts across two runs of the same source**.

Local models are for bulk descriptive text — documentation, comment blocks, first-pass
condensation clearly marked as unverified. Any figure that reaches the report is verified
against the source by targeted extraction.

---

## 9. The flow analysis is pre-tax — declare it, and give the IRR a hurdle

When a brief specifies the expected gain as an **EBITDA** uplift, the model that consumes
it carries no tax on the gain and no depreciation on the spend. That is a legitimate
simplification for a rough analysis. It is not legitimate to leave it unsaid: a reader who
sees "IRR 38.3 %" assumes a return they could actually keep.

> **Annahme:** Die Rechnung ist eine Vorsteuerbetrachtung. Der vorgegebene EBITDA-Zuwachs
> geht ungekürzt in den operativen Cash-Flow ein; Ertragsteuern und Abschreibungen auf die
> Investitionssummen bleiben unberücksichtigt, weil die Aufgabenstellung die
> Ergebnissteigerung als EBITDA-Größe vorgibt und weder einen Steuersatz noch eine
> Nutzungsdauer nennt. Die ausgewiesenen internen Zinsfüße sind deshalb Vorsteuerrenditen.
> Da alle Optionen gleich behandelt werden, bleibt die Rangfolge unberührt.

The last sentence is the one that saves the analysis: a uniform simplification distorts
the level but not the ranking, and the report is a ranking exercise.

**Then give the IRR something to beat.** An internal rate of return alone supports no
verdict; "clearly above any plausible minimum return" is an assertion, not a comparison.
Anchor the hurdle in a figure the report has already sourced — the subject's current
return on equity is usually available and defensible (net profit ÷ equity, both from the
key-figures page). Do **not** construct a WACC: beta, risk-free rate and market premium
appear in none of the primary documents, so a WACC is exactly the plausible-looking
placeholder rule 5 forbids. If no hurdle can be sourced, say so, and confine the verdict
to the ranking.

The hurdle carries a second load, and it is the heavier one: it converts the IRR into a
**break-even price**. "IRR 10.6 % against a hurdle of 11.1 %" is a judgement the reader
has to trust; "clears the hurdle up to 67.80, and it trades at 71.39" is one they can
check against a screen. `break_even_investment()` in `cashflow_irr.py` does the bisection,
and it refuses to run on a hurdle whose `hurdle_source` is empty — an unsourced hurdle
produces an authoritative-looking price that rests on nothing.

---

## 10. A summary table summarises its own section

Reports of this shape carry two tracks: **your own estimates**, derived from the company's
data, and the **figures the brief prescribes** for the flow analysis. Both are legitimate
and they will differ. What is not legitimate is a comparison matrix that silently swaps
one for the other.

In the reference case the per-option grids reported an investment need of 42.2 / 21.1 /
55.3 (own estimates), and the summary matrix two pages later reported 35.0 / 30.0 / 60.0
(the brief's figures) under the same column heading. Both sets were correct; together they
read as a contradiction, and the reader cannot tell which one the recommendation rests on.

- The summary of a section summarises **that** section. If it must import an anchor from
  the flow analysis, label it per column or say so in the caption, not vaguely.
- **Fill the cell the heading asks for.** A column headed *annual liquidity requirement*
  takes an annual requirement. The reference case filled it with the year-end reserve —
  the right number under the wrong heading, in the one table a hurried reader looks at.

---

## 11. A ratio states its reference date, and both sides share it

Market capitalisation over equity, net debt over EBITDA, dividend over price: each mixes
two figures that exist only at a date. Pin it, and check that numerator and denominator
carry the *same* one.

The reference case produced one real instance and one false alarm, and both cost time:

- **Real.** A proposal to compare a year-end-2025 market capitalisation of 159.1 m€ with
  equity of 201.7 m€ — which is the **2024** figure. The 2025 equity was 195.4 m€. The
  corrected comparison turned out to be the more interesting one.
- **False alarm.** That same market capitalisation was twice suspected of multiplying a
  2025 price by a 2026 share count. It did not: the annual report states the reduced count
  was already in force at 31 December 2025. What invited the suspicion was the *citation* —
  the share count was sourced to an IR web page retrieved in 2026 rather than to the annual
  report. Cite the dated source, and a correct figure stops looking wrong.

Related, and cheap to get right: **gloss a term correctly or not at all.** EBITDA explained
to a lay reader as "profit" is false; it is earnings before interest, tax, depreciation and
amortisation, and "operating result before depreciation" is both understandable and true.
Likewise gloss *net* debt as net debt, or the sentence explaining that a negative value
means a net cash position stops making sense.

---

## 12b. A level model states where its level sits in history

This rule exists because a report that obeys every other rule here can still be wrong in
one specific, recurring way: it reads a cyclical trough as the base case.

The scenario ladder in section 6.3 is built from the subject's own published figures -
last full year, last twelve months, a prior year. That is the right discipline and it has
a blind spot: **those are levels, and a level carries no information about where it sits
in its own distribution.** At the bottom of a cycle every one of them is low, the ladder is
low, the break-even price is low, and the verdict reads "expensive" at the exact moment the
asset is cheap.

Two cheap corrections, both mechanical:

1. **Publish the percentile.** With the seven-to-ten-year series that
   `report-template.md` section 2 now requires, state where the base case sits: "the base
   case is the 35th percentile of this company's own decade" is a sentence a reader can
   act on. A base case in the lowest decile is a warning printed on the verdict itself.
2. **Invert the price into a growth rate** (`required_growth()`), and compare it to the
   realized rates in that same series. A constant-level model cannot forecast, but it can
   say what the price assumes and whether the company has ever done it.

The failure this prevents is documented and expensive. In one reference case a subject
reported free cash flow of -341 in a trough year, cut its dividend by 83 %, and fell 29.7 %
in a month; a ladder built on that year and the next twelve months would have said "avoid"
within weeks of a low from which the share went on to rise more than fourfold. Nothing in
the arithmetic was wrong. The report simply never said which part of the cycle it was
standing in.

**Corollary for the drawdown case.** When a report is written after a sharp fall, say
explicitly what fell: the company's own leading indicator, or its multiple. Those are
different events and only the first belongs in the earnings scenarios. A move that is
mostly market or sector - measurable against an index and two peers in one line of code -
is a valuation event, and modelling it as a deterioration double-counts it.

---

## 12. The same cash cannot be spent twice

A projection without an assumed use of the surplus is not conservative, it is broken. In
the reference case the first group model deducted interest, tax, capex, dividend and
earn-outs, then let whatever remained pile up: opening liquidity 1,079, closing 5,492,
net debt −561. Every line was right and the result was meaningless, because a business
throwing off that much cash repays debt, buys companies or pays it out — it does not
hoard it for five years.

Fix it by naming the use, and by keeping the uses apart:

| Question | Surplus goes to | Net debt falls by |
|---|---|---|
| How fast can it deleverage? | early repayment | scheduled + surplus |
| What does an owner earn? | distribution to owners | scheduled only |

Both are legitimate; together they are not. Publish them as **alternatives** with a
sentence saying so, or a reader adds a deleveraging path to a shareholder return and
double-counts the money.

Two smaller consequences. A deleveraging path is a *capacity* calculation, not a forecast
— say so in its caption, particularly when "no further acquisitions" is the assumption
holding it up and acquisitions are what the subject does for a living. And an investor
series should carry the **distributable** cash flow, not the dividend alone: modelling
only the declared dividend while cash accumulates unowned understates the return by
whatever the buyback would have been.

---
name: startup-investment-analyzer
description: "Produce a rigorous, fully sourced investment or corporate-finance analysis report from primary documents (annual reports, investor decks, data rooms, filings). Use when asked to evaluate a company or startup as an investment, compare strategic options with a cash-flow and IRR analysis, assess whether to fund or pass on a round, or write a structured financial assessment that must cite every figure. Triggers on: /startup-investment-analyzer, 'analyse this startup', 'should I invest', 'due diligence', 'evaluate these strategic options', 'cash-flow and IRR comparison', 'investment memo', 'should I buy or hold this share', 'at what price is it worth buying', 'judge an acquisition already closed', 'Unternehmensanalyse', 'Handlungsoptionen bewerten', 'Flow-Analyse', 'Verdikt', 'lohnt sich der Einstieg', 'analyse cette startup', 'faut-il investir', 'jusqu'a quel prix'."
allowed-tools: [Read, Write, Edit, Bash]
permissions: [read, write, network]
---

# Startup & Corporate Investment Analyzer

Turns primary documents into a structured, defensible assessment. The method comes from a
graded German Master's financial analysis (Aumann AG, HS Kaiserslautern): every figure
traceable to a page, every assumption declared before the calculation that uses it, and
every gap named rather than filled.

**The rule that makes the report worth reading: a number you cannot source is a gap you
declare, never an estimate you slip in.**

## When to use

- Evaluating a company or startup as an investment (screening or full assessment)
- Comparing strategic options (organic growth vs. adjacency vs. acquisition) on cash flow
- Judging a decision already taken — an acquisition closed, a round raised — where there
  is nothing to compare and the question is whether to own the result
- Any report where a reader will check your figures against the source

Not for: quick company summaries, market sizing alone, or pitch-deck feedback with no
financial analysis.

## How to run an analysis

### 1. Frame it

Establish, before any research:

- **Subject and scope.** Listed company or private startup? Which fiscal years?
- **Primary sources.** Name them and pin them down. Download them locally so page numbers
  stay stable. A source you cannot re-open is a source you cannot cite.
- **Decision at stake.** Invest / pass / which option to fund. The report exists to answer
  a question — state it in one line at the top.
- **Number of options** to compare in section 5 (the template is written for *n*; the
  reference case used 3). **n may be zero**: when the decision is already made, section 5
  disappears and section 6 splits into two mutually exclusive calculations. See "The
  no-options variant" in `references/report-template.md`.
- **The hurdle.** Which sourced figure the return has to beat, and on which page it
  stands. Decide this early: it determines whether section 7 can state a price or only a
  ranking.

### 2. Research in parallel, summarize into files

Dispatch one subagent per research domain. Each writes a structured file and returns only
a path plus a ten-line digest, so raw source material never enters the main thread.
Delegate the bulk reading to a local model where speed matters more than precision.

Full protocol, dispatch prompts, and the digest schema: **`references/orchestration.md`**.

### 3. Compute, don't estimate

Run `scripts/cashflow_irr.py` for the cash-flow tables, IRR, and every sensitivity table.
Never hand-compute an IRR, and never publish one without stating its horizon.

Configure `hurdle` and `hurdle_source` and you also get the **break-even table**: the
price at which the return still meets the hurdle, per option and horizon. That is the
table section 7 is built on, because it answers "up to what price" instead of "is it
good". The script refuses a hurdle whose source is blank.

For an **ownership position rather than a project** - the whole n = 0 case - set
`terminal_value` on the option: the stake is still worth something at the end of the
horizon, and leaving it out turns the break-even into a distributions-only payback that
sits far below any market price. `required_terminal_value()` inverts it and reports what
today's price already presupposes, which is the counter-check 7.3 needs. Details in
`report-template.md`, "The owner's series needs a terminal value".

**`required_growth()` is the second inversion, and it repairs the method's standing
weakness.** Holding the cash flow flat forecasts nothing, which is the point - and it also
makes every growing company look expensive, because the case sits in the term the model
dropped. Inverting the price into the growth rate it presupposes gives that term back
without forecasting, and turns "do not buy" into a claim the reader can check against the
company's own history. It needs the multi-year series from `report-template.md` section 2
to be an argument rather than a number.

Offline tests, stdlib only, no config and no network:

```bash
python3 scripts/Test/test_cashflow_irr.py
```

Before writing a word of prose, run the page verifier over every stored figure
(`rigor-and-assumptions.md` rule 1). In the reference case 23 of 176 citations were wrong
on the first pass, every one of them inferred rather than read.

```bash
python3 scripts/cashflow_irr.py --config analysis.json --format markdown
python3 scripts/cashflow_irr.py --config analysis.json --format latex --outdir data/
```

Stdlib only. The config schema is documented in the script's header and in
`references/report-template.md` section 6.

### 4. Write against the skeleton

Follow **`references/report-template.md`** — seven sections, fixed order, fixed tables.
The six-factor evaluation grid and the option-comparison matrix are not suggestions: a
reader comparing two of your reports should find the same rows in the same order.

**For a report you will revisit** — a holding re-examined each quarter, anything past
ten pages, anything a reader will audit — build it as a project rather than a file:
**`references/latex-project-pattern.md`**, with the copyable skeleton in
`assets/scaffold/`. It buys a data layer where every figure is declared once with its
printed page, and five guards that fail the build when prose and tables disagree. It
costs about a day and pays back on the second pass.

For a private startup, read **`references/startup-adaptations.md`** first. Listed-company
inputs (share price, analyst coverage, dividend yield) have defined substitutes
(traction, cap table, round terms). The rigor requirements do not change.

### 5. Hold the line on rigor

**`references/rigor-and-assumptions.md`** is the part that survives scrutiny. Read it
before writing section 6, and again before writing the verdict. It covers: page-level
citation as a *guard* rather than a habit, the `Annahme:` paragraph, sensitivity tables
for any metric resting on a non-neutral base, the three-colour separation of fact from
assessment from verdict, the rule that the same cash cannot be spent twice, and the
explicit gaps section.

### 6. Don't relearn the plumbing

**`references/sources-and-typesetting.md`** holds what costs hours once and nothing
afterwards: opening an EDGAR filing and recovering its *printed* page numbers (the proxy
footer differs from the 10-K one), which price source answers when Yahoo returns 429, and
the six LaTeX traps that produce a wrong page without any error — among them siunitx
grouping the decimals, so an exact 43.4679 is typeset as `43,467.9`.

## Non-negotiables

1. **Every derived figure cites report + page, or URL + retrieval date.** Verify the
   *printed* page, not the PDF page — they differ, and guessing produces citations that
   collapse when a reader opens the document.
2. **Every assumption not deducible from the data gets a declared `Annahme:` paragraph**
   before the calculation that consumes it.
3. **No IRR without a stated horizon**, and a sensitivity table across at least three
   horizons. Over a short horizon most investment cases are negative; the horizon is the
   result.
4. **Any metric resting on a non-neutral base gets a sensitivity table.** A margin computed
   on a collapsed revenue year flatters itself through the denominator. The same applies to
   the scenario ladder itself: state which percentile of the company's own multi-year
   history the base case occupies, or the report can read a cyclical trough as normal -
   see `rigor-and-assumptions.md` rule 12b.
5. **Missing data becomes a "Nicht verfügbare Angaben" / "Data not available" section.**
   Never an estimate, never a plausible-looking placeholder. But test the gap first:
   *not in the primary documents* is not *not obtainable*. Analyst consensus, ratings,
   price targets, short interest and current market data are never in a company's own
   reports and are always obtainable elsewhere - retrieve them, label them secondary,
   and check them for staleness. An unfinished search declared as a gap is a defect.
6. **The verdict states a time horizon, a price and what would change it.** Section 7 has
   five fixed sub-sections; 7.3 gives the break-even price per scenario, and 7.5 gives the
   arguments *against* the verdict. A verdict that cannot be argued against is not one.
7. **The flow analysis is pre-tax unless the brief says otherwise — say so.** An uplift
   given as EBITDA carries no tax, and the spend carries no depreciation. Declare it in
   its own `Annahme:` paragraph. An undeclared pre-tax IRR will be read as an after-tax
   one, and it flatters every option in the comparison.
8. **An IRR needs something to beat.** Name the hurdle and where it comes from — a figure
   already documented in the report, such as the subject's current return on equity.
   Never invent a WACC: rule 1 applies to the hurdle exactly as it applies to every other
   number. "Clearly above any plausible minimum return" is not a comparison.

## Optional blocks

Include only when the deliverable is academic coursework:

- **Hinweis zur Nutzung von KI-Werkzeugen** — which AI tools were used and for what.
- **Eidesstattliche Erklärung** — signed declaration of independent authorship.

Omit both for an investment memo. If the report will be graded, check whether the
assignment restricts AI use, and make the disclosure match what actually happened.

## Working language

The template ships German section headings because the reference case was a German
Hausarbeit. Translate them for an English deliverable — the structure is what matters,
not the language. Keep one language throughout, including table headers.

## Files

| File | Read it when |
|---|---|
| `references/report-template.md` | Before writing anything — the seven-section skeleton |
| `references/latex-project-pattern.md` | Before starting a *recurring* or long report — the buildable project: data layer, five guards, page splitting, footnote traps |
| `references/rigor-and-assumptions.md` | Before section 6 and before the verdict |
| `references/sources-and-typesetting.md` | Opening filings, price sources, LaTeX traps, the guards |
| `references/startup-adaptations.md` | Subject is a private company |
| `references/orchestration.md` | Before dispatching research subagents |
| `scripts/cashflow_irr.py` | Section 6 — cash-flow tables, IRR, sensitivities, break-even and required terminal value |
| `scripts/Test/test_cashflow_irr.py` | Offline tests for the above; run after any change to it |
| `scripts/local_llm.sh` | Called by subagents; probes llmfit → ollama → sentinel |
| `assets/scaffold/` | Copyable project skeleton for the above — guards, page splitter, `projekt.py` |

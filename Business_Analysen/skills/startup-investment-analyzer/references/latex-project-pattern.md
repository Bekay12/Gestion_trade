# The LaTeX project pattern

`report-template.md` says what the seven sections contain. This file says how to
build a project that can actually hold them: where the numbers live, which scripts
stop you from lying by accident, and which traps cost a day each the first time.

The pattern comes from two runs — a graded Master's Hausarbeit on Aumann AG and an
investment analysis of Brown & Brown that reused its method — and was hardened on a
third (Alamos Gold, a gold producer filing Form 40-F instead of 10-K). Every rule
below exists because its absence produced a real defect in one of them. A copyable
skeleton sits in `assets/scaffold/`.

**Use it when** the subject will be revisited (a holding you re-examine each
quarter), when a reader will check figures against the source, or when the report
runs past ten pages. **Skip it** for a one-off screening memo: the setup costs
about a day and pays back on the second pass.

---

## Layout

```
<Firma>/
  analyse.tex          nur \input-Zeilen, keine Inhalte
  preamble.tex         Satz, Farben, Betragsmakros, ZITIERMAKROS
  build.sh             latexmk -halt-on-error, bricht bei jedem Fehler ab
  sections/            ein Abschnitt je Teil - enthaelt KEINE Zahl
  data/                erzeugte Zahlenschicht + Tabellenkoerper - nie von Hand
  scripts/             Beschaffung, Zahlenschicht, Rechnung, Wachhunde
  refs/                Einreichungen als .htm UND als seitenweiser .txt (gitignoriert)
  docs/                Befunde und getroffene Entscheidungen
  out/                 Bauergebnis (gitignoriert)
```

Two invariants carry everything else:

1. **`sections/*.tex` contains no digit that is a figure.** Every value arrives as a
   macro from `data/kennzahlen.tex` or inside a generated table body.
2. **`data/` is generated.** Re-running the scripts must reproduce it byte for byte.

---

## The data layer

`scripts/kennzahlen.py` holds three blocks and nothing else.

```python
ROH = {                       # Rohwerte: genau einmal, mit gedruckter Seite
    "Umsatz":     (1808.8, "Abschluss 2025, S. 7"),
    "Produktion": (545400,  "MD&A 2025, S. 4"),
}
EXTERN = {                    # Sekundaerquellen: KEINE gedruckte Seite
    "KonsensZiel": (46.25, "stockanalysis.com/..., abgerufen 16.09.2026"),
}
ABGELEITET = [                # gerechnet, nie getippt
    ("UmsatzDelta", "(Umsatz / UmsatzVJ - 1) * 100"),
]
```

`EXTERN` is a separate dict, not a flag on `ROH`. The page guard iterates `ROH`;
a web figure has no page, and mixing the two means either the guard weakens or the
secondary data quietly acquires an authority it does not have. The guard prints the
`EXTERN` count so the split stays visible, and every table carrying such a value
says *Sekundärquelle* in its caption.

**Roman-numeral suffixes are not an affectation.** LaTeX macro names cannot contain
digits, so a ten-year series becomes `UmsatzXVI … UmsatzXXV`. Generate those entries
with a script and paste them in; typing thirty of them by hand is how a transposed
digit enters a report.

---

## The five guards

Judgement fails at this scale. A script does not.

| Script | Catches |
|---|---|
| `pruefe_seiten.py` | a value that is not on the page it cites |
| `reihe.py` | a multi-year series whose columns silently misaligned |
| `check_literals.py` | a figure typed into prose instead of coming from the data layer |
| `check_footnote_pages.py` | the same source+page footnoted twice on one printed page |
| `check_footnote_groups.py` | a reused footnote whose anchor sits on another page |

Run `pruefe_seiten.py` **before writing a sentence of prose**. In the reference run
23 of 176 citations were wrong on the first pass — without exception the ones
inferred from context rather than read off the page. In the Alamos run every one of
203 values passed first time, for exactly one reason: they were extracted
mechanically from the page text rather than transcribed.

### A guard that finds nothing is worse than no guard

`check_footnote_pages.py` matches specific citation macros. Ported to a subject
filing Form 40-F, whose macros are named differently, it reported *"0 Zitationen
geprüft — OK"* and looked green for as long as nobody read the count. Make the
macro names configurable (`scripts/projekt.py` in the scaffold) and have the script
fail loudly when it finds no citations at all.

---

## Splitting filings into printed pages

SEC filings arrive as one HTML file. Printed page breaks are `<hr>` elements and the
printed number sits in the footer before the break. **Read it; never infer it from
the block index** — they coincide often enough to lull you and diverge exactly where
it matters.

Three traps, all met in practice:

**The footer form differs per issuer and per document.** `12` alone, `12 | FIRMA`,
`FIRMA | 12`, and `12 |` on a line of its own with the name on the next. Checking
only the last line left 135 of 137 blocks of one Annual Information Form
uncitable. Check the **last three** non-empty lines against every known form.

**Measure page continuity, not the share of blocks that carry a number.** Older
filings emit two `<hr>` blocks per printed page — a running header and the body —
and only the body carries the footer. That reads as 58 % coverage while pages 3 to
49 are in fact complete. The right metric is: are there gaps in the printed
sequence?

**A figure in the text can look exactly like a page number.** In one Annual
Information Form the reserve figure `452` sat in the last three lines of a block and
matched every bare-number pattern; read as a page it turned "pages 1–68" into
"pages 1–452" and would have anchored every citation from that block to a page that
does not exist. The defence is not a tighter pattern but a **sequence filter**: keep
only the longest chain in which each number repeats the previous or exceeds it by
one, and discard the rest. A page number is recognisable by continuing the
pagination, not by its shape.

---

## The chain check on a multi-year series

A ten-year series is assembled from ten reports, and each report's key-figure page
prints **four** columns: current quarter, prior quarter, current year, prior year.
Take the wrong one and you get a quarterly figure where an annual belongs — off by a
factor of four, and on the cited page, so the page guard passes.

Two mechanical defences:

**Footnote markers shift the columns.** A row carrying a footnote reference emits an
extra small integer before the values: `[1, 105676, 104734, 392000, 380000]`. Drop a
leading single digit when the row yields five values and the rest are orders of
magnitude larger.

**Then check the chain.** The prior-year column of report *Y* must equal the
current-year column of report *Y−1*. Ten reports give nine such equations per metric.
In the Alamos run 45 equations held and exactly one broke — and that break was a
**finding**, not a fault: the company had restated its unit costs to exclude
mark-to-market effects of share-based compensation. Record acknowledged breaks in a
`BEKANNT` dict with the reason and the page, so the check stays green *and* a new
break still fails. Never silence the check itself.

---

## Reusing a footnote number — and the bug that hides in it

Citing the same source and page repeatedly on one printed page prints the identical
footnote several times. `\quelleMerken{key}` stores the number, `\quelleErneut{key}`
refers back to it. The mechanism is only valid **within one printed page**, so
`fussnoten_gruppen.py` assigns the keys from the SyncTeX page map and loops until
both checkers fall silent.

**The trap is in the key.** The original implementation built it as

```python
key = f"s{seite}" + re.sub(r"[^A-Za-z]", "", quelle.lower())[:14]
```

which strips exactly the digits that distinguish one source from another:
`MDA|2025|7` and `MDA|2025|13` both became `mda`. Two different pages of the same
document cited on one printed page received the same key, the second `\quelleMerken`
overwrote the first, and every `\quelleErneut` then pointed at the wrong page.

It produces no warning. It is not caught by the page guard, which checks the data
layer and never looks at prose footnotes. In the Brown & Brown document it hit
**7 keys and 12 reuses**: three footnotes on the management section cite page 28 of
the 10-K, the page that states the stock exchange listing.

Encode the digits instead of deleting them (`0→a … 9→j`) and add the regression test
in `assets/scaffold/scripts/test_fussnoten_gruppen.py`.

A second, smaller trap in the same script: a citation macro that takes only a page
(`\quelleZB{4}`) yields `None` for the missing second argument, and the key becomes
`...zbnone` with no matching `\quelleMerken`. LaTeX then aborts with *Missing number,
treated as zero*. Keep single-argument macros in their own set.

---

## Three colours, three responsibilities

| Colour | Content | Who stands behind it |
|---|---|---|
| black | figures, tables, sources, pages | the primary source |
| blue `\bk{}` | assessments inside the running text | the author, after checking |
| red `\vd{}` | the verdict in part 7 | a recommendation under declared assumptions |

The test: a reader who reads only the black text gets the verified position with no
judgement mixed in. Keep both switchable in one place so a version for third parties
needs no edit to the text. Use `\color` inside a group, never `\textcolor` — the
latter is not `long` and a multi-paragraph verdict aborts the run. And a colour macro
must never wrap a figure, or the number stops being traceable.

---

## Order of operations

1. Acquire sources → `refs/`, split with `edgar_seiten.py`, **demand a gapless page
   sequence per file**.
2. Build `ROH` by extracting mechanically from the page text.
3. `pruefe_seiten.py` — green before any prose.
4. `reihe.py` — chain check on the multi-year series.
5. Compute in `flow.py`; never hand-compute an IRR.
6. Write `sections/`; `check_literals.py` after each.
7. `./build.sh`, then `fussnoten_gruppen.py --schleife`, then both footnote checkers.
8. `python3 -m unittest discover -p 'test_*.py'`.

Steps 3 and 4 before step 6 is the whole point. In the other order you build
paragraphs around citations that turn out to be wrong, and the rewrite costs more
than the check ever would.

---

## Adapting to a new subject

Only `scripts/projekt.py` and the citation macros in `preamble.tex` are
company-specific among the reusable parts. `kennzahlen.py`, `pruefe_seiten.py` and
`flow.py` are written fresh each time and deliberately have no template: they carry a
particular company's figures and a particular company's assumptions, and a template
of those is an invitation to leave the previous subject's assumptions standing.

One structural warning. The pattern's part 5 was built for a leveraged acquirer and
asks "how fast can it deleverage?". Applied to a subject holding more cash than debt
that question is empty, and answering it anyway produces a page of arithmetic that
decides nothing. Replace it with the capacity question the subject actually faces —
for a miner mid-build, "can the published capital programme be funded from operating
cash flow, and at which price does it stop being funded?" The *shape* transfers: one
capacity calculation, one owner's calculation, stated as alternatives because the
same cash cannot be spent twice.

### Calibrate the model against a known year

A cash-flow model built on a company's own cost metric must be held against an actual
year before it is used to price anything. In the Alamos run the model produced 498.7
against a reported 288.2 — and the 210.5 gap resolved to 209.6 of named one-off items
(a hedge unwind, a prepayment, a working-capital build, capitalised interest), leaving
0.9 unexplained on a 981.7 gross margin. Publish that reconciliation as a table. A
model that reproduces a real year once you name the exceptions is usable; one that
was never checked is a claim.

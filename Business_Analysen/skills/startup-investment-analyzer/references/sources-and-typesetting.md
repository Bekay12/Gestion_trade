# Sources and typesetting

The parts of a run that cost hours and teach nothing the second time. Nothing here is
about judgement; it is about getting figures out of filings and onto a page without
silently corrupting them.

---

## Opening up a US filer (EDGAR)

### Finding the filings

```bash
curl -s -A "Name mail@example.com" \
  "https://data.sec.gov/submissions/CIK0000079282.json"
```

The SEC requires a real User-Agent with a contact address; without one you get throttled
or blocked. `filings.recent` gives form type, filing date, accession number and primary
document; the fetch URL is
`https://www.sec.gov/Archives/edgar/data/<cik>/<accession-without-dashes>/<primaryDocument>`.

Four filings usually carry a full analysis: the two most recent **10-K** (current year and
comparatives), the latest **10-Q** (post-balance-sheet development, often the sharpest
part of the story), and the **DEF 14A** proxy (board, executive officers, beneficial
ownership). `companyfacts` from the XBRL API is useful for long series such as dividends
per share, but treat it as a cross-check: it will not give you a page number.

### Recovering the printed page

A filing arrives as one HTML file, and the citation rule needs printed pages. The page
breaks of the printed document are the `<hr>` elements; the printed number is the last
line before each break. Split there, strip tags, read the footer.

**The footer format differs by document type**, and this is the trap:

| Document | Last line of the block |
|---|---|
| 10-K, 10-Q | `49` |
| DEF 14A | `76 \| BROWN & BROWN, INC.` or `BROWN & BROWN, INC. \| 77` (alternating) |

Handle only the first form and the proxy yields 3 pages out of 109 — and every figure from
it becomes uncitable. Check the ratio after splitting: if many blocks come back without a
number, the document has a footer you have not taught the splitter yet.

Do **not** infer the printed page from the PDF page or from a line number in a full-text
dump. See `rigor-and-assumptions.md` rule 1.

### Where the interesting numbers sit in a 10-K

Roughly, and worth knowing before you start grepping: Item 1 carries headcount and segment
descriptions; Item 5 the share count, holders of record and buyback authorisations; Item 7
(MD&A) the reconciliation of non-GAAP measures and the year-over-year table with percentage
changes — usually the densest page in the filing; Item 8 the statements themselves; the
notes carry business combinations (purchase price allocation, pro-forma revenue) and the
debt schedule with maturities and coupons. The contractual-obligations table in Item 7 is
gold for a flow analysis: it hands you the interest and repayment profile by period, so
neither has to be assumed.

---

## Price series

Yahoo Finance is the obvious source and the least reliable one: it answered HTTP 429 for
every request during the reference run, from several endpoints, with and without cookies.
Stooq served a JavaScript challenge. What worked:

```bash
curl -s -A "<browser UA>" -H "Referer: https://www.nasdaq.com/" \
  "https://api.nasdaq.com/api/quote/BRO/historical?assetclass=stocks&fromdate=2020-12-01&todate=2026-08-28&limit=3000"
```

Values arrive as `"$110.57"` with thousands separators; strip `$` and `,` before parsing.
Write the retrieval date into the header of every generated file, and name the source in
the report — a price series with no date is not a citation.

Keep a second source in the script and say in the source list which one answered. Deriving
the year-end close, the yearly high and low **and the trading day each extreme fell on**
costs nothing extra and gives section 4 its evidence: an annual high that predates an
announcement is an argument, an annual high alone is a number.

---

## Typesetting traps (LaTeX)

Each of these produced a wrong or broken page in the reference run, and none of them
announces itself in the source.

**Macro names cannot contain digits.** `\KursSchluss2025` is not a valid control word.
Spell the year: `ZFI`…`ZFVI` for 2021–2026, `Fuenf`/`Zehn`/`Fuenfzehn` for IRR horizons.
The error message points at the definition, not at the year, so it reads as nonsense until
you know.

**siunitx groups the decimals too.** With `group-separator={.}` the exact stored value
43.4679 was typeset as `43,467.9` — wrong by three orders of magnitude and entirely
plausible-looking. Set `group-digits = integer`. Then round on **output**
(`round-mode=places`), never in the stored value, so derived figures still come from full
precision.

**TeX eats the space after a control word.** `\KurstagHoch und` sets `1.4.2025und`. Wrap
bare value macros in a trivial formatting macro (`\Datum{\KurstagHoch}`) rather than
remembering `{}` at every call site.

**`\textcolor` is not `long`.** A colour macro wrapping more than one paragraph aborts the
run with `Paragraph ended before \@textcolor was complete`. Use `{\color{x}...}` in a
group. This bites exactly when the verdict section grows past one paragraph.

**`\input` appends `\relax`.** Inside a table that starts the next cell and `\bottomrule`
fails with `Misplaced \noalign`. Read generated table bodies with the primitive
`\@@input` instead.

**pgfplots dates.** `date coordinates in=x` needs `\usepgfplotslibrary{dateplot}`, or the
first date is parsed as a float. And `xtick distance={365 days}` does not work there —
give explicit tick dates.

**Verify in the rendered PDF, not the source.** Two of the above were invisible in the
`.tex` and obvious in the image. Render the pages you changed and look at them. A missing
`\par` after a `tabular` put a whole line of text on the same baseline as the last table
row, and `\vspace` silently acted horizontally.

---

## Guards worth having from the start

Four scripts paid for themselves within one document:

| Guard | Catches |
|---|---|
| page verifier | a figure cited to a page it does not appear on (23 of 176 on the first pass) |
| literal check | any number typed into prose instead of read from the data layer |
| footnote duplicates | the same source cited twice on one printed page with two footnotes |
| footnote groups | a reused footnote number whose anchor sits on the previous page |

The last two need the real page break, so they read it from SyncTeX rather than guessing.
Both are page-sensitive: any edit that moves a line can create or invalidate a group, so
they run after every build, and a small fixer that re-splits the groups automatically is
worth more than doing it by hand.

Expect the literal check to catch **your own** figures in the methodology and verdict
sections. It should. Move them into the data layer like any other number; a document that
states how many figures it verifies should not hard-code the count.

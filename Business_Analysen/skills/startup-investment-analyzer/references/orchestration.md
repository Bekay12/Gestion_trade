# Orchestration

Research is split across subagents so raw source material never enters the main thread.
Each subagent reads heavily, writes a structured file, and returns a ten-line digest.

**Mechanism: the native Agent tool**, one dispatch per research domain. Not
`superpowers:subagent-driven-development` — that skill exists to execute a code plan task
by task with review packages, a five-round fix loop, and a quality gate. Those have no
natural counterpart in documentary research, and their apparatus costs more than it
returns here.

---

## The split

Three domains for a listed company, four for a startup. They are chosen to be
**independent** — no subagent needs another's output.

| # | Domain | Covers | Feeds |
|---|---|---|---|
| 1 | Company & governance | Management, board, mandates, ownership / cap table, directors' dealings (12 months), legal structure, sites | §1 |
| 2 | Financials & metrics | All figures with page anchors, multi-year series, year-end prices for the historical multiples, guidance vs. actual | §2, §3, §6, §7.3 |
| 3 | Market & options | Competitors, market size, evidence for the competitive position (market share, two peers' margins and multiples, same year and definition), the strategic options or investment criteria | §1.6, §5, §7.3 |
| 4 | Traction & financing *(startup only)* | ARR/MRR, cohorts, burn, runway, round history, terms | §2, §5 |

Section 4 (explaining the trajectory), section 6 (flow analysis) and section 7 (verdict)
are **not** delegated. They are synthesis across domains, and they are where the report's
judgement lives — keep them in the main thread on a capable model.

---

## Model assignment

| Role | Model | Why |
|---|---|---|
| Research subagents | Haiku | They frame the task and drive the local model; they do not reason deeply |
| Bulk reading / condensation | **local** via `scripts/local_llm.sh` | Free, and this is screening, not final diligence |
| Verification of any cited figure | Haiku or Sonnet, targeted extraction | Never the local model — see below |
| Synthesis, §4/§6/§7, final report | Sonnet or better | Where the judgement is |

**Always name the model explicitly in a dispatch.** An omitted model inherits the session's
most capable and most expensive one.

### The local-model boundary

Local models condense; they do not source. Measured behaviour: a 14B model stopped
honouring a strict output format at roughly 14k prompt tokens, dropped the required page
anchors, injected banned evaluative wording, and returned two different employee counts
across two runs of the same document.

So the rule inside every subagent prompt: **the local model may produce candidate facts;
every figure that will be cited is re-verified against the source by targeted extraction
before it enters the digest.** Facts that could not be verified go in the gaps list.

### Calling the local model

```bash
scripts/local_llm.sh --model gemma4:12b < prompt.txt        # stdin, never an argument
scripts/local_llm.sh --model gemma4:12b --probe             # runtime check only
```

The script probes `llmfit`, then `ollama` (starting the server if needed), and exits **3**
with `NO_LOCAL_RUNTIME` on stderr if neither answers. On exit 3 the subagent redoes the
condensation itself on its own (cheap) model and says so in its digest.

Passing the prompt as a command argument hangs indefinitely in a non-interactive shell.
Always stdin.

---

## Dispatch prompt skeleton

```
You are researching [DOMAIN] for an investment analysis of [SUBJECT].

## Sources
[explicit local paths / URLs. Downloaded first so page numbers are stable.]

## Your job
1. Read the sources. Delegate bulk condensation to the local model:
   scripts/local_llm.sh --model gemma4:12b < your_prompt.txt
   If it exits 3 (NO_LOCAL_RUNTIME), do the condensation yourself and note it.
2. RE-VERIFY every figure that will be cited, by targeted extraction against the
   source. The local model's output is candidate material, never evidence.
3. Confirm the PRINTED page for each citation (see rigor-and-assumptions.md §1) —
   PDF page and printed page differ.
4. Write your findings to [WORKDIR]/[domain].md using the schema below.
5. Return ONLY the digest — under 10 lines.

## Hard rules
- Never invent a figure. Unverifiable → the Gaps list.
- Facts only. No evaluation, no adjectives like "strong" or "promising".
- Every figure: value, source, printed page (or URL + retrieval date).

## File schema
# [Domain] — [Subject]
## Verified facts
| Fact | Value | Source | Page/URL | Retrieved |
## Gaps
- [what is missing, where you looked, why it matters]
## Contradictions
- [figures that disagree across sources, both cited]
## Raw notes
[anything the synthesis might need; not size-limited]

## Digest to return (max 10 lines)
STATUS: OK | PARTIAL | BLOCKED
CONFIDENCE: high | medium | low
FILE: [path]
- [up to 4 headline findings, each with its figure]
- GAP: [each blocking gap]
COUNTS: [n] verified facts, [n] sources, [n] gaps
LOCAL_MODEL: used | unavailable (fell back)
```

---

## Why the digest is capped

Everything a subagent prints stays in the main context for the rest of the session and is
re-read every turn. Four subagents returning 40-line JSON blobs cost 160 lines carried to
the end; four digests cost 40. The synthesis reads the files on demand, once, when it needs
them.

The same reason applies to dispatches: **never paste prior-domain summaries into a later
dispatch.** The domains are independent by design. A fresh subagent needs its domain, its
sources, and the rules — nothing else.

---

## Progress file

Keep `[WORKDIR]/progress.md` in the analysis directory, **not** in a gitignored scratch
path. In the reference case the progress ledger lived in a gitignored directory and would
not have survived the session closing.

Append one line per completed domain:

```
[domain]: complete -> [file] ([n] facts, [n] gaps, confidence [level])
[domain]: GAP CARRIED -> [what synthesis must handle]
```

Any gap a subagent could not close is carried here so the synthesis handles it explicitly
rather than silently inheriting it.

---

## Sequence

1. Frame the analysis; download and pin the sources locally.
2. Dispatch domains 1–3 (or 1–4). Do not run implementation subagents in parallel with
   each other if they write to the same files — these do not, so parallel is fine.
3. Read the digests. Any `BLOCKED` or `PARTIAL` status: decide whether to re-dispatch with
   more context or to record the gap and continue.
4. Run `scripts/cashflow_irr.py` for section 6.
5. Write sections 4, 6 and 7 in the main thread, reading the domain files as needed.
6. Assemble, then verify: every figure cited, every assumption declared, every gap listed.

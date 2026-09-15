# Local agent workflow for this workspace

This workspace is already set up for a local-first workflow. Use the local Ollama bridge for drafting, rewriting, and explanation work, and reserve cloud tokens for tasks that truly require them.

## Default routing

- Use local models for:
  - drafting prose in German or English
  - rewriting sections of the thesis
  - summarizing notes and plan updates
  - lightweight code or LaTeX edits
- Use cloud models only for:
  - final polishing of highly sensitive academic prose
  - tasks that need stronger reasoning than the local model can provide
  - anything that explicitly requires a cloud-only capability

## Required startup step

Before using the local bridge, run:

```bash
./scripts/ensure_ollama.sh
```

This starts Ollama if needed and keeps it reachable for repeated calls.

## Preferred pattern

1. Read the relevant file or section.
2. Use the local model to draft or revise content.
3. Validate the result with the project checks.
4. Only escalate to a cloud model if the local output is insufficient.

## Delegation limits (measured 2026-08-02, not assumed)

**Always set `num_ctx`.** Ollama truncates the prompt to `num_ctx` and reserves about half
that window for the output, cutting from the *start* — where the format spec sits. On a
15,403-token prompt the same model produced 0 of 8 required output lines at `num_ctx=4096`
(only 2,050 tokens read) and 8 of 8 at `num_ctx=16384` (all 15,403 read). Pass at least twice
the expected prompt length, then compare the response's `prompt_eval_count` with the real
prompt length; if they differ, the prompt was cut and the answer must be discarded.

**Never take a cited figure from an image model.** On the "Aumann in figures" table
(`refs/gb2024.pdf`, printed p. 2, 29 values verified against `data/kennzahlen.tex`):

| Method | Correct | Fabricated amounts |
|---|---|---|
| `pdftotext -layout` | 29/29 | 0 |
| PaddleOCR-VL 1.6 | 29/29 | 0 |
| `qwen3.5:9b` (image mode) | 29/29 | 0 |
| `llama3.2-vision:11b` | 15/29 | 4 |
| `ibm/granite-docling:258m` | 7/29 | 2 |
| `gemma4:12b` (image mode) | 1/29 | **39** |

All eight PDFs in `refs/` are born-digital (1,400–3,100 characters of text layer per page),
so OCR is never needed here. Take figures with `pdftotext -layout -f N -l N`, one page at a
time, and verify each against the printed page. `qwen3:14b` reads figures correctly but
silently strips thousands separators and percent signs — check the formatting, not just the
value.

**The blue zones stay with the author.** A local model may draft `%` comments and
documentation. It does not write the LaTeX prose of this Hausarbeit: rule 1 in
[../CLAUDE.md](../CLAUDE.md) reserves the evaluative passages for the author.

## Project checks to run after edits

```bash
./build.sh
python3 scripts/check_literals.py sections/*.tex
cd scripts && python3 -m unittest discover -p 'test_*.py'
```

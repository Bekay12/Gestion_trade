# Skill-Kopien

Archivkopien der Skills, die die Analysen in diesem Verzeichnis tragen. Sie liegen
hier, damit sie **versioniert** sind und mit dem Repository reisen.

> **Diese Kopien sind nicht aktiv.** Claude Code lädt Skills aus
> `~/.claude/skills/` (persönlich) und aus `<repo>/.claude/skills/` (Projekt).
> Dieses Verzeichnis ist keines von beiden und wird bewusst nicht geladen: Zwei
> aktive Fassungen desselben Namens laufen mit der Zeit auseinander, und dann
> entscheidet der Ladepfad darüber, welche Regeln gelten.

| Skill | Maßgebliche Fassung | Wofür |
|---|---|---|
| `startup-investment-analyzer` | `~/.claude/skills/startup-investment-analyzer/` | Belegpflichtige Unternehmens- und Anlageanalysen; Muster der Arbeiten in `../Aumann AG/`, `../Brown & Brown/` und `../Alamos/` |

## Kopie auffrischen

```bash
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.gitkeep' \
  ~/.claude/skills/startup-investment-analyzer/ \
  Business_Analysen/skills/startup-investment-analyzer/
```

Bei jeder Änderung an der maßgeblichen Fassung mitlaufen lassen, sonst ist die
Kopie eine Momentaufnahme mit dem Anschein von Aktualität. Stand dieser Kopie:
**19.09.2026**.

## Was drinsteckt

- `SKILL.md` — Einstieg und die acht nicht verhandelbaren Regeln
- `references/report-template.md` — das Sieben-Teile-Gerüst des Berichts
- `references/latex-project-pattern.md` — der baubare Projektaufbau: Zahlenschicht,
  fünf Wachhunde, Seitenzerlegung, die Fußnotenfallen
- `references/rigor-and-assumptions.md` — Belegpflicht, Annahmen, Sensitivitäten
- `assets/scaffold/` — kopierbares Gerüst; `scripts/projekt.py` ist die einzige
  firmenspezifische Datei darin
- `scripts/cashflow_irr.py` — IRR, Schwellenkurs, erforderliches Wachstum und
  erforderlicher Endwert, je durch Bisektion

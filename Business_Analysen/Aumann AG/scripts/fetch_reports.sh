#!/usr/bin/env bash
# Laedt die Aumann-Finanzberichte nach refs/ und schreibt refs/MANIFEST.md.
# Quelle: https://www.aumann.com/en/investor-relations/financial-reports
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p refs
BASE="https://www.aumann.com/fileadmin/templates/downloads/finanzberichte"
ABRUF="$(date +%d.%m.%Y)"

# Deterministische Reihenfolge (Geschaeftsberichte chronologisch, dann
# Zwischenberichte chronologisch) statt einem assoziativen Array, dessen
# Iterationsreihenfolge nicht spezifiziert ist. Format: "key=dateiname".
DOCS=(
  "gb2022=2022_aag-geschaeftsbericht-hp_englisch.pdf"
  "gb2023=2023_aag-geschaeftsbericht-hp_englisch.pdf"
  "gb2024=2024_aag-geschaeftsbericht-eng-hp.pdf"
  "gb2025=2025_AAG_Gesch%C3%A4ftsbericht_FINAL_eng.pdf"
  "q1-2025=20250513-aumann-quartalsmitteilung-q1-2025-eng.pdf"
  "h1-2025=20250814-aumann-halbjahresfinanzbericht-h1-2025-eng.pdf"
  "q3-2025=20251113-aumann-quartalsmitteilung-q3-2025-eng.pdf"
  "q1-2026=20260512-aumann-quartalsmitteilung-q1-2026-eng.pdf"
)

{
  echo "# Manifest der Aumann-Quelldokumente"
  echo
  echo "Abrufdatum aller Dokumente: ${ABRUF}"
  echo
  echo "| Datei | Quelle (URL) | Seiten | SHA-256 |"
  echo "|---|---|---|---|"
} > refs/MANIFEST.md

for entry in "${DOCS[@]}"; do
  key="${entry%%=*}"
  filename="${entry#*=}"
  url="${BASE}/${filename}"
  out="refs/${key}.pdf"
  echo "-> ${key}"
  curl -sSL --fail --max-time 120 "$url" -o "$out"
  if ! head -c 5 "$out" | grep -q '%PDF-'; then
    echo "FEHLER: ${out} ist kein PDF" >&2; exit 1
  fi
  pages="$(pdfinfo "$out" | awk '/^Pages:/{print $2}')"
  sha="$(sha256sum "$out" | cut -c1-16)"
  echo "| \`refs/${key}.pdf\` | <${url}> | ${pages} | \`${sha}\` |" >> refs/MANIFEST.md
done

echo "OK -> refs/MANIFEST.md"

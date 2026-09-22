"""Lecture déterministe du tableau « 50 Aktien fürs Leben » (Capital 10/2026, p. 80-99)."""
import json, re
from pathlib import Path
P = Path(__file__).parent / "pages"
NUM = r"([\d,–-]+)"
# Nom sans double espace ; pays entre parenthèses (imbrication "SLB (Schlumberger, Antillen)") ;
# appel de note facultatif ("Fresenius (Deutschland) 1") ; les cinq chiffres peuvent déborder
# sur la ligne suivante (Shin-Etsu).
ROW = re.compile(r"(?:^|\s{2,})([A-ZÄÖÜ0-9](?:[^()\s]|\s(?!\s))*?) \(([^()]+)\)(?: \d)?((?:\s+[\d,–-]+)*)\s*$")
ISIN = re.compile(r"(?:^|\s)(\d{2})\s+([A-Z]{2} ?[0-9A-Z]{3} ?[0-9A-Z]{6} ?\d)\b")
CAT = re.compile(r"^\s*(DOMINATOR|[A-ZÄÖÜ]{5,})\s*$")
out, cat = [], None
for p in range(85, 92):  # tableau principal ; p. 94-95 = "Junge Stars" (candidats, exclus)
    lines = (P / f"Capital_10-2026_p{p:03d}.txt").read_text().splitlines()
    for i, l in enumerate(lines):
        m = ROW.search(l)
        if not m:
            continue
        nums = m.group(3).split()
        if len(nums) < 5 and i + 1 < len(lines):  # débordement : chiffres sur la ligne suivante
            nums = lines[i + 1].split() + nums
        if len(nums) != 5:
            continue
        rank = isin = sector = None
        for j in range(i + 1, min(i + 5, len(lines))):
            mi = ISIN.search(lines[j])
            if mi:
                rank, isin = mi.group(1), mi.group(2)
                seg = [s for s in re.split(r"\s{2,}", lines[j + 1].strip()) if s] if j + 1 < len(lines) else []
                sector = seg[-1] if seg else None
                break
        out.append({"page": p, "rang": rank, "entreprise": m.group(1).strip(), "pays": m.group(2),
                    "isin": isin, "secteur": sector, "div_rendite": nums[0], "ausschuettung": nums[1],
                    "kcv": nums[2], "kgv": nums[3], "rendite_25j": nums[4]})
json.dump(out, open(Path(__file__).parent / "capital50.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "lignes;", len({o['rang'] for o in out if o['rang']}), "rangs distincts")
for o in out: print(o["rang"], o["entreprise"], "|", o["pays"], "|", o["isin"], "|", o["secteur"], "|", o["div_rendite"], o["kgv"], o["rendite_25j"], "p", o["page"])

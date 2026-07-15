"""
Genere Academy_Germain_AVEC_TOC.pdf avec pages de TOC au theme identique
aux PDFs de la formation (header/footer navy+gold, filigrane, boite titre).
"""

import sys, subprocess, re
from pathlib import Path

try:
    import fitz
except ImportError:
    print("Installation de PyMuPDF...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

print(f"PyMuPDF {fitz.__version__}")

BASE_DIR = Path(r"C:\Users\Berkam\Documents\Projets_perso\Gestion_trade\Academy-Germain-final")
OUTPUT   = BASE_DIR / "Academy_Germain_AVEC_TOC.pdf"

# ── Dimensions exactes issues des PDFs sources ────────────────
W = 595.28
H = 841.89
ML = 56.69   # marge gauche contenu
MR = 538.58  # marge droite contenu

# ── Couleurs extraites des PDFs (theme Academy Germain) ───────
C_NAVY  = (0.102, 0.102, 0.180)   # bleu marine fonce
C_GOLD  = (0.788, 0.659, 0.298)   # or / ambre
C_WHITE = (1.0,   1.0,   1.0  )
C_LGREY = (0.87,  0.89,  0.93 )   # gris clair (filigrane)
C_MGREY = (0.80,  0.80,  0.80 )   # gris moyen
C_DARK  = (0.20,  0.20,  0.20 )   # texte courant

# ── Coordonnees header/footer (mesures des PDFs sources) ──────
HDR_H      = 60.94   # hauteur bande header
HDR_GOLD_Y = 64.63   # fin du filet gold sous header
FTR_GOLD_Y = 797.95  # debut filet gold au-dessus du footer
FTR_BAR_Y  = 800.79  # debut bande footer navy

# ── Titres des parties ─────────────────────────────────────────
PARTIE_TITLES = {
    0:  "Partie 0  -  Introduction au Trading",
    1:  "Partie 1  -  Mecanismes Economiques Fondamentaux",
    2:  "Partie 2  -  Analyse Fondamentale de l'Entreprise",
    3:  "Partie 3  -  Financement & Instruments Financiers",
    4:  "Partie 4  -  IPO & Modes de Cotation",
    5:  "Partie 5  -  Architecture des Marches Financiers",
    6:  "Partie 6  -  Acteurs des Marches",
    7:  "Partie 7  -  Formation des Prix & Mecanismes",
    8:  "Partie 8  -  EDGAR & Analyse des Filings",
    9:  "Partie 9  -  Catalyseurs de Marche",
    10: "Partie 10 -  Analyse Graphique",
    11: "Partie 11 -  Psychologie & Gestion du Risque",
    12: "Partie 12 -  Outils, Workflow & Routine",
    13: "Partie 13 -  Cryptomonnaies",
}

# ── Helpers ────────────────────────────────────────────────────
def partie_num(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 999

def chapter_num(filename):
    m = re.search(r'_Ch(\d+)_', filename, re.I)
    return int(m.group(1)) if m else 999

def chapter_title(filename):
    stem = Path(filename).stem
    m = re.match(r'P\d+_Ch(\d+)_(.*)', stem)
    if m:
        ch, rest = m.groups()
        return f"Ch.{ch.zfill(2)}  {rest.replace('_', ' ')}"
    return stem.replace('_', ' ')

def tw(text, fontname, fontsize):
    """Largeur d'un texte."""
    return fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)

# ── Collecte des PDFs ──────────────────────────────────────────
dirs = sorted(
    [d for d in BASE_DIR.iterdir() if d.is_dir() and (d / "PDF").exists()],
    key=lambda d: partie_num(d.name)
)
chapters = []
for d in dirs:
    pn = partie_num(d.name)
    for pdf in sorted((d / "PDF").glob("*.pdf"), key=lambda f: chapter_num(f.name)):
        chapters.append({"pnum": pn, "title": chapter_title(pdf.name), "path": pdf})

print(f"{len(chapters)} chapitres trouves")

# ── Fusion des PDFs sources ────────────────────────────────────
merged = fitz.open()
pg = 0
for ch in chapters:
    src = fitz.open(str(ch["path"]))
    ch["pg0"] = pg
    ch["pgc"] = len(src)
    merged.insert_pdf(src)
    src.close()
    pg += ch["pgc"]
print(f"{pg} pages de contenu fusionnees")

# ── Layout de la TOC ───────────────────────────────────────────
# Page 0 : titre occupe y=93..180, contenu commence a y=185
# Pages suivantes : contenu commence a y=HDR_GOLD_Y+14 = ~79
CONTENT_Y0   = 185    # debut contenu page 0 (apres boite titre)
CONTENT_Y    = 80     # debut contenu pages suivantes
CONTENT_YBOT = 791    # fin contenu (au-dessus du filet footer)
LH_P = 18             # hauteur ligne partie
LH_C = 13             # hauteur ligne chapitre
GAP_P = 8             # espace avant nouvelle partie

def compute_layout():
    entries, pidx, last_pnum = [], 0, None
    y = CONTENT_Y0
    ytop = CONTENT_Y
    ybot = CONTENT_YBOT
    for ch in chapters:
        if ch["pnum"] != last_pnum:
            if last_pnum is not None:
                y += GAP_P
            if y + LH_P + LH_C > ybot:
                pidx += 1
                y = ytop
            entries.append(("partie", pidx, y, ch["pnum"]))
            y += LH_P
            last_pnum = ch["pnum"]
        if y + LH_C > ybot:
            pidx += 1
            y = ytop
        entries.append(("chapter", pidx, y, ch))
        y += LH_C
    return entries, pidx + 1

layout, n_toc = compute_layout()
print(f"{n_toc} page(s) de table des matieres")

# ── Decalage final des references de pages ────────────────────
for ch in chapters:
    ch["fp0"] = ch["pg0"] + n_toc
    ch["fp1"] = ch["fp0"] + 1

# ── Insertion des pages vierges de TOC en tete ────────────────
for i in range(n_toc):
    merged.insert_page(i, width=W, height=H)

# ══════════════════════════════════════════════════════════════
# Fonctions de dessin correspondant au theme de la formation
# ══════════════════════════════════════════════════════════════

def draw_header(page, label_right, sub_right):
    """Bande header navy + filet gold + textes (identique aux PDFs source)."""
    # Bande navy
    page.draw_rect(fitz.Rect(0, 0, W, HDR_H),
                   fill=C_NAVY, color=None, overlay=True)
    # Filet gold
    page.draw_rect(fitz.Rect(0, HDR_H, W, HDR_GOLD_Y),
                   fill=C_GOLD, color=None, overlay=True)
    # Gauche : ACADEMY GERMAIN
    page.insert_text((15, 23), "ACADEMY GERMAIN",
                     fontname="hebo", fontsize=11, color=C_WHITE)
    page.insert_text((15, 34), "Ecole de Formation aux Marches Financiers",
                     fontname="helv", fontsize=7.5, color=C_GOLD)
    # Droite : label dynamique
    lw = tw(label_right, "hebo", 8.5)
    page.insert_text((W - 15 - lw, 23), label_right,
                     fontname="hebo", fontsize=8.5, color=C_WHITE)
    sw = tw(sub_right, "helv", 7.5)
    page.insert_text((W - 15 - sw, 34), sub_right,
                     fontname="helv", fontsize=7.5, color=C_MGREY)


def draw_footer(page, page_num):
    """Bande footer navy + filet gold + textes (identique aux PDFs source)."""
    # Filet gold au-dessus
    page.draw_rect(fitz.Rect(0, FTR_GOLD_Y, W, FTR_BAR_Y),
                   fill=C_GOLD, color=None, overlay=True)
    # Bande navy
    page.draw_rect(fitz.Rect(0, FTR_BAR_Y, W, H),
                   fill=C_NAVY, color=None, overlay=True)
    baseline = FTR_BAR_Y + 14
    # Gauche
    page.insert_text((15, baseline),
                     "Academy Germain - Formation aux Marches Financiers",
                     fontname="hebo", fontsize=7.5, color=C_WHITE)
    # Centre : numero de page
    pg_str = f"- {page_num} -"
    pgw = tw(pg_str, "hebo", 8)
    page.insert_text((W / 2 - pgw / 2, baseline), pg_str,
                     fontname="hebo", fontsize=8, color=C_GOLD)
    # Droite : auteur
    author = "Fait par Germain Lionel"
    aw = tw(author, "hebo", 7.5)
    page.insert_text((W - 15 - aw, baseline), author,
                     fontname="hebo", fontsize=7.5, color=C_GOLD)


def draw_title_block(page):
    """Filigrane + boite titre navy avec titre gold (page 0 seulement)."""
    # Filigrane en fond (derriere tout)
    wm = "ACADEMY GERMAIN"
    wmw = tw(wm, "hebo", 40)
    page.insert_text(((W - wmw) / 2, 260), wm,
                     fontname="hebo", fontsize=40, color=C_LGREY, overlay=False)
    # Boite titre navy
    page.draw_rect(fitz.Rect(ML, 93.87, MR, 168.87),
                   fill=C_NAVY, color=None, overlay=True)
    # Titre principal : gold bold
    ttl = "TABLE DES MATIERES"
    ttlw = tw(ttl, "hebo", 21)
    page.insert_text(((W - ttlw) / 2, 130), ttl,
                     fontname="hebo", fontsize=21, color=C_GOLD)
    # Sous-titre : blanc italique
    sub = "Formation Complete aux Marches Financiers"
    subw = tw(sub, "heit", 10)
    page.insert_text(((W - subw) / 2, 152), sub,
                     fontname="heit", fontsize=10, color=C_WHITE)
    # Separateur gold sous la boite
    page.draw_rect(fitz.Rect(ML, 168.87, MR, 172.0),
                   fill=C_GOLD, color=None, overlay=True)


# ── Dessin de toutes les pages TOC ────────────────────────────
for pidx in range(n_toc):
    page = merged[pidx]
    draw_header(page, "TABLE DES MATIERES", f"page {pidx + 1} / {n_toc}")
    draw_footer(page, pidx + 1)
    if pidx == 0:
        draw_title_block(page)

# ── Dessin des entrees de la TOC ──────────────────────────────
for etype, pidx, ey, data in layout:
    page = merged[pidx]

    if etype == "partie":
        pnum  = data
        title = PARTIE_TITLES.get(pnum, f"Partie {pnum}")
        # Bande navy pour la ligne de partie
        bg = fitz.Rect(ML - 4, ey - LH_P + 4, MR + 4, ey + 4)
        page.draw_rect(bg, fill=C_NAVY, color=None, overlay=True)
        # Filet gold a gauche
        page.draw_rect(fitz.Rect(ML - 4, ey - LH_P + 4, ML - 1, ey + 4),
                       fill=C_GOLD, color=None, overlay=True)
        # Texte gold bold
        page.insert_text((ML + 4, ey), title,
                         fontname="hebo", fontsize=11, color=C_GOLD)
        # Lien vers le 1er chapitre de la partie
        for ch in chapters:
            if ch["pnum"] == pnum:
                page.insert_link({"kind": fitz.LINK_GOTO, "from": bg,
                                  "page": ch["fp0"], "to": fitz.Point(0, 0)})
                break

    else:  # chapitre
        ch    = data
        title = ch["title"]
        if len(title) > 72:
            title = title[:69] + "..."

        x_txt  = ML + 14
        tlen   = tw(title, "helv", 9)
        dot_xs = x_txt + tlen + 3
        dot_xe = MR - 26

        # Titre du chapitre
        page.insert_text((x_txt, ey), title,
                         fontname="helv", fontsize=9, color=C_DARK)
        # Pointilles
        if dot_xe > dot_xs + 8:
            dot_w  = tw(".", "helv", 9)
            n_dots = max(0, int((dot_xe - dot_xs) / dot_w))
            page.insert_text((dot_xs, ey), "." * n_dots,
                             fontname="helv", fontsize=9, color=C_MGREY)
        # Numero de page en gold bold
        pg_str = str(ch["fp1"])
        pgw    = tw(pg_str, "hebo", 9)
        page.insert_text((MR - pgw, ey), pg_str,
                         fontname="hebo", fontsize=9, color=C_GOLD)
        # Lien cliquable sur toute la ligne
        page.insert_link({"kind": fitz.LINK_GOTO,
                          "from": fitz.Rect(ML, ey - 10, MR, ey + 2),
                          "page": ch["fp0"], "to": fitz.Point(0, 0)})

# ── Signets PDF (volet lateral) ────────────────────────────────
outline, seen = [], set()
for ch in chapters:
    if ch["pnum"] not in seen:
        outline.append([1, PARTIE_TITLES.get(ch["pnum"], f"Partie {ch['pnum']}"),
                        ch["fp1"]])
        seen.add(ch["pnum"])
    outline.append([2, ch["title"], ch["fp1"]])
merged.set_toc(outline)

# ── Sauvegarde ─────────────────────────────────────────────────
merged.save(str(OUTPUT), garbage=4, deflate=True)
merged.close()

sz = OUTPUT.stat().st_size
print(f"\nFichier cree  : {OUTPUT}")
print(f"Taille        : {sz / 1_000_000:.1f} Mo")
print("Termine !")

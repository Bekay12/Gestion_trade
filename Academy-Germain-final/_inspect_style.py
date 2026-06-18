import fitz, json

paths = [
    r"C:\Users\Berkam\Documents\Projets_perso\Gestion_trade\Academy-Germain-final\Partie0_Academy_Germain\PDF\P0_Ch01_Le_Vrai_Cout_Ignorance_Financiere.pdf",
    r"C:\Users\Berkam\Documents\Projets_perso\Gestion_trade\Academy-Germain-final\Partie1_Academy_Germain\PDF\P1_Ch01_Mecanismes_Fondamentaux_Economie.pdf",
]

for path in paths:
    doc = fitz.open(path)
    print(f"\n=== {doc.name.split(chr(92))[-1]} ===")
    page = doc[0]
    print(f"Page size: {page.rect}")
    
    # Dessins / rectangles de fond
    print("-- Drawings (first 6) --")
    for d in page.get_drawings()[:6]:
        print(f"  type={d['type']} fill={d.get('fill')} color={d.get('color')} rect={d.get('rect')}")

    # Texte
    print("-- Text spans --")
    for b in page.get_text("dict")["blocks"][:10]:
        if b["type"] == 0:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    col = s["color"]
                    r = ((col >> 16) & 0xFF) / 255
                    g = ((col >>  8) & 0xFF) / 255
                    b2 = ( col        & 0xFF) / 255
                    print(f"  sz={s['size']:.1f} rgb=({r:.2f},{g:.2f},{b2:.2f}) font={s['font']} text={repr(s['text'][:50])}")
    doc.close()

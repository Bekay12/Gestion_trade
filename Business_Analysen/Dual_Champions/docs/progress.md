# Fortschritt

Rahmen (2026-09-19): Vertiefter Filter (nicht Vollbericht) fuer sechs Dual Champions
aus `Magasines/Combined_scan_recos_2026-09-19.csv`, in drei Paaren:

| Paar | Titel | Gemeinsame Frage |
|---|---|---|
| 1 Oel | OXY, TTE.PA | Was ist der Cashflow ueber den Oelzyklus wert, nicht an seiner Spitze? |
| 2 Tanker | TNK, FRO | Sind Rekordfrachtraten Normalfall oder Spitze? |
| 3 Ausruester | G1A.DE, BESI.AS | Qualitaetswachstum - zu welchem Preis? |

Sprache Deutsch. Teile je Titel: 1.2 (mit Directors' Dealings), 1.6, 2 (Reihe 7-10 J.),
4, 7 (Schwelle, erforderliches Wachstum, erforderlicher Endwert, Multiple im Kontext);
Teil 6 nur als Anlegerrechnung, die 7.3 speist. Teile 1.4, 3 und 5 bewusst nicht
bearbeitet (Filtertiefe) - im Dokument als Umfangsgrenze ausweisen.

Lokale Modelle: qwen3.5:9b (Seitenfindung, Zahlenkandidaten), gemma4:12b (Verdichtung
von Fliesstext). Kein Wert gelangt ohne pruefe_seiten.py ins Dokument.

## Quellen
- SEC (hole_quellen.py): OXY 10-K 2024/2025; TTE, TNK, FRO 20-F 2024/2025;
  companyfacts aller vier (Reihen, Beleg per URL + Abrufdatum 19.09.2026).
- GEA Geschaeftsbericht 2025 (DE, cdn.gea.com); BESI Annual Report 2025 und 2020
  (besi.com). GEA 2020 unter der vermuteten URL nicht vorhanden - noch offen.

## Seitenzerlegung (scripts/seiten.py) - fertig, alle Dateien lueckenlos
- 20-F: Abschlussteil "F-n" als EIGENE Folge. Die uebernommene Zerlegung haette F-1
  die naechste gewoehnliche Seitenzahl gegeben (TTE: F-1 als "39") - 191 Abschluss-
  seiten mit falscher Seitenangabe, ohne Fehlermeldung.
- TTE setzt Null-Breite-Zeichen zwischen Leerzeichen um die Seitenzahl.
- TTE-20-F hat nur 38 gewoehnliche Seiten (Querverweise auf das URD); Unternehmens-
  beschreibung ggf. aus dem Universal Registration Document nachladen.

## Stand
- Quellenbeschaffung: SEC fertig, GEA/BESI 2025 fertig; GEA-Reihe vor 2021 offen
- Seitenzerlegung: fertig
- Naechster Schritt: Seitenfindung + Zahlenkandidaten mit qwen3.5:9b

## Stand 19.09.2026 (Abend)
- Extraktion (abschluss_extrakt.py, qwen3.5:9b fuer Etiketten, Code fuer Werte): 28 Berichte.
- Reihen (reihe.py): Kettenpruefung + XBRL-Abgleich. Korrigiert: Einheiten (BESI), Notenspalte
  (FRO, Werte um eine Spalte verrutscht), zweideutige Etiketten (TTE Dividenden), Zwischen-
  ueberschriften (TTE CFO), TNK "Vessel acquisitions", GEA Leasing, OXY fortgefuehrte Taetigkeit.
- XBRL bestaetigt: OXY CFO/Capex 12 J., TTE CFO/Capex/Dividenden 6 J. (fuellt 2015-2019),
  FRO alle Posten. Verbleibende Abweichungen = erklaerte Anpassungen (OxyChem, TNK 2022,
  GEA 2020, TTE-Umsatz 2015/16) oder Konzeptumfang (TNK XBRL nur Schiffskaeufe).
- Dossier (dossier.py): Wettbewerb OXY 6, TTE 31 (URD), GEA 5, BESI 11 belegte Aussagen;
  TNK/FRO keine Vorteilsbehauptung (FRO 20-F S. 29: "highly fragmented and competitive").
- Eigentuemer: OXY DEF 14A S. 65; BESI GB S. 159 (Applied Materials 9,00 %); TNK/FRO/GEA/TTE
  aus Dossier bzw. URD.
- GAP UEBERNOMMEN: TTE weist keine Leasingtilgung in der Kapitalflussrechnung aus - FCF ohne
  Leasingabzug (ueberzeichnet gegen GEA/BESI/FRO). Als Annahme ausweisen.
- GAP UEBERNOMMEN: OXY Capex nicht nach Taetigkeit getrennt; Naeherung Capex fortgefuehrt =
  Capex gesamt - Investitions-CF aufgegeben (Annahme).
- Naechster Schritt: kennzahlen.py (ROH mit Seiten), pruefe_seiten.py, flow.py (7.3), LaTeX.

## Stand 19.09.2026 (fertig)
- Dokument: out/analyse.pdf, 26 Seiten; 512 Werte seitengeprueft, keine Zahl im Fliesstext.
- Lesefallen gefunden und behoben: Aenderungsspalte GEA 2025 (EPS 2024 = "5,8" = Veraenderung
  in %), zweispaltiger Satz GEA 2020 (Wert der Nachbarspalte), Notenspalte FRO, F-Seiten 20-F,
  Null-Breite-Zeichen TTE, Zwischenueberschrift TTE, OXY-Dividendenzeile mit "preferred".
- Voten: alle sechs ueber der Schwelle; GEA beobachten, uebrige nicht kaufen.

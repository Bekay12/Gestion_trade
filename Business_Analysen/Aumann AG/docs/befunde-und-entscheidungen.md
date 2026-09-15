# Befunde und getroffene Entscheidungen

Inhaltliche Ergebnisse und Festlegungen, die eine neue Sitzung sonst neu herleiten müsste.
Stand: 29.07.2026. Ergänzt die Spezifikation und den Plan in `superpowers/`.

## Entscheidungen mit Begründung

### Operatives EBITDA = das bereinigte EBITDA

Der Geschäftsbericht 2024 (S. 2) weist **beide** Größen nebeneinander aus:

| | 2023 | 2024 | Marge 2024 |
|---|---|---|---|
| EBITDA (berichtet) | 20,647 | 35,804 | 11,5 % |
| EBITDA (bereinigt) | 21,294 | 36,417 | 11,7 % |

Der Ausdruck „operating EBITDA" kommt im Bericht nicht wörtlich vor. Als *operatives
EBITDA* im Sinne der Aufgabenstellung wird das bereinigte geführt; das berichtete steht im
Dokument daneben. Differenz 0,613 Mio. €, im Wesentlichen bereinigte Personalaufwendungen
(S. 17). Als Steuerungskennzahl definiert der Bericht allerdings das *berichtete* EBITDA
(S. 18) — deshalb werden beide gezeigt.

**Im Geschäftsjahr 2025 kehrt sich das Verhältnis um:** berichtet 28,215 > bereinigt
27,278. Das ist im Dokument vermerkt.

### Kursreihe — am 31.07.2026 doch monatlich beschafft

Am 29.07.2026 war keine Quelle erreichbar (Yahoo 429 auf `query1` und `query2`, stooq mit
JavaScript-Verifikation, boerse-frankfurt und onvista ohne Ergebnis). Das war ein
vorübergehender Zustand: Am **31.07.2026 antwortete `query1` normal**. Es liegen nun 48
Monatsbars 2021–2025, spaeter ergaenzt um Tagesbars fuer die exakten Handelstage der
Extrema. Die 13 Stuetzpunkte der Abbildung stehen in `data/kursverlauf.csv`.

**Validierung gegen die Primärquelle** — der entscheidende Punkt, der die externe Quelle
zitierfähig macht: Die vier Dezember-Schlusskurse der Yahoo-Reihe stimmen auf den Cent mit
den XETRA-Jahresschlusskursen der Geschäftsberichte überein, und die Schnittstelle weist im
`meta`-Block `"fullExchangeName": "XETRA"` aus. Die Reihe steht damit nicht *neben* den
Berichten, sondern auf ihnen.

| | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| Schluss | 13,68 €¹ | 11,48 € (−16,1 %) | 18,58 € (+61,8 %) | 10,62 € (−42,9 %) | 12,32 € (+16,0 %) |
| Hoch | 19,06 € | 17,68 € | 18,98 € | 18,96 € | 14,66 € |
| Tief | 11,14 € | 10,10 € | 11,28 € | 9,42 € | 9,87 € |

¹ nicht gegen eine Primärquelle geprüft, siehe unten.

**Die Abbildung zeigt Jahrespunkte, keine Monatsreihe — und das ist eine Korrektur.** Ein
erster Entwurf zeichnete die 48 Monatsschlusskurse und setzte Hoch und Tief als Dreiecke
darauf. Das Ergebnis war unbrauchbar: Die Extrema sind *Intraday*-Werte, die Linie besteht
aus Schlusskursen, also lagen die Marken frei neben der Kurve und sahen nach Zeichenfehler
aus. Ersetzt durch fünf Jahrespunkte (31.12.) mit einem senkrechten Balken von Tief bis
Hoch, technisch über pgfplots-Fehlerbalken. Die Spanne hängt damit sichtbar am Jahrespunkt.
Spezifikation (Stand vor der Korrektur):
[superpowers/specs/2026-07-31-kursextrema-abbildung-design.md](superpowers/specs/2026-07-31-kursextrema-abbildung-design.md).

**2021 ist ergänzt, aber nicht gegengeprüft.** Der Geschäftsbericht 2022 enthält keine
Angabe zum Jahresschlusskurs (geprüft am 31.07.2026, `refs/gb2022.pdf`), der Bericht 2023
führt als Vorjahreswert bereits 2022. Der Wert 13,68 € stützt sich allein auf Yahoo Finance
und ist als einziger der fünf nicht gegen eine Primärquelle gehalten. Im Dokument steht das
in Abschnitt 2.7 und im Quellenverzeichnis.

Hoch und Tief sind **Intraday**-Werte, nicht Extrema der Schlusskurse; im Dokument ist das
benannt. Inhaltlich trägt vor allem 2024: Das Jahreshoch von 18,96 € fiel in den Januar,
also **vor** die Veröffentlichung der Rekordzahlen, das Tief von 9,42 € in den November —
ein Rückgang um 50,3 % innerhalb des Jahres, in dem Umsatz, EBITDA und Jahresüberschuss
Rekordwerte erreichten. Das stützt die These aus Teil 4 (der Markt liest den
Auftragseingang, nicht die GuV) deutlich schärfer als der reine Jahresvergleich.

### Annahmen der Flow-Analyse

Die Aufgabenstellung gibt Beträge vor, nicht deren zeitliche Verteilung. Festgelegt und im
Dokument offenzulegen:

1. Anfangsbestand 138,2 Mio. € (31.12.2024) laut Vorgabe; der tatsächliche Wert zum
   31.12.2025 beträgt 148,1 Mio. € und gehört in eine Fußnote.
2. Investitionsprofil A und B gleichmäßig, **C: 45 / 7,5 / 7,5** — ein Kaufpreis wird im
   Vollzugsjahr fällig, nicht in Dritteln.
3. Kapitalbindung im Jahr vor dem Ergebnisbeitrag; bei C mit dem Vollzug 2026.
4. **Inkrementelle Sicht** des operativen Cash-Flows, damit die Wirkung der Entscheidung
   nicht im Konzern-Cash-Flow untergeht.
5. IRR über den erklärten Horizont 2026–2035, mit Sensitivität 5/10/15 Jahre.

## Ergebnisse der Flow-Analyse

| Ende 2028 | A – Next Automation | B – Batterie/BZ | C – Akquisition |
|---|---|---|---|
| Investition kumuliert | 35,0 | 30,0 | 60,0 Mio. € |
| EBITDA-Zuwachs | +16,0 ab 2028 | +14,0 ab 2027 | +10,0 ab 2027 |
| IRR (2026–2035) | 29,6 % | 38,3 % | 6,3 % |
| Liquiditätsreserve | 104,2 | 121,2 | 78,2 Mio. € |
| Net Debt/EBITDA | nicht relevant | nicht relevant | −2,10 |

IRR nach Horizont: 5 Jahre A 11,7 / B 27,4 / **C −9,9 %** · 10 Jahre 29,6 / 38,3 / 6,3 ·
15 Jahre 32,0 / 39,6 / 10,2. Tiefpunkt der Liquidität bei C: 73,2 Mio. € in 2026.

### Die Marge Ende 2028 ist ein Effekt des Nenners — erledigt

Die EBITDA-Marge Ende 2028 (A 21,2 % · B 20,2 % · C 18,3 %) beruht auf dem konstant
fortgeschriebenen Umsatz 2025. Der war um 34,7 % eingebrochen, der kleine Nenner
schmeichelt der Marge. Abschnitt 6 legt das im Absatz „Vorbehalt zur EBITDA-Marge Ende
2028" offen; `scripts/cashflow.py` erzeugt dazu `data/marge_sensitivitaet.tex`:

| Umsatzbasis 2028 | A | B | C |
|---|---|---|---|
| Umsatz 2025 konstant (204,0 Mio. €, Basisfall) | 21,2 % | 20,2 % | 18,3 % |
| Umsatz +3 % p. a. bis 2028 (222,9 Mio. €) | 19,4 % | 18,5 % | 16,7 % |
| Umsatzniveau 2024 (312,3 Mio. €) | 13,9 % | 13,2 % | 11,9 % |

Die Rangfolge der Optionen bleibt in allen drei Szenarien erhalten, das Niveau nicht. Die
Marge taugt daher zum Vergleich der Optionen untereinander, nicht als absolute Prognose.

## Inhaltliche Befunde

### Prognose 2025 gegen Ist (Aufgabe 3.6)

Prognose im GB 2024 (S. 5): Umsatz 210–230 Mio. €, EBITDA-Marge 8–10 %. **Unverändert
bestätigt** in Q1 (S. 2), H1 (S. 3) und Q3 2025 (S. 3). Ist: Umsatz 203,985 Mio. €, also
2,9 % unter der Untergrenze; berichtete EBITDA-Marge 13,8 %, deutlich über der Obergrenze.
Umsatz verfehlt, Marge übertroffen, Prognose nie angepasst.

### Die Lücken in Teil 1 — Stand 31.07.2026

**Die IR-Seite „Shares" hatte zwei Fassungen, und der Entwurf hat sie vermischt.** Das ist
der wichtigste Befund dieser Runde. Belegt durch rohes HTML beider Fassungen:

| Feld | Archiv 14.09.2025 | Live 31.07.2026 |
|---|---|---|
| Aktienzahl | 14.345.231 | 12.917.048 |
| MBB SE | 44,49 % | 47,81 % (`* as of 31 December 2025`) |
| Streubesitz | 45,5 % | 42,9 % (ohne Stichtag) |
| Analysten | Berenberg, EQUI.TS, **Hauck & Aufhäuser** | Berenberg, EQUI.TS |

Der Entwurf führte die **alten Prozentsätze** zusammen mit der **neuen Aktienzahl**. Die
daraus abgeleitete Differenz von 10,01 % war folglich zum Teil ein Rechenartefakt der
zwischenzeitlichen Kapitalherabsetzung. Korrigiert auf den Live-Stand; die alte Fassung ist
über das Internet Archive zitiert
(`web.archive.org/web/20250914061406/...`), weil die Gesellschaft die Änderung nicht
kennzeichnet.

**Die Restdifferenz bleibt, ist aber jetzt zuzuordnen.** 47,81 % + 42,9 % = 90,71 %, es
fehlen 9,29 %. Eine Position dieser Größenordnung ist belegt: Aumann hielt am 16.07.2026
**1.291.200 eigene Aktien = 9,996 % der Stimmrechte** (eigene Veröffentlichung nach § 40
Abs. 1 Satz 2 WpHG), aus dem Rückkaufangebot, das am 14.07.2026 zum auf 17,80 €
angehobenen Preis abgeschlossen wurde (9.358.558 Stück angedient, Zuteilung ≈ 6,07 %).

Sie schließt die Lücke aber **nicht stichtagsgenau**: Der MBB-Anteil ist auf den 31.12.2025
datiert, und zu diesem Stichtag hielt die Gesellschaft keine eigenen Aktien (GB 2025 S. 29).
Der Streubesitz trägt gar kein Datum. Im Dokument steht deshalb: Größenordnung zugeordnet,
Differenz weiter ausgewiesen, nicht geglättet. Das ist Regel 3, nicht Bequemlichkeit.

**Analystenaussagen bleiben unbelegbar.** Die Abdeckung ist von drei auf zwei Häuser
gesunken (Hauck & Aufhäuser entfernt), was im Dokument mit Archivbeleg vermerkt ist. Weder
Aumann noch die Häuser veröffentlichen Rating oder Kursziel.

**Korrektur 04.08.2026:** Diese Zeile enthielt bis eben eine konkrete Konsenszahl
(„boerse-express: 2 × Halten, Ziel 13,95 €, Spanne 10,10–16,00 €, Stand 27.07.2026") als
bewusst nicht verwendeten Befund. Der Wert ließ sich bei einer erneuten Prüfung am
04.08.2026 nicht bestätigen: Ein direkter Abruf derselben boerse-express-Seite zeigt
wörtlich „Eine belastbare Konsens-Einschätzung mit Kursziel liegt nicht vor", und
boerse.de zeigt unter „Empfehlungen zur Aumann Aktie" ebenfalls keine Einträge. Die
frühere Zahl war entweder ein Fehlgriff der damaligen Recherche oder durch eine
inzwischen geänderte Seite überholt — in beiden Fällen war sie nicht mehr verlässlich und
ist deshalb hier entfernt. Der Befund im Dokument (Abschnitt 1.5) ist jetzt mit zwei
unabhängig geprüften Aggregatoren (boerse-express, boerse.de, beide 04.08.2026) belegt,
zusätzlich zur Unternehmensquelle: **keine belastbare Konsensangabe zugänglich, an
mehreren unabhängigen Quellen bestätigt.**

**Interactive Brokers scheidet als Quelle aus.** Der Kontrakt existiert (`AAG`, IBIS,
`contract_id` 270900387), aber `get_price_history` antwortet mit `No market data
permissions` — kein Marktdaten-Abo für deutsche Aktien.

### Material für die blauen Zonen

- Der Kurs folgt dem **Auftragseingang**, nicht der Gewinn- und Verlustrechnung: Hoch Ende
  2023 bei Rekord-Auftragseingang (339,4 Mio. €), Einbruch 2024 (−42,9 %) trotz Rekorden
  bei Umsatz und EBITDA — weil der Auftragseingang um 41,1 % fiel.
- Der Emissionspreis von 2017 (42,00 €) liegt über allen Schlusskursen 2022–2025.
- **Next Automation läuft gegenläufig:** Umsatz 60,5 → 53,8 → 40,2 Mio. €, aber
  Auftragseingang 2025 wieder 56,5 nach 36,6 Mio. €, Marge 10,2 → 10,8 → 12,8 %. Das
  Segment schrumpft in der Fakturierung und füllt sich zugleich mit Aufträgen.
- **Klumpenrisiko beziffert:** E-Mobility trägt rund 80 % des Geschäfts und praktisch die
  gesamte Ergebnisverbesserung 2024 (Segment-EBITDA 17,1 → 33,8 Mio. €).
- Kapitalallokation: Dividende 0,22 €, Rückkaufangebot über bis zu 1.434.523 Aktien zu
  12,37 €, anschließend eingezogen; Aktienzahl 15.250.000 → 14.345.231 → 12.917.048.
- Personalabbau: 891 → 773 Mitarbeitende.
- Akquisition: Aumann Lauchheim GmbH, Post-Merger-Integration im Berichtsjahr.

## Schlussprüfung der Aufgaben 4 bis 8 (29.07.2026)

Die Aufgaben 4 bis 8 waren ohne unabhängige Zweitprüfung ausgeführt worden, weil das
Ausgabelimit des Kontos die Prüf-Subagenten beendet hatte. Sie sind jetzt nachgeprüft:
jedes der 64 mit einer Seitenangabe versehenen Makros aus `../data/kennzahlen.tex` wurde
seitenweise gegen den Bericht gehalten, und jede abgeleitete Rate wurde nachgerechnet.

**Drei echte Befunde, alle behoben:**

1. **Die ISIN war falsch zugeschrieben.** `DE000A2DAM03` war als „GB 2024, S. 13" geführt,
   kommt aber in keinem der acht Berichte vor. Quelle ist die IR-Seite „Shares"; im
   Dokument steht dafür jetzt eine `\quelleWeb`-Fußnote. Seite 13 belegt nur die
   Notierung seit März 2017 — und zwar im **Prime Standard**, was der Text vorher
   ungenau als „regulierter Markt" wiedergab.
2. **Der Segmentumsatz E-Mobility 2024** (`258.530`) steht auf S. 2 unter „thereof
   E-mobility", nicht auf S. 13. Seite 13 nennt nur den gerundeten Wert 258,5 Mio. €.
3. **Die Nummerierung in Teil 2 war um eins verschoben.** Der eingefügte
   Kennzahlenüberblick war eine nummerierte Unterabschnitt, wodurch die Kursentwicklung —
   Frage 2.7 der Aufgabenstellung — im Dokument als 2.8 erschien. Der Überblick ist jetzt
   unnummeriert.

**Zwei Scheinbefunde, bewusst so belassen:** `\KursDeltaZFIV` (−42,9 %) und
`\EbitdaMargeNextAutoZFV` (12,8 %) lassen sich aus den ebenfalls veröffentlichten,
gerundeten Eingangsgrößen nicht exakt nachrechnen (−42,8 % bzw. 12,7 %). Hinterlegt ist
jeweils der **im Bericht gedruckte** Wert.

**Seitenzuordnung nachgeprüft:** Bei den Geschäftsberichten 2023–2025 ist die gedruckte
Seite gleich der PDF-Seite; bei allen drei Zwischenberichten gilt PDF = gedruckt + 1. Die
drei `\quelleQM`-Zitate (Q1 S. 2, H1 S. 3, Q3 S. 3) sind damit korrekt.

Die Aufgaben 1 bis 3 waren bereits geprüft; dort fand die Prüfung einen echten Fehler im
Wachhund (deutsch gruppierte Zahlen), der behoben ist.

Ein zweiter Fehler wurde in Aufgabe 7 selbst gefunden: fünf Seitenangaben waren aus
Zeilennummern abgeleitet statt gegen die gedruckte Seite geprüft. Alle Zitate sind
inzwischen nachgeprüft; die Methode steht in `../refs/CLAUDE.md`.

## Layout: Kopfzeile und Fußnotenapparat (30.07.2026)

**Kopfzeile.** Links steht jetzt Nummer *und* Titel des laufenden Abschnitts, rechts der
feste Dokumenttitel „Analyse Aumann AG“, darunter eine Trennlinie (`headsepline`). Der
Kopf ist auf `\small` gesetzt: der längste Abschnittstitel („Strategische Maßnahmen im
Geschäftsbericht 2024“) und der Dokumenttitel müssen zusammen in eine Zeile passen. Auf
den unnummerierten Anhangseiten bleibt nach dem `\markright{}` in `hausarbeit.tex` nur der
Dokumenttitel stehen; das Deckblatt bleibt kopfzeilenfrei.

**Fußnotenapparat: `dblfnote` ist hier unbenutzbar.** Gewünscht war ein zweispaltiger
Apparat, um seine Höhe zu halbieren. Das einzige Paket, das das in einem einspaltigen
Dokument leistet, ist `dblfnote` (aus `yafoot`) — es ist installiert, aber mit diesem
Dokument nicht kombinierbar:

- `pgfplots` + `hyperref` + `dblfnote` treibt die Ausgaberoutine in eine Endlosschleife:
  LaTeX gibt bis zum Abbruch leere Seiten aus (im Log Seitenzähler bis ~4900) und endet mit
  `TeX capacity exceeded, sorry [input stack size=5000]`.
- Reproduziert mit `scrartcl` **und** `article`, unabhängig von der Ladereihenfolge und
  auch mit `hyperfootnotes=false`. Ohne `hyperref` oder ohne `pgfplots` läuft es; beide
  werden hier gebraucht (Kursdiagramm, `\url` in den Web-Belegen).
- Ausweg wäre nur der Verzicht auf `hyperref` — dafür sind zwei Spalten im Fußnotenapparat
  kein hinreichender Grund.

Gewählt ist deshalb `\usepackage[para]{footmisc}`: alle Fußnoten einer Seite stehen in
**einem durchlaufenden Absatz** über die volle Breite. Das spart dieselbe Höhe wie zwei
Spalten (auf belegdichten Seiten ein bis zwei Zeilen statt bis zu einem Drittel des
Satzspiegels) und ist der übliche Satz in deutschen wirtschaftswissenschaftlichen Arbeiten.
Nebenwirkung: `microtype` kann seinen `footnote`-Patch nicht anlegen; er ist über
`patch={item,toc,eqnum}` abgeschaltet, damit der Lauf warnungsfrei bleibt.

**Folge für die Fußnotengruppen.** Der flachere Apparat verkürzt das Dokument von 28 auf
26 Seiten und verschiebt damit die Seitenumbrüche. Vier der zwölf `\quelleMerken`/
`\quelleErneut`-Gruppen lagen danach über einer Seitengrenze — ihr `\footnotemark` verwies
auf eine Fußnote, die auf der Seite nicht mehr abgedruckt war. `check_footnote_pages.py`
findet diesen Fall **nicht** (es sieht nur die `\quelle*`-Aufrufe). Dafür gibt es jetzt
`scripts/check_footnote_groups.py`; alle zwölf Gruppen sind damit nachgeprüft, ebenso die
Schlüsselnamen (`p<Seite>-...`), die nach der Umstellung auf die neuen Seiten zeigen.

## Seitenumbrüche (31.07.2026)

Beim Einbau der Kursmarken sind zwei Fehlumbrüche aufgefallen, die schon vorher bestanden:

- **Seite 7:** Die Überschrift „Überblick der Kennzahlen" stand mit einer einzigen Zeile am
  Seitenfuß, die angekündigte Tabelle folgte erst überblättert.
- **Seite 22:** Die Überschriften 6.2 und 6.2.1 standen ohne jeden Rumpf am Seitenfuß.

Behoben mit `needspace` und angehobenen Umbruchstrafen (`\clubpenalty`, `\widowpenalty`,
`\displaywidowpenalty` je 10000 statt der LaTeX-Vorgabe 150). Details in
[../sections/CLAUDE.md](../sections/CLAUDE.md), Abschnitt „Seitenumbrueche".

**Eine Falle dabei:** `\needspace{6\baselineskip}` vor dem Kennzahlenüberblick hat nichts
bewirkt — sechs Zeilen waren frei, die Überschrift passte hinein, die Tabelle blieb
trotzdem zurück. Der Wert muss für **alles** bemessen sein, was zusammenbleiben soll; hier
`16\baselineskip` für Überschrift, Ankündigungssatz und Tabellenkörper.

**Prüfmethode.** Aus dem Quelltext ist das nicht zu sehen. Die Überschriften aus
`out/hausarbeit.toc` gegen ihre y-Koordinaten aus `pdftotext -bbox` halten; das Kriterium
ist nicht die Position allein, sondern wie viele Textzeilen der Überschrift auf derselben
Seite noch folgen. Weniger als zwei ist ein Fehlumbruch. Eine reine Positionsschwelle von
75 % hätte den Fall auf Seite 7 knapp durchgehen lassen — er lag exakt bei 75 %.

**Folgekosten, die einzuplanen sind.** Das Dokument wuchs von 27 auf 28 Seiten, und die
verschobene Paginierung hat drei Fußnotengruppen ungültig gemacht. Beim Reparieren habe ich
zunächst geraten und dabei eine Gruppe selbst zerschossen. Der verlässliche Weg ist, sich
für jede Zitation die tatsächliche Seite über `check_footnote_pages.resolve_page()` ausgeben
zu lassen und die Schlüssel danach in einem Zug zu vergeben, statt Meldung für Meldung
nachzubessern.

## Abbildung und Zitierweise (01.08.2026)

**Die Tiefstkurse waren mit einem nach oben weisenden Dreieck gezeichnet.** Der naheliegende
Weg, `mark=triangle*` über `mark options={rotate=180}` zu drehen, funktioniert innerhalb von
`scatter/classes` **nicht**: pgfplots setzt `mark options` je Klasse neu und verwirft die
Drehung stillschweigend — kein Fehler, keine Warnung, nur ein falsches Bild. Behoben mit
einer eigenen Markenform (`\pgfdeclareplotmark{dreieckab}` in `preamble.tex`), die nicht
überschrieben werden kann.

**Lehre:** Eine Markenform ist am gesetzten PDF zu prüfen, nicht am Quelltext. Der Fehler
stand drei Fassungen lang unbemerkt im Dokument, weil er bei Bildgröße nicht auffällt; erst
eine 600-dpi-Vergrößerung hat ihn gezeigt.

Farben: Jahreshoch grün, Jahrestief rot, Schlusskurs blau (`kurshoch` / `kurstief` /
`kursschluss` in `preamble.tex`). Die Farbe kommt **zur** Form hinzu und ersetzt sie nicht —
im Schwarzweißdruck bleiben die drei Arten über Kreis, Dreieck-auf und Dreieck-ab
unterscheidbar.

**Webquellen werden als Kurzbeleg zitiert.** Die Fußnote nennt Urheber, Titel und Abrufdatum
und verweist mit der Nummer auf das Quellenverzeichnis; die Adresse steht dort. Zuvor druckte
jede Fußnote die volle URL — auf belegdichten Seiten füllte das mehrere Zeilen, und der
Kurzbeleg der Geschäftsberichte stand unvermittelt neben einem Volltextlink.

Das zweite Argument von `\quelleWeb` ist seitdem der `\label`-Schlüssel des
Verzeichniseintrags, nicht die URL. Damit steht jede Adresse genau einmal im Dokument —
dieselbe Regel, die für Zahlenwerte gilt. Eine neue Webquelle braucht jetzt zwei Schritte:
erst `\item\label{quelle:...}` im Quellenverzeichnis, dann den Aufruf.

### Nachtrag: die Belege sind Sprungziele, keine ausgeschriebenen Nummern

Die erste Umsetzung druckte „Quellenverzeichnis Nr. 7" als Text in die Fußnote. Gemeint war
aber eine **Verknüpfung**: Der Kurztitel selbst ist jetzt per `\hyperref` mit seinem Eintrag
verbunden, ein Klick führt dorthin. Das gilt für alle drei Zitiermakros, nicht nur für die
Webquellen — bei `\quelleGB` und `\quelleQM` leitet sich der Schlüssel aus dem ersten
Argument ab, sodass die rund vierzig bestehenden Aufrufe unverändert bleiben konnten.

Zwei Punkte, die dabei zu wissen sind:

- **`hidelinks` macht Verknüpfungen unsichtbar.** Ohne eigene Einfärbung wäre der Beleg zwar
  anklickbar, aber nicht als Verweis erkennbar. Die Farbe `quellenlink` in `preamble.tex`
  behebt das und lässt sich für einen einfarbigen Ausdruck auf `black` setzen, ohne die
  Verknüpfung zu verlieren.
- **Ein Label darf Leerzeichen enthalten.** `quelle:Quartalsmitteilung Q1 2025` funktioniert
  und erspart es, die Signatur von `\quelleQM` zu ändern.

Geprüft wurde nicht nur, ob LaTeX zufrieden ist, sondern ob das PDF tatsächlich
Verknüpfungen trägt: 57 GoTo-Ziele zeigen auf Einträge des Quellenverzeichnisses, ihre
Häufigkeit entspricht der Zitierhäufigkeit (Geschäftsbericht 2024 mit 28 Belegen an der
Spitze).

## Offener Befund (31.07.2026): Teil 5 braucht eigene Schätzungen

Teil 5 liest für Investitionsbedarf, Kapitalbindung und Liquiditätsbedarf die Makros aus
`data/cf_makros.tex` — also die von der Aufgabenstellung **für Teil 6** vorgegebenen
Beträge. Gefragt ist dort aber die eigene Einschätzung: 5.x.2 verlangt „eine Tabelle, in
der Sie **Ihre** (qualitativen und quantitativen) **Einschätzungen** darstellen", und
Teil 6 sagt ausdrücklich „**(ggf. abweichend von Ihren vorausgegangenen Analysen)**". Die
Klammer ergibt nur Sinn, wenn Teil 5 eigene Zahlen enthält.

Betroffen sind neun Tabellenzellen (3 Optionen × 3 Zeilen) und die Kapitalbedarfs-Aussagen
in 5.x.1 und 5.x.3. Die Verweise auf IRR, Liquiditätsreserve und Tiefpunkt bleiben, da sie
Rechenergebnisse des Teils 6 sind — sofern als solche gekennzeichnet.

Auftrag mit Belegstellen, Umfang, Ankern und fertigem Prompt:
[prompt-teil5-eigene-schaetzung.md](prompt-teil5-eigene-schaetzung.md).

**Geschlossen am 04.08.2026.** Neun eigene Makros in `data/schaetzung_teil5.tex`,
hergeleitet aus zwei bislang ungenutzten, seitengenau geprüften Ankern: aktivierte
Entwicklungskosten 2023 (2,7 Mio. € = 1,5 % vom Umsatz, GB 2023 S. 14) als F&E-Kernrate,
und die Nettofinanzliquidität 2024 (138,2 Mio. €) als Kapazitätsgrenze für Option C, deren
Zielobjekt die Aufgabenstellung offenlässt. Methode: Kernrate (1,5 % × Konzernumsatz 2024
× 3 Jahre = 14,1 Mio.) × Markterschließungsfaktor (A: verdreifacht, branchenfremder
Markteintritt; B: Eineinhalbfaches, bekannter Markt) für Investitionsbedarf A/B;
Kapitalbindung einheitlich ein Viertel des Investitionsbedarfs; Liquiditätsbedarf jährlich
= Investitionsbedarf/3 (A/B, gleichmäßig verteilt) bzw. Investitionsbedarf + Kapitalbindung
im Vollzugsjahr (C, Kaufpreis nicht in Raten). Ergebnis: Investitionsbedarf A 42,2 / B 21,1
/ C 55,3 Mio. €, Kapitalbindung A 10,6 / B 5,3 / C 13,8 Mio. €, Liquiditätsbedarf jährlich
A 14,1 / B 7,0 / C 69,1 Mio. € — durchgängig niedriger als die für Teil 6 vorgegebenen
Beträge (35/30/60 bzw. 15/15/20), bei Option C am nächsten an der Vorgabe. Der
Abweichungssatz steht als neue `\annahme{}` in Teil 6. Umfang bewusst auf die neun Zellen
plus 5.x.1 (volle Ersetzung) und 5.x.3 (Kennzeichnungssatz statt Ersetzung, da die dortige
Statement-Erzählung strukturell an der Teil-6-Zeitachse hängt) begrenzt; Übersichtstabelle
5.4 und Teil 6 selbst bleiben unverändert bei den vorgegebenen Beträgen.

**Nebenbefund, nicht behoben:** `sections/01-unternehmen.tex` enthält unabhängig von
diesem Auftrag eine bereits vor dieser Sitzung uncommittete Änderung, die `100\,\%` als
Ziffer statt als Wort schreibt und `check_literals.py` einen Treffer liefert (Zeile 81).
Nicht Teil dieses Befundes, siehe `git diff sections/01-unternehmen.tex`.

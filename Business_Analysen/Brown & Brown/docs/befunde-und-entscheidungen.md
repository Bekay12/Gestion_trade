# Befunde und getroffene Entscheidungen

Stand 28.08.2026. Was hier steht, ist das Ergebnis der Quellenarbeit, nicht der Weg dorthin.

## Der Fall in einem Satz

Brown & Brown hat zum 1.8.2025 die Accession Risk Management Group für 9.608 Mio. USD
übernommen, dafür 4.200 Mio. USD Anleihen begeben und 43.137.254 Aktien zu 102,00 USD
platziert — und im selben Jahr brach das organische Umsatzwachstum von 10,4 % auf 2,8 %
ein, im ersten Halbjahr 2026 auf −0,3 %.

## Der Frühindikator

Die Analyse folgt dem Aumann-Muster: Nicht die GuV erklärt den Kurs, sondern eine
vorlaufende Größe. Bei Aumann war es der Auftragseingang, hier ist es die vom Unternehmen
selbst quartalsweise veröffentlichte **Organic Revenue growth rate**.

| Jahr | Organisch | Umsatz | Kurs (Jahresschluss) |
|---|---|---|---|
| 2024 | +10,4 % | +12,9 % | +43,5 % |
| 2025 | +2,8 % | +22,8 % | −21,9 % |
| H1 2026 | −0,3 % | +33,0 % | −10,4 % (Stand 27.08.) |

2025 fielen Rekordumsatz und Rekord-EBITDAC mit einem Kursverlust von 21,9 % zusammen.
Nur der Frühindikator erklärt das. Der Einbruch liegt fast vollständig im Segment Specialty
Distribution: dort 17,8 % → 2,8 %, im Retail nur 5,8 % → 2,8 %.

## Die Umkehr der Bereinigung

2024 lag das **bereinigte** EBITDAC (1.689) **unter** dem berichteten (1.720), weil ein
Veräußerungsgewinn von 31 herausgerechnet wurde. 2025 liegt es **darüber** (2.121 gegen
2.060), weil 113 Akquisitions- und Integrationskosten hinzugerechnet werden. Wer nur die
bereinigte Reihe liest, sieht eine steigende Marge; wer nur die berichtete liest, eine
fallende. Beides ist richtig.

## Warum die Flow-Analyse zwei Rechnungen führt

Ohne Handlungsoptionen gibt es nichts zu vergleichen. Statt dessen zwei Fragen:

1. **Entschuldungspfad** (Kapazitätsrechnung): Alle freien Mittel gehen in die Tilgung.
   Ergebnis: Nettoverschuldung/EBITDAC fällt von 2,30 auf 0,00 bis 2030; selbst im
   schlechtesten Szenario auf 0,15. Die Verschuldung ist tragbar und **nicht** das Risiko
   des Falls.
2. **Anlegerrechnung**: Nur die planmäßige Tilgung wird bedient, der Rest fließt an die
   Eigentümer. IRR bei Einstieg zu 71,39 USD: 11,5 % / 10,8 % / 10,6 % über 5/10/15 Jahre.

**Beide Rechnungen verwenden dasselbe Geld für verschiedene Zwecke und sind Alternativen,
keine Ergänzungen.** Das steht als Annahme im Dokument. Der erste Modellentwurf ließ die
Liquidität bis 5.492 Mio. USD auflaufen, weil kein Mittelverwendungspfad unterstellt war —
arithmetisch richtig, inhaltlich unsinnig.

## Der Maßstab

Der IRR wird gegen die **belegte Eigenkapitalrendite auf das durchschnittliche
Eigenkapital 2025 von 11,1 %** gemessen, nicht gegen einen konstruierten WACC: Beta,
risikofreier Zins und Marktrisikoprämie stehen in keiner Primärquelle.

Das Ergebnis ist unbequem knapp: 11,5 % über fünf Jahre liegt knapp über dem Maßstab,
10,8 % und 10,6 % über zehn und fünfzehn Jahre knapp darunter. Der heutige Kurs preist
das Unternehmen also ungefähr auf seine eigene Eigenkapitalrendite, ohne Aufschlag.

Die Rangfolge ist **nicht** stabil: Bei −0,3 % Wachstum sinkt der IRR auf 6,3–6,6 %, bei
10,4 % steigt er auf 19,9–22,6 %. Rund sechzehn Prozentpunkte Spannweite über fünf Jahre.

## Die weitreichendste Annahme

**Keine weiteren Zukäufe.** Das Unternehmen hat 2025 allein 43 Gesellschaften erworben;
Zukäufe sind sein Geschäftsmodell. Die Rechnung ist deshalb ausdrücklich als
Kapazitätsrechnung ausgewiesen und nicht als Prognose.

## Quellentechnik

- SEC-Einreichungen kommen als eine HTML-Datei. `scripts/edgar_pages.py` splittet an den
  `<hr>`-Elementen (= gedruckte Seitenumbrüche) und liest die Seitenzahl aus der letzten
  Zeile. Das 10-K setzt sie allein („49“), das **Proxy Statement** dagegen als
  „76 | BROWN & BROWN, INC.“ bzw. „BROWN & BROWN, INC. | 77“ — ohne diesen zweiten Fall
  blieben 106 von 109 Blöcken ohne Seitenzahl.
- **Yahoo Finance antwortete am 28.08.2026 durchgehend mit HTTP 429**, Stooq mit einem
  JS-Challenge, IBKR lieferte keine Treffer. Die Kursreihe kommt von der
  Nasdaq-API (`api.nasdaq.com/api/quote/BRO/historical`, UA- und Referer-Header nötig).
- Ratings und Kursziele sind auf der IR-Seite **nicht** veröffentlicht, nur 18 Häuser mit
  Namen des Analysten. Wie bei Aumann wird deshalb kein Konsens gebildet.

## Fallen, die Zeit gekostet haben

- **23 von 176 Seitenangaben waren beim ersten Durchgang falsch** — alle die, die ich nicht
  unmittelbar auf der Seite gelesen, sondern aus dem Zusammenhang geschlossen hatte.
  `scripts/pruefe_seiten.py` existiert deswegen.
- LaTeX-Makronamen dürfen **keine Ziffern** enthalten: `\KursSchluss2025` ist ungültig.
  Daher `ZFI`…`ZFVI` für 2021–2026 und `Fuenf`/`Zehn`/`Fuenfzehn` für die IRR-Horizonte.
- siunitx gruppiert per Voreinstellung auch die **Nachkommastellen**: Aus dem exakten
  Rohwert 43.4679 wurde „43,467.9 %“. Fix: `group-digits = integer`.
- TeX überliest das Leerzeichen nach einem Kontrollwort: „`\KurstagHochZFV` und“ wurde als
  „1.4.2025und“ gesetzt. Daher der Wrapper `\Datum{}`.
- `\textcolor` ist nicht `long`; die mehrabsätzigen blauen Zonen brechen damit ab. `\bk`
  benutzt `\color` in einer Gruppe.
- `pgfplots` braucht `\usepgfplotslibrary{dateplot}` für `date coordinates in=x`, und
  `xtick distance={365 days}` funktioniert dort nicht — explizite Datums-Ticks stattdessen.

## Das Votum (Teil 7)

**Nicht kaufen zu 71,39 USD; beobachten.** Kein Verkaufsvotum.

Der Schwellenkurs, bei dem der IRR die Hürde von 11,1 % trifft:

| Szenario | 5 J | 10 J | 15 J |
|---|---|---|---|
| −0,3 % (H1 2026, **gemessen**) | 58,81 | 50,00 | 44,94 |
| +2,8 % (GJ 2025) | 72,48 | 70,12 | 67,80 |
| +10,4 % (GJ 2024) | 113,34 | 146,02 | 174,56 |

Der heutige Kurs liegt 5,3 % über der Schwelle des mittleren Szenarios (15 Jahre) und
58,9 % über der des zuletzt gemessenen. **Das Szenario, das die aktuellen Zahlen stützen,
ist dasjenige, unter dem die Aktie deutlich zu teuer ist.**

Zwei Auslöser, beide kostenlos beobachtbar: organische Rate zwei Quartale positiv (→ Kauf
bis 72,48), oder Kurs unter 44,94 (→ selbst das untere Szenario fair). Zwei Quartale
weiter negativ → meiden.

Gegen das eigene Votum spricht: die Rechnung schließt weitere Zukäufe aus (das
Geschäftsmodell), und der Kurs ist bereits 32,7 % über dem Tief vom 13.5.2026.

## Offen

- Der getrennte Ergebnisbeitrag von Accession allein ist nicht veröffentlicht; die Angabe
  von 1.792 Mio. USD Jahresumsatz gilt für alle 43 Zukäufe zusammen und ist aus zwei
  veröffentlichten Werten hergeleitet (Formel steht im Dokument).
- Die blauen Zonen sind vom Verfasser noch nicht geprüft.

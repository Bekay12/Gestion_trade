# Insider-Käufe und -Verkäufe: Prüfung des Börse-Online-Beitrags vom 28.09.2026

**Frage:** Was steckt hinter „Kaufen: GameStop, Salesforce“ und „Verkaufen: Nvidia, Oracle,
CrowdStrike, Palo Alto, CRISPR“, und lohnen die beiden Käufe einen eigenen Einstieg?

**Stand:** 28.09.2026, Kurse vom 28.09.2026 (yfinance, Tagesdaten).
**Primärquelle:** SEC Form 4 aller sieben Emittenten, eingereicht 14.–28.09.2026
(`data/form4.json`, je Meldung mit URL), für GME und CRM zusätzlich zwölf Monate
(`data/form4_2025-09-28.json`). Finanzdaten aus den XBRL-Daten der 10-K und 10-Q
(`data/xbrl_*.json`, je Wert mit Aktenzeichen), GameStop-Quartalsbericht Q2 FY2026 im Wortlaut
(`refs/gme-10q-q2-2026.txt`). Sekundärquellen sind als solche gekennzeichnet.

Legende: Fakt aus der Quelle · **Einschätzung des Verfassers** · *Votum*.

---

## 1. Was die Formulare zeigen, das Bild aber nicht

Form 4 unterscheidet Kauf am Markt (Code P), Verkauf am Markt (S), Optionsausübung (M),
Steuereinbehalt (F) und Zuteilung (A). Nur P und S sind eine Entscheidung mit eigenem Geld;
10b5-1 kennzeichnet einen **vorab festgelegten** Verkaufsplan, der kein aktuelles Urteil ausdrückt.

| Titel | Markt-Käufe 14.–28.09. | Markt-Verkäufe 14.–28.09. | Davon im 10b5-1-Plan | Lesart |
|---|---|---|---|---|
| **GameStop** | 2 Personen, 26,8 Mio. $ | keine | – | Echter Kaufcluster (Abschnitt 3) |
| **Salesforce** | 1 Person, 1,0 Mio. $ | keine in diesem Fenster; sonst nur M/F | – | Einzelkauf, klein |
| Nvidia | – | 3 Personen, 314,6 Mio. $ | CFO, General Counsel | 300,2 Mio. $ entfallen auf **einen** Aufsichtsrat (Mark Stevens), ohne Plan |
| CrowdStrike | – | 7 Personen, 232,0 Mio. $ | CEO, CFO (24.09.), ein Aufsichtsrat | Breit gestreut, Aktie beim 3-Fachen ihres 52-Wochen-Tiefs |
| Palo Alto | – | 3 Aufsichtsräte, 9,0 Mio. $ | keiner | James Goetz verkauft den gemeldeten indirekten Bestand vollständig (20.000 → 0) |
| Oracle | – | 2 Personen, 5,3 Mio. $ | CEO | Klein, CEO im Plan; Aktie 58 % unter dem 52-Wochen-Hoch |
| CRISPR | – | 2 Personen, 1,4 Mio. $ | General Counsel | Klein, nach Optionsausübung |

Zahlenbasis: Summe von Stückzahl × gemeldetem Durchschnittskurs je Transaktionszeile.

**Einzelheiten zu den Verkäufen**

- **Nvidia.** Mark A. Stevens (Aufsichtsrat) verkaufte am 18.09.2026 1.366.000 Aktien zu rund
  219,7–220,4 $ aus indirektem Besitz, ohne 10b5-1-Kennzeichnung; danach verbleiben in dieser
  Besitzform 970.531 Stück. CFO Colette Kress (34.918 Stück, 7,65 Mio. $) und General Counsel
  Timothy Teter (30.460 Stück, 6,79 Mio. $) verkauften im Plan. Kurs heute 230,06 $, nahe dem
  52-Wochen-Hoch von 235,74 $.
  **Ein einzelner großer Planlos-Verkauf eines Aufsichtsrats nach mehrjährigem Anstieg ist
  Vermögensumschichtung, solange Management und Führung nicht mitziehen. Kein Verkaufssignal
  für das Unternehmen.**
- **CrowdStrike.** Sieben Verkäufer in zwei Wochen, darunter CEO George Kurtz (154.515 Stück,
  37,4 Mio. $, täglicher Plan), CFO Burt Podbere (513.882 Stück, 133,6 Mio. $; am 21.09. ohne,
  am 24.09. mit Plan-Kennzeichen), President Michael Sentonas (11,6 Mio. $, ohne Plan) und drei
  Aufsichtsräte (Watzinger im Plan, Austin und Davis ohne). Kurs 259,42 $, 52-Wochen-Spanne
  87,56–262,49 $.
  **Das ist der einzige Fall mit Breite: fast die ganze Führung verkauft nach einer
  Verdreifachung. Das ist ein Bewertungssignal der Insider, kein Hinweis auf ein operatives
  Problem, und es ist das stärkste Verkaufsbild der Liste.**
  Eine Meldung von Kurtz (16.09.) nennt 354 Stück zu 364,26 $, weit über allen anderen Kursen des
  Tages (236–245 $); vermutlich ein Tippfehler in der Einreichung, betragsmäßig ohne Gewicht.
- **Palo Alto.** Drei Aufsichtsräte, alle ohne Plan; Goetz (Sequoia) schließt den gemeldeten
  indirekten Bestand. Kurs 389,69 $ nahe dem 52-Wochen-Hoch.
- **Oracle, CRISPR.** Beträge unter 5 Mio. $, überwiegend im Plan oder nach Optionsausübung.
  **Kein Informationsgehalt.**

**Fazit Abschnitt 1:** *Das Bild wirft planmäßige Verkäufe und eigene Entscheidungen in einen
Topf. Belastbar als Signal sind auf der Verkaufsseite nur CrowdStrike (Breite) und mit
Abstand Palo Alto; auf der Kaufseite nur GameStop.*

---

## 2. Maßstab für die Vertiefung

Wie im Neste-Bericht: Hürde **13,56 %** p. a. (MSCI World, Gross Returns USD, 10 Jahre,
Factsheet 31.08.2026), Sensitivität **9,08 %** (seit Auflage 1987). Vorsteuer auf Anlegerebene.

---

## 3. Kauf 1: GameStop (GME), Kurs 23,68 $

### 3.1 Die Käufe

| Datum | Person | Stück | Ø-Kurs | Volumen | Bestand danach |
|---|---|---|---|---|---|
| 20.–21.01.2026 | Ryan Cohen, Chairman & CEO | 1.000.000 | 21,12 / 21,60 | 21,4 Mio. $ | 38.347.842 |
| 20.–21.01.2026 | Alain Attal, Aufsichtsrat | 24.000 | 20,90 / 21,63 | 0,5 Mio. $ | 596.464 |
| 23.01.2026 | Lawrence Cheng, Aufsichtsrat | 5.000 | 22,87 | 0,1 Mio. $ | 88.000 (indirekt) |
| 08.09.2026 | Lawrence Cheng | 55.000 | 18,80 | 1,0 Mio. $ | 143.000 (indirekt) |
| 09.09.2026 | James Grube, Aufsichtsrat | 10.255 | 19,12 | 0,2 Mio. $ | 39.694 |
| 10.09.2026 | Ryan Cohen | 1.000.000 | 20,38 | 20,4 Mio. $ | 39.347.842 |
| 10.09.2026 | Alain Attal | 5.000 | 20,00 | 0,1 Mio. $ | 601.464 |
| 21.09.2026 | Ryan Cohen | 1.150.680 | 22,94 | 26,4 Mio. $ | 40.498.522 |
| 21.09.2026 | Alain Attal | 17.500 | 22,97 | 0,4 Mio. $ | 618.964 |

Zwölf Monate: **Cohen 3,15 Mio. Aktien für 68,1 Mio. $, kein Verkauf.** Vier Insider kauften
im September 2026 am Markt. Verkäufe gab es nur von zwei Führungskräften (General Counsel im Plan,
Finanzchef), zusammen 1,6 Mio. $. Die Fußnote der Attal-Meldung vom 21.09. schreibt „sold“, Code
(P) und Richtungsfeld (A) belegen einen Kauf; das ist ein Formfehler der Einreichung.

### 3.2 Was man damit kauft: GameStop ist heute eine Beteiligungsgesellschaft

Aus dem Quartalsbericht Q2 FY2026 (Bilanz 01.08.2026), fortgeschrieben um den Anleihetausch vom
03.09.2026:

| Baustein | Mio. $ | $ je Aktie |
|---|---|---|
| Kasse und Wertpapiere (4.854,3 + 206,0), abzüglich 358,4 Barteil des Anleihetauschs | 4.702 | 9,32 |
| **eBay-Beteiligung**: 43.390.383 Aktien zu 108,39 $ (Einstand 101,09 $) | 4.703 | 9,32 |
| Bitcoin: 4.710 zu 83.393 $, **an Coinbase verpfändet** (Covered Calls) | 393 | 0,78 |
| Wandelanleihen: 4.167,8 abzüglich 1.400 getauscht (Buchwert, Näherung) | −2.768 | −5,49 |
| **Netto-Finanzvermögen** | **7.030** | **13,93** |
| Börsenwert (504,5 Mio. Aktien laut Deckblatt 10-Q, Stand 03.09.2026) | 11.947 | 23,68 |
| **Vom Kurs dem Einzelhandel zugeschriebener Rest** | **4.917** | **9,75** |

Das operative Geschäft verdient wieder: Betriebsergebnis FY2025 232,1 Mio. $, erstes Halbjahr
FY2026 303,5 Mio. $ nach 55,6 Mio. $; die letzten zwölf Monate ergeben 480,0 Mio. $. Der Rest von
4.917 Mio. $ entspricht dem **10,2-Fachen** davon.

Die entscheidende Tatsache: Am 03.05.2026 bot GameStop **125 $ je eBay-Aktie** in bar und Aktien
für 100 % von eBay (10-Q, Anhang 10). eBay lehnte am 12.05.2026 ab (CNBC, Sekundärquelle). Laut
Rule-425-Mitteilung vom 20.07.2026 (FT-Interview mit Cohen) verfolgte GameStop die Übernahme
danach weiter, mit knapp 10 % Anteil. Cohen zog im Juni seinen eigenen Leistungsbonus zurück, um
„fully focused on … its proposed eBay acquisition“ zu sein (8-K 23.06.2026).

Weitere Punkte aus dem 10-Q:
- 59,1 Mio. Optionsscheine zu **32,00 $**, Verfall **30.10.2026**; derzeit aus dem Geld.
  Würden sie ausgeübt, flössen rund 1,9 Mrd. $ zu, bei entsprechender Verwässerung.
- Anleihetausch: 55,5 Mio. neue Aktien plus 358,4 Mio. $ bar für 1,4 Mrd. $ Nominal.
- Rückkaufermächtigung über 2,0 Mrd. $ bis 02.06.2029, bis 01.08.2026 nicht genutzt.

### 3.3 Einordnung

- **Cohen zahlte am 10.09. 20,38 $ und am 21.09. 22,94 $, also 46 % bzw. 65 % über dem
  Netto-Finanzvermögen.** Er kauft nicht die Kasse, sondern den Einzelhandel plus die Option
  auf eBay. Ein Kaufcluster (CEO, eigenes Geld, mehrere Personen, wiederholt) ist die stärkste Form
  eines Insidersignals, aber hier ist der größte Käufer zugleich derjenige, der
  die eBay-Strategie treibt. Sein Kauf misst seine Überzeugung von dieser Strategie, keine
  unabhängige Bewertung.
- **Sensitivität zu eBay:** Zu 125 $ steigt das Netto-Finanzvermögen auf 15,36 $ je Aktie,
  bei einem Rückgang von eBay um 20 % (86,71 $) fällt es auf 12,07 $. Das eBay-Paket wird zu
  Marktpreisen gehalten; ein endgültiges Scheitern des Angebots träfe den Kurs daher zweimal:
  über das Paket und über die eingepreiste Option.
- **Abwärtspuffer:** Etwa 59 % des Kurses sind durch Finanzvermögen gedeckt (13,93 von 23,68 $).
  Das ist weit mehr als bei jedem anderen Titel der Liste.

### 3.4 Votum GameStop

*Nicht kaufen zu 23,68 $, beobachten.* **Der Kaufcluster ist echt, aber er bezahlt eine
Übernahmeoption, deren Stand seit Juli nicht mehr primär belegt ist, und einen Einzelhandel
zum 10-fachen Betriebsergebnis, dessen Aufschwung zwei Quartale alt ist.**

| Horizont | Marke | Begründung |
|---|---|---|
| bis 12 Monate | ~14 $ (Netto-Finanzvermögen) | Darunter bekommt man den Einzelhandel umsonst |
| bis 12 Monate | ~20 $ (Cohens Einstand 10.09.) | Mit dem größten Insider gleichziehen, ~43 % Aufschlag auf das Finanzvermögen |

- *Halter:* halten bis zum Verfall der Optionsscheine (30.10.2026) und zur Klärung des
  eBay-Angebots.
- *Nichthalter:* kaufen erst unter etwa 20 $; wer auf den Deal spekulieren will, kann das
  direkter über eBay tun.

**Auslöser:** (1) Ein verbindliches Angebot oder Einstieg eines Finanzierers → Deal-Wert rückt
näher; (2) Rückzug des Angebots oder Verkauf des eBay-Pakets → Netto-Finanzvermögen wird
Kassenwert, Option fällt weg; (3) Bericht Q3 FY2026 (Dezember): hält das Betriebsergebnis das
Halbjahresniveau?

**Gegen das Votum:** Cohen ist mit 40,5 Mio. Aktien der größte Einzelaktionär und hat seit Januar
nur gekauft. Der Einzelhandel verdient seit FY2025 wieder operativ Geld, im ersten Halbjahr FY2026 mehr als im ganzen Vorjahr. Und ein
Unternehmen, dessen Kurs zu knapp 60 % aus Kasse und börsennotierten Werten besteht, hat
einen Boden, den kein anderer Titel der Liste hat.

---

## 4. Kauf 2: Salesforce (CRM), Kurs 228,46 $

### 4.1 Die Käufe

Einziger Marktkauf im Fenster: David Blair Kirk (Aufsichtsrat), 18.09.2026, 4.176 Aktien zu
239,33 $ = 1,0 Mio. $; es ist sein dritter Kauf nach Dezember 2025 (1.936 zu 258,64 $) und März
2026 (2.570 zu 194,62 $). Über zwölf Monate:

| Person | Käufe | Verkäufe |
|---|---|---|
| G. Mason Morfit, Aufsichtsrat (indirekt) | 96.000 Aktien, 25,0 Mio. $ (05.12.2025) | – |
| David Blair Kirk, Aufsichtsrat | 8.682 Aktien, 2,0 Mio. $ | – |
| Laura Alber, Aufsichtsrätin | 2.571 Aktien, 0,5 Mio. $ | – |
| Parker Harris, Mitgründer | – | 134.662 Aktien, 31,6 Mio. $ (Plan) |
| Marc Benioff, Chair & CEO | – | 58.622 Aktien, 14,5 Mio. $ (Plan) |
| Craig Conway, Aufsichtsrat | – | 4.500 Aktien, 1,2 Mio. $ (**September 2026**) |
| Neelie Kroes, Aufsichtsrätin | – | 3.893 Aktien, 0,9 Mio. $ |

**Ein Aufsichtsrat kauft, im selben Monat verkauft ein anderer; Management verkauft planmäßig.
Das ist kein Kaufcluster. Der Einzelkauf von 1,0 Mio. $ ist gemessen an einem Börsenwert von
188 Mrd. $ ein Datenpunkt, kein Signal.**

### 4.2 Was sich 2026 verändert hat: fremdfinanzierter Rückkauf

Im ersten Halbjahr FY2027 (Februar bis Juli 2026) kaufte Salesforce **eigene Aktien für
27.332 Mio. $** und begab dafür **Anleihen über 24.842 Mio. $** (10-Q, Kapitalflussrechnung).
Folgen: langfristige Schulden 14.439 → 39.288 Mio. $, Eigenkapital 59.142 → 38.378 Mio. $,
Aktien 823 Mio. (Deckblatt 10-Q, 20.08.2026); Zinsaufwand im zweiten Quartal 473 Mio. $ gegen
135 Mio. $ im gesamten Vorjahreshalbjahr.

### 4.3 Eigentümerrechnung

Freier Zufluss (operativer Mittelzufluss − Investitionen), Mio. $:

| FY | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | LTM 07/26 |
|---|---|---|---|---|---|---|---|---|---|---|
| FCF | 2.204 | 2.803 | 3.688 | 4.091 | 5.283 | 6.313 | 9.498 | 12.434 | 14.402 | **15.154** |

Die letzten zwölf Monate liegen auf dem Höchststand der Reihe (Rang 100 %). Realisiertes
Wachstum (Mittel der ersten gegen die letzten drei Jahre): **26,9 % p. a.** Aktienvergütung in
den letzten zwölf Monaten: 3.665 Mio. $.

> **Annahme:** Einstieg = Börsenwert (823 Mio. × 228,46 $ = 188.023 Mio. $), Zufluss ab 2027
> konstant, Endwert = Buchwert des Eigenkapitals zum 31.07.2026 (38.378 Mio. $). Die Zeile
> „volle neue Zinslast“ unterstellt, dass der Zins im Vorjahr je Quartal auf dem Niveau des
> Vorjahreshalbjahrs lag, und zieht den Unterschied zur Laufrate (4 × 473 Mio. $) vor Steuern
> ab. Die Zeile „abzgl. Aktienvergütung“ behandelt die Vergütung in Aktien wie Lohn in bar.

| Szenario | Zufluss | Rendite auf Börsenwert | Schwelle 5 J. | 10 J. | 15 J. | 10 J. bei 9,08 % | Verlangtes Wachstum (10 J.) |
|---|---|---|---|---|---|---|---|
| LTM | 15.154 | 8,1 % | 88,58 | 110,79 | 122,55 | 137,31 | 16,8 % |
| LTM, volle neue Zinslast | 14.187 | 7,5 % | 84,50 | 104,56 | 115,17 | 129,79 | 18,1 % |
| LTM abzgl. Aktienvergütung | 11.489 | 6,1 % | 73,13 | 87,16 | 94,59 | 108,83 | 22,4 % |
| Median FY2018–2026 | 5.283 | 2,8 % | 46,96 | 47,14 | 47,23 | 60,61 | 37,7 % |

Konsens (Sekundärquelle, stockanalysis.com, 28.09.2026; Referenzkurs 228,28 $, also aktuell):
54 Analysten, „Buy“, Ziel Ø 281,08 $ (160–475 $). Erwartet werden für FY2027 11,5 % und für
FY2028 9,7 % Umsatzwachstum, der Gewinn je Aktie soll im FY2028 um 4,5 % sinken.

### 4.4 Votum Salesforce

*Nicht kaufen zu 228,46 $.* **Der Kurs verlangt 17 bis 22 % jährliches Wachstum des freien
Zuflusses über zehn Jahre. Die Firma hat das in der Vergangenheit geliefert (26,9 %), aber aus
einem Margensprung, der sich nicht wiederholen lässt, während der Umsatz nur noch um rund 10 %
wächst. Dazu kommen 25 Mrd. $ neue Schulden. Der Insiderkauf ändert daran nichts.**

- *Halter:* halten. Das Geschäft wirft 8 % Zufluss auf den Börsenwert ab, die Rückkäufe
  konzentrieren ihn auf weniger Aktien.
- *Nichthalter:* nicht kaufen. Marke auf zehn Jahre 104,56–110,79 $ (bei 9,08 %:
  129,79–137,31 $).

**Auslöser:** Umsatzwachstum wieder über 12 % bei stabiler Marge; oder ein Kurs in der Nähe von
130 $, der beim milderen Maßstab trägt.

**Gegen das Votum:** Die Konstantrechnung bewertet ein Unternehmen, das seinen freien Zufluss
sieben Jahre lang jedes Jahr gesteigert hat, als wüchse es nicht mehr. Der Konsens liegt 23 %
über dem Kurs. Und wer 27 Mrd. $ in die eigenen Aktien steckt, hält sie offenbar selbst für zu
billig; allerdings ist das ein Urteil des Vorstands, nicht der kaufenden Insider.

---

## 5. Gesamturteil

| Titel | Richtung im Bild | Befund aus den Formularen | Votum |
|---|---|---|---|
| GameStop | Kauf | Kaufcluster, CEO 68 Mio. $ in 12 Monaten | *beobachten, Marke ~20 $ / ~14 $* |
| Salesforce | Kauf | Einzelkauf 1 Mio. $, Gegenverkäufe | *nicht kaufen, Marke ~105–110 $ (10 J.)* |
| CrowdStrike | Verkauf | Breite Verkäufe nach Verdreifachung | stärkstes Warnsignal der Liste |
| Palo Alto | Verkauf | drei Aufsichtsräte ohne Plan | schwaches Warnsignal |
| Nvidia | Verkauf | ein großer Aufsichtsratsverkauf, sonst Pläne | kein Signal |
| Oracle, CRISPR | Verkauf | klein, planmäßig | kein Signal |

## 6. Nicht verfügbare Angaben

- **Stand des eBay-Angebots nach dem 20.07.2026:** weder zurückgezogen noch erneuert in einer
  Primärquelle gefunden; der 10-Q (Bilanzstichtag 01.08.2026) beschreibt nur das Angebot vom 03.05.
  Die Aussage „seit Mai tot“ stammt aus einer Suchzusammenfassung und ist nicht belegt.
- **Umwandlungspreise der verbleibenden GameStop-Wandelanleihen:** im Text nicht gefunden;
  der verwässerte Durchschnitt des zweiten Quartals lag bei 592,6 Mio. Aktien.
- **GameStop-Konsens:** Die Quellen widersprechen sich (Ziel 13,50 $ bei 8 Analysten laut Simply
  Wall St, anderswo ein Ziel mit Referenzkurs um 25 $); nicht verwendet.
- **Salesforce-Zinsaufwand des gesamten Vorjahres:** kein XBRL-Jahreswert; ersetzt durch die
  erklärte Annahme in 4.3.
- **Anteil der verkauften Bestände:** nur für eine Besitzform genau bestimmbar; wo Personen direkt
  und über Trusts halten, ist er nicht angegeben.

## 7. Quellen

- SEC EDGAR, Form 4 der sieben Emittenten, abgerufen 28.09.2026 (URLs in `data/form4.json`).
- GameStop Corp., Form 10-Q Q2 FY2026, `gme-20260801.htm`, Anhänge 5, 10, 11, 13, 14, 15.
- GameStop Corp., 8-K vom 23.06.2026 (Rückzug CEO-Bonus) und Rule-425-Mitteilung vom 20.07.2026.
- SEC XBRL companyfacts, GameStop (CIK 1326380) und Salesforce (CIK 1108524), abgerufen 28.09.2026.
- CNBC, 04.05. und 12.05.2026 (eBay-Ablehnung), Sekundärquelle.
- stockanalysis.com, Salesforce Forecast, abgerufen 28.09.2026, Sekundärquelle.
- Kurse: yfinance, Schlusskurse 28.09.2026.
- Hürde: MSCI World Index Factsheet, Stand 31.08.2026.

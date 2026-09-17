# Geruest einer belegpflichtigen Unternehmensanalyse

Kopiervorlage fuer eine Analyse nach dem in
`references/latex-project-pattern.md` beschriebenen Muster. Sie enthaelt die
Teile, die bei jeder Firma gleich sind: die vier Wachhunde, den
Seitenzerleger, den Werteleser und das Bauskript.

## In Betrieb nehmen

```bash
cp -r <skill>/assets/scaffold <ziel>/<Firma>
cd <ziel>/<Firma>
$EDITOR scripts/projekt.py      # die EINZIGE anzupassende Datei in scripts/
```

Danach in dieser Reihenfolge:

1. **Quellen beschaffen** und nach `refs/` legen.
2. `python3 scripts/edgar_seiten.py` - zerlegt sie in gedruckte Seiten. Die
   Ausgabe muss fuer jede Datei `lueckenlos` melden. Meldet sie Luecken, ist
   eine Fusszeilenform in `scripts/projekt.py` nachzutragen; meldet sie eine
   unplausible Seitenspanne, hat eine Zahl aus dem Fliesstext als Seitenzahl
   durchgeschlagen und der Folgefilter greift nicht.
3. **`scripts/kennzahlen.py` anlegen** (Vorlage siehe `references/`), jeden
   Rohwert mit Quelle und gedruckter Seite.
4. `python3 scripts/pruefe_seiten.py` - **vor der ersten Zeile Prosa.** In der
   Referenzarbeit waren beim ersten Durchgang 23 von 176 Seitenangaben falsch,
   ausnahmslos die aus dem Zusammenhang geschlossenen.
5. **Abschnitte schreiben**, dann `./build.sh`.
6. `python3 scripts/fussnoten_gruppen.py --schleife`, danach beide
   Fussnotenpruefer, bis sie schweigen.
7. `cd scripts && python3 -m unittest discover -p 'test_*.py'`

## Was hier absichtlich fehlt

`kennzahlen.py`, `pruefe_seiten.py` und `flow.py` sind **nicht** im Geruest.
Sie tragen die Zahlen und die Annahmen einer bestimmten Firma, und eine
Vorlage davon waere eine Einladung, die Annahmen der Vorgaengerfirma
stehenzulassen. Ihr Aufbau steht in `references/latex-project-pattern.md`.

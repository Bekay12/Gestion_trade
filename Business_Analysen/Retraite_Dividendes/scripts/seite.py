#!/usr/bin/env python3
"""seite.py <textdatei> <gedruckte Seite> [...] - gibt die gedruckten Seiten aus."""
import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
parts = re.split(r'(?m)^=== SEITE (\S+) \(Block (\d+)\) ===$', txt)
for i in range(1, len(parts), 3):
    if parts[i] in sys.argv[2:]:
        print(f'--- {sys.argv[1]} gedruckte Seite {parts[i]} ---')
        print(re.sub(r'\n{2,}', '\n', parts[i + 2]).strip())

#!/usr/bin/env bash
# ollama.sh <modele> <num_ctx> < prompt.txt  ->  reponse sur stdout
# Regles du depot: prompt par stdin, num_ctx explicite, think=false, API HTTP (pas de
# codes ANSI). Sortie code 3 si Ollama ne repond pas; code 4 si le prompt a ete tronque.
set -euo pipefail
MODELE="$1"; CTX="$2"
PROMPT=$(cat)
curl -s -m 5 localhost:11434/api/tags >/dev/null || { echo "NO_LOCAL_RUNTIME" >&2; exit 3; }
REP=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.stdin.read(),"stream":False,"think":False,"options":{"num_ctx":int(sys.argv[2]),"temperature":0}}))' "$MODELE" "$CTX" <<<"$PROMPT" \
  | curl -s -m 900 localhost:11434/api/generate -d @-)
# NOTE (tache 0): le script du brief lisait le JSON via <<<"$REP" en meme temps que le
# corps du script via <<'EOF' sur le meme fd (stdin) ; bash ne garde que la derniere
# redirection, donc $REP etait toujours vide et json.loads("") echouait. Correction :
# passer $PROMPT et $REP en argv plutot que sur stdin, semantique inchangee sinon.
python3 -c '
import json, sys
r = json.loads(sys.argv[2])
lu, attendu = r.get("prompt_eval_count", 0), len(sys.argv[1]) // 4
if lu < attendu * 0.8:
    print(f"PROMPT_TRONQUE lu={lu} attendu~{attendu}", file=sys.stderr); sys.exit(4)
print(r.get("response", ""))
' "$PROMPT" "$REP"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$ROOT_DIR/out/ollama.log"
PID_FILE="$ROOT_DIR/out/ollama.pid"
HOST="127.0.0.1"
PORT="11434"
BASE_URL="http://${HOST}:${PORT}"

mkdir -p "$ROOT_DIR/out"

is_server_ready() {
  curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1
}

start_server() {
  if is_server_ready; then
    echo "Ollama server already available."
    return 0
  fi

  echo "Starting Ollama server..."
  nohup ollama serve >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"

  for _ in $(seq 1 30); do
    if is_server_ready; then
      echo "Ollama server is ready."
      return 0
    fi
    sleep 1
  done

  echo "Ollama server did not become ready." >&2
  echo "Log file: $LOG_FILE" >&2
  return 1
}

ensure_watchdog() {
  local watchdog_script="$ROOT_DIR/out/ollama_watchdog.sh"
  cat >"$watchdog_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/out/ollama.pid"
LOG_FILE="$ROOT_DIR/out/ollama.log"
while true; do
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      if ! curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
        echo "Restarting Ollama server..." >>"$LOG_FILE"
        kill "$pid" 2>/dev/null || true
        nohup ollama serve >>"$LOG_FILE" 2>&1 &
        echo $! >"$PID_FILE"
      fi
    else
      echo "Restarting Ollama server..." >>"$LOG_FILE"
      nohup ollama serve >>"$LOG_FILE" 2>&1 &
      echo $! >"$PID_FILE"
    fi
  else
    echo "Restarting Ollama server..." >>"$LOG_FILE"
    nohup ollama serve >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  fi
  sleep 10
 done
EOF
  chmod +x "$watchdog_script"

  if ! pgrep -f "$watchdog_script" >/dev/null 2>&1; then
    nohup "$watchdog_script" >/dev/null 2>&1 &
  fi
}

if [[ $# -eq 0 ]]; then
  start_server
  ensure_watchdog
  ollama list
  exit 0
fi

start_server
ensure_watchdog
MODEL="${1:-gemma4:12b}"
PROMPT="${2:-Say hello in one short sentence}"
NUM_CTX="${OLLAMA_NUM_CTX:-32768}"

# Ueber die HTTP-API statt "ollama run": nur so laesst sich num_ctx setzen. Ollama
# kuerzt einen laengeren Prompt still vom ANFANG her - also genau dort, wo die
# Formatvorgabe steht - und reserviert einen Teil des Fensters fuer die Ausgabe.
# Gemessen 02.08.2026: 15.403 Tokens bei num_ctx=4096 auf 2.050 gekuerzt, Ausgabe
# ohne jeden Bezug zur Anweisung. "think:false" verhindert ausserdem, dass die
# Antwort im Feld "thinking" statt in "response" landet.
curl -s http://localhost:11434/api/generate -d "$(jq -n \
  --arg m "$MODEL" --arg p "$PROMPT" --argjson c "$NUM_CTX" \
  '{model:$m, prompt:$p, stream:false, think:false,
    options:{temperature:0, num_ctx:$c}}')" | jq -r '.response'

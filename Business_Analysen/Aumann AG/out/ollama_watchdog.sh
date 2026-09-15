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

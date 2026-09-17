#!/usr/bin/env bash
# local_llm.sh - Run a prompt against a local model, or signal that none is available.
#
# The prompt is read from STDIN. Passing it as a command argument hangs indefinitely
# in a non-interactive shell (measured), so this script never does that.
#
#   scripts/local_llm.sh --model gemma4:12b < prompt.txt
#   scripts/local_llm.sh --probe                      # runtime check only, no inference
#
# Probe order:
#   1. ollama            - stdin-driven, starts its server if needed
#   2. OpenAI-compatible endpoint on $LOCAL_LLM_ENDPOINT (default localhost:8080)
#      This is how llmfit exposes a model: `llmfit run <model> --server --port 8080`.
#      The endpoint is used only if it is ALREADY listening; this script never
#      launches llama.cpp, because that needs llama-cli plus a downloaded GGUF.
#   3. nothing           - prints NO_LOCAL_RUNTIME to stderr and exits 3.
#
# Exit codes:
#   0  inference succeeded, completion on stdout
#   2  usage error
#   3  NO_LOCAL_RUNTIME - caller must fall back to a cheap cloud model
#   4  a runtime was found but the call failed
#
# On exit 3 the caller redoes the work itself and says so in its report. A local
# model is a cost optimisation, never a correctness dependency.
set -uo pipefail

MODEL="${LOCAL_LLM_MODEL:-gemma4:12b}"
# Ollama truncates a prompt longer than num_ctx from the START, taking the
# instructions with it, and reserves part of the window for the output. Measured
# 2026-08-02: a 15,403-token prompt at num_ctx=4096 was cut to 2,050 tokens and the
# format spec was lost. Size this at >= 2x the longest prompt you send.
NUM_CTX="${LOCAL_LLM_NUM_CTX:-32768}"
ENDPOINT="${LOCAL_LLM_ENDPOINT:-http://localhost:8080}"
TIMEOUT="${LOCAL_LLM_TIMEOUT:-900}"
PROBE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model)    MODEL="${2:?--model needs a value}"; shift 2 ;;
    --endpoint) ENDPOINT="${2:?--endpoint needs a value}"; shift 2 ;;
    --timeout)  TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
    --num-ctx)  NUM_CTX="${2:?--num-ctx needs a value}"; shift 2 ;;
    --probe)    PROBE=1; shift ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *)          echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[local_llm] %s\n' "$*" >&2; }

# --- 1. ollama --------------------------------------------------------------
ollama_ready() {
  command -v ollama >/dev/null 2>&1 || return 1
  ollama list >/dev/null 2>&1 && return 0
  # Server not up: start it detached and wait briefly.
  log "starting ollama server"
  (ollama serve >/dev/null 2>&1 &) || return 1
  for _ in $(seq 1 20); do
    sleep 1
    ollama list >/dev/null 2>&1 && return 0
  done
  return 1
}

ollama_has_model() {
  ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx -- "$1"
}

# --- 2. OpenAI-compatible endpoint (llmfit --server, llama-server, …) -------
endpoint_ready() {
  command -v curl >/dev/null 2>&1 || return 1
  curl -sf --max-time 3 "${ENDPOINT}/v1/models" >/dev/null 2>&1
}

endpoint_complete() {
  # Reads the prompt from $1 (a file), prints the completion on stdout.
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - "$ENDPOINT" "$MODEL" "$1" "$TIMEOUT" <<'PY'
import json, sys, urllib.request
endpoint, model, path, timeout = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
with open(path, encoding="utf-8") as fh:
    prompt = fh.read()
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "stream": False}).encode()
req = urllib.request.Request(f"{endpoint}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=timeout) as resp:
    payload = json.load(resp)
print(payload["choices"][0]["message"]["content"])
PY
}

# --- probe mode -------------------------------------------------------------
if [ "$PROBE" -eq 1 ]; then
  if ollama_ready; then
    if ollama_has_model "$MODEL"; then
      echo "ollama: ready, model '$MODEL' present"; exit 0
    fi
    echo "ollama: ready, model '$MODEL' NOT pulled (ollama pull $MODEL)" >&2; exit 4
  fi
  if endpoint_ready; then echo "endpoint: ready at $ENDPOINT"; exit 0; fi
  echo "NO_LOCAL_RUNTIME" >&2; exit 3
fi

# --- inference --------------------------------------------------------------
PROMPT_FILE="$(mktemp)"
trap 'rm -f "$PROMPT_FILE"' EXIT
cat > "$PROMPT_FILE"

if [ ! -s "$PROMPT_FILE" ]; then
  echo "empty prompt on stdin" >&2; exit 2
fi

if ollama_ready; then
  if ! ollama_has_model "$MODEL"; then
    log "model '$MODEL' not pulled; run: ollama pull $MODEL"
    exit 4
  fi
  # The HTTP API rather than `ollama run`: it takes num_ctx, it can turn thinking off
  # (the CLI leaves it on, and the answer then arrives in a separate field), and it does
  # not emit ANSI escapes into a redirected stream. It also reports prompt_eval_count,
  # which is the only way to detect a silently truncated prompt.
  if out="$(timeout "$TIMEOUT" curl -sf http://localhost:11434/api/generate \
        -d "$(jq -n --rawfile p "$PROMPT_FILE" --arg m "$MODEL" --argjson c "$NUM_CTX" \
              '{model:$m, prompt:$p, stream:false, think:false,
                options:{temperature:0, num_ctx:$c}}')")"; then
    # Truncation guard. Comparing prompt_eval_count with num_ctx does NOT work:
    # Ollama reserves part of the window for the output, so a cut prompt reports a
    # count well BELOW num_ctx (measured: num_ctx=2048 -> 1024 tokens read, answer
    # silently unrelated to the instructions). Compare against the prompt itself
    # instead. 6 bytes per token deliberately UNDER-estimates the real count
    # (measured 5.3-5.7 on layout text), so the check errs towards not firing.
    read_tok="$(printf '%s' "$out" | jq -r '.prompt_eval_count // 0')"
    est_tok=$(( $(wc -c < "$PROMPT_FILE") / 6 ))
    if [ "$read_tok" -lt $(( est_tok * 8 / 10 )) ]; then
      log "prompt truncated: $read_tok tokens read, ~$est_tok expected (num_ctx=$NUM_CTX)"
      log "raise --num-ctx to at least $(( est_tok * 5 / 2 )) and rerun"
      exit 4
    fi
    printf '%s\n' "$out" | jq -r '.response'
    exit 0
  fi
  log "ollama call failed"; exit 4
fi

if endpoint_ready; then
  log "using endpoint $ENDPOINT"
  if endpoint_complete "$PROMPT_FILE"; then exit 0; fi
  log "endpoint call failed"; exit 4
fi

echo "NO_LOCAL_RUNTIME" >&2
exit 3

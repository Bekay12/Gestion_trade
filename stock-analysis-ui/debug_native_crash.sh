#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
VENV_PY="$PROJECT_ROOT/.venv_new/bin/python"
TRADING_C_DIR="$ROOT_DIR/src/trading_c_acceleration"
UI_ENTRY="$ROOT_DIR/src/ui/main_window.py"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[ERREUR] Python venv introuvable: $VENV_PY"
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./debug_native_crash.sh build-debug   # Recompile trading_c avec -g -O0 -fno-omit-frame-pointer
  ./debug_native_crash.sh build-asan    # Recompile trading_c avec debug + -fsanitize=address
  ./debug_native_crash.sh run-gdb       # Lance l'UI dans gdb
  ./debug_native_crash.sh run-asan      # Lance l'UI avec runtime ASan preload

Notes:
  - run-gdb: reproduire le crash dans l'UI puis taper: bt
  - run-asan: reproduire le crash et lire le rapport ASan dans stderr
EOF
}

build_debug() {
  echo "[INFO] Build debug trading_c"
  cd "$TRADING_C_DIR"
  QSI_DEBUG_C_MODE=1 QSI_USE_ASAN=0 "$VENV_PY" setup.py build_ext --inplace
}

build_asan() {
  echo "[INFO] Build ASan trading_c"
  cd "$TRADING_C_DIR"
  QSI_DEBUG_C_MODE=1 QSI_USE_ASAN=1 "$VENV_PY" setup.py build_ext --inplace
}

run_gdb() {
  echo "[INFO] Launch gdb (reproduire le segfault puis taper 'bt')"
  cd "$PROJECT_ROOT"
  gdb --args "$VENV_PY" "$UI_ENTRY"
}

run_asan() {
  local asan_lib
  asan_lib="$(gcc -print-file-name=libasan.so)"
  if [[ ! -f "$asan_lib" ]]; then
    echo "[ERREUR] libasan introuvable via gcc -print-file-name=libasan.so"
    exit 1
  fi

  echo "[INFO] Launch UI with ASan runtime preload"
  cd "$PROJECT_ROOT"
  export LD_PRELOAD="$asan_lib"
  export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:halt_on_error=1:strict_init_order=1:check_initialization_order=1"
  "$VENV_PY" "$UI_ENTRY"
}

cmd="${1:-}"
case "$cmd" in
  build-debug)
    build_debug
    ;;
  build-asan)
    build_asan
    ;;
  run-gdb)
    run_gdb
    ;;
  run-asan)
    run_asan
    ;;
  *)
    usage
    exit 1
    ;;
esac

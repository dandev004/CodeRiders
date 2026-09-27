#!/usr/bin/env bash
# Secure MOM — pornește tot stack-ul local: Mailpit (SMTP intern) + n8n (rutare) + Ollama (LLM) + serverul web.
# Totul ascultă doar pe 127.0.0.1; telemetria n8n/HF e dezactivată. Oprire: Ctrl+C.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/data/logs"; mkdir -p "$LOG"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

wait_for() {  # url, nume
  for _ in $(seq 1 60); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done
  echo "  ! $2 nu a pornit (vezi $LOG)"; return 1
}

# ---- n8n: fără telemetrie, fără șabloane/versiuni descărcate de pe internet
export N8N_USER_FOLDER="$ROOT/n8n/data" N8N_HOST=127.0.0.1 N8N_LISTEN_ADDRESS=127.0.0.1 N8N_PORT=5678
export N8N_DIAGNOSTICS_ENABLED=false N8N_VERSION_NOTIFICATIONS_ENABLED=false N8N_TEMPLATES_ENABLED=false
export N8N_PERSONALIZATION_ENABLED=false N8N_HIRING_BANNER_ENABLED=false N8N_AI_ENABLED=false
export N8N_SECURE_COOKIE=false N8N_DEFAULT_BINARY_DATA_MODE=default N8N_PAYLOAD_SIZE_MAX=64
export N8N_ENCRYPTION_KEY="${N8N_ENCRYPTION_KEY:-secure-mom-local-demo-key}" N8N_RUNNERS_ENABLED=true N8N_EXPRESSION_ENGINE=legacy N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=false
export N8N_DIAGNOSTICS_CONFIG_FRONTEND="" N8N_DIAGNOSTICS_CONFIG_BACKEND="" EXTERNAL_FRONTEND_HOOKS_URLS=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1
N8N="$ROOT/tools/n8n/node_modules/.bin/n8n"

echo "▶ Mailpit (SMTP intern :1025, căsuță web :8025)"
if ! curl -sf -o /dev/null http://127.0.0.1:8025; then
  mailpit --listen 127.0.0.1:8025 --smtp 127.0.0.1:1025 --disable-version-check >"$LOG/mailpit.log" 2>&1 &
  PIDS+=($!)
  wait_for http://127.0.0.1:8025 Mailpit
fi

echo "▶ Ollama (LLM local)"
if ! curl -sf -o /dev/null http://127.0.0.1:11434/api/tags; then
  OLLAMA_HOST=127.0.0.1:11434 ollama serve >"$LOG/ollama.log" 2>&1 &
  PIDS+=($!)
  wait_for http://127.0.0.1:11434/api/tags Ollama
fi

if [ -x "$N8N" ]; then
  echo "▶ n8n (workflow de rutare :5678)"
  if [ ! -f "$ROOT/n8n/data/.imported" ]; then
    mkdir -p "$ROOT/n8n/data"
    "$N8N" import:credentials --input="$ROOT/n8n/credentials.json" >"$LOG/n8n-import.log" 2>&1
    "$N8N" import:workflow --input="$ROOT/n8n/workflow.json" >>"$LOG/n8n-import.log" 2>&1
    "$N8N" update:workflow --id=SecureMomRoute1 --active=true >>"$LOG/n8n-import.log" 2>&1 || true
    touch "$ROOT/n8n/data/.imported"
  fi
  if ! curl -sf -o /dev/null http://127.0.0.1:5678/healthz; then
    "$N8N" start >"$LOG/n8n.log" 2>&1 &
    PIDS+=($!)
    wait_for http://127.0.0.1:5678/healthz n8n || true
  fi
else
  echo "  ! n8n nu e instalat (scripts/setup.sh) — livrarea merge direct prin SMTP intern"
fi

echo "▶ Secure MOM — http://127.0.0.1:8000"
# pe laptop: macOS adoarme sistemul după câteva minute de inactivitate și oprește procesarea în mijlocul ședinței;
# `caffeinate -i` îl ține treaz cât rulează serverul (ecranul se poate stinge)
if command -v caffeinate >/dev/null; then
  caffeinate -i "$ROOT/.venv/bin/python" -m app.main
else
  "$ROOT/.venv/bin/python" -m app.main
fi

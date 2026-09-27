#!/usr/bin/env bash
# Antrenarea de peste noapte: fiecare pas pornește doar după ce s-a terminat cel anterior (16 GB -> fără paralelism).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=training/results/overnight.log
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }
S=${SMOKE_DIR:?}

say "aștept generarea textelor"
while pgrep -f make_medical_texts >/dev/null; do sleep 30; done
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen3.5:9b","keep_alive":0}' >/dev/null
say "texte: $(wc -l < data/medical_text/spoken_medical.jsonl)"

say "test scurt de antrenare (3 pași)"
if .venv/bin/python -m training.finetune_whisper_mlx --xtts-dir "$S" --n-real 40 --n-babble 20 --epochs 1 \
     --max-steps 3 --out "$S/smoke-model" >>"$LOG" 2>&1; then SMOKE=ok; else SMOKE=fail; fi
say "test antrenare: $SMOKE"

say "sinteză vocală XTTS"
.venv-tts/bin/python training/make_tts_xtts.py --n-simonero 400 >>"$LOG" 2>&1
say "sinteză: $(wc -l < data/tts_xtts/metadata.jsonl) clipuri"

say "verificare dus-întors"
.venv/bin/python -m training.filter_tts --max-cer 0.4 >>"$LOG" 2>&1
say "filtru gata"

if [ "$SMOKE" != ok ]; then say "STOP: testul de antrenare a eșuat"; exit 1; fi
say "antrenare completă"
.venv/bin/python -m training.finetune_whisper_mlx --out models/mlx-whisper-turbo-moldovan-med2 >>"$LOG" 2>&1 \
  || { say "STOP: antrenarea a eșuat"; exit 1; }

say "evaluare Medpark"
.venv/bin/python -m training.eval_reference --ro-model models/mlx-whisper-turbo-moldovan-med2 --tag med2 \
  --out training/results/ref_med2.json >>"$LOG" 2>&1
say "evaluare dialect"
.venv/bin/python -m training.eval_wer --role ro --model models/mlx-whisper-turbo-moldovan-med2 \
  --out training/results/wer_med2.json >>"$LOG" 2>&1
say "evaluare code-switching"
.venv/bin/python -m training.eval_codeswitch --model models/mlx-whisper-large-v3-turbo \
  --ro-model models/mlx-whisper-turbo-moldovan-med2 --out training/results/codeswitch_med2.json >>"$LOG" 2>&1
say "GATA"

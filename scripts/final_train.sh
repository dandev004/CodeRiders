#!/usr/bin/env bash
# Rundă finală scurtată (termen: 3 ore). Pași secvențiali, verificați; rulează sub caffeinate.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=training/results/final.log
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
n=$(lines data/tts_xtts/metadata.jsonl)

say "verificare dus-întors ($n clipuri) + set de test cu voci nevăzute (50), în paralel"
.venv-tts/bin/python training/make_tts_xtts.py --texts data/medical_text/spoken_medical_test50.jsonl --n-simonero 0 \
  --test-voices --out data/tts_test >>"$LOG" 2>&1 &
T=$!
.venv/bin/python -m training.filter_tts --max-cer 0.4 >>"$LOG" 2>&1
wait $T
[ "$(lines data/tts_xtts/filter.jsonl)" -ge "$n" ] || { say "STOP: filtru incomplet"; exit 1; }
say "filtru gata: $(grep -c '"keep": true' data/tts_xtts/filter.jsonl) păstrate; test: $(lines data/tts_test/metadata.jsonl)"

say "antrenare (2 epoci)"
.venv/bin/python -m training.finetune_whisper_mlx --epochs 2 --n-real 1200 --n-babble 400 \
  --out models/mlx-whisper-turbo-moldovan-med2 >>"$LOG" 2>&1 || { say "STOP: antrenarea a eșuat"; exit 1; }

say "evaluare Medpark"
.venv/bin/python -m training.eval_reference --ro-model models/mlx-whisper-turbo-moldovan-med2 --tag med2 \
  --out training/results/ref_med2.json >>"$LOG" 2>&1
say "evaluare test voci nevăzute (actual vs nou)"
.venv/bin/python -m training.eval_synth --ro-model models/mlx-whisper-turbo-moldovan-med --out training/results/synth_med.json >>"$LOG" 2>&1
.venv/bin/python -m training.eval_synth --ro-model models/mlx-whisper-turbo-moldovan-med2 --out training/results/synth_med2.json >>"$LOG" 2>&1
say "evaluare code-switching"
.venv/bin/python -m training.eval_codeswitch --model models/mlx-whisper-large-v3-turbo \
  --ro-model models/mlx-whisper-turbo-moldovan-med2 --out training/results/codeswitch_med2.json >>"$LOG" 2>&1
say "evaluare dialect"
.venv/bin/python -m training.eval_wer --role ro --model models/mlx-whisper-turbo-moldovan-med2 \
  --out training/results/wer_med2.json >>"$LOG" 2>&1
say "GATA"

#!/usr/bin/env bash
# Rundă finală (termen 11:30): așteaptă verificarea în curs, apoi antrenare 3 epoci + evaluări.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=training/results/final.log
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }
while pgrep -f "training.filter_tts|make_tts_xtts.py" >/dev/null; do sleep 20; done
say "filtru gata: $(grep -c '"keep": true' data/tts_xtts/filter.jsonl)/$(wc -l < data/tts_xtts/filter.jsonl | tr -d ' ') păstrate; test: $(wc -l < data/tts_test/metadata.jsonl | tr -d ' ')"
say "antrenare (3 epoci, 1500 clipuri reale)"
.venv/bin/python -m training.finetune_whisper_mlx --epochs 3 --n-real 1500 --n-babble 400 \
  --out models/mlx-whisper-turbo-moldovan-med2 >>"$LOG" 2>&1 || say "antrenarea s-a oprit (se folosește cea mai bună epocă salvată, dacă există)"
[ -f models/mlx-whisper-turbo-moldovan-med2/weights.safetensors ] || { say "STOP: niciun model salvat"; exit 1; }
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

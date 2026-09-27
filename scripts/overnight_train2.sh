#!/usr/bin/env bash
# Antrenarea de peste noapte (v2): pași strict secvențiali, fiecare verificat înainte de următorul.
# Rulează sub `caffeinate` (Mac-ul nu adoarme). Oprire sigură: pkill -f overnight_train2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=training/results/overnight.log
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

say "v2: sinteză vocală XTTS (continuă de la $(lines data/tts_xtts/metadata.jsonl))"
.venv-tts/bin/python training/make_tts_xtts.py --n-simonero 400 >>"$LOG" 2>&1
n=$(lines data/tts_xtts/metadata.jsonl); say "sinteză: $n clipuri"
[ "$n" -ge 1850 ] || { say "STOP: sinteză incompletă"; exit 1; }

say "sinteză set de test (voci nevăzute)"
.venv-tts/bin/python training/make_tts_xtts.py --texts data/medical_text/spoken_medical_test.jsonl --n-simonero 0 \
  --test-voices --out data/tts_test >>"$LOG" 2>&1
say "test: $(lines data/tts_test/metadata.jsonl) clipuri"

say "verificare dus-întors"
.venv/bin/python -m training.filter_tts --max-cer 0.4 >>"$LOG" 2>&1
[ "$(lines data/tts_xtts/filter.jsonl)" -ge "$n" ] || { say "STOP: filtru incomplet"; exit 1; }
say "filtru gata"

say "evaluare de referință pe setul de test (modelul actual)"
.venv/bin/python -m training.eval_synth --ro-model models/mlx-whisper-turbo-moldovan-med \
  --out training/results/synth_med.json >>"$LOG" 2>&1

say "antrenare completă"
.venv/bin/python -m training.finetune_whisper_mlx --out models/mlx-whisper-turbo-moldovan-med2 >>"$LOG" 2>&1 \
  || { say "STOP: antrenarea a eșuat"; exit 1; }

say "evaluare set de test (model nou)"
.venv/bin/python -m training.eval_synth --ro-model models/mlx-whisper-turbo-moldovan-med2 \
  --out training/results/synth_med2.json >>"$LOG" 2>&1
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

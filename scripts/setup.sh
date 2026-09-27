#!/usr/bin/env bash
# Secure MOM — pregătire (SINGURUL pas care folosește internetul): descarcă modelele open-weights și dependențele.
# După acest pas, sistemul rulează complet deconectat de la rețea.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

python3.11 -m venv .venv
.venv/bin/pip install -U pip -q
.venv/bin/pip install -r requirements.txt -q

.venv/bin/python - <<'PY'
import platform
from huggingface_hub import snapshot_download
apple = platform.system() == "Darwin" and platform.machine() == "arm64"
# Whisper adaptat pe româna moldovenească (FraPiz, Apache 2.0) — convertit local pentru MLX și CTranslate2
snapshot_download("FraPiz/whisper-large-v3-turbo-moldovan-romanian", local_dir="models/hf-whisper-turbo-moldovan",
                  allow_patterns=["*.json", "*.safetensors", "LICENSE", "README.md"])
if apple:
    snapshot_download("mlx-community/whisper-large-v3-turbo", local_dir="models/mlx-whisper-large-v3-turbo")
else:
    snapshot_download("mobiuslabsgmbh/faster-whisper-large-v3-turbo", local_dir="models/faster-whisper-large-v3-turbo")
snapshot_download("speechbrain/spkrec-ecapa-voxceleb", local_dir="models/spkrec-ecapa-voxceleb")
PY
if [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ]; then
  .venv/bin/python -m training.convert_hf_to_mlx models/hf-whisper-turbo-moldovan models/mlx-whisper-turbo-moldovan \
      --reference models/mlx-whisper-large-v3-turbo
else
  .venv/bin/pip install -q "transformers>=4.45"
  .venv/bin/ct2-transformers-converter --model models/hf-whisper-turbo-moldovan --output_dir models/ct2-whisper-turbo-moldovan \
      --quantization float16 --copy_files tokenizer.json preprocessor_config.json
fi

mkdir -p models/lexicon data/moldovan_corpus
for l in ro ru en; do
  curl -sfL -o "models/lexicon/${l}_50k.txt" "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/$l/${l}_50k.txt"
done
curl -sfL -o data/moldovan_corpus/metadata.jsonl \
  https://huggingface.co/datasets/FraPiz/moldovan-dialectal-romanian-speech-corpus/resolve/main/metadata.jsonl
.venv/bin/python -m training.build_lexicon

# corpusuri medicale text (vocabular medical RO/RU)
mkdir -p data/medical_text
for f in train dev test; do
  curl -sfL -o "data/medical_text/simonero-$f.conllu" \
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-SiMoNERo/master/ro_simonero-ud-$f.conllu"
done
curl -sfL -o data/medical_text/medical_qa_ru.csv \
  https://huggingface.co/datasets/blinoff/medical_qa_ru_data/resolve/main/medical_qa_ru_data.csv
curl -sfL -o data/medical_text/rus_med_dialogues.parquet \
  https://huggingface.co/datasets/Mykes/rus_med_dialogues/resolve/main/data/train-00000-of-00001.parquet
.venv/bin/python -m training.build_medical_vocab

ollama pull qwen3.5:9b

# n8n self-hosted (fără compilarea isolated-vm; folosim motorul de expresii „legacy”)
(cd tools/n8n && npm install --no-audit --no-fund --ignore-scripts && npm rebuild sqlite3)
command -v mailpit >/dev/null || brew install mailpit
echo "Gata. Deconectați rețeaua și porniți: ./scripts/start.sh"

"""Evaluare WER / CER pe româna moldovenească (FraPiz/moldovan-dialectal-romanian-speech-corpus).

Compară configurații ASR pe fraze reale cu accent moldovenesc, de la vorbitori NEFOLOSIȚI la adaptare:
  - modelul (large-v3 vs large-v3-turbo vs model fine-tunat)
  - decodare forțată pe română vs pipeline-ul hibrid RO/RU/EN (arată cât de des româna e confundată cu rusa)

    python -m training.eval_wer --shard data/moldovan_corpus/train-00017.parquet --n 150 \
        --model models/mlx-whisper-large-v3 --model models/mlx-whisper-large-v3-turbo

Normalizare: litere mici, fără punctuație, ş/ţ -> ș/ț. Corpusul scrie numerele în litere („zero virgulă doi”),
Whisper în cifre — raportăm și WER fără frazele cu cifre, ca eroarea de format să nu fie numărată ca eroare ASR.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import asr  # noqa: E402
from app.config import get_config  # noqa: E402


def norm(t: str) -> str:
    t = t.lower().replace("ş", "ș").replace("ţ", "ț").replace("-", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_clips(shard: str, n: int, seed: int = 0, min_dur: float = 1.5, max_dur: float = 12.0):
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

    t = pq.read_table(shard)
    rows = [r for r in range(t.num_rows) if min_dur <= t.column("duration")[r].as_py() <= max_dur]
    random.Random(seed).shuffle(rows)
    clips = []
    for r in rows[:n]:
        row = t.slice(r, 1).to_pylist()[0]
        a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = resample_poly(a, 16000, sr).astype(np.float32)
        clips.append((a, row["text"], row["audio_filename"]))
    return clips


def run(model: str, clips, languages: list[str], use_prompt: bool, role: str = "main"):
    import jiwer

    cfg = get_config()
    cfg["asr"]["profile"] = ""  # altfel profilul din config suprascrie modelul testat
    # role="ro": modelul testat e specialistul pentru română (arhitectura cu două modele); altfel e modelul principal
    cfg["asr"]["ro_mlx_model" if role == "ro" else "mlx_model"] = model
    cfg["asr"]["use_glossary_prompt"] = use_prompt
    gap = np.zeros(8000, dtype=np.float32)
    audio, spans, pos = [], [], 0
    for a, _, _ in clips:
        audio += [a, gap]
        spans.append((pos, pos + len(a)))
        pos += len(a) + len(gap)
    audio = np.concatenate(audio)
    t0 = time.time()
    backend = asr.load_backend()
    spec = asr.load_specialist() if role == "ro" else None
    segs, st = asr.decode_spans(audio, spans, "medical", backend=backend, languages=languages, specialist=spec)
    backend.release()
    if spec:
        spec.release()
    dt = time.time() - t0
    by_start = {round(s.start * 16000): s for s in segs}
    refs, hyps, refs_nd, hyps_nd, wrong_lang = [], [], [], [], 0
    for (s, _), (_, ref, _) in zip(spans, clips):
        seg = by_start.get(s)
        hyp = seg.text if seg else ""
        if seg and seg.lang != "ro":
            wrong_lang += 1
        refs.append(norm(ref))
        hyps.append(norm(hyp))
        if not re.search(r"\d", hyp):
            refs_nd.append(norm(ref))
            hyps_nd.append(norm(hyp))
    audio_s = sum(len(a) for a, _, _ in clips) / 16000
    return {
        "model": Path(model).name, "role": role, "languages": "+".join(languages), "prompt": use_prompt, "n": len(clips),
        "WER": round(jiwer.wer(refs, hyps) * 100, 2), "CER": round(jiwer.cer(refs, hyps) * 100, 2),
        "WER_no_digits": round(jiwer.wer(refs_nd, hyps_nd) * 100, 2) if refs_nd else None,
        "ro_as_other_lang": wrong_lang, "seconds": round(dt, 1), "rtf": round(dt / audio_s, 3),
        "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:5]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/moldovan_corpus/train-00017.parquet")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", action="append")
    ap.add_argument("--out", default="training/results/wer.json")
    ap.add_argument("--role", choices=["main", "ro"], default="main",
                    help="ro = modelele date sunt testate ca specialist pentru română")
    a = ap.parse_args()
    models = a.model or ["models/mlx-whisper-large-v3", "models/mlx-whisper-large-v3-turbo"]
    clips = load_clips(a.shard, a.n)
    print(f"{len(clips)} fraze, {sum(len(c[0]) for c in clips) / 16000 / 60:.1f} min audio", file=sys.stderr)
    results = []
    for m in models:
        for langs in (["ro"], ["ro", "ru", "en"]):
            r = run(m, clips, langs, True, a.role)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "examples"}, ensure_ascii=False), file=sys.stderr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

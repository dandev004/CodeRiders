"""Test de code-switching RO / RU / EN, reproductibil și complet offline (vocile TTS locale ale macOS).

Construiește o „ședință” sintetică: fraze medicale în română, rusă și engleză, alternate, plus fraze cu schimbare
de limbă în mijloc (fără pauză). Măsoară, pe fiecare limbă, WER-ul și dacă limba a fost detectată corect.

    python -m training.eval_codeswitch --model models/mlx-whisper-turbo-moldovan --model models/mlx-whisper-large-v3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import asr  # noqa: E402
from app.config import get_config  # noqa: E402
from training.eval_reference import norm  # noqa: E402

VOICES = {"ro": "Ioana", "ru": "Milena", "en": "Eddy (English (US))"}
ITEMS = [
    [("ro", "Pacientul de pe patul patru primește meropenem și amikacină de mâine dimineață.")],
    [("ru", "Хорошо, я сделаю компьютерную томографию до пятницы и позвоню урологу.")],
    [("en", "Okay, let's schedule the follow-up for next week and check the troponin levels.")],
    [("ru", "Сатурация девяносто два процента, давление восемьдесят на сорок.")],
    [("ro", "Doamna doctor Rusu discută cu familia până joi despre nefrostomă.")],
    [("ru", "Нужно снизить дозу норадреналина и продолжить антибиотикотерапию.")],
    [("ro", "Deci am hotărât că pacientul"), ("ru", "идёт на операцию завтра утром.")],
    [("ro", "Hemoglobina a scăzut la optzeci și șase, am comandat o pungă de sânge.")],
    [("ro", "Facem ecocardiografie și"), ("en", "the ejection fraction is around forty percent.")],
    [("ru", "Посев крови положительный, клебсиелла чувствительна к меропенему.")],
]


def tts(lang: str, text: str, tmp: Path) -> np.ndarray:
    aiff = tmp / "x.aiff"
    wav = tmp / "x.wav"
    subprocess.run(["say", "-v", VOICES[lang], "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    a, _ = sf.read(wav, dtype="float32")
    return a


def build():
    tmp = Path(tempfile.mkdtemp())
    clips = []
    for parts in ITEMS:
        audio = np.concatenate([tts(l, t, tmp) for l, t in parts])
        clips.append((audio, parts))
    return clips


def run(model: str, clips, ro_model: str | None = None):
    import jiwer

    get_config()["asr"]["profile"] = "none"
    get_config()["asr"]["mlx_model"] = model
    gap = np.zeros(16000, dtype=np.float32)
    audio, spans, pos = [], [], 0
    for a, _ in clips:
        audio += [a, gap]
        spans.append((pos, pos + len(a)))
        pos += len(a) + len(gap)
    backend = asr.load_backend()
    spec = asr.MLXBackend(ro_model) if ro_model else None
    segs, _ = asr.decode_spans(np.concatenate(audio), spans, "medical", backend=backend, specialist=spec)
    backend.release()
    if spec:
        spec.release()
    by = {round(s.start * 16000): s for s in segs}
    per = {"ro": [[], []], "ru": [[], []], "en": [[], []], "mixed": [[], []]}
    lang_ok, rows = 0, []
    for (s, _), (_, parts) in zip(spans, clips):
        seg = by.get(s)
        hyp = seg.text if seg else ""
        ref = " ".join(t for _, t in parts)
        key = parts[0][0] if len(parts) == 1 else "mixed"
        per[key][0].append(norm(ref))
        per[key][1].append(norm(hyp))
        if len(parts) == 1 and seg and seg.lang == parts[0][0]:
            lang_ok += 1
        rows.append({"ref": ref, "hyp": hyp, "lang": seg.lang if seg else None})
    single = sum(1 for p in ITEMS if len(p) == 1)
    return {"model": Path(model).name + (f" + RO:{Path(ro_model).name}" if ro_model else ""),
            "WER": {k: round(jiwer.wer(r, h) * 100, 1) for k, (r, h) in per.items() if r},
            "language_detected_correctly": f"{lang_ok}/{single}", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--out", default="training/results/codeswitch.json")
    ap.add_argument("--ro-model", default=None)
    a = ap.parse_args()
    clips = build()
    res = []
    for m in a.model or [get_config().asr.mlx_model]:
        r = run(m, clips, a.ro_model)
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, ensure_ascii=False), file=sys.stderr)
        for row in r["rows"]:
            print(f"   [{row['lang']}] {row['hyp']}", file=sys.stderr)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

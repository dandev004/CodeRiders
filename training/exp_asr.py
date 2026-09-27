"""Experimente ASR pe ferestrele de referință Medpark (partea românească a pipeline-ului).

Variază: beam search, varianta de audio (brut / filtrat / curățat), promptul, contextul frazei anterioare,
segmentarea. Folosește faster-whisper (CTranslate2) — singurul backend local cu beam search.

    python -m training.exp_asr --model models/ct2-whisper-turbo-moldovan --beam 1 --beam 5 --audio asr --audio clean
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import asr, audio  # noqa: E402
from app.config import ROOT, get_config  # noqa: E402
from app.glossary import get_glossary  # noqa: E402
from app.medcorrect import phonetic_fix  # noqa: E402
from app.ronum import to_digits  # noqa: E402
from training.eval_reference import REF, norm, term_recall  # noqa: E402


def load_variants():
    src = ROOT / json.loads(REF.read_text())["audio"]
    work = ROOT / "data" / "reference" / "_audio"
    a, c, sr, _ = audio.prepare(src, work)
    raw_wav = work / "raw.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(raw_wav)], check=True)
    raw, _ = sf.read(raw_wav, dtype="float32")
    n = min(len(a), len(raw))
    out = {"asr": a[:n], "clean": c[:n], "raw": raw[:n]}
    for extra in ("wpe", "dfn", "dfnmix"):  # variante pregătite separat (dereverberare / DeepFilterNet)
        p = work / f"{extra}.wav"
        if p.exists():
            x, _ = sf.read(p, dtype="float32")
            out[extra] = np.pad(x, (0, max(0, n - len(x))))[:n]
    return out, c[:n]


def spans_for(vad_audio, windows, max_seg):
    get_config()["vad"]["max_segment_s"] = max_seg
    out = []
    for s0, e0 in asr.vad_segments(vad_audio):
        for w in windows:
            a0, b0 = max(s0, int(w["start"] * 16000)), min(e0, int(w["end"] * 16000))
            if b0 - a0 > 0.3 * 16000:
                out.append((a0, b0, windows.index(w)))
    return out


def main():
    import jiwer
    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ct2-whisper-turbo-moldovan")
    ap.add_argument("--beam", type=int, action="append")
    ap.add_argument("--audio", action="append")
    ap.add_argument("--prompt", action="append", help="none | glossary")
    ap.add_argument("--context", action="append", type=int, help="1 = fraza anterioară ca prompt")
    ap.add_argument("--maxseg", action="append", type=float)
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--no-repeat", type=int, default=0)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--out", default="training/results/exp_asr.jsonl")
    a = ap.parse_args()
    ref = json.loads(REF.read_text())
    windows, terms = ref["windows"], ref["medical_terms"]
    R = norm(" ".join(w["text"] for w in windows))
    variants, vad_audio = load_variants()
    model = WhisperModel(str(ROOT / a.model), device="cpu", compute_type=a.compute, cpu_threads=8)
    seed = get_glossary().whisper_prompt("ro", "medical")
    for beam, av, pr, ctx, ms in itertools.product(a.beam or [1], a.audio or ["asr"], a.prompt or ["glossary"],
                                                   a.context or [0], a.maxseg or [25]):
        spans = spans_for(vad_audio, windows, ms)
        x = variants[av]
        t0 = time.time()
        texts = [[] for _ in windows]
        prev = ""
        for s, e, wi in spans:
            p = seed if pr == "glossary" else ""
            if ctx and prev:
                p = (p + " " + prev[-200:]).strip()
            segs, _ = model.transcribe(x[s:e], language="ro", beam_size=beam, initial_prompt=p or None,
                                       condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
                                       temperature=[0.0, 0.2, 0.4], compression_ratio_threshold=2.2,
                                       no_repeat_ngram_size=a.no_repeat, repetition_penalty=a.rep_penalty)
            t = " ".join(z.text.strip() for z in segs).strip()
            if asr.compression_ratio(t) > 2.2:
                t = asr.collapse_repeats(t)
            t, _ = phonetic_fix(to_digits(t), "ro", 92)
            t, _ = get_glossary().correct(t, "ro", 88)
            texts[wi].append(t)
            prev = t
        hyp = " ".join(" ".join(tt) for tt in texts)
        H = norm(hyp)
        rec, miss = term_recall(terms, hyp)
        r = {"model": Path(a.model).name, "compute": a.compute, "no_repeat": a.no_repeat, "rep_pen": a.rep_penalty,
             "beam": beam, "audio": av, "prompt": pr, "context": ctx, "maxseg": ms,
             "WER": round(jiwer.wer(R, H) * 100, 1), "CER": round(jiwer.cer(R, H) * 100, 1),
             "MED": round(rec * 100, 1), "missing": miss, "seconds": round(time.time() - t0, 1), "hyp": hyp}
        print(json.dumps({k: v for k, v in r.items() if k != "hyp"}, ensure_ascii=False), flush=True)
        with open(ROOT / a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

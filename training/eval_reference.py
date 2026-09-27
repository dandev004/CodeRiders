"""Evaluare pe înregistrarea reală Medpark, față de transcrierea manuală a echipei (data/reference/medpark_ref.json).

Măsoară WER, CER și RECALL-ul termenilor medicali (câți dintre termenii din referință apar corect în transcriere).

    python -m training.eval_reference --model models/mlx-whisper-large-v3 --model models/mlx-whisper-turbo-moldovan
    python -m training.eval_reference --model ... --correct      # + corectarea medicală (app/medcorrect.py)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import asr, audio  # noqa: E402
from app.config import ROOT, get_config  # noqa: E402

REF = ROOT / "data" / "reference" / "medpark_ref.json"


def norm(t: str) -> str:
    from app.ronum import to_digits

    t = to_digits(t.lower().replace("ş", "ș").replace("ţ", "ț"))
    t = re.sub(r"\[[^\]]*\]", " ", t)              # [neinteligibil]
    t = re.sub(r"(\d)[.,](\d)", r"\1 virgulă \2", t)  # 0.22 / 0,22 -> aceeași formă
    t = t.replace("-", " ").replace("–", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def term_recall(terms: list[str], hyp: str) -> tuple[float, list[str]]:
    h = " " + norm(hyp) + " "
    missing = [t for t in terms if " " + norm(t) not in h]
    return 1 - len(missing) / len(terms), missing


def prepare_audio():
    src = ROOT / json.loads(REF.read_text())["audio"]
    work = ROOT / "data" / "reference" / "_audio"
    a, c, sr, _ = audio.prepare(src, work)
    return a, c


def transcribe_windows(a, c, windows, model: str, correct: bool, ro_model: str | None = None):
    cfg = get_config()
    cfg["asr"]["profile"] = "none"  # evaluăm exact modelele date în linia de comandă
    cfg["asr"]["mlx_model"] = model
    # segmentele VAD tăiate exact la marginile ferestrelor de referință (altfel cuvintele din afara ferestrei
    # sunt numărate ca erori)
    spans = []
    for s0, e0 in asr.vad_segments(c):
        for w in windows:
            a0, b0 = max(s0, int(w["start"] * 16000)), min(e0, int(w["end"] * 16000))
            if b0 - a0 > 0.3 * 16000:
                spans.append((a0, b0))
    backend = asr.load_backend()
    spec = asr.MLXBackend(str(ROOT / ro_model)) if ro_model else None
    t0 = time.time()
    segs, st = asr.decode_spans(a, spans, "medical", backend=backend, specialist=spec)
    backend.release()
    if spec:
        spec.release()
    if correct:
        from app import medcorrect
        segs_d = [s.to_dict() for s in segs]
        segs_d, cst = medcorrect.correct_segments(segs_d, "medical")
        texts = [(s["start"], s["end"], s["text"]) for s in segs_d]
    else:
        texts = [(s.start, s.end, s.text) for s in segs]
    out = []
    for w in windows:
        out.append(" ".join(t for s, e, t in texts if s < w["end"] and e > w["start"]))
    return out, round(time.time() - t0, 1), st


def main():
    import jiwer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--out", default="training/results/reference.json")
    ap.add_argument("--set", action="append", default=[], help="suprascrie config: asr.x=valoare / vad.y=valoare")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ro-model", default=None, help="model specialist pentru română (arhitectura cu două modele)")
    a = ap.parse_args()
    import yaml
    for kv in a.set:
        k, v = kv.split("=", 1)
        sec, key = k.split(".", 1)
        get_config()[sec][key] = yaml.safe_load(v)
    ref = json.loads(REF.read_text())
    windows, terms = ref["windows"], ref["medical_terms"]
    au, cl = prepare_audio()
    results = []
    for m in a.model or [get_config().asr.mlx_model]:
        for corr in ([False, True] if a.correct else [False]):
            hyps, secs, st = transcribe_windows(au, cl, windows, m, corr, a.ro_model)
            R = norm(" ".join(w["text"] for w in windows))
            H = norm(" ".join(hyps))
            rec, missing = term_recall(terms, " ".join(hyps))
            r = {"model": Path(m).name + (f" + RO:{Path(a.ro_model).name}" if a.ro_model else ""), "tag": a.tag, "set": a.set, "medical_correction": corr, "WER": round(jiwer.wer(R, H) * 100, 1),
                 "CER": round(jiwer.cer(R, H) * 100, 1), "medical_term_recall": round(rec * 100, 1),
                 "missing_terms": missing, "seconds": secs, "lang_seconds": st.get("lang_seconds"), "hyp": hyps}
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "hyp"}, ensure_ascii=False), file=sys.stderr)
            for h in hyps:
                print("   HYP:", h[:600], file=sys.stderr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

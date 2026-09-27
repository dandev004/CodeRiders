"""Test de generalizare: replici medicale cu code-switching, rostite de VORBITORI NEVĂZUȚI la antrenare.

Setul (data/tts_test/) e generat cu training/make_tts_xtts.py --test-voices din 120 de replici scoase din antrenare
(data/tts_xtts/holdout.json), cu vocile profesorilor din data/moldovan_test/ — care nu apar nicăieri în antrenare.
Fiecare clip trece prin TOT sistemul (specialist moldovenesc + multilingv + alegerea limbii + numerale + corecturi),
în două condiții: audio curat și „sală” (reverberație, voci de fundal nevăzute, zgomot, bandă de telefon).

Metrici:
  - WER latin (RO + EN) și WER chirilic (RU): rusa scrisă cu litere latine apare imediat ca eroare pe chirilic;
  - numere exacte: dozele, paturile, tensiunile din replicile românești (după conversia în cifre);
  - termeni critici: medicamente/proceduri (app/medcorrect.is_critical) regăsiți exact.

    python -m training.eval_synth --ro-model models/mlx-whisper-turbo-moldovan-med2 --out training/results/synth_med2.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CYR = re.compile(r"[а-яё]")


def words(t: str) -> list[str]:
    t = t.lower().replace("ş", "ș").replace("ţ", "ț").replace("ё", "е")
    return re.findall(r"[\wăâîșț]+(?:[.,]\d+)?", t)


def proj(ws: list[str], cyr: bool) -> str:
    return " ".join(w for w in ws if bool(CYR.search(w)) == cyr and not w.isdigit())


def main() -> None:
    import jiwer

    from app import asr
    from app.config import get_config
    from app.medcorrect import is_critical
    from app.ronum import to_digits
    from training.eval_wer import load_clips
    from training.finetune_whisper_mlx import augment

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "tts_test"))
    ap.add_argument("--ro-model", default=None, help="specialistul pentru română (implicit: cel din config)")
    ap.add_argument("--out", default=str(ROOT / "training" / "results" / "synth.json"))
    a = ap.parse_args()
    d = Path(a.dir)
    meta = [json.loads(line) for line in open(d / "metadata.jsonl", encoding="utf-8")]
    babble = []
    for s in sorted((ROOT / "data" / "moldovan_test").glob("train-*.parquet")):
        babble += [c[0] for c in load_clips(str(s), 25, seed=5, max_dur=20)]

    cfg = get_config()
    cfg["asr"]["profile"] = ""
    if a.ro_model:
        cfg["asr"]["ro_mlx_model"] = a.ro_model
    backend, spec = asr.load_backend(), asr.load_specialist()
    results = {}
    for cond in ("curat", "sala"):
        audio, spans, pos = [], [], 0
        gap = np.zeros(8000, np.float32)
        for i, m in enumerate(meta):
            x, _ = sf.read(d / m["audio"], dtype="float32")
            if cond == "sala":
                x = augment(x, np.random.default_rng(1000 + i), babble, 1.0)
            audio += [x, gap]
            spans.append((pos, pos + len(x)))
            pos += len(x) + len(gap)
        t0 = time.time()
        segs, _ = asr.decode_spans(np.concatenate(audio), spans, "medical", backend=backend, specialist=spec)
        by_start = {round(s.start * 16000): s for s in segs}
        R_lat, H_lat, R_cyr, H_cyr = [], [], [], []
        n_num = ok_num = n_term = ok_term = 0
        rows = []
        for (s0, _), m in zip(spans, meta):
            seg = by_start.get(s0)
            hyp = seg.text if seg else ""
            ref = " ".join(to_digits(p["text"]) if p["lang"] == "ro" else p["text"] for p in m["parts"])
            rw, hw = words(ref), words(hyp)
            if proj(rw, False):
                R_lat.append(proj(rw, False))
                H_lat.append(proj(hw, False))
            if proj(rw, True):
                R_cyr.append(proj(rw, True))
                H_cyr.append(proj(hw, True))
            hs = set(hw)
            for p in m["parts"]:
                if p["lang"] != "ro":
                    continue
                for num in re.findall(r"\d+(?:,\d+)?", to_digits(p["text"])):
                    n_num += 1
                    ok_num += num in hyp
            for w in set(rw):
                if not w.isdigit() and is_critical(w):
                    n_term += 1
                    ok_term += w in hs
            rows.append({"ref": ref, "hyp": hyp, "langs": "+".join(p["lang"] for p in m["parts"])})
        results[cond] = {
            "WER_latin": round(jiwer.wer(R_lat, H_lat) * 100, 1),
            "WER_chirilic": round(jiwer.wer(R_cyr, H_cyr) * 100, 1) if R_cyr else None,
            "numere_exacte": f"{ok_num}/{n_num}", "numere_pct": round(100 * ok_num / max(1, n_num), 1),
            "termeni_critici": f"{ok_term}/{n_term}", "termeni_pct": round(100 * ok_term / max(1, n_term), 1),
            "seconds": round(time.time() - t0, 1), "examples": rows[:15],
        }
        print(cond, json.dumps({k: v for k, v in results[cond].items() if k != "examples"}, ensure_ascii=False),
              file=sys.stderr, flush=True)
    backend.release()
    if spec:
        spec.release()
    out = {"ro_model": a.ro_model or str(cfg.asr.get("ro_mlx_model")), "n": len(meta), **results}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

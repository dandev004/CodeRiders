"""Verificare dus-întors a vorbirii sintetice (training/make_tts_xtts.py) înainte de antrenare.

Sinteza vocală greșește uneori: sare cuvinte, repetă, „bâlbâie” sau produce sunete fără sens. Antrenat pe așa ceva,
ASR-ul ar învăța să scrie text care nu a fost rostit (halucinații) — inacceptabil pentru un proces-verbal medical.

Fiecare bucată (ro/ru/en) e tăiată după pozițiile salvate la sinteză și transcrisă cu Whisper multilingv, forțat pe
limba ei. Comparăm literele (fără cifre/numerale, care se scriu diferit: „optzeci” vs „80”). Pragul e LARG: vrem să
eliminăm doar sinteza ratată, nu frazele cu termeni medicali grei (tocmai pe acelea le vrem în antrenare).

    python -m training.filter_tts --max-cer 0.35
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NUMW = r"\b(zero|unu|una|doi|două|trei|patru|cinci|șase|șapte|opt|nouă|zece|\w+sprezece|\w+șpe|\w+zeci|sută|sute|" \
       r"mie|mii|virgulă|ноль|один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|\w+надцать|" \
       r"\w+дцать|сорок|девяносто|сто|\w+сот|тысяч\w*|zero|one|two|three|four|five|six|seven|eight|nine|ten|" \
       r"\w+teen|\w+ty|hundred|thousand|percent|la sută|процент\w*|%)\b"


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t.lower()).replace("ş", "ș").replace("ţ", "ț").replace("ё", "е")
    t = re.sub(NUMW, " ", t)
    t = re.sub(r"[\d\W_]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    import jiwer
    import mlx_whisper

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "tts_xtts"))
    ap.add_argument("--model", default=str(ROOT / "models" / "mlx-whisper-large-v3-turbo"))
    ap.add_argument("--max-cer", type=float, default=0.35)
    a = ap.parse_args()
    d = Path(a.dir)
    out_p = d / "filter.jsonl"
    done = {json.loads(line)["audio"] for line in open(out_p, encoding="utf-8")} if out_p.exists() else set()
    meta = [json.loads(line) for line in open(d / "metadata.jsonl", encoding="utf-8")]
    t0, kept, n = time.time(), 0, 0
    with open(out_p, "a", encoding="utf-8") as f:
        for m in meta:
            if m["audio"] in done:
                continue
            x, _ = sf.read(d / m["audio"], dtype="float32")
            parts = []
            for p, (s, e) in zip(m["parts"], m.get("bounds") or [[0, len(x)]]):
                seg = x[max(0, s - 800):min(len(x), e + 800)]
                dur = len(seg) / 16000
                hyp = mlx_whisper.transcribe(seg, path_or_hf_repo=a.model, language=p["lang"], temperature=0.0,
                                             condition_on_previous_text=False, verbose=None)["text"]
                r, h = norm(p["text"]), norm(hyp)
                cer = jiwer.cer(r, h) if r else (0.0 if not h else 1.0)
                cps = len(p["text"]) / max(dur, 0.1)
                parts.append({"lang": p["lang"], "cer": round(cer, 3), "cps": round(cps, 1), "hyp": hyp.strip()})
            ok = all(q["cer"] <= a.max_cer and 5 <= q["cps"] <= 30 for q in parts)
            f.write(json.dumps({"audio": m["audio"], "keep": ok, "parts": parts}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            kept += ok
            if n % 100 == 0:
                print(f"{n} verificate, {kept} păstrate ({kept / n:.0%}), {(time.time() - t0) / 60:.1f} min",
                      file=sys.stderr, flush=True)
    rows = [json.loads(line) for line in open(out_p, encoding="utf-8")]
    by = {}
    for r_ in rows:
        for q in r_["parts"]:
            by.setdefault(q["lang"], []).append(q["cer"])
    print(json.dumps({"total": len(rows), "keep": sum(r_["keep"] for r_ in rows),
                      "median_cer": {k: round(float(np.median(v)), 3) for k, v in by.items()}}), file=sys.stderr)


if __name__ == "__main__":
    main()

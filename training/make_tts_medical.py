"""Date de antrenare medicale sintetice, generate LOCAL: fraze medicale românești citite de vocea TTS a macOS,
apoi „murdărite” ca să semene cu o ședință reală (zgomot de fond din pauzele înregistrărilor + reverberație).

Surse de text (fără audio):
  - SiMoNERo / MoNERo (CC BY-SA 4.0): fraze medicale reale, cu termeni ANAT/CHEM/DISO/PROC
  - glosarul medical al proiectului, în șabloane de vorbire de consiliu („pacientul de pe patul X primește ...”)

NU folosește transcrierea de referință Medpark (aceea rămâne exclusiv pentru evaluare).

    python -m training.make_tts_medical --n 2500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "tts_medical"
TEMPLATES = [
    "pacientul de pe patul {n} primește {drug} de mâine dimineață",
    "am scăzut doza de {drug} la {d}",
    "pacienta are {diso} și trebuie consult la {spec}",
    "facem {proc} și vedem mai departe tactica",
    "la pacientul de pe patul {n} s-a confirmat {diso}",
    "noi suntem pe {drug} și {drug2}, așteptăm antibiograma",
    "dacă nu se stabilizează, discutăm cu {spec} despre {proc}",
    "dânsul e după {proc}, hemodinamic stabil, fără suport vasopresor",
    "am comandat {proc} pentru mâine și revenim cu rezultatul",
    "se menține {diso}, continuăm {drug}",
]
SPECS = ["cardiolog", "urolog", "chirurg", "neurolog", "oncolog", "infecționist", "nefrolog", "anestezist"]


def simonero_sentences() -> list[str]:
    sents = []
    for f in sorted((ROOT / "data" / "medical_text").glob("simonero-*.conllu")):
        for line in open(f, encoding="utf-8"):
            if line.startswith("# text = "):
                t = re.sub(r"\s*\([^)]*\)", "", line[9:].strip())  # trimiteri „(1)”, abrevieri „(mec)” — nu se rostesc
                t = re.sub(r"\s+([,.;:])", r"\1", t).strip()
                n = len(t.split())
                if 5 <= n <= 22 and not re.search(r"[\[\]{}=<>|/\\]|http", t):
                    sents.append(t)
    return sents


def template_sentences(k: int, rng: random.Random) -> list[str]:
    import yaml

    g = yaml.safe_load(open(ROOT / "config" / "medical_glossary.yaml", encoding="utf-8"))
    cats = g["categories"]
    drugs = cats["pharmacology"]["ro"] + [x for x in cats["infectious"]["ro"] if x[0].islower() and x.endswith(("em", "ină", "ol"))]
    diso = cats["cardiology"]["ro"] + cats["intensive_care"]["ro"] + cats["infectious"]["ro"]
    proc = cats["surgery_urology"]["ro"] + cats["laboratory_imaging"]["ro"] + ["ecocardiografie", "coronarografie",
                                                                              "hemodializă", "intubație"]
    out = []
    for _ in range(k):
        t = rng.choice(TEMPLATES)
        out.append(t.format(n=rng.choice(["unu", "doi", "trei", "patru", "cinci", "șase", "șapte", "opt", "nouă"]),
                            drug=rng.choice(drugs), drug2=rng.choice(drugs), d=rng.choice(["zero virgulă zero opt",
                            "jumătate", "o sută cincizeci", "cinci miligrame"]), diso=rng.choice(diso),
                            proc=rng.choice(proc), spec=rng.choice(SPECS)))
    return out


def noise_bank() -> np.ndarray:
    """Zgomot de fond real: pauzele (fără vorbire) din înregistrările din data/samples."""
    from app import asr

    chunks = []
    for f in sorted((ROOT / "data" / "samples").glob("*16k*.wav")):
        a, _ = sf.read(f, dtype="float32")
        sp = asr.vad_segments(a)
        prev = 0
        for s, e in sp:
            if s - prev > 8000:
                chunks.append(a[prev + 1600:s - 1600])
            prev = e
    return np.concatenate([c for c in chunks if len(c) > 1600]) if chunks else np.zeros(16000, np.float32)


def room(x: np.ndarray, rng: random.Random) -> np.ndarray:
    rt60 = rng.uniform(0.2, 0.7)
    n = int(16000 * rt60)
    t = np.arange(n) / 16000
    rir = np.random.default_rng(rng.randint(0, 10 ** 6)).standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    rir /= np.abs(rir).sum() ** 0.5
    y = np.convolve(x, rir)[:len(x)]
    return y / (np.abs(y).max() + 1e-9) * 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--templates", type=int, default=600)
    a = ap.parse_args()
    rng = random.Random(7)
    sents = simonero_sentences()
    rng.shuffle(sents)
    texts = sents[:max(0, a.n - a.templates)] + template_sentences(a.templates, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    noise = noise_bank()
    tmp = Path(tempfile.mkdtemp())

    def job(i_t):
        i, text = i_t
        wav = OUT / f"{i:05d}.wav"
        if wav.exists():
            return i, text
        aiff = tmp / f"{i}.aiff"
        rate = rng.randint(165, 215)
        subprocess.run(["say", "-v", "Ioana", "-r", str(rate), "-o", str(aiff), text], check=True)
        raw = tmp / f"{i}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(raw)], check=True)
        x, _ = sf.read(raw, dtype="float32")
        r = random.Random(i)
        if r.random() < 0.8:
            x = room(x, r)
        if r.random() < 0.85 and len(noise) > len(x):
            o = r.randint(0, len(noise) - len(x) - 1)
            nz = noise[o:o + len(x)]
            snr = r.uniform(3, 20)
            ps, pn = np.mean(x ** 2) + 1e-9, np.mean(nz ** 2) + 1e-9
            x = x + nz * np.sqrt(ps / (pn * 10 ** (snr / 10)))
        x = x / (np.abs(x).max() + 1e-9) * r.uniform(0.3, 0.9)
        sf.write(wav, x.astype(np.float32), 16000, subtype="PCM_16")
        aiff.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        return i, text

    meta = []
    with ThreadPoolExecutor(6) as ex:
        for k, (i, text) in enumerate(ex.map(job, enumerate(texts))):
            meta.append({"audio": f"{i:05d}.wav", "text": text.lower()})
            if k % 250 == 0:
                print(f"{k}/{len(texts)}", file=sys.stderr, flush=True)
    (OUT / "metadata.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in meta), encoding="utf-8")
    print(f"{len(meta)} fraze sintetice -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()

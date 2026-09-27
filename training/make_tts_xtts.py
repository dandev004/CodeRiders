"""Vorbire sintetică medicală cu VOCI MOLDOVENEȘTI REALE (rulează în .venv-tts, separat de aplicație).

Model: FraPiz/xtts-v2-moldovan-romanian — XTTS-v2 adaptat pe 44 h de română moldovenească (licență Coqui CPML,
doar pentru cercetare/prototip). Vocea e clonată din clipuri scurte ale vorbitorilor din corpusul moldovenesc,
deci fiecare replică medicală e rostită de alt vorbitor real, cu accentul lui.

Code-switching: fiecare bucată a replicii (ro/ru/en) e sintetizată în limba ei, cu ACEEAȘI voce, apoi lipită —
exact situația din ședințe („...și dânsul, ну короче, nu mai urinează”).

Ieșire: audio CURAT 16 kHz în data/tts_xtts/ + metadata.jsonl. Zgomotul, reverberația și vocile de fundal se
adaugă aleator la antrenare (training/finetune_whisper_mlx.py), deci fiecare epocă vede altă „sală”.

    .venv-tts/bin/python training/make_tts_xtts.py --n-simonero 800
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
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xtts-moldovan"        # româna (moldovenească)
BASE = ROOT / "models" / "xtts-v2-base"          # rusa și engleza: modelul moldovenesc le-a uitat (măsurat)
OUT = ROOT / "data" / "tts_xtts"
REFS = ROOT / "data" / "xtts_refs"
EVAL_SHARD = "train-00017"  # rămâne pentru evaluare (training/eval_wer.py) — nicio voce de acolo


def build_refs(per_speaker: int, rng: random.Random, corpus: Path | None = None, refs: Path = REFS,
               exclude: set[str] | None = None) -> list[Path]:
    """Clipuri de referință (6-11 s) pentru fiecare vorbitor din corpus, fără shard-ul de evaluare."""
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly

    refs.mkdir(parents=True, exist_ok=True)
    have = sorted(refs.glob("*.wav"))
    if have:
        return have
    for shard in sorted((corpus or ROOT / "data" / "moldovan_corpus").glob("train-*.parquet")):
        if EVAL_SHARD in shard.name:
            continue
        t = pq.read_table(shard, columns=["audio", "duration", "audio_filename"])
        by_spk: dict[str, list[int]] = {}
        for i, (d, f) in enumerate(zip(t.column("duration").to_pylist(), t.column("audio_filename").to_pylist())):
            m = re.search(r"teacher-([a-z0-9-]+?)__", f)
            if m and 6.0 <= d <= 11.0:
                by_spk.setdefault(m.group(1), []).append(i)
        for spk, rows in by_spk.items():
            if exclude and spk in exclude:
                continue
            rng.shuffle(rows)
            for k, r in enumerate(rows[:per_speaker]):
                a, sr = sf.read(io.BytesIO(t.column("audio")[r].as_py()["bytes"]), dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                if sr != 16000:
                    a = resample_poly(a, 16000, sr).astype(np.float32)
                sf.write(refs / f"{spk}_{k}.wav", a, 16000)
    return sorted(refs.glob("*.wav"))


def simonero(n: int, rng: random.Random) -> list[dict]:
    sents = []
    for f in sorted((ROOT / "data" / "medical_text").glob("simonero-*.conllu")):
        for line in open(f, encoding="utf-8"):
            if line.startswith("# text = "):
                t = re.sub(r"\s*\([^)]*\)", "", line[9:].strip())
                t = re.sub(r"\s+([,.;:])", r"\1", t).strip()
                if 5 <= len(t.split()) <= 25 and not re.search(r"[\[\]{}=<>|/\\%]|http|\d", t):
                    sents.append(t)
    rng.shuffle(sents)
    return [{"type": "medical", "spec": "simonero", "parts": [{"lang": "ro", "text": s}]} for s in sents[:n]]


def load_model(device: str, mdir: Path = MODEL, ckpt: str = "best_model.pth"):
    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    def _sf_load(path, **_):  # torchaudio fără ffmpeg/torchcodec
        d, sr = sf.read(path, dtype="float32")
        d = d[np.newaxis, :] if d.ndim == 1 else d.T
        return torch.from_numpy(d), sr
    torchaudio.load = _sf_load

    cfg = XttsConfig()
    cfg.load_json(str(mdir / "config.json"))
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=str(mdir / ckpt), vocab_path=str(mdir / "vocab.json"),
                      use_deepspeed=False)
    m.to(device)
    m.eval()
    return m


def patch_tokenizer() -> None:
    """Româna nu e în lista de limbi a XTTS-v2: aceleași modificări ca în generate_tts.py al modelului FraPiz."""
    import TTS.tts.layers.xtts.tokenizer as tk

    tk._abbreviations.setdefault("ro", [])
    if hasattr(tk, "_symbols_multilingual"):
        tk._symbols_multilingual.setdefault("ro", [])
    if hasattr(tk, "_ordinal_re"):
        tk._ordinal_re.setdefault("ro", re.compile(r"([0-9]+)\.(?=\s|$)"))
    orig = tk.VoiceBpeTokenizer.preprocess_text

    def pre(self, txt, lang):
        if lang == "ro":
            txt = txt.translate(str.maketrans("şţŞŢ", "șțȘȚ"))
            return tk.multilingual_cleaners(txt, "ro") if hasattr(tk, "multilingual_cleaners") else txt
        return orig(self, txt, lang)
    tk.VoiceBpeTokenizer.preprocess_text = pre
    if hasattr(tk.VoiceBpeTokenizer, "char_limits"):
        tk.VoiceBpeTokenizer.char_limits = {**getattr(tk.VoiceBpeTokenizer, "char_limits"), "ro": 250}


def main() -> None:
    import torch
    from scipy.signal import resample_poly

    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(ROOT / "data" / "medical_text" / "spoken_medical.jsonl"))
    ap.add_argument("--n-simonero", type=int, default=800)
    ap.add_argument("--per-speaker", type=int, default=3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--test-voices", action="store_true", help="voci nevăzute la antrenare (data/moldovan_test)")
    a = ap.parse_args()
    out = Path(a.out)
    rng = random.Random(11)

    if a.test_voices:  # vorbitori care NU apar nicăieri în antrenare (nici ca voce TTS, nici ca replay)
        seen = {r.stem.rsplit("_", 1)[0] for r in REFS.glob("*.wav")}
        refs = build_refs(a.per_speaker, rng, ROOT / "data" / "moldovan_test", ROOT / "data" / "xtts_refs_test", seen)
    else:
        refs = build_refs(a.per_speaker, rng)
    print(f"{len(refs)} clipuri de referință, {len({r.stem.rsplit('_', 1)[0] for r in refs})} vorbitori",
          file=sys.stderr)
    items = [json.loads(line) for line in open(a.texts, encoding="utf-8")] + simonero(a.n_simonero, rng)
    if a.limit:
        items = items[:a.limit]
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    done = {json.loads(line)["audio"] for line in open(meta_path, encoding="utf-8")} if meta_path.exists() else set()

    patch_tokenizer()
    models = {"ro": load_model(a.device)}
    models["ru"] = models["en"] = load_model(a.device, BASE, "model.pth")
    latents: dict[tuple, tuple] = {}
    t0, n_new, audio_s = time.time(), 0, 0.0
    with open(meta_path, "a", encoding="utf-8") as meta:
        for i, it in enumerate(items):
            name = f"{i:05d}.wav"
            if name in done:
                continue
            r = random.Random(i)
            ref = r.choice(refs)
            speed = r.uniform(0.95, 1.2)  # la ședințe se vorbește repede
            pieces, bounds, pos = [], [], 0
            try:
                for p in it["parts"]:
                    model = models[p["lang"]]
                    key = (id(model), ref)
                    if key not in latents:  # aceeași voce (același vorbitor), latente calculate de fiecare model
                        with torch.no_grad():
                            latents[key] = model.get_conditioning_latents(audio_path=[str(ref)], gpt_cond_len=6)
                    gpt, spk = latents[key]
                    with torch.no_grad():
                        o = model.inference(p["text"], p["lang"], gpt, spk, temperature=0.7, top_p=0.8, top_k=40,
                                            length_penalty=1.0, repetition_penalty=5.0, speed=speed)
                    w = np.asarray(o["wav"].cpu() if hasattr(o["wav"], "cpu") else o["wav"], dtype=np.float32).squeeze()
                    gap = np.zeros(int(24000 * r.uniform(0.03, 0.2)), np.float32)
                    bounds.append([round(pos * 2 / 3), round((pos + len(w)) * 2 / 3)])  # în eșantioane 16 kHz
                    pos += len(w) + len(gap)
                    pieces += [w, gap]
            except Exception as e:  # noqa: BLE001
                print(f"{i}: eroare {e}", file=sys.stderr)
                continue
            x = resample_poly(np.concatenate(pieces), 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz
            if len(x) > 16000 * 29:
                continue  # peste fereastra Whisper de 30 s
            sf.write(out / name, x / (np.abs(x).max() + 1e-9) * 0.9, 16000, subtype="PCM_16")
            text = " ".join(p["text"] for p in it["parts"])
            meta.write(json.dumps({"audio": name, "text": text, "parts": it["parts"], "bounds": bounds, "speaker": ref.stem,
                                   "type": it.get("type"), "spec": it.get("spec")}, ensure_ascii=False) + "\n")
            meta.flush()
            n_new += 1
            audio_s += len(x) / 16000
            # cache-ul MPS reține buffere pentru fiecare lungime de secvență: fără golire procesul ajunge la 20 GB
            # în 7 minute și sistemul intră în swap (măsurat)
            del pieces, o
            if a.device == "mps":
                torch.mps.empty_cache()
            if n_new % 25 == 0 or n_new <= 3:
                el = time.time() - t0
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
                mps = torch.mps.driver_allocated_memory() / 2**30 if a.device == "mps" else 0
                print(f"{n_new} noi ({i + 1}/{len(items)}), {audio_s / 60:.1f} min audio, RTF {el / audio_s:.2f}, "
                      f"{el / 60:.1f} min, RAM vârf {rss:.1f}G, MPS {mps:.1f}G", file=sys.stderr, flush=True)
    print(f"gata: {n_new} fraze noi -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

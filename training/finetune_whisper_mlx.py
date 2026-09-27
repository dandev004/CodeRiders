"""Adaptarea Whisper-ului moldovenesc la ședințe medicale reale — antrenare locală pe GPU-ul Mac-ului (MLX).

Față de training/finetune_decoder_mlx.py (doar decoderul, voce TTS macOS):
  - se antrenează și ULTIMELE straturi ale encoderului („urechea”): acustica sălii, vocile suprapuse,
    pronunția termenilor medicali — adaptarea medicală a Whisper se face în principal în encoder;
  - date: vorbire medicală sintetică rostită de VOCI MOLDOVENEȘTI REALE, cu code-switching RO/RU/EN în frază
    (training/make_tts_xtts.py), filtrată prin verificare dus-întors (training/filter_tts.py) + vorbire reală
    din corpusul moldovenesc (replay, ca să nu se piardă dialectul și vorbirea spontană);
  - fiecare clip trece, la fiecare pas, printr-o „sală” aleatoare: reverberație, 1-3 voci reale care vorbesc pe
    fundal, zgomot, bandă de telefon, volum variabil — exact condițiile unui telefon pus pe masă la consiliu.

Encoderul înghețat rulează în fp16 și e evaluat separat (fără graf); se antrenează în fp32 doar ce e deblocat.
Pe 16 GB unificați: batch 4 x acumulare 4, cache MLX limitat (altfel swap și viteză de ~4x mai mică).

    python -m training.finetune_whisper_mlx --out models/mlx-whisper-turbo-moldovan-med2
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SR = 16000
EVAL_SHARD = "train-00017"  # nu se antrenează pe el (training/eval_wer.py)


# ------------------------------------------------------------------------------------------------
# Augmentare: „sala de consiliu”
# ------------------------------------------------------------------------------------------------

def rir(rng: np.random.Generator, rt60: float, drr_db: float) -> np.ndarray:
    n = int(SR * rt60)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.003 * SR)] = 0
    tail /= np.sqrt((tail ** 2).sum()) + 1e-9
    h = tail * 10 ** (-drr_db / 20)
    h[0] = 1.0
    return h.astype(np.float32)


def at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps, pn = np.mean(sig ** 2) + 1e-9, np.mean(noise ** 2) + 1e-9
    return noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def colored_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32)
    kind = rng.integers(3)
    if kind == 1:  # roz-maro: ventilație, aparatură
        w = np.cumsum(w)
        w -= np.convolve(w, np.ones(400) / 400, mode="same")
    elif kind == 2:  # hum de rețea + armonici
        t = np.arange(n) / SR
        w = w * 0.2 + sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6)) / k for k in (1, 2, 3))
    return w.astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator, babble: list[np.ndarray], strength: float) -> np.ndarray:
    from scipy.signal import butter, fftconvolve, sosfilt

    y = x.astype(np.float32)
    if rng.random() < 0.75 * strength:  # vorbitorul la 1-4 m de telefon
        y = fftconvolve(y, rir(rng, rng.uniform(0.25, 0.9), rng.uniform(-3, 12)))[: len(y)].astype(np.float32)
    if babble and rng.random() < 0.55 * strength:  # alți medici vorbesc în același timp / pe fundal
        for _ in range(rng.integers(1, 4)):
            b = babble[rng.integers(len(babble))]
            b = fftconvolve(b, rir(rng, rng.uniform(0.3, 0.9), rng.uniform(-6, 3)))[: len(b)]
            seg = np.zeros_like(y)
            o = rng.integers(-len(b) // 2, len(y))
            s0, s1 = max(0, o), min(len(y), o + len(b))
            if s1 > s0:
                seg[s0:s1] = b[s0 - o:s1 - o]
                y = y + at_snr(y, seg, rng.uniform(4, 20))
    if rng.random() < 0.6 * strength:
        y = y + at_snr(y, colored_noise(rng, len(y)), rng.uniform(8, 35))
    if rng.random() < 0.4 * strength:  # microfon de telefon / compresie
        lo, hi = rng.uniform(80, 300), rng.uniform(3400, 7500)
        y = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y).astype(np.float32)
    y = y / (np.abs(y).max() + 1e-9) * rng.uniform(0.05, 0.95)
    if rng.random() < 0.05:
        y = np.clip(y * 2.5, -0.9, 0.9)
    return y.astype(np.float32)


# ------------------------------------------------------------------------------------------------
# Date
# ------------------------------------------------------------------------------------------------

def load_data(a, rng: random.Random):
    from training.eval_wer import load_clips

    items = []  # (sursă, audio|cale, text, limbă)
    xdir = Path(a.xtts_dir)
    meta = xdir / "metadata.jsonl"
    keep = None
    filt = xdir / "filter.jsonl"
    if filt.exists():
        keep = {j["audio"] for j in map(json.loads, open(filt, encoding="utf-8")) if j["keep"]}
    hold_p = xdir / "holdout.json"
    hold = set(json.load(open(hold_p, encoding="utf-8"))["audio"]) if hold_p.exists() else set()
    for m in map(json.loads, open(meta, encoding="utf-8")):
        if (keep is not None and m["audio"] not in keep) or m["audio"] in hold:
            continue
        langs = {p["lang"] for p in m["parts"]}
        lang = langs.pop() if len(langs) == 1 else "ro"
        items.append(("xtts", xdir / m["audio"], m["text"], lang))
    if a.n_mac:
        mac = [json.loads(line) for line in open(ROOT / "data" / "tts_medical" / "metadata.jsonl", encoding="utf-8")]
        rng.shuffle(mac)
        items += [("mac", ROOT / "data" / "tts_medical" / m["audio"], m["text"], "ro") for m in mac[:a.n_mac]]
    shards = [s for s in sorted((ROOT / "data" / "moldovan_corpus").glob("train-*.parquet")) if EVAL_SHARD not in s.name]
    real, babble = [], []
    per = max(1, a.n_real // max(1, len(shards)))
    for s in shards:
        clips = load_clips(str(s), per + a.n_babble // len(shards), seed=3, max_dur=28)
        real += [("real", c[0], c[1], "ro") for c in clips[:per]]
        babble += [c[0] for c in clips[per:]]
    return items + real, babble


# ------------------------------------------------------------------------------------------------

def main() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.load_models import load_model
    from mlx_whisper.tokenizer import get_tokenizer

    from app.glossary import get_glossary

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/mlx-whisper-turbo-moldovan")
    ap.add_argument("--out", default="models/mlx-whisper-turbo-moldovan-med2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5, help="decoder")
    ap.add_argument("--lr-enc", type=float, default=4e-6, help="straturile deblocate ale encoderului")
    ap.add_argument("--enc-layers", type=int, default=4, help="câte straturi finale ale encoderului se antrenează")
    ap.add_argument("--n-real", type=int, default=2500)
    ap.add_argument("--n-babble", type=int, default=600)
    ap.add_argument("--n-mac", type=int, default=0, help="fraze cu vocea TTS macOS (runda anterioară)")
    ap.add_argument("--aug", type=float, default=1.0, help="intensitatea augmentării (0 = audio curat)")
    ap.add_argument("--clean-prob", type=float, default=0.15, help="fracțiunea de clipuri lăsate curate")
    ap.add_argument("--prompt-prob", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--xtts-dir", default=str(ROOT / "data" / "tts_xtts"))
    ap.add_argument("--mem-limit-gb", type=float, default=9.0)
    a = ap.parse_args()
    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    mx.set_memory_limit(int(a.mem_limit_gb * 2**30))
    mx.set_cache_limit(1 * 2**30)

    model = load_model(str(ROOT / a.base), dtype=mx.float16)
    enc_blocks = model.encoder.blocks
    split = len(enc_blocks) - a.enc_layers
    to32 = lambda m: m.update(tree_map(lambda x: x.astype(mx.float32), m.parameters()))  # noqa: E731
    to32(model.decoder)
    for b in enc_blocks[split:]:
        to32(b)
    to32(model.encoder.ln_post)
    model.freeze()
    model.decoder.unfreeze()
    model.decoder.freeze(keys=["token_embedding", "positional_embedding"])
    for b in enc_blocks[split:]:
        b.unfreeze()
    model.encoder.ln_post.unfreeze()
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"parametri antrenați: {n_train / 1e6:.1f}M (decoder + ultimele {a.enc_layers} straturi ale encoderului)",
          file=sys.stderr)

    tok = get_tokenizer(True, num_languages=model.num_languages, language="ro", task="transcribe")
    gl = get_glossary()
    prompts = {l: tok.encode(" " + gl.whisper_prompt(l, "medical"))[-150:] for l in ("ro", "ru", "en")}
    sot = {l: [tok.sot, tok.special_tokens[f"<|{l}|>"], tok.transcribe, tok.no_timestamps] for l in ("ro", "ru", "en")}
    eot = tok.eot

    data, babble = load_data(a, rng)
    kinds = {k: sum(1 for d in data if d[0] == k) for k in ("xtts", "mac", "real")}
    print(f"date: {kinds}, voci de fundal: {len(babble)}", file=sys.stderr)
    rng.shuffle(data)
    n_val = min(120, len(data) // 20)
    val, data = data[:n_val], data[n_val:]

    def audio_of(item):
        return sf.read(item[1], dtype="float32")[0] if isinstance(item[1], Path) else item[1]

    def make_batch(items, train: bool):
        mels, seqs, starts = [], [], []
        for it in items:
            x = audio_of(it)[:N_SAMPLES]
            if train and a.aug > 0 and rng.random() > a.clean_prob:
                x = augment(x, nrng, babble, a.aug)
            elif not train:
                x = augment(x, np.random.default_rng(hash(it[2]) % 2**32), babble, a.aug)
            mels.append(log_mel_spectrogram(pad_or_trim(mx.array(x), N_SAMPLES), n_mels=model.dims.n_mels))
            lang = it[3]
            text_ids = tok.encode(" " + it[2].strip().lower())[:220]
            pre = ([tok.sot_prev] + prompts[lang] if (train and rng.random() < a.prompt_prob) else []) + sot[lang]
            seqs.append(pre + text_ids + [eot])
            starts.append(len(pre))
        L = -(-max(len(s) for s in seqs) // 32) * 32 + 1
        inp = np.full((len(seqs), L - 1), eot, dtype=np.int32)
        tgt = np.full((len(seqs), L - 1), eot, dtype=np.int32)
        mask = np.zeros((len(seqs), L - 1), dtype=np.float32)
        for i, (s, st) in enumerate(zip(seqs, starts)):
            inp[i, :len(s) - 1] = s[:-1]
            tgt[i, :len(s) - 1] = s[1:]
            mask[i, st - 1:len(s) - 1] = 1.0
        return mx.stack(mels).astype(mx.float16), mx.array(inp), mx.array(tgt), mx.array(mask)

    def frozen_encoder(mel):
        """Straturile înghețate ale encoderului, câte 2 clipuri, evaluate imediat (fără graf)."""
        e = model.encoder
        parts = []
        for c in range(0, mel.shape[0], 2):
            x = nn.gelu(e.conv1(mel[c:c + 2]))
            x = nn.gelu(e.conv2(x))
            x = x + e._positional_embedding
            for b in enc_blocks[:split]:
                x, _, _ = b(x)
            mx.eval(x)
            parts.append(x)
        return mx.stop_gradient(mx.concatenate(parts)).astype(mx.float32)

    def loss_fn(m, h, inp, tgt, mask):
        for b in m.encoder.blocks[split:]:
            h, _, _ = b(h)
        feats = m.encoder.ln_post(h)
        logits = m.decoder(inp, feats)[0].astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, tgt, reduction="none")
        return (ce * mask).sum() / mask.sum()

    def is_enc(path: str) -> bool:
        return path.startswith("encoder.")

    steps_total = a.epochs * (len(data) // (a.batch * a.accum))
    warm = min(60, max(1, steps_total // 10))
    sched = lambda base: optim.join_schedules([optim.linear_schedule(base / 100, base, warm),  # noqa: E731
                                             optim.cosine_decay(base, max(1, steps_total - warm))], [warm])
    opt_dec = optim.AdamW(learning_rate=sched(a.lr), weight_decay=0.01)
    opt_enc = optim.AdamW(learning_rate=sched(a.lr_enc), weight_decay=0.01)
    grad_fn = nn.value_and_grad(model, loss_fn)

    def evaluate() -> float:
        losses = []
        for b in range(0, len(val) - a.batch + 1, a.batch):
            mel, inp, tgt, mask = make_batch(val[b:b + a.batch], train=False)
            losses.append(loss_fn(model, frozen_encoder(mel), inp, tgt, mask).item())
            mx.clear_cache()
        return float(np.mean(losses)) if losses else float("nan")

    print(f"val loss inițial: {evaluate():.3f} ({n_val} clipuri, cu augmentare fixă)", file=sys.stderr, flush=True)
    step, t0, run = 0, time.time(), []
    t_win, s_win, best = t0, 0, float("inf")
    out = ROOT / a.out
    for ep in range(a.epochs):
        rng.shuffle(data)
        acc, n_acc = None, 0
        for b in range(0, len(data) - a.batch + 1, a.batch):
            mel, inp, tgt, mask = make_batch(data[b:b + a.batch], train=True)
            loss, grads = grad_fn(model, frozen_encoder(mel), inp, tgt, mask)
            if not np.isfinite(loss.item()):
                print(f"pas {step}: loss invalid, batch sărit", file=sys.stderr, flush=True)
                continue
            acc = grads if acc is None else tree_map(lambda x, y: x + y, acc, grads)
            n_acc += 1
            run.append(loss.item())
            mx.eval(acc)
            mx.clear_cache()
            if n_acc < a.accum:
                continue
            g = tree_map(lambda x: mx.clip(x / n_acc, -1.0, 1.0), acc)
            flat = dict(tree_flatten(g))
            from mlx.utils import tree_unflatten
            opt_dec.update(model, tree_unflatten([(k, v) for k, v in flat.items() if not is_enc(k)]))
            opt_enc.update(model, tree_unflatten([(k, v) for k, v in flat.items() if is_enc(k)]))
            mx.eval(model.trainable_parameters(), opt_dec.state, opt_enc.state)
            acc, n_acc = None, 0
            mx.clear_cache()
            step += 1
            if step % 25 == 0 or step <= 3:
                now = time.time()
                print(f"ep {ep + 1} pas {step}/{steps_total} loss {np.mean(run[-100:]):.3f} "
                      f"{(now - t_win) / max(1, step - s_win):.1f}s/pas vârf {mx.get_peak_memory() / 2**30:.1f}G",
                      file=sys.stderr, flush=True)
                t_win, s_win = now, step
            if a.max_steps and step >= a.max_steps:
                break
        vl = evaluate()
        print(f"== epoca {ep + 1}: val loss {vl:.3f} ({(time.time() - t0) / 60:.0f} min)", file=sys.stderr, flush=True)
        if vl < best:
            best = vl
            save(model, out, a, step, n_train, t0, vl, ep + 1)
        if a.max_steps and step >= a.max_steps:
            break
    print(f"cel mai bun val loss {best:.3f} -> {out}", file=sys.stderr)


def save(model, out: Path, a, step: int, n_train: int, t0: float, vl: float, ep: int) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    out.mkdir(parents=True, exist_ok=True)
    weights = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
    base_w = mx.load(str(ROOT / a.base / "weights.safetensors"))
    if "alignment_heads" in base_w:
        weights["alignment_heads"] = base_w["alignment_heads"]
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    shutil.copy(ROOT / a.base / "config.json", out / "config.json")
    (out / "TRAINING.json").write_text(json.dumps({**vars(a), "steps": step, "epoch": ep, "val_loss": round(vl, 4),
                                                   "trainable_M": round(n_train / 1e6, 1),
                                                   "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"salvat (val {vl:.3f}) -> {out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

"""Adaptare pe domeniul medical a Whisper-ului moldovenesc — antrenare locală pe GPU-ul Mac-ului (MLX).

Ce se antrenează: DOAR blocurile decoderului (partea de „limbaj”); encoderul („urechea” — robustețea acustică,
accentul moldovenesc învățat pe 70 h) și vocabularul de tokeni rămân înghețate. Astfel modelul învață cuvintele și
construcțiile medicale fără să-și piardă adaptarea la dialect.

Date:
  - fraze medicale sintetice (training/make_tts_medical.py) — SiMoNERo/MoNERo + glosar, cu zgomot și reverberație
  - vorbire moldovenească REALĂ din FraPiz (replay), ca modelul să nu uite dialectul și vorbirea spontană

    python -m training.finetune_decoder_mlx --base models/mlx-whisper-turbo-moldovan \
        --out models/mlx-whisper-turbo-moldovan-med --epochs 2
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


def load_data(n_real: int, real_shard: str):
    from training.eval_wer import load_clips

    items = []
    meta = ROOT / "data" / "tts_medical" / "metadata.jsonl"
    for line in open(meta, encoding="utf-8"):
        m = json.loads(line)
        items.append(("tts", ROOT / "data" / "tts_medical" / m["audio"], m["text"]))
    for a, text, _ in load_clips(real_shard, n_real, seed=1, max_dur=28):
        items.append(("real", a, text.lower()))
    return items


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
    ap.add_argument("--out", default="models/mlx-whisper-turbo-moldovan-med")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-real", type=int, default=1000)
    ap.add_argument("--real-shard", default="data/moldovan_corpus/train-00000.parquet")
    ap.add_argument("--prompt-prob", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--enc-chunk", type=int, default=2, help="clipuri trecute simultan prin encoder")
    ap.add_argument("--mem-limit-gb", type=float, default=8.0)
    a = ap.parse_args()
    rng = random.Random(0)
    try:  # pe 16 GB unificați, fără limită MLX împinge sistemul în swap și antrenarea devine de ~10x mai lentă
        mx.set_memory_limit(int(a.mem_limit_gb * 2**30))
    except AttributeError:
        mx.metal.set_memory_limit(int(a.mem_limit_gb * 2**30))
    mx.set_cache_limit(1 * 2**30)  # cache-ul de buffere altfel crește la fiecare pas (12 GB după 100 de pași)

    model = load_model(str(ROOT / a.base), dtype=mx.float16)
    # encoderul rămâne în fp16 (înghețat, doar forward); decoderul se antrenează în fp32
    model.decoder.update(tree_map(lambda x: x.astype(mx.float32), model.decoder.parameters()))
    model.freeze()
    model.decoder.unfreeze()
    model.decoder.freeze(keys=["token_embedding", "positional_embedding"])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"parametri antrenați: {n_train / 1e6:.1f}M (doar blocurile decoderului)", file=sys.stderr)

    tok = get_tokenizer(True, num_languages=model.num_languages, language="ro", task="transcribe")
    sot = list(tok.sot_sequence_including_notimestamps)
    prompt_ids = tok.encode(" " + get_glossary().whisper_prompt("ro", "medical"))[-150:]
    eot = tok.eot

    data = load_data(a.n_real, str(ROOT / a.real_shard))
    print(f"{sum(1 for d in data if d[0] == 'tts')} fraze medicale sintetice + "
          f"{sum(1 for d in data if d[0] == 'real')} fraze reale moldovenești", file=sys.stderr)

    def audio_of(item):
        kind, src, _ = item
        if kind == "tts":
            x, _ = sf.read(src, dtype="float32")
            return x
        return src

    def make_batch(items):
        mels, seqs, starts = [], [], []
        for it in items:
            x = audio_of(it)[:N_SAMPLES]
            mels.append(log_mel_spectrogram(pad_or_trim(mx.array(x), N_SAMPLES), n_mels=model.dims.n_mels))
            text_ids = tok.encode(" " + it[2].strip())[:200]
            pre = ([tok.sot_prev] + prompt_ids if rng.random() < a.prompt_prob else []) + sot
            seqs.append(pre + text_ids + [eot])
            starts.append(len(pre))
        L = -(-max(len(s) for s in seqs) // 32) * 32 + 1  # lungimi fixe (multipli de 32): mai puține forme noi de buffere
        inp = np.full((len(seqs), L - 1), eot, dtype=np.int32)
        tgt = np.full((len(seqs), L - 1), eot, dtype=np.int32)
        mask = np.zeros((len(seqs), L - 1), dtype=np.float32)
        for i, (s, st) in enumerate(zip(seqs, starts)):
            inp[i, :len(s) - 1] = s[:-1]
            tgt[i, :len(s) - 1] = s[1:]
            mask[i, st - 1:len(s) - 1] = 1.0  # pierderea doar pe textul transcris (+ EOT), nu pe prompt
        return mx.stack(mels).astype(mx.float16), mx.array(inp), mx.array(tgt), mx.array(mask)

    def loss_fn(m, feats, inp, tgt, mask):
        logits = m.decoder(inp, feats)[0].astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, tgt, reduction="none")
        return (ce * mask).sum() / mask.sum()

    steps_total = a.epochs * (len(data) // a.batch)
    sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, 50),
                                  optim.cosine_decay(a.lr, max(1, steps_total - 50))], [50])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    grad_fn = nn.value_and_grad(model, loss_fn)

    step, t0 = 0, time.time()
    t_win, s_win = t0, 0
    for ep in range(a.epochs):
        rng.shuffle(data)
        run = []
        for b in range(0, len(data) - a.batch + 1, a.batch):
            mel, inp, tgt, mask = make_batch(data[b:b + a.batch])
            # encoderul pe bucăți mici, evaluat imediat: graful lui (atenție 1500x1500 pe 32 de straturi) nu mai
            # stă în memorie în timpul backward-ului decoderului
            parts = []
            for c in range(0, mel.shape[0], a.enc_chunk):
                f = model.encoder(mel[c:c + a.enc_chunk])
                mx.eval(f)
                parts.append(f)
            feats = mx.stop_gradient(mx.concatenate(parts)).astype(mx.float32)
            mx.eval(feats)
            loss, grads = grad_fn(model, feats, inp, tgt, mask)
            if not np.isfinite(loss.item()):
                bad = [(d[0], d[2][:60]) for d in data[b:b + a.batch]]
                print(f"pas {step}: loss invalid, batch sărit: feats_nan={mx.isnan(feats).any().item()} {bad}",
                      file=sys.stderr, flush=True)
                continue
            grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state, loss)
            run.append(loss.item())
            mx.clear_cache()
            step += 1
            t_last = time.time()
            if a.max_steps and step >= a.max_steps:
                break
            if step % 25 == 0 or step <= 5:
                print(f"ep {ep + 1} pas {step}/{steps_total} loss {np.mean(run[-25:]):.3f} "
                      f"{(t_last - t_win) / max(1, step - s_win):.2f}s/pas mem activ {mx.get_active_memory() / 2**30:.1f}G "
                      f"cache {mx.get_cache_memory() / 2**30:.1f}G vârf {mx.get_peak_memory() / 2**30:.1f}G",
                      file=sys.stderr, flush=True)
                t_win, s_win = t_last, step
        try:
            mx.clear_cache()
        except AttributeError:
            mx.metal.clear_cache()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    weights = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
    base_w = mx.load(str(ROOT / a.base / "weights.safetensors"))
    if "alignment_heads" in base_w:
        weights["alignment_heads"] = base_w["alignment_heads"]
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    shutil.copy(ROOT / a.base / "config.json", out / "config.json")
    (out / "TRAINING.json").write_text(json.dumps({**vars(a), "steps": step, "trainable_M": round(n_train / 1e6, 1),
                                                   "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"salvat -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
